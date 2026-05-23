"""Mini-Omni3 streaming web backend.

Per-frame flow:
    frontend mic --[400ms int16 mono PCM @ 16kHz]--> POST /audio
        pcm bytes -> float32[6400] (padded) -> log-mel -> audio_tower
                  -> last (10, n_embd) feat chunk
        -> InferenceSession.step_audio_frame() -> one sampled token
            KEEP_SILENCE  : do nothing, await next frame
            TEXT_BEGIN    : tell frontend to stop sending; run continuous
                            text-only decode loop; stream each detokenized
                            piece over SSE; send <|end|> on TEXT_END;
                            reset KV cache; tell frontend it's ready again.

The model and audio_tower are loaded once at startup and held as singletons.

Endpoints:
    GET  /        -> serve the frontend HTML
    POST /audio   -> body is one 400ms 16-bit-LE mono PCM frame
    POST /audio_end -> no-op signal that mic detected silence; the model
                       decides on its own when to BREAK_SILENCE
    GET  /stream  -> SSE stream of decoded text pieces + control markers
                     (<|busy|> when model starts speaking, <|end|> when done)
"""

import os
import queue
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import whisper
from flask import Flask, Response, request, send_from_directory

from mini_omni3.dataset.TOKENS import (
    ASSISTANT, AUDIO_BEGIN, ENGLISH, KEEP_SILENCE, ONLINE, PAD,
    SYSTEM, TEXT_BEGIN, TEXT_END,
)
from mini_omni3.generate.base import (
    AUDIO_TOKENS_PER_CHUNK, SYSTEM_PROMPT,
    load_audio_encoder, load_model, resolve_checkpoint_paths, sample, set_seed,
)
from mini_omni3.tokenizer import Tokenizer
from mini_omni3.utils import get_default_supported_precision


# Single source of truth for all weights — see README for the expected layout.
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoint"
MODEL_CONFIG_DIR, TRAINED_CHECKPOINT, QWEN_OMNI_CKPT, AUDIO_TOWER_CKPT = \
    resolve_checkpoint_paths(str(CHECKPOINT_DIR))


# === Audio framing constants (must match what the frontend sends) ===
SAMPLE_RATE   = 16000
SAMPLE_WIDTH  = 2                                   # int16 -> 2 bytes
CHANNELS      = 1
FRAME_SAMPLES = SAMPLE_RATE * 400 // 1000           # 6400 samples = 400 ms
MAX_TEXT_TOKENS = 512
DEVICE        = os.environ.get("DEVICE", "cuda:0")

# Where to drop one folder of per-frame wavs per utterance (for debugging).
RECORDINGS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "recordings"

HOST          = "0.0.0.0"
PORT          = int(os.environ.get("PORT", "5001"))
FRONTEND_FILE = "mini_omni3.html"
END_TOKEN_STR = "<|end|>"
BUSY_TOKEN_STR = "<|busy|>"


# === Audio: PCM -> log-mel -> audio_tower -> 10-frame feature chunk ===

# Accumulated PCM history for the current utterance. We re-run the encoder on
# the FULL accumulated audio each frame so STFT windows at chunk boundaries
# see real neighbouring samples (not zeros), then take only the newest 10
# feature frames. This matches offline encoding behavior.
_pcm_history: List[np.ndarray] = []


def reset_pcm_history() -> None:
    global _pcm_history
    _pcm_history = []


def pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """int16-LE mono PCM -> float32 in [-1, 1], padded/truncated to FRAME_SAMPLES."""
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(pcm) < FRAME_SAMPLES:
        pcm = np.pad(pcm, (0, FRAME_SAMPLES - len(pcm)))
    elif len(pcm) > FRAME_SAMPLES:
        pcm = pcm[:FRAME_SAMPLES]
    return pcm


def save_pcm_as_wav(pcm_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pad to exactly FRAME_SAMPLES so every wav is 400 ms.
    expected = FRAME_SAMPLES * SAMPLE_WIDTH
    if len(pcm_bytes) < expected:
        pcm_bytes = pcm_bytes + b"\x00" * (expected - len(pcm_bytes))
    elif len(pcm_bytes) > expected:
        pcm_bytes = pcm_bytes[:expected]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def encode_frame(pcm_f32: np.ndarray) -> torch.Tensor:
    """Append pcm_f32 to history, re-encode the full accumulated audio,
    return the newest AUDIO_TOKENS_PER_CHUNK features as (10, n_embd)."""
    _pcm_history.append(pcm_f32)
    full_audio = np.concatenate(_pcm_history)
    mel = whisper.log_mel_spectrogram(full_audio, n_mels=128)
    mel_t = mel if isinstance(mel, torch.Tensor) else torch.from_numpy(mel)
    mel_t = mel_t.to(DEVICE)

    T = mel_t.shape[-1]
    chunk_size = 40                                   # mel frames per 400 ms
    n_chunks = T // chunk_size
    feature_lens = torch.tensor([chunk_size] * n_chunks, device=DEVICE)
    aftercnn_lens = torch.tensor((T - 1) // 2 + 1, device=DEVICE)

    with torch.no_grad():
        feat = audio_encoder(mel_t, feature_lens, aftercnn_lens).last_hidden_state

    return feat[-AUDIO_TOKENS_PER_CHUNK:]              # (10, n_embd)


# === Boot: load model + audio encoder once ===

set_seed(1337)

print(f"[boot] device={DEVICE}")
fabric = L.Fabric(
    devices=1, num_nodes=1, strategy="auto",
    precision=get_default_supported_precision(training=False),
)

print(f"[boot] loading model from {TRAINED_CHECKPOINT}")
model = load_model(fabric, MODEL_CONFIG_DIR, TRAINED_CHECKPOINT)

print(f"[boot] loading audio_tower from {AUDIO_TOWER_CKPT}")
audio_encoder = load_audio_encoder(QWEN_OMNI_CKPT, AUDIO_TOWER_CKPT, DEVICE)

tokenizer = Tokenizer(MODEL_CONFIG_DIR)
print("[boot] model + audio encoder ready")


# === Streaming inference session (holds KV cache + rolling state) ===

class InferenceSession:
    """Tracks the model's KV cache + input_pos cursor across an utterance.

    One utterance = listening loop (one model step per audio frame) followed by
    a speaking loop (autoregressive token generation until TEXT_END).
    """

    def __init__(self):
        self.device = torch.device(DEVICE)
        self.token: Optional[torch.Tensor] = None
        self.input_pos: Optional[torch.Tensor] = None
        self.input_pos_maxp1: Optional[torch.Tensor] = None
        self.listening = True
        self._emitted_len = 0
        # Per-utterance recording state.
        self.session_dir: Optional[Path] = None
        self.frame_idx: int = 0
        self._init_kv_and_prompt()

    def _init_kv_and_prompt(self) -> None:
        with fabric.init_tensor():
            model.set_kv_cache(batch_size=1)
        model.eval()

        sys_ids = tokenizer.encode(SYSTEM_PROMPT).cpu().tolist()
        prefix = [ONLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + sys_ids + [TEXT_END]
        token = torch.LongTensor(prefix).to(self.device)
        prompt_size = token.size(0)

        # The first audio step prefills [prompt | AUDIO_BEGIN | pads | ASSISTANT].
        self.token = token
        self.input_pos = torch.arange(0, prompt_size, device=self.device, dtype=torch.int64)
        self.input_pos_maxp1 = torch.tensor(prompt_size, device=self.device)
        self.listening = True
        self._emitted_len = 0
        self.session_dir = None
        self.frame_idx = 0

    def reset(self) -> None:
        try:
            model.clear_kv_cache()
        except Exception:
            pass
        reset_pcm_history()
        self._init_kv_and_prompt()

    def ensure_session_dir(self) -> Path:
        if self.session_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            self.session_dir = RECORDINGS_DIR / ts
            self.session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_dir

    def _forward_sample(self, audio_feat: Optional[torch.Tensor]) -> int:
        """Run one model forward + sample step. Returns the sampled token id."""
        logits = model(
            self.token.view(1, -1), None, 1, audio_feat, self.input_pos,
            input_pos_maxp1=self.input_pos_maxp1,
            audio_tokens_per_chunk=AUDIO_TOKENS_PER_CHUNK,
        )
        nxt = sample(logits).to(torch.int64)
        return int(nxt.item()), nxt

    @torch.inference_mode()
    def step_audio_frame(self, audio_feat: torch.Tensor) -> int:
        """Append one [AUDIO_BEGIN, PAD*10, ASSISTANT] block and sample one token.
        Returns the sampled token id; flips self.listening if TEXT_BEGIN arrives.
        """
        if not self.listening:
            raise RuntimeError("step_audio_frame called while not listening")

        new_tokens = torch.LongTensor(
            [AUDIO_BEGIN] + [PAD] * AUDIO_TOKENS_PER_CHUNK + [ASSISTANT]
        ).to(self.device)
        self.token = torch.cat((self.token, new_tokens))

        added = 2 + AUDIO_TOKENS_PER_CHUNK
        last = self.input_pos[-1]
        new_pos = torch.tensor([last + i for i in range(1, added + 1)], device=self.device)
        self.input_pos = torch.cat((self.input_pos, new_pos))
        self.input_pos_maxp1.add_(added)

        int_token, nxt = self._forward_sample(audio_feat.to(self.device))
        # Advance one position for the next forward.
        self.input_pos = self.input_pos[-1].unsqueeze(0).add_(1)
        self.input_pos_maxp1.add_(1)
        self.token = nxt

        if int_token == TEXT_BEGIN:
            self.listening = False
        elif int_token != KEEP_SILENCE:
            print(f"[warn] unexpected token while listening: {int_token}")
        return int_token

    @torch.inference_mode()
    def stream_text(self, on_piece, max_tokens: int = MAX_TEXT_TOKENS) -> None:
        """Autoregressive text decoding; calls on_piece(str) for each new chunk."""
        produced: List[int] = []
        for _ in range(max_tokens):
            int_token, nxt = self._forward_sample(None)
            self.input_pos.add_(1)
            self.input_pos_maxp1.add_(1)
            self.token = nxt

            if int_token == TEXT_END:
                self.listening = True
                return

            produced.append(int_token)
            # Detokenize the whole buffer and emit only the new suffix to avoid
            # broken bytes when BPE splits a multi-byte char across tokens.
            text_so_far = tokenizer.decode(torch.tensor(produced))
            piece = text_so_far[self._emitted_len:]
            self._emitted_len = len(text_so_far)
            if piece:
                on_piece(piece)


# === Flask + SSE plumbing ===

app = Flask(__name__)

session: Optional[InferenceSession] = None
session_lock = threading.Lock()

# True when the frontend may send audio frames (i.e. not mid-text-decode).
ready_flag = threading.Event()
ready_flag.set()

sse_clients: List[queue.Queue] = []
sse_lock = threading.Lock()


def sse_broadcast(text: str) -> None:
    with sse_lock:
        for q in list(sse_clients):
            try:
                q.put_nowait(text)
            except Exception:
                pass


def run_text_decode_in_thread() -> None:
    try:
        with session_lock:
            session.stream_text(sse_broadcast)
    except Exception as e:
        print(f"[error] text decode crashed: {e}")
    finally:
        sse_broadcast(END_TOKEN_STR)
        with session_lock:
            session.reset()
        ready_flag.set()


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), FRONTEND_FILE)


@app.route("/audio", methods=["POST"])
def audio():
    if not ready_flag.is_set():
        return ("not ready", 409)

    pcm = request.get_data()
    if not pcm:
        return ("empty", 400)

    # Persist the raw 400 ms frame for debugging.
    if session is not None:
        sdir = session.ensure_session_dir()
        session.frame_idx += 1
        try:
            save_pcm_as_wav(pcm, sdir / f"{session.frame_idx}.wav")
        except Exception as e:
            print(f"[warn] failed to save wav: {e}")

    pcm_f32 = pcm_bytes_to_float32(pcm)
    audio_feat = encode_frame(pcm_f32)

    with session_lock:
        tok = session.step_audio_frame(audio_feat)

    if tok == TEXT_BEGIN:
        ready_flag.clear()
        sse_broadcast(BUSY_TOKEN_STR)
        threading.Thread(target=run_text_decode_in_thread, daemon=True).start()
    return ("", 204)


@app.route("/audio_end", methods=["POST"])
def audio_end():
    """The frontend reports trailing silence. The model decides on its own when
    to start speaking — this endpoint just absorbs the signal."""
    return ("", 204)


@app.route("/stream")
def stream():
    q: queue.Queue = queue.Queue()
    with sse_lock:
        sse_clients.append(q)

    def gen():
        try:
            yield ": connected\n\n"
            while True:
                try:
                    msg = q.get(timeout=15)
                    safe = msg.replace("\r", "").replace("\n", "\\n")
                    yield f"data: {safe}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    print("[boot] creating inference session")
    session = InferenceSession()
    print(f"Mini-Omni3 backend on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True, debug=False)
