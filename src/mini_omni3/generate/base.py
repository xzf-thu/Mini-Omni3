"""Streaming offline inference primitives for the audio-enhanced GPT.

The model alternates between two states inside `streaming_generate`:
  - LISTENING: each step consumes one encoder-output chunk of audio. The model
    emits either KEEP_SILENCE (keep listening) or TEXT_BEGIN (start replying).
  - SPEAKING:  autoregressive text generation until TEXT_END, then back to
    LISTENING for the next audio chunk.

Public surface (used by `infer.py` at the repo root and by `web/server.py`):
    SYSTEM_PROMPT, AUDIO_TOKENS_PER_CHUNK
    sample, encode_audio_chunks, streaming_generate
    set_seed, load_model, load_audio_encoder
    run_inference (end-to-end entry point)
"""

import random
from pathlib import Path
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import whisper
from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration

from mini_omni3.dataset.TOKENS import (
    ASSISTANT, AUDIO_BEGIN, ENGLISH, KEEP_SILENCE, ONLINE, PAD,
    SYSTEM, TEXT_BEGIN, TEXT_END,
)
from mini_omni3.model import GPT, Config
from mini_omni3.tokenizer import Tokenizer
from mini_omni3.utils import get_default_supported_precision, load_checkpoint


SYSTEM_PROMPT = (
    "You are a helpful assistant. When there is no user text, if the audio contains a question, "
    "please answer it. If it is a sound effect, determine based on the sound whether help is needed."
)

# Encoder-output frames per [AUDIO_BEGIN, PAD*N, ASSISTANT, ...] block.
AUDIO_TOKENS_PER_CHUNK = 10


# === Sampling ===

def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=False)
    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumprobs <= (1 - top_p)
    remove[-1:] = 0  # always keep the most probable token
    return logits.masked_fill(remove.scatter(0, sorted_idx, remove), float("-inf"))


def sample(logits: torch.Tensor, *, temperature=1.0, top_k=None, top_p=1.0) -> torch.Tensor:
    """Sample one token id from the last position of `logits` ([1, T, V])."""
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    logits = logits[0, -1]
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    if temperature <= 0.0 and top_p <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if temperature > 0.0:
        logits = logits / temperature
    if top_p < 1.0:
        logits = _top_p_filter(logits, top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# === Audio feature extraction ===

def _split_into_chunks(n: int, chunk_size: int) -> List[int]:
    chunks = [chunk_size] * (n // chunk_size)
    if n % chunk_size:
        chunks.append(n % chunk_size)
    return chunks


def encode_audio_chunks(audio_path: str, audio_encoder: torch.nn.Module, device) -> List[torch.Tensor]:
    """Run the audio_tower on `audio_path` and split the output into AUDIO_TOKENS_PER_CHUNK chunks."""
    audio = whisper.load_audio(audio_path, sr=16000).tolist()
    # Pad to a 0.4-s boundary (6400 samples @ 16 kHz).
    if len(audio) % 6400 != 0:
        audio += [0] * (6400 - len(audio) % 6400)
    mel = whisper.log_mel_spectrogram(np.array(audio, dtype=np.float32), n_mels=128)
    len_feature = mel.shape[1]

    with torch.no_grad():
        feat = audio_encoder(
            torch.tensor(mel).to(device),
            torch.tensor(_split_into_chunks(len_feature, 40)).to(device),
            torch.tensor((len_feature - 1) // 2 + 1).to(device),
        ).last_hidden_state

    # Drop any trailing partial chunk so each chunk is exactly AUDIO_TOKENS_PER_CHUNK frames.
    keep = feat.shape[0] - feat.shape[0] % AUDIO_TOKENS_PER_CHUNK
    return [feat[i: i + AUDIO_TOKENS_PER_CHUNK] for i in range(0, keep, AUDIO_TOKENS_PER_CHUNK)]


# === Streaming generation ===

def _forward(model, tokens, input_pos, *, input_pos_maxp1, audio_feat):
    return model(
        tokens, None, 1, audio_feat, input_pos,
        input_pos_maxp1=input_pos_maxp1,
        audio_tokens_per_chunk=AUDIO_TOKENS_PER_CHUNK,
    )


def _init_input_pos_maxp1(model, prompt_size, device):
    # input_pos_maxp1 introduces data-dependent shapes; skip if a Thunder module is involved.
    if any(m.__class__.__name__ == "ThunderModule" for m in model.modules()):
        return None
    return torch.tensor(prompt_size, device=device)


def _append_listening_block(token, input_pos, input_pos_maxp1, device):
    """Append `[AUDIO_BEGIN, PAD*N, ASSISTANT]` to the running context."""
    new_tokens = torch.LongTensor([AUDIO_BEGIN] + [PAD] * AUDIO_TOKENS_PER_CHUNK + [ASSISTANT]).to(device)
    new_positions = input_pos[-1] + torch.arange(1, len(new_tokens) + 1, device=device)
    token = torch.cat([token, new_tokens])
    input_pos = torch.cat([input_pos, new_positions])
    if input_pos_maxp1 is not None:
        input_pos_maxp1.add_(len(new_tokens))
    return token, input_pos, input_pos_maxp1


def _advance_one(input_pos, input_pos_maxp1):
    """Move input_pos forward by one (for the next single-token call)."""
    new_pos = input_pos[-1].unsqueeze(0).add_(1)
    if input_pos_maxp1 is not None:
        input_pos_maxp1.add_(1)
    return new_pos, input_pos_maxp1


def streaming_generate(
    model: GPT,
    audio_encoder: torch.nn.Module,
    tokenizer: Tokenizer,
    prefix_ids: torch.Tensor,
    *,
    rounds: int = 10,
    audio_paths: Optional[List[str]] = None,
    max_returned_tokens: int = 4096,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
):
    """Stream audio→text. If `audio_paths` is given, run one round per path
    non-interactively (offline); otherwise prompt stdin each round (online)."""
    device = prefix_ids.device
    token = prefix_ids
    input_pos = torch.arange(0, prefix_ids.size(0), device=device, dtype=torch.int64)
    input_pos_maxp1 = _init_input_pos_maxp1(model, prefix_ids.size(0), device)

    turns: List[List[int]] = []  # one inner list per assistant turn, starting at TEXT_BEGIN
    n_rounds = len(audio_paths) if audio_paths is not None else rounds

    for round_idx in range(n_rounds):
        if audio_paths is not None:
            audio_path = audio_paths[round_idx]
            print(f"Round {round_idx} — audio: {audio_path}")
        else:
            audio_path = input(f"Round {round_idx} — enter audio path: ").strip()
        audio_chunks = encode_audio_chunks(audio_path, audio_encoder, device)
        print(f"[{len(audio_chunks)} audio chunks]")

        listening, audio_idx = True, -1
        for _ in range(max_returned_tokens - input_pos.numel()):
            if listening:
                audio_idx += 1
                if audio_idx >= len(audio_chunks):
                    break
                token, input_pos, input_pos_maxp1 = _append_listening_block(
                    token, input_pos, input_pos_maxp1, device
                )
                logits = _forward(
                    model, token.view(1, -1), input_pos,
                    input_pos_maxp1=input_pos_maxp1,
                    audio_feat=audio_chunks[audio_idx].to(device),
                )
            else:
                logits = _forward(
                    model, token.view(1, -1), input_pos,
                    input_pos_maxp1=input_pos_maxp1,
                    audio_feat=None,
                )

            token = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p).to(torch.int64)
            int_token = token.item()
            input_pos, input_pos_maxp1 = _advance_one(input_pos, input_pos_maxp1)

            if listening:
                if int_token == TEXT_BEGIN:
                    listening = False
                    turns.append([int_token])
                elif int_token == KEEP_SILENCE:
                    turns.append([int_token])
                else:
                    raise ValueError(f"Unexpected token {int_token} while listening")
            else:
                turns[-1].append(int_token)
                if int_token == TEXT_END:
                    listening = True

        # Decode and print this round's assistant turns.
        # Each turn looks like [TEXT_BEGIN, EMOTION, ...text..., TEXT_END]; strip both ends.
        # In offline mode (audio_paths supplied) also print silent decisions so the
        # demo viewer sees the model's "no reply" outcomes explicitly.
        for piece_idx, turn in enumerate(turns, start=1):
            if turn[0] == TEXT_BEGIN:
                decoded = tokenizer.decode(torch.tensor(turn[2:-1]))
                if decoded:
                    print(f"\n=== Audio piece {piece_idx} ===\n{decoded}\n")
            elif audio_paths is not None:
                print(f"\n=== Audio piece {piece_idx} === (silent — kept listening)\n")

    return turns


# === Setup ===

def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(fabric, model_config_dir, trained_checkpoint):
    config = Config.from_file(Path(model_config_dir) / "model_config.yaml")
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
    model = fabric.setup(model)
    load_checkpoint(fabric, model, trained_checkpoint, strict=True)
    return model


def load_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, device):
    # Instantiate via the full Omni model and pluck out thinker.audio_tower,
    # then load the wrapped audio_tower ckpt (proj.* baked in from the trained
    # audio_adapter.*; see finetune/wrap_audio_tower.py).
    cfg = AutoConfig.from_pretrained(qwen_omni_ckpt)
    encoder = Qwen2_5OmniForConditionalGeneration._from_config(cfg).thinker.audio_tower
    encoder.load_state_dict(torch.load(audio_tower_ckpt, map_location=device))
    encoder.to(device).requires_grad_(False).eval()
    return encoder


def resolve_checkpoint_paths(checkpoint_dir: str):
    """Map a single checkpoint root → (model_config_dir, trained_checkpoint,
    qwen_omni_ckpt, audio_tower_ckpt). The release layout is:

        <checkpoint_dir>/
            model_config.yaml + tokenizer.json + ...   ← model_config_dir = root
            MiniOmni3_LM.pt
            MiniOmni3_ChunkwisedEncoder.pth
            qwen_2_5_omni_config/
    """
    ckpt = Path(checkpoint_dir)
    return (
        str(ckpt),
        str(ckpt / "MiniOmni3_LM.pt"),
        str(ckpt / "qwen_2_5_omni_config"),
        str(ckpt / "MiniOmni3_ChunkwisedEncoder.pth"),
    )


def run_inference(
    *,
    checkpoint_dir: str,
    rounds: int = 10,
    audio_paths: Optional[List[str]] = None,
    seed: int = 1337,
    max_new_tokens: int = 4096,
    device: str = "cuda:0",
):
    """End-to-end: build fabric, load model + audio encoder, run streaming_generate.

    If `audio_paths` is given, runs one round per path non-interactively
    (offline mode). Otherwise prompts stdin each round (online mode).
    """
    if not checkpoint_dir:
        raise RuntimeError("`checkpoint_dir` is empty — set it before calling run_inference().")
    model_config_dir, trained_checkpoint, qwen_omni_ckpt, audio_tower_ckpt = \
        resolve_checkpoint_paths(checkpoint_dir)

    set_seed(seed)
    fabric = L.Fabric(
        devices=1, num_nodes=1, strategy="auto",
        precision=get_default_supported_precision(training=False),
        loggers="tensorboard",
    )
    model = load_model(fabric, model_config_dir, trained_checkpoint)
    audio_encoder = load_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, device)
    tokenizer = Tokenizer(model_config_dir)

    system_ids = tokenizer.encode(SYSTEM_PROMPT).cpu().tolist()
    prefix_ids = torch.LongTensor(
        [ONLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + system_ids + [TEXT_END]
    ).to(model.device)

    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    model.eval()
    try:
        with torch.inference_mode():
            return streaming_generate(
                model, audio_encoder, tokenizer, prefix_ids,
                rounds=rounds, audio_paths=audio_paths,
                max_returned_tokens=max_new_tokens,
            )
    finally:
        model.clear_kv_cache()
