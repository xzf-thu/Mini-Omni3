"""Build online (streaming) SFT samples in 4 inspectable steps.

Pipeline:
  step1 — sample a leading-silence length per turn + a tail silence
          (encoder-output frames). No audio touched.
  step2 — load every turn's audio, prepend the sampled silence, concat into
          one waveform, write `<wavs_dir>/<idx>.wav` (16 kHz mono int16).
  step3 — run the Qwen2.5-Omni audio_tower on each wav and dump
          `<features_dir>/<idx>/AudioFeat.pt`.
  step4 — lay out the token-level streaming sequence (input_ids / labels /
          audio_pos) and write a training-ready jsonl.

Edit the constants below, then `python cons_online_data.py`. Each step's
intermediate jsonl is left on disk so you can re-run any step independently
or inspect the artifacts.
"""

import glob
import json
import os
import random
import wave

import numpy as np
import soundfile as sf
import whisper
from tqdm import tqdm

from mini_omni3.dataset.utils.extract_online_feature import extract_audio_features
from mini_omni3.dataset.utils.load_audio import SAMPLES_PER_FRAME, _load_mel
from mini_omni3.dataset.TOKENS import (
    ASSISTANT, AUDIO_BEGIN, EMOTION_TO_ID, ENGLISH, KEEP_SILENCE, MASK,
    NORMAL, ONLINE, PAD, SYSTEM, TEXT_BEGIN, TEXT_END,
)
from mini_omni3.generate.base import resolve_checkpoint_paths
from mini_omni3.tokenizer import Tokenizer


# ============================================================
# Fill these in before running.
# ============================================================
CHECKPOINT_DIR  = ""    # checkpoint root (tokenizer + qwen_2_5_omni_config + MiniOmni3_ChunkwisedEncoder.pth)
INPUT_JSONL     = ""    # user-supplied raw jsonl
WORK_DIR        = ""    # holds intermediate step1/2/3 jsonls + wavs/ + features/
OUT_TRAIN_JSONL = ""    # final training-ready jsonl
NOISE_DIR       = ""    # dir of background-noise audio files (wav/flac/ogg/mp3); recursed

MIN_NOISE_LEN = 20      # min leading/tail noise length per turn (encoder frames)
MAX_NOISE_LEN = 60      # max leading/tail noise length per turn
CHUNK_SIZE    = 10      # encoder-output frames per audio chunk
SEED          = 1337
DEVICE        = "cuda"
# ============================================================


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. When there is no user text, if the audio contains a question, "
    "please answer it. If it is a sound effect, determine based on the sound whether help is needed."
)

NO_RESPONSE_MARKERS = {"<no need to response>", "no need to response"}


# === Step 1 — sample silence (filled from a noise library) ===

NOISE_AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3")


def _scan_noise_dir(noise_dir):
    """Walk noise_dir and return [(path, duration_s), ...] for every audio file."""
    paths = []
    for root, _, files in os.walk(noise_dir):
        for f in files:
            if f.lower().endswith(NOISE_AUDIO_EXTS):
                paths.append(os.path.join(root, f))
    index = []
    for p in tqdm(paths, desc="scan noise"):
        try:
            index.append((p, sf.info(p).duration))
        except Exception:
            # sf.info won't read mp3; fall back to librosa.
            try:
                import librosa
                index.append((p, librosa.get_duration(path=p)))
            except Exception as e:
                print(f"[warn] skip noise {p}: {e}")
    return index


def _pick_noise(noise_index, needed_s, *, margin_s=0.0, max_tries=100):
    """Pick a random noise file. If the draw is shorter than needed, re-draw.
    Returns (path, start_s) — a slice of `needed_s + margin_s` is guaranteed to fit."""
    budget = needed_s + margin_s
    for _ in range(max_tries):
        path, dur = random.choice(noise_index)
        if dur >= budget:
            start_s = random.uniform(0.0, dur - budget)
            return path, round(start_s, 3)
    raise RuntimeError(
        f"Couldn't find a noise file long enough for {budget:.2f}s after {max_tries} tries — "
        f"add longer files to NOISE_DIR."
    )


def _extract_turns(data_item):
    """Return list of {audio_path, assistant, emotion} dicts."""
    convs = data_item.get("conversation", [])
    if convs:
        return [
            {"audio_path": c["audio_path"],
             "assistant":  c["assistant"],
             "emotion":    c.get("emotion") or "normal"}
            for c in convs
        ]
    if "merge_path" in data_item and "assistant" in data_item:
        return [{
            "audio_path": data_item["merge_path"],
            "assistant":  data_item["assistant"],
            "emotion":    data_item.get("emotion", "normal"),
        }]
    raise ValueError("missing 'conversation' or single-turn fields")


def step1_sample_silence(input_jsonl, output_jsonl, *,
                         noise_dir, min_noise_len, max_noise_len, chunk_size, seed):
    """For each turn, pick a leading noise slice (file + start_s); also one for tail.

    The tail length is adjusted to a chunk boundary inside step 2, so its noise
    slice is selected with a `chunk_size`-frame margin to guarantee the eventual
    actual length fits in the picked file.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)
    noise_index = _scan_noise_dir(noise_dir)
    if not noise_index:
        raise RuntimeError(f"No audio files found under {noise_dir}")
    tail_margin_s = chunk_size * 0.04   # one chunk = 10 frames × 40 ms

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin.readlines(), desc="step1")):
            try:
                data_item = json.loads(line)
                turns = _extract_turns(data_item)
                for t in turns:
                    t["leading_silence_frames"] = random.randint(min_noise_len, max_noise_len)
                    needed_s = t["leading_silence_frames"] * 0.04
                    t["leading_noise_path"], t["leading_noise_start_s"] = \
                        _pick_noise(noise_index, needed_s)
                tail_frames = random.randint(min_noise_len, max_noise_len)
                tail_path, tail_start_s = _pick_noise(
                    noise_index, tail_frames * 0.04, margin_s=tail_margin_s,
                )
                rec = {
                    "idx": idx,
                    "turns": turns,
                    "tail_silence_frames": tail_frames,
                    "tail_noise_path": tail_path,
                    "tail_noise_start_s": tail_start_s,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[step1 idx {idx}] {type(e).__name__}: {e}")


# === Step 2 — concatenate audio per sample ===

def _load_audio_aligned(audio_path):
    """Load audio @ 16 kHz, pad/crop to (output_len * SAMPLES_PER_FRAME) samples.
    Returns (np.float32 array, output_len_frames)."""
    audio, _, _, _, output_len = _load_mel(audio_path)
    arr = np.asarray(audio, dtype=np.float32)
    target = output_len * SAMPLES_PER_FRAME
    if len(arr) < target:
        arr = np.concatenate([arr, np.zeros(target - len(arr), dtype=np.float32)])
    elif len(arr) > target:
        max_start = (len(arr) - target) // SAMPLES_PER_FRAME
        start = random.randint(0, max_start) * SAMPLES_PER_FRAME
        arr = arr[start: start + target]
    return arr, output_len


def _load_noise_segment(noise_path, start_s, n_samples):
    """Pull exactly n_samples from noise_path starting at start_s (16 kHz, mono float).
    Raises if the file actually yields fewer samples than asked (step 1 chose this
    file based on metadata duration — caller should treat it as a data error)."""
    full = whisper.load_audio(noise_path, sr=16000)   # float32 in [-1, 1]
    start_idx = int(round(start_s * 16000))
    seg = full[start_idx: start_idx + n_samples]
    if len(seg) < n_samples:
        raise ValueError(
            f"Noise {noise_path} only yields {len(seg)} samples from start {start_s}s, "
            f"need {n_samples}."
        )
    return seg


def _write_wav(path, float_samples):
    """Write a 16 kHz mono int16 wav from float samples in [-1, 1]."""
    arr = np.asarray(float_samples, dtype=np.float32)
    arr_i16 = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(arr_i16.tobytes())


def step2_concat_audio(input_jsonl, output_jsonl, wavs_dir, *, chunk_size, seed):
    """For each step-1 record, splice [noise → audio → noise → audio → ... → tail noise]
    into one wav using the noise slices step 1 picked. Records audio_frames per turn."""
    random.seed(seed)
    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)
    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in tqdm(fin.readlines(), desc="step2"):
            rec = json.loads(line)
            try:
                segments = []
                total_frames = 0
                for t in rec["turns"]:
                    n = t["leading_silence_frames"] * SAMPLES_PER_FRAME
                    segments.append(_load_noise_segment(
                        t["leading_noise_path"], t["leading_noise_start_s"], n,
                    ))
                    seg, n_frames = _load_audio_aligned(t["audio_path"])
                    segments.append(seg)
                    t["audio_frames"] = n_frames
                    total_frames += t["leading_silence_frames"] + n_frames

                # Round tail length so the total lands on a chunk boundary.
                tail = rec["tail_silence_frames"] - (total_frames + rec["tail_silence_frames"]) % chunk_size
                if tail < 0:
                    tail = (chunk_size - (total_frames % chunk_size)) % chunk_size
                segments.append(_load_noise_segment(
                    rec["tail_noise_path"], rec["tail_noise_start_s"], tail * SAMPLES_PER_FRAME,
                ))
                rec["tail_silence_frames_actual"] = tail

                wav_path = os.path.join(wavs_dir, f"{rec['idx']}.wav")
                _write_wav(wav_path, np.concatenate(segments))
                rec["concat_wav_path"] = wav_path
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[step2 idx {rec.get('idx')}] {type(e).__name__}: {e}")


# === Step 3 — run audio_tower on each wav ===

def step3_extract_features(input_jsonl, output_jsonl, features_dir, *,
                           qwen_omni_ckpt, audio_tower_ckpt, device):
    """For each step-2 record, encode the concatenated wav into AudioFeat.pt."""
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)
    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in tqdm(fin.readlines(), desc="step3"):
            rec = json.loads(line)
            try:
                audio = whisper.load_audio(rec["concat_wav_path"], sr=16000)
                pt_path_dir = os.path.join(features_dir, str(rec["idx"]))
                extract_audio_features(
                    audio, pt_path_dir,
                    qwen_omni_ckpt=qwen_omni_ckpt,
                    audio_tower_ckpt=audio_tower_ckpt,
                    device=device,
                )
                rec["pt_path_dir"] = pt_path_dir
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[step3 idx {rec.get('idx')}] {type(e).__name__}: {e}")


# === Step 4 — build token-level training samples ===

def _silence_chunk(chunk_size):
    """Mid-turn or tail chunk: model should keep waiting."""
    ids = [AUDIO_BEGIN] + [PAD] * chunk_size + [ASSISTANT, KEEP_SILENCE]
    labels = [MASK] + [MASK] * chunk_size + [MASK, KEEP_SILENCE]
    return ids, labels


def _response_chunk(assistant_text, emotion, tokenizer, chunk_size):
    """Last chunk of a turn: actual reply, or silence if marked no-response."""
    if isinstance(assistant_text, str) and assistant_text.strip().lower() in NO_RESPONSE_MARKERS:
        return _silence_chunk(chunk_size)
    emotion_tok = EMOTION_TO_ID.get(emotion.lower(), NORMAL) if isinstance(emotion, str) else NORMAL
    assistant_ids = tokenizer.encode(assistant_text).cpu().tolist()
    ids    = [AUDIO_BEGIN] + [PAD] * chunk_size + [ASSISTANT, TEXT_BEGIN, emotion_tok] + assistant_ids + [TEXT_END]
    labels = [MASK]        + [MASK] * chunk_size + [MASK,      TEXT_BEGIN, emotion_tok] + assistant_ids + [TEXT_END]
    return ids, labels


def step4_build_tokens(input_jsonl, output_jsonl, *, tokenizer_dir, chunk_size):
    """For each step-3 record, lay out input_ids / labels / audio_pos."""
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)
    tokenizer = Tokenizer(tokenizer_dir)
    system_ids = tokenizer.encode(DEFAULT_SYSTEM_PROMPT).cpu().tolist()
    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in tqdm(fin.readlines(), desc="step4"):
            rec = json.loads(line)
            try:
                input_ids = [ONLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + system_ids + [TEXT_END]
                labels    = [MASK] * len(input_ids)
                audio_pos = []

                total_frames = 0
                for i, t in enumerate(rec["turns"]):
                    new_chunks = (
                        (total_frames + t["leading_silence_frames"] + t["audio_frames"]) // chunk_size
                        - total_frames // chunk_size
                    )
                    total_frames += t["leading_silence_frames"] + t["audio_frames"]
                    # First turn gets one extra trailing chunk so the reply has room to land.
                    if i == 0:
                        new_chunks += 1
                    for j in range(new_chunks):
                        pos_start = len(input_ids) + 1
                        audio_pos.append((pos_start, pos_start + chunk_size))
                        if j == new_chunks - 1:
                            chunk_ids, chunk_labels = _response_chunk(t["assistant"], t["emotion"], tokenizer, chunk_size)
                        else:
                            chunk_ids, chunk_labels = _silence_chunk(chunk_size)
                        input_ids += chunk_ids
                        labels    += chunk_labels

                tail = rec["tail_silence_frames_actual"]
                tail_chunks = (total_frames + tail) // chunk_size - total_frames // chunk_size - 1
                for _ in range(max(0, tail_chunks)):
                    pos_start = len(input_ids) + 1
                    audio_pos.append((pos_start, pos_start + chunk_size))
                    chunk_ids, chunk_labels = _silence_chunk(chunk_size)
                    input_ids += chunk_ids
                    labels    += chunk_labels

                fout.write(json.dumps({
                    "tasks":       "online",
                    "idx":         rec["idx"],
                    "input_ids":   input_ids,
                    "labels":      labels,
                    "audio_pos":   audio_pos,
                    "pt_path_dir": rec["pt_path_dir"],
                }, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[step4 idx {rec.get('idx')}] {type(e).__name__}: {e}")


# === Main driver ===

def main():
    for name, value in [("CHECKPOINT_DIR", CHECKPOINT_DIR), ("INPUT_JSONL", INPUT_JSONL),
                        ("WORK_DIR", WORK_DIR), ("OUT_TRAIN_JSONL", OUT_TRAIN_JSONL),
                        ("NOISE_DIR", NOISE_DIR)]:
        if not value:
            raise SystemExit(f"Set {name} at the top of cons_online_data.py before running.")

    tokenizer_dir, _, qwen_omni_ckpt, audio_tower_ckpt = resolve_checkpoint_paths(CHECKPOINT_DIR)
    step1_jsonl = os.path.join(WORK_DIR, "step1.jsonl")
    step2_jsonl = os.path.join(WORK_DIR, "step2.jsonl")
    step3_jsonl = os.path.join(WORK_DIR, "step3.jsonl")
    wavs_dir    = os.path.join(WORK_DIR, "wavs")
    features_dir = os.path.join(WORK_DIR, "features")

    step1_sample_silence(
        INPUT_JSONL, step1_jsonl,
        noise_dir=NOISE_DIR,
        min_noise_len=MIN_NOISE_LEN, max_noise_len=MAX_NOISE_LEN,
        chunk_size=CHUNK_SIZE, seed=SEED,
    )
    step2_concat_audio(
        step1_jsonl, step2_jsonl, wavs_dir,
        chunk_size=CHUNK_SIZE, seed=SEED,
    )
    step3_extract_features(
        step2_jsonl, step3_jsonl, features_dir,
        qwen_omni_ckpt=qwen_omni_ckpt, audio_tower_ckpt=audio_tower_ckpt, device=DEVICE,
    )
    step4_build_tokens(
        step3_jsonl, OUT_TRAIN_JSONL,
        tokenizer_dir=tokenizer_dir, chunk_size=CHUNK_SIZE,
    )


if __name__ == "__main__":
    main()
