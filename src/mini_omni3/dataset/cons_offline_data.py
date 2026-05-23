"""Build offline single-turn SFT samples.

Token layout: prefix + [audio_block?] + [user_block?] + assistant_block.
Three task variants come from which blocks are present:
  - audio + user  -> A_T_T
  - audio         -> A_T
  -         user  -> T_T

Input  (per line): {"conversation": [{"user", "assistant", "audio_path"}, ...]} (first turn only)
                   or flat {"user", "assistant", "audio_path"}.
Output (per line): {"tasks": "offline", "idx", "input_ids", "labels", "audio_pos", "pt_path_dir"}.

Edit the constants below, then `python cons_offline_data.py`.
Re-running picks up where the previous run stopped (already-written idx are skipped).
"""

import json
import os

import torch
from tqdm import tqdm
from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration

from mini_omni3.dataset.utils.load_audio import _load_mel
from mini_omni3.dataset.TOKENS import (
    ASSISTANT, AUDIO_BEGIN, AUDIO_END, AUDIO_PAD, ENGLISH, MASK,
    OFFLINE, SYSTEM, TEXT_BEGIN, TEXT_END, USER,
)
from mini_omni3.generate.base import resolve_checkpoint_paths
from mini_omni3.tokenizer import Tokenizer


# ============================================================
# Fill these in before running.
# ============================================================
CHECKPOINT_DIR = ""   # checkpoint root (tokenizer + qwen_2_5_omni_config + MiniOmni3_ChunkwisedEncoder.pth)
INPUT_JSONL    = ""   # raw input jsonl
OUTPUT_JSONL   = ""   # training-ready output jsonl
ERROR_LOG      = ""   # path to append per-sample error messages
FEATURE_DIR    = ""   # dir to save audio feature .pt files (one subdir per sample)
DEVICE         = "cuda"
# ============================================================


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. When there is no user text, if the audio contains a question, "
    "please answer it. If it is a sound effect, determine based on the sound whether help is needed."
)

# Cap offline audio length at 20 seconds (matches the prior behavior).
MAX_AUDIO_SECONDS = 20


# === Audio encoder (lazy, cached per device) ===
_audio_encoder_cache = {}


def _get_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, device):
    key = (qwen_omni_ckpt, audio_tower_ckpt, str(device))
    if key not in _audio_encoder_cache:
        print(f"[offline] loading audio_encoder on {device} ...")
        cfg = AutoConfig.from_pretrained(qwen_omni_ckpt)
        enc = Qwen2_5OmniForConditionalGeneration._from_config(cfg).thinker.audio_tower
        enc.load_state_dict(torch.load(audio_tower_ckpt, map_location=device))
        enc.to(device).eval()
        _audio_encoder_cache[key] = enc
    return _audio_encoder_cache[key]


def _extract_and_save_audio_feature(audio_path, save_dir, *,
                                    qwen_omni_ckpt, audio_tower_ckpt, device):
    """Encode audio, save to `<save_dir>/AudioFeat.pt`, return output_len."""
    _, mel, len_feature, input_len, _ = _load_mel(audio_path, max_seconds=MAX_AUDIO_SECONDS)
    encoder = _get_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, device)
    with torch.no_grad():
        hidden = encoder(
            mel.to(device),
            torch.tensor([len_feature]).to(device),
            torch.tensor([input_len]).to(device),
        ).last_hidden_state
    hidden = hidden.squeeze(0).cpu()
    output_len = hidden.shape[0]
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden, os.path.join(save_dir, "AudioFeat.pt"))
    return output_len


# === Input parsing ===

def _extract_first_turn(data_item):
    """Return (user_text, assistant_text, audio_path). Offline is single-turn,
    so a `conversation` list contributes only its first turn.
    """
    convs = data_item.get("conversation", [])
    src = convs[0] if convs else data_item
    return (
        src.get("user"),
        src.get("assistant"),
        src.get("audio_path") or src.get("merge_path"),
    )


# === Sample builder ===

def build_offline_sample(data_item, idx, *,
                         tokenizer, system_ids, feature_dir,
                         qwen_omni_ckpt, audio_tower_ckpt, device):
    """Build one offline SFT sample. Returns a dict with input_ids/labels/audio_pos/pt_path_dir."""
    user_text, assistant_text, audio_path = _extract_first_turn(data_item)
    if not assistant_text or assistant_text == "None":
        raise ValueError("missing assistant text")

    has_audio = audio_path not in (None, "", "None")
    has_user = user_text not in (None, "", "None")
    assistant_ids = tokenizer.encode(assistant_text).cpu().tolist()
    user_ids = tokenizer.encode(user_text).cpu().tolist() if has_user else []

    if has_audio:
        pt_path_dir = os.path.join(feature_dir, str(idx))
        output_len = _extract_and_save_audio_feature(
            audio_path, pt_path_dir,
            qwen_omni_ckpt=qwen_omni_ckpt, audio_tower_ckpt=audio_tower_ckpt, device=device,
        )
    else:
        pt_path_dir = None
        output_len = 0

    prefix          = [OFFLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + system_ids + [TEXT_END]
    audio_block     = [AUDIO_BEGIN] + [AUDIO_PAD] * output_len + [AUDIO_END]    if has_audio else []
    user_block      = [USER, TEXT_BEGIN] + user_ids + [TEXT_END]                if has_user  else []
    assistant_block = [ASSISTANT, TEXT_BEGIN] + assistant_ids + [TEXT_END]

    input_ids = prefix + audio_block + user_block + assistant_block
    labels    = [MASK] * (len(prefix) + len(audio_block) + len(user_block)) + assistant_block

    # AUDIO_PAD sits right after [AUDIO_BEGIN] (which is at index len(prefix)).
    audio_pos = [(len(prefix) + 1, len(prefix) + 1 + output_len)] if has_audio else None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "audio_pos": audio_pos,
        "pt_path_dir": pt_path_dir,
    }


# === Main driver ===

def _load_processed_indices(output_path):
    """For resume: collect idx of all valid records already in output_path."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as fr:
        for line in fr:
            try:
                done.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    for name, value in [("CHECKPOINT_DIR", CHECKPOINT_DIR), ("INPUT_JSONL", INPUT_JSONL),
                        ("OUTPUT_JSONL", OUTPUT_JSONL), ("ERROR_LOG", ERROR_LOG),
                        ("FEATURE_DIR", FEATURE_DIR)]:
        if not value:
            raise SystemExit(f"Set {name} at the top of cons_offline_data.py before running.")

    tokenizer_dir, _, qwen_omni_ckpt, audio_tower_ckpt = resolve_checkpoint_paths(CHECKPOINT_DIR)
    os.makedirs(FEATURE_DIR, exist_ok=True)

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        data_lines = f.readlines()

    processed = _load_processed_indices(OUTPUT_JSONL)
    todo = [i for i in range(len(data_lines)) if i not in processed]
    print(f"Total {len(data_lines)} | already done {len(processed)} | to process {len(todo)}")

    tokenizer = Tokenizer(tokenizer_dir)
    system_ids = tokenizer.encode(DEFAULT_SYSTEM_PROMPT).cpu().tolist()

    with open(OUTPUT_JSONL, "a", encoding="utf-8", buffering=1) as fout, \
         open(ERROR_LOG,   "a", encoding="utf-8", buffering=1) as ferr:
        for idx in tqdm(todo):
            try:
                data_item = json.loads(data_lines[idx])
                sample = build_offline_sample(
                    data_item, idx,
                    tokenizer=tokenizer, system_ids=system_ids,
                    feature_dir=FEATURE_DIR,
                    qwen_omni_ckpt=qwen_omni_ckpt,
                    audio_tower_ckpt=audio_tower_ckpt,
                    device=DEVICE,
                )
                fout.write(json.dumps(
                    {"tasks": "offline", "idx": idx, **sample},
                    ensure_ascii=False,
                ) + "\n")
            except Exception as e:
                ferr.write(f"[idx {idx}] {type(e).__name__}: {e}\n")
                ferr.write(data_lines[idx])


if __name__ == "__main__":
    main()
