"""DataLoaders for SFT training.

Reads samples produced by `mini_omni3/dataset/get_dataset_{online,offline}.py`:
    {"tasks", "idx", "input_ids", "labels", "audio_pos", "pt_path_dir", "turn_audio_end_map"?}

Each jsonl is scanned once to build an index of (file_path, byte_offset, length) tuples;
the Dataset reads one line on demand. Audio features come from `<pt_path_dir>/AudioFeat.pt`.
"""

import glob
import json
import os
import random
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mini_omni3.dataset.TOKENS import MASK, PAD, TEXT_END


# Fallback for any None / () that slips into input_ids/labels from upstream data prep.
PLACEHOLDER_TOKEN = 151618

# Max turn count considered when picking a random truncation point for multi-turn online data.
MAX_TURNS = 10


# ---- Dataset ----

class SFTAudioDataset(Dataset):
    """Lazily loads SFT samples from indexed jsonls.

    Each index entry is (file_path, byte_offset, length, data_type, use_turn_map):
      - data_type    : free-form label propagated to the batch (e.g. "train_data" or an
                       eval-source name); used by the training loop to bucket validation loss.
      - use_turn_map : if True and the sample has a `turn_audio_end_map`, randomly truncate
                       to one of the recorded turn boundaries (online multi-turn data aug).
    """

    def __init__(self, data_index, max_seq_len=4096):
        self.data_index = data_index
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, index):
        file_path, offset, length, data_type, use_turn_map = self.data_index[index]
        item = _read_jsonl_chunk(file_path, offset, length)

        input_ids = item.get("input_ids", [])
        labels = item.get("labels", [])
        audio_pos = item.get("audio_pos")
        pt_path_dir = item.get("pt_path_dir", "")

        # Replace bad tokens defensively.
        input_ids = [PLACEHOLDER_TOKEN if x is None or x == () else x for x in input_ids]
        labels = [PLACEHOLDER_TOKEN if x is None or x == () else x for x in labels]

        # Random truncation to a turn boundary for online multi-turn data.
        turn_map = item.get("turn_audio_end_map")
        if use_turn_map and audio_pos and turn_map:
            input_ids, labels, audio_pos = _truncate_to_random_turn(
                input_ids, labels, audio_pos, turn_map
            )

        # Load precomputed audio features; placeholder if missing.
        pt_tensor = [-1]
        if audio_pos and pt_path_dir:
            pt_file = os.path.join(pt_path_dir, "AudioFeat.pt")
            if os.path.exists(pt_file):
                pt_tensor = torch.load(pt_file, map_location="cpu")

        # Pad / truncate to max_seq_len.
        input_ids = input_ids[:self.max_seq_len] + [PAD] * (self.max_seq_len - len(input_ids))
        labels = labels[:self.max_seq_len] + [MASK] * (self.max_seq_len - len(labels))

        return {
            "tasks": item.get("tasks", "online"),
            "type": data_type,
            "input_ids": input_ids,
            "labels": labels,
            "audio_pos": audio_pos,
            "pt_list": pt_tensor,
        }


def _read_jsonl_chunk(file_path, offset, length):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            return json.loads(f.read(length))
    except Exception as e:
        print(f"Error reading {file_path} at offset {offset}: {e}")
        return {"input_ids": [PAD], "labels": [MASK], "audio_pos": None, "pt_path_dir": ""}


def _truncate_to_random_turn(input_ids, labels, audio_pos, turn_map):
    """Pick a recorded turn boundary, truncate input_ids/labels up to the next TEXT_END
    after that turn's last audio chunk, and clip audio_pos to that many segments.
    """
    candidates = [f"turn_{i}" for i in range(MAX_TURNS) if f"turn_{i}" in turn_map]
    if not candidates:
        return input_ids, labels, audio_pos

    truncate_pos = turn_map[random.choice(candidates)]
    if not 0 < truncate_pos <= len(audio_pos):
        return input_ids, labels, audio_pos[:truncate_pos]

    last_audio_end = audio_pos[truncate_pos - 1][1]
    truncate_idx = next(
        (i + 1 for i in range(last_audio_end, len(input_ids)) if input_ids[i] == TEXT_END),
        None,
    )
    if truncate_idx is not None:
        input_ids = input_ids[:truncate_idx]
        labels = labels[:truncate_idx]
    return input_ids, labels, audio_pos[:truncate_pos]


# ---- Index scanning ----

def _scan_and_split(folders, ratio, train_type, eval_type, use_turn_map, check_audio_file):
    """Scan jsonls under `folders`, skipping samples whose AudioFeat.pt is missing
    (if check_audio_file), and split into (train, eval) index lists by `ratio`.
    """
    if isinstance(folders, str):
        folders = [folders]

    files = []
    for folder in folders:
        files.extend(glob.glob(os.path.join(folder, "*.jsonl")))

    entries, skipped = [], 0
    for file_path in tqdm(files, desc="Scanning"):
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                if check_audio_file and not _has_audio_feature(line):
                    skipped += 1
                    continue
                entries.append((file_path, offset, len(line), use_turn_map))

    random.shuffle(entries)
    split_idx = int(len(entries) * ratio)
    train = [(p, o, l, train_type, utm) for p, o, l, utm in entries[:split_idx]]
    eval_ = [(p, o, l, eval_type, utm) for p, o, l, utm in entries[split_idx:]]

    print(f"  Scanned {len(entries)} items (skipped {skipped} missing AudioFeat.pt) "
          f"-> train={len(train)}, eval={len(eval_)}")
    return train, eval_


def _has_audio_feature(json_line):
    """True if the sample has no audio (text-only) or its AudioFeat.pt is on disk."""
    try:
        item = json.loads(json_line)
    except json.JSONDecodeError:
        return False
    pt_path_dir = item.get("pt_path_dir")
    if not pt_path_dir:
        return True
    return os.path.exists(os.path.join(pt_path_dir, "AudioFeat.pt"))


# ---- Collate ----

def _collate_fn(batch):
    return {
        "batch_size": len(batch),
        "tasks":     [item["tasks"]     for item in batch],
        "types":     [item["type"]      for item in batch],
        "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch]),
        "labels":    torch.stack([torch.tensor(item["labels"])    for item in batch]),
        "pt_list":   [item["pt_list"]   for item in batch],
        "audio_pos": [item["audio_pos"] for item in batch],
    }


# ---- Main entry point ----

def get_dataloaders(
    data_sources: Dict[str, Dict],
    eval_data_percentage: float = 0.1,
    max_seq_len: int = 3200,
    train_batchsize: int = 1,
    eval_batchsize: int = 1,
    seed: int = 1337,
    train_type: str = "train_data",
    check_audio_file: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[str, str]]:
    """Build (train_dataloader, eval_dataloader, type_to_name) from a multi-source config.

    `data_sources` keys are display names; each value is:
        {
            "folders":      [path, ...],     # jsonl roots to scan
            "type_name":    "<eval_type>",   # label attached to this source's eval split
            "enabled":      True,            # optional, default True
            "use_turn_map": True,            # optional, default True (online multi-turn aug)
        }

    `type_to_name` maps every `type_name` (and `train_type`) back to a display name so
    the training loop can bucket validation loss per source.
    """
    random.seed(seed)

    train_indices, eval_indices = [], []
    type_to_name = {train_type: "train"}

    for name, cfg in data_sources.items():
        if not cfg.get("enabled", True):
            print(f"Skipping disabled source: {name}")
            continue
        folders = cfg.get("folders", [])
        if not folders:
            print(f"Warning: no folders for {name}, skipping")
            continue

        eval_type = cfg.get("type_name", f"{name}_eval")
        use_turn_map = cfg.get("use_turn_map", True)
        print(f"\n[{name}]  folders={folders}  eval_type={eval_type}  use_turn_map={use_turn_map}")

        train, eval_ = _scan_and_split(
            folders,
            ratio=1 - eval_data_percentage,
            train_type=train_type,
            eval_type=eval_type,
            use_turn_map=use_turn_map,
            check_audio_file=check_audio_file,
        )
        train_indices.extend(train)
        eval_indices.extend(eval_)
        type_to_name[eval_type] = name

    # Carve a generic eval slice off the merged training pool (kept under train_type).
    random.shuffle(train_indices)
    extra = int(len(train_indices) * eval_data_percentage)
    eval_indices.extend(train_indices[-extra:])
    train_indices = train_indices[:-extra]

    print(f"\nFinal: train={len(train_indices)}, eval={len(eval_indices)}, type_map={type_to_name}")

    train_loader = DataLoader(
        SFTAudioDataset(train_indices, max_seq_len=max_seq_len),
        batch_size=train_batchsize, shuffle=True, num_workers=4, collate_fn=_collate_fn,
    )
    eval_loader = DataLoader(
        SFTAudioDataset(eval_indices, max_seq_len=max_seq_len),
        batch_size=eval_batchsize, shuffle=True, num_workers=4, collate_fn=_collate_fn,
    )
    return train_loader, eval_loader, type_to_name
