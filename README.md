# mini-omni3 123

Audio-enhanced GPT for streaming speech-to-text dialogue.

> Built on top of [LitGPT](https://github.com/Lightning-AI/litgpt) (Apache 2.0).

&nbsp;

## Install

```bash
pip install -e '.[all]'
```

&nbsp;

## 1. Prepare training data

Edit the path constants at the top of each script first:

| File | Constants to fill in |
|---|---|
| `src/mini_omni3/dataset/extract_audio_features.py` | `QWEN_OMNI_CKPT`, `AUDIO_TOWER_CKPT` |
| `src/mini_omni3/dataset/build_online.py` | `QWEN_OMNI_CKPT` |
| `src/mini_omni3/dataset/build_offline.py` | `QWEN_OMNI_CKPT`, `AUDIO_TOWER_CKPT` |

### Input JSONL format

**Online** (streaming, multi-turn audio). One JSON per line:

```json
{"conversation": [
    {"audio_path": "/path/to/turn1.wav", "assistant": "reply 1", "emotion": "normal"},
    {"audio_path": "/path/to/turn2.wav", "assistant": "reply 2", "emotion": "happy"}
]}
```

- `audio_path`, `assistant`: required per turn.
- `emotion`: optional, defaults to `"normal"`. Allowed: `happy`, `sad`, `angry`, `surprise`, `normal`, `urgent`.
- For "model should stay silent on this turn", set `assistant` to `"<no need to response>"`.

Single-turn shorthand also accepted:

```json
{"merge_path": "/path/to/audio.wav", "assistant": "reply", "emotion": "normal"}
```

**Offline** (single-turn; three task variants based on which fields are present). One JSON per line — either:

```json
{"user": "user text", "assistant": "reply", "audio_path": "/path/to/audio.wav"}
```

or the online-style multi-turn shape (only the **first** turn is used):

```json
{"conversation": [{"user": "...", "assistant": "...", "audio_path": "..."}, ...]}
```

`assistant` is required. The three task variants come from what else is present:

| Has `audio_path`? | Has `user`? | Task |
|---|---|---|
| ✓ | ✓ | A_T_T (audio + user text → assistant) |
| ✓ |   | A_T (audio → assistant) |
|   | ✓ | T_T (user text → assistant) |

### Run

```bash
# Online:
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python src/mini_omni3/dataset/build_online.py \
    <input.jsonl> <output.jsonl> <error.log> <feature_dir>

# Offline:
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python src/mini_omni3/dataset/build_offline.py \
    <input.jsonl> <output.jsonl> <error.log> <feature_dir>
```

Both scripts are resumable — re-running picks up where the previous run stopped (skips already-written `idx`). See `src/mini_omni3/dataset/build_dataset.sh` for a parallel multi-GPU template.

&nbsp;

## 2. Train

Training cold-starts from a Qwen2.5-Omni Thinker checkpoint converted to this
repo's GPT state-dict layout (`paths.init_checkpoint` in `config.yaml`). We
will release the converted `.pt` on HuggingFace; point `init_checkpoint` at it.

```bash
# 1. Set the two data roots referenced by config.yaml
export DATA_ROOT=/path/to/your/jsonl/data
export CHECKPOINT_ROOT=/path/to/your/checkpoints

# 2. Edit hyperparameters / data sources in src/mini_omni3/finetune/config.yaml
#    Make sure paths.init_checkpoint points at the converted Qwen2.5-Omni .pt
#    (downloaded from our HuggingFace release).

# 3. Launch
PYTHONPATH=src python src/mini_omni3/finetune/train.py --config src/mini_omni3/finetune/config.yaml
```

&nbsp;

## 3. Inference

Both `assets/infer_*.py` and `web/server.py` read everything from a single
`checkpoint/` folder at the repo root. Create it with the following layout:

```
Mini-Omni3/
├── assets/
│   ├── infer_online.py         streaming entry (prompts for an audio path each round)
│   └── infer_offline.py        one-shot entry (edit AUDIO_PATH at the top)
├── checkpoint/                 (you create this)
│   ├── model_config/           model_config.yaml + tokenizer files (our HF release)
│   ├── qwen2.5-omni_config/    Qwen2.5-Omni-3B from the official HF repo
│   ├── state_dict.pt           trained GPT weights (our HF release)
│   └── audio_tower.pth         wrapped audio_tower, proj.* baked in (our HF release)
└── ...
```

Sources:

| Item | Where to get it |
|---|---|
| `checkpoint/model_config/`  | Our HuggingFace release |
| `checkpoint/state_dict.pt`  | Our HuggingFace release (or extract from your own training: `python src/mini_omni3/finetune/extract_state_dict.py <out_dir>/step-NNNNNN/lit_model.pth checkpoint/state_dict.pt`) |
| `checkpoint/audio_tower.pth`| Our HuggingFace release (produced by `src/mini_omni3/finetune/wrap_audio_tower.py`) |
| `checkpoint/qwen2.5-omni_config/` | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) |

Once `checkpoint/` is filled in, run either mode:

```bash
# Online streaming — interactive, one audio file per round on stdin.
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python assets/infer_online.py

# Offline single-shot — edit AUDIO_PATH at the top of the file first.
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python assets/infer_offline.py
```

Online mode prompts for an audio file path each round and streams replies.
Offline mode runs the given audio once and prints the reply, or
`(silent — kept listening)` if the model chose not to respond.

&nbsp;

## 4. Web demo

A browser-based streaming UI is provided under `web/`. It captures mic audio
in the browser, ships 400 ms PCM frames to a Flask backend that runs the same
model as `infer.py`, and streams the text reply back over SSE. It reads the
same `checkpoint/` folder as §3.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python web/server.py
# Open http://<host>:5001/ in a browser, allow microphone access, and talk.
```

Endpoints (used by `web/mini_omni3.html`; documented for reference):

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/`          | Serves the frontend HTML |
| `POST` | `/audio`     | Body is one 400 ms 16-bit-LE mono PCM frame @ 16 kHz |
| `POST` | `/audio_end` | No-op signal that the mic detected trailing silence |
| `GET`  | `/stream`    | Server-Sent Events: text pieces, `<\|busy\|>` when the model starts speaking, `<\|end\|>` when it finishes |

Each utterance's raw audio frames are saved under `web/recordings/<timestamp>/`
for debugging. Override the port with `PORT=… python web/server.py`.

&nbsp;

## Background

_TODO_
