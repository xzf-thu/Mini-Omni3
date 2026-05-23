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

```bash
# 1. Extract the model weights from a training checkpoint
#    (training writes <out_dir>/step-NNNNNN/lit_model.pth alongside optimizer state).
python src/mini_omni3/finetune/extract_state_dict.py \
    <out_dir>/step-NNNNNN/lit_model.pth state_dict.pt

# 2. Edit the four path constants at the top of src/mini_omni3/generate/infer.py
#    (MODEL_CONFIG_DIR, TRAINED_CHECKPOINT, QWEN_OMNI_CKPT, AUDIO_TOWER_CKPT).
#    Point TRAINED_CHECKPOINT at state_dict.pt from step 1.

# 3. Launch
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python src/mini_omni3/generate/infer.py
```

The script prompts for an audio file path per round and prints the reply.

&nbsp;

## 4. Web demo

A browser-based streaming UI is provided under `web/`. It captures mic audio
in the browser, ships 400 ms PCM frames to a Flask backend that runs the same
model as `infer.py`, and streams the text reply back over SSE.

```bash
# 1. Edit the four path constants at the top of web/server.py
#    (MODEL_CONFIG_DIR, TRAINED_CHECKPOINT, QWEN_OMNI_CKPT, AUDIO_TOWER_CKPT) —
#    same values as infer.py.

# 2. Launch
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python web/server.py

# 3. Open http://<host>:5001/ in a browser, allow microphone access, and talk.
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
