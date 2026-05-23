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

Both builders read their config from constants at the top of the file
(no command-line args). `CHECKPOINT_DIR` is the same checkpoint folder from §3,
so `qwen_2_5_omni_config/` and `MiniOmni3_ChunkwisedEncoder.pth` get picked up
automatically.

| File | Constants to fill in |
|---|---|
| `src/mini_omni3/dataset/cons_offline_data.py` | `CHECKPOINT_DIR`, `INPUT_JSONL`, `OUTPUT_JSONL`, `ERROR_LOG`, `FEATURE_DIR` |
| `src/mini_omni3/dataset/cons_online_data.py`  | `CHECKPOINT_DIR`, `INPUT_JSONL`, `WORK_DIR`, `OUT_TRAIN_JSONL`, `NOISE_DIR` |

One-line JSON samples for every stage live under
`src/mini_omni3/dataset/examples/{offline,online}/`. The shapes are documented
inline below.

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
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python src/mini_omni3/dataset/cons_offline_data.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python src/mini_omni3/dataset/cons_online_data.py
```

Lengths in `*_frames` fields are encoder-output frames
(1 frame = 40 ms @ 16 kHz audio_tower → 640 audio samples).

#### Offline output (`OUTPUT_JSONL`, one line per input sample)

| Field | What it is |
|---|---|
| `tasks` | always `"offline"` |
| `idx` | line index in `INPUT_JSONL` |
| `input_ids` | `prefix + [AUDIO_BEGIN, AUDIO_PAD×N, AUDIO_END]? + [USER, TEXT_BEGIN, ...user, TEXT_END]? + [ASSISTANT, TEXT_BEGIN, ...reply, TEXT_END]` |
| `labels`    | same length as `input_ids`; `-100` masks prefix / audio / user blocks; the assistant block is left unmasked |
| `audio_pos` | `[[start, end]]` — index range of the `AUDIO_PAD` slots in `input_ids`. `null` for T_T (text-only) samples |
| `pt_path_dir` | dir containing this sample's `AudioFeat.pt`. `null` for T_T |

#### Online pipeline

The online builder runs four `stepN_*` functions in order, each appending new
fields onto the previous step's jsonl so every intermediate artifact is
inspectable on disk. Re-run the script for everything, or import a single
`stepN_*` function to redo just that stage.

**Step 1 — `step1_sample_silence`** → `<WORK_DIR>/step1.jsonl`
Draws a random leading-noise length for each turn plus a tail-noise length,
and picks a noise file from `NOISE_DIR` (with a random start offset) for each
slot. No audio is loaded.

| Field | What it is |
|---|---|
| `idx` | input line index |
| `turns[].audio_path`, `assistant`, `emotion` | carried over from input |
| `turns[].leading_silence_frames` | sampled length of the leading noise gap for this turn |
| `turns[].leading_noise_path`     | noise file chosen for this turn's leading gap |
| `turns[].leading_noise_start_s`  | start offset (seconds) inside the noise file |
| `tail_silence_frames`            | sampled length of the trailing noise |
| `tail_noise_path` / `tail_noise_start_s` | noise file + start for the tail |

**Step 2 — `step2_concat_audio`** → `<WORK_DIR>/step2.jsonl` + `<WORK_DIR>/wavs/<idx>.wav`
Loads each turn's audio, splices `noise → turn audio → noise → turn audio → ... → tail noise`
into one waveform and writes it as a wav. Adds:

| Field | What it is |
|---|---|
| `turns[].audio_frames`         | actual encoder-frame count of the turn's audio after pad/crop |
| `tail_silence_frames_actual`   | tail length after rounding the total to a chunk boundary |
| `concat_wav_path`              | path of the per-sample concatenated wav |

**Step 3 — `step3_extract_features`** → `<WORK_DIR>/step3.jsonl` + `<WORK_DIR>/features/<idx>/AudioFeat.pt`
Runs the Qwen2.5-Omni audio_tower on each wav and saves the feature tensor.

| Field | What it is |
|---|---|
| `pt_path_dir` | dir holding `AudioFeat.pt` for this sample |

**Step 4 — `step4_build_tokens`** → `OUT_TRAIN_JSONL`
Lays out the streaming token sequence. Output is the training-ready shape
consumed by `SFTAudioDataset`.

| Field | What it is |
|---|---|
| `tasks` | always `"online"` |
| `idx`   | same as step 1 |
| `input_ids` | `prefix + N × [AUDIO_BEGIN, PAD×chunk_size, ASSISTANT, KEEP_SILENCE]`; the chunk that ends each turn is replaced by `[..., ASSISTANT, TEXT_BEGIN, emotion, ...reply ids, TEXT_END]` |
| `labels`    | same length; only the per-chunk emission (`KEEP_SILENCE`, or the reply tokens on the response chunk) is unmasked |
| `audio_pos` | one `[start, end]` per chunk — index range of that chunk's `PAD×chunk_size` slot |
| `pt_path_dir` | same as step 3 |

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

`infer_online.py`, `infer_offline.py`, and `web/server.py` all read from a
single `checkpoint/` folder. Create it and then point the `CHECKPOINT_DIR`
constant at the top of each entry file at it.

```
Mini-Omni3/
├── infer_online.py                      streaming entry (prompts for an audio path each round)
├── infer_offline.py                     one-shot entry (edit AUDIO_PATH at the top)
├── checkpoint/                          (you create this)
│   ├── model_config.yaml                ┐
│   ├── tokenizer.json                   │  our HF release (drop the files at the root)
│   ├── tokenizer_config.json            │
│   ├── config.json                      │
│   ├── generation_config.json           ┘
│   ├── MiniOmni3_LM.pt                  trained GPT weights (our HF release)
│   ├── MiniOmni3_ChunkwisedEncoder.pth  wrapped audio_tower, proj.* baked in (our HF release)
│   └── qwen_2_5_omni_config/            Qwen2.5-Omni-3B config dir from the official HF repo
└── ...
```

Sources:

| Item | Where to get it |
|---|---|
| Files at `checkpoint/` root (`model_config.yaml`, `tokenizer*.json`, `config.json`, `generation_config.json`) | Our HuggingFace release |
| `checkpoint/MiniOmni3_LM.pt`                 | Our HuggingFace release (or extract from your own training: `python src/mini_omni3/finetune/extract_state_dict.py <out_dir>/step-NNNNNN/lit_model.pth checkpoint/MiniOmni3_LM.pt`) |
| `checkpoint/MiniOmni3_ChunkwisedEncoder.pth` | Our HuggingFace release (produced by `src/mini_omni3/finetune/wrap_audio_tower.py`) |
| `checkpoint/qwen_2_5_omni_config/`           | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) |

Once `checkpoint/` is filled in and `CHECKPOINT_DIR` is set in the entry file,
run either mode:

```bash
# Online streaming — interactive, one audio file per round on stdin.
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python infer_online.py

# Offline single-shot — edit AUDIO_PATH at the top of the file first.
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python infer_offline.py
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
