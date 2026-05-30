<p align="center">
  <img src="assets/figures/top.png" alt="Mini-Omni3 Logo" width="80%">
</p>


<h1 align="center">Mini-Omni3: An Always-On Streaming Audio Language Model for the Real World</h1>

We introduce **MINI-OMNI3**, the first **always-on Streaming Audio Language Model (SALM)** that follows **every audio task — understanding, transcription, translation, full-spectrum conversation, and proactive intervention — in a single streaming session**, deciding frame by frame via `⟨Silent⟩` / `⟨Speak⟩` control tokens when and how to respond. Trained on **260,000 hours of streaming audio** with our unified **SoundFlow** framework, it stays **competitive with strong offline baselines** — and sometimes beats them. If you like us, please give us a star ✨.

<p align="center"><u><em>The world never stops making sound — neither should your model.</em></u></p>


<p align="center">
  <a href="https://arxiv.org/abs/2605.XXXXX">Technical Report 📖</a> /
  <a href="https://huggingface.co/datasets/mini-omni3/SoundFlow-260K">SoundFlow-260K 🤗</a> /
  <a href="https://huggingface.co/mini-omni3/Mini-Omni3">Mini-Omni3 Weights 🤗</a> /
  <a href="https://github.com/mini-omni3/Streaming-Audio-Bench">Streaming-Audio-Bench 🏆</a>
</p>

<p align="center">
  <a href="https://github.com/mini-omni3/Mini-Omni3/raw/main/assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>&nbsp;<a href="https://mini-omni3.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>&nbsp;<a href="https://x.com/"><img src="https://img.shields.io/badge/X-@MiniOmni3-black?logo=x&logoColor=white" alt="X"></a>
</p>


<p align="center">
  <a href="https://www.youtube.com/watch?v=r1S4xiUBg9s">
    <img src="https://img.youtube.com/vi/r1S4xiUBg9s/maxresdefault.jpg" alt="Watch Mini-Omni3 running live" width="95%">
  </a>
</p>
<p align="center"><em>▶ Click to watch Mini-Omni3 listen, decide, and speak — live (YouTube)</em></p>



## 🔥 News

- [Coming]: We will release the full SoundFlow data construction pipeline.
- [Coming]: WebUI for live streaming interaction will be open-sourced.
- [Coming]: Dataset and benchmark will be reformatted to be clearer.
- **May 28, 2026**: 🔥 We add Mini-Omni3 low-latency streaming inference with FIFO asynchronous encode/decode.
- **May 20, 2026**: 🔥 We release **Streaming-Audio-Bench**, a benchmark for always-on SALM evaluation.
- **May 20, 2026**: 🔥 We release **SoundFlow-260K**.
- **May 20, 2026**: 🔥 We release the **Mini-Omni3 Inference and Training Codebase**.
- **May 19, 2026**: 🔥 **Mini-Omni3** model weights are now available on Hugging Face.
- **May 19, 2026**: 🔥 We release the **Mini-Omni3 Technical Report**.


## Contents

* **[Quick Start — play first](#quick-start)**
* **[Everything, at once](#everything-at-once)** — the one-session demo + head-to-head comparisons
* **[How it works: SoundFlow](#how-it-works)**
* **[Finetuning](#finetuning)**
* **[Evaluation](#evaluation)**
* **[License, Citation & Stars](#citation)**


## <a id="quick-start"></a>⚡ Quick Start — play first

Mini-Omni3 is an always-on model: it keeps listening to incoming audio frames and **decides for itself when to speak**. By default it stays in a `⟨Silent⟩` state and only emits output when the task or the acoustic context warrants it — so you can open a single session, stream audio into it continuously, and watch every capability take turns on its own.

**Installation**
```bash
git clone https://github.com/mini-omni3/Mini-Omni3.git
cd Mini-Omni3

conda create -n mini-omni3 python=3.10 -y
conda activate mini-omni3
pip install -r requirements.txt
```

**Download Weights**
```bash
python scripts/download.py
```

**Offline Inference**
```bash
# infer with default audio
bash scripts/inference.sh

# Use your own audio:
bash scripts/inference.sh --audio /path/to/audio.wav
```

**Streaming Inference**

Mini-Omni3 adopts a FIFO-style streaming mechanism with asynchronous encoding and decoding plus a lightweight cache, so latency and compute stay stable in the always-on setting. Feed audio frames chunk by chunk and the model will emit partial output as soon as it transitions from `⟨Silent⟩` to `⟨Speak⟩`:

```bash
python infer_streaming.py \
  --audio assets/example/streaming_long_example.wav \
  --step_ms 1000 \
  --reset_interval_sec 120 \
  --overlap_sec 2 \
  --max_new_tokens 32
```

The script prints partial text after each streaming step and a final transcript once the stream finishes. For long audio, it periodically resets the streaming state so the internally accumulated context does not grow without bound. Set `--reset_interval_sec 0` to disable state resets.

On smaller GPUs, the CLI automatically uses conservative defaults unless you override them: `gpu_memory_utilization=0.85`, `max_model_len=8192`, `max_num_seqs=1`, and `max_num_batched_tokens=2048`.

```bash
python infer_streaming.py \
  --gpu_memory_utilization 0.85 \
  --max_model_len 8192 \
  --max_num_seqs 1 \
  --max_num_batched_tokens 2048 \
  --audio /path/to/audio.wav
```


## <a id="everything-at-once"></a>🎬 Everything, at once

Most audio models do one job and wait to be asked. Mini-Omni3's defining trait is that **all of its abilities live in the same continuous stream**, and the model itself decides which one is needed at each moment. The demo below is **one unbroken session, one model, no mode switches, no prompts** — transcription, understanding, conversation, and proactive intervention simply happen as the soundscape changes.

<div align="center">
  <video src="assets/demo/all_in_one_session.mp4" controls width="320"></video>
</div>

#### One stream, one model, capabilities firing on their own

| Time | What's happening in the stream | Model state | What Mini-Omni3 does |
|---|---|---|---|
| 0:00 | User starts talking through a plan out loud | `⟨Speak⟩` | Streams a live transcript + on-the-fly EN→ZH translation, chunk by chunk |
| 0:18 | A song fades in on a nearby speaker; user asks "what's this track?" | `⟨Speak⟩` | Fuses speech **and** background music, answers in-context without losing the conversation |
| 0:31 | User goes quiet and keeps working | `⟨Silent⟩` | Stays silent — no filler, no hallucinated turns |
| 0:52 | A smoke alarm starts beeping | `⟨Silent⟩` → `⟨Speak⟩` | Flips to speaking on its own and warns the user — **no wake word, no prompt** |

> The point isn't four features. It's **one always-on policy** over `⟨Silent⟩` / `⟨Speak⟩` that quietly routes between perception, conversation, and intervention. Pipelines that bolt together an ASR model, a chat model, and a wake-word detector can't make these decisions jointly — Mini-Omni3 does it in a single forward stream.

---

Below, the same four abilities are broken out individually and put **head-to-head** against `gpt-audio`, `doubao-voicechat`, and `gemini-omni`. Legend: ✅ handles it natively · ⚠️ partial / degraded · ❌ can't do it in a streaming setting.

#### Capability 1 — Online audio understanding

<table>
  <tr>
    <th valign="top">Input (streaming)</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">Mini-Omni3 (Ours)</th>
  </tr>
  <tr>
    <td valign="top">Continuous ambient audio: footsteps, a door opening, distant traffic.</td>
    <td valign="top">❌ Record-then-infer: waits for the clip to end, then returns one summary — no incremental narration.</td>
    <td valign="top">⚠️ Speech-centric: lumps non-speech into "background noise" and misses individual events.</td>
    <td valign="top">⚠️ Buffers a fixed window first, so narration lags several seconds behind the sound.</td>
    <td valign="top">✅ Detects each event incrementally and narrates the scene in real time, without waiting for the clip to end.</td>
  </tr>
</table>

<details>
<summary><strong>Capabilities 2 – 4 (transcription &amp; translation · full-spectrum chat · proactive intervention)</strong></summary>

<br>

#### Capability 2 — Real-time transcription &amp; translation

<table>
  <tr>
    <th valign="top">Input (streaming)</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">Mini-Omni3 (Ours)</th>
  </tr>
  <tr>
    <td valign="top">A speaker talking continuously while the model listens.</td>
    <td valign="top">⚠️ Clean transcript, but only after the utterance finishes — no mid-sentence partials.</td>
    <td valign="top">⚠️ Streams ASR well, but translation is turn-based and only fires at sentence boundaries.</td>
    <td valign="top">⚠️ Emits chunks but re-decodes aggressively, causing flicker and unstable partials.</td>
    <td valign="top">✅ Emits partial transcripts and translations chunk by chunk with low latency, correcting incrementally as context arrives.</td>
  </tr>
</table>

#### Capability 3 — Voice chat beyond speech

<table>
  <tr>
    <th valign="top">Input (streaming)</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">Mini-Omni3 (Ours)</th>
  </tr>
  <tr>
    <td valign="top">A user asks about a song playing in the background while talking.</td>
    <td valign="top">⚠️ Hears the speech but ignores the music — answers as if no song were playing.</td>
    <td valign="top">❌ Treats the music as noise to suppress; can't reason about it.</td>
    <td valign="top">⚠️ Can ID the song in isolation, but can't fuse it with the ongoing conversation.</td>
    <td valign="top">✅ Jointly perceives speech, music, and general audio, and responds in a context-aware, full-spectrum conversation.</td>
  </tr>
</table>

#### Capability 4 — Proactive intervention

<table>
  <tr>
    <th valign="top">Input (streaming)</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">Mini-Omni3 (Ours)</th>
  </tr>
  <tr>
    <td valign="top">A smoke alarm starts beeping while the user is silent.</td>
    <td valign="top">❌ Stays silent — only responds when prompted; no self-initiated speech.</td>
    <td valign="top">❌ Waits for a wake word / user turn; never volunteers a warning.</td>
    <td valign="top">❌ No notion of <em>when</em> to speak; requires an explicit query.</td>
    <td valign="top">✅ Holds <code>⟨Silent⟩</code> until the acoustic cue appears, then switches to <code>⟨Speak⟩</code> and warns the user — no prompt required.</td>
  </tr>
</table>

</details>

### Why it adds up

✅ **One always-on model, not a pipeline** — continuously ingests audio frames and decides **when** and **how** to respond via `⟨Silent⟩` / `⟨Speak⟩` control tokens, so every capability above shares the same context.

✅ **Full-spectrum perception** — jointly handles **speech, music, and general audio** (background sounds, pauses, non-verbal cues), which is exactly what lets understanding, chat, and intervention coexist.

✅ **Low-latency by design** — FIFO-style streaming with **asynchronous encoding and decoding** and a lightweight cache for stable latency and compute utilization.

✅ **No accuracy tax for going streaming** — trained with chunked inputs and streaming objectives, yet stays **competitive with strong offline baselines** (e.g., **78.4 vs. 77.9**, **82.1 vs. 81.5**, **69.8 vs. 69.2** against Qwen2.5-Omni-3B).

<p align="center">
  <img src="assets/figures/radar_results.png" alt="Results" width="100%">
</p>


## <a id="how-it-works"></a>🧠 How it works: SoundFlow

**Mini-Omni3** is trained with the **SoundFlow** framework, which reformulates long audio into chunked streaming sequences and supervises the model to predict both **linguistic content** and **intervention behavior** through dedicated control tokens. That dual objective is what bundles the four abilities into one policy: instead of only predicting *what was said*, the model also learns *whether this is a moment to stay `⟨Silent⟩` or to `⟨Speak⟩`* — enabling instruction following and audio-triggered proactive responses within a single streaming language-modeling loss.

The framework unifies long-form data construction, token-level temporal annotation, streaming-aware training, and low-latency inference — and supports training patterns well beyond transcription.

<p align="center">
  <img src="/docs/assets/training.png" alt="Mini-Omni3 Training" width="100%">
</p>


## <a id="finetuning"></a>🔧 Finetuning

You can fine-tune Mini-Omni3 on your own streaming scenarios and data, and you can also use this repository to train standard offline audio language models.

`src/MiniOmni3/SoundFlow` contains the core training code built around the SoundFlow framework.

```text
src/MiniOmni3/SoundFlow/
├── arguments.py      # Defines command-line arguments and training hyperparameters.
├── chunking.py       # Reformulates long audio into chunked streaming sequences.
├── dataloader.py     # Loads JSONL data, reads audio, builds streaming inputs, masks non-target tokens.
├── finetune.py       # Main entry point for launching SoundFlow training.
├── modeling.py       # Loads the audio encoder + LLM and defines the streaming objective.
├── trainer.py        # Defines the streaming-aware trainer with control-token supervision.
```

Training data is in JSONL format. Token-level temporal annotation lets the model learn intervention behavior (when to stay `⟨Silent⟩` vs. `⟨Speak⟩`):

```json
{
  "audio": ".../wavs/stream/0001.wav",
  "text": "task transcription<speak>THE TRANSCRIPT TEXT",
  "prompt": ""
}
```

We can use the following command to start it.

```bash
torchrun --nproc_per_node=2 SoundFlow/finetune.py \
  --model_path Mini-Omni3-Base --train_file ${TRAIN_JSONL} \
  --eval_file ${VAL_JSONL} --output_dir ${OUT_DIR} \
  --batch_size 8 --grad_acc 8 \
  --lr 1e-6 --lr_encoder 1e-6 --lr_aligner 1e-6 --lr_llm 1e-6 \
  --epochs 2 --save_steps 200 --save_total_limit 300 \
  --chunk_ms 1000 --overlap_ms 200 --streaming 1 \
  --warmup_ratio 0.05 --max_grad_norm 1.0 --weight_decay 0.01 \
  --run_name ${RUN_NAME} --report_to wandb \
  2>&1 | tee -a ${LOG_FILE}
```


## <a id="evaluation"></a>📊 Evaluation

We provide a simple evaluation script for running Mini-Omni3 inference and computing streaming metrics (WER/CER for transcription, plus latency). The input file should be a JSONL file. Each line needs `audio` (or `audio_path`) plus `answer` as the ground-truth output:

```json
{"audio": "examples/audio/stream.wav", "answer": "I usually take the quieter road home because the main street gets crowded after work."}
```

The script keeps all original fields and appends the following to the output JSONL:

```text
prediction  # model output
metric      # "wer" for English samples, "cer" for Chinese samples
wer         # WER/CER score value; CER is also stored in this field for compatibility
latency_ms  # average emit latency under streaming decoding
num_edits   # edit distance between prediction and ground truth
ref_len     # number of reference words or characters
```

**Offline metrics**
```bash
python src/MiniOmni3/eval/evaluate.py \
  --ckpt_dir ckpt/Mini-Omni3 \
  --input_jsonl examples/test.jsonl \
  --output_jsonl outputs/pred_with_metrics.jsonl
```

**Streaming metrics** — measure incremental, low-latency performance under the FIFO mechanism:
```bash
python src/MiniOmni3/eval/evaluate_streaming.py \
  --ckpt_dir ckpt/Mini-Omni3 \
  --input_jsonl examples/test.jsonl \
  --output_jsonl outputs/pred_with_metrics.streaming.jsonl \
  --step_ms 1000 --overlap_sec 2
```

Mini-Omni3 is evaluated across both **standard offline audio benchmarks** and our **streaming-tailored analyses**, showing that moving from offline understanding to a fully streaming regime does not inherently sacrifice general capability.

<p align="center">
  <img src="/assets/tables/offline_benchmarks.png" alt="Mini-Omni3 Offline Results" width="100%">
</p>

<p align="center">
  <img src="/assets/tables/streaming_breakdown.png" alt="Mini-Omni3 Streaming Results" width="100%">
</p>


## Acknowledgements

We sincerely thank the creators, maintainers, and contributors of the public datasets and resources used in this work. We also thank the broader large audio language model community for laying the groundwork that made streaming audio modeling possible.


## <a id="citation"></a>License, Citation & Stars

This project will be released under the **Apache-2.0 License**. You can do everything with Mini-Omni3 🎉

**Citation**: You can cite Mini-Omni3 using the following BibTeX entry. Thank you for your kindness 🙂

```bibtex
@misc{miniomni3,
      title={Mini-Omni3: An Always-On Streaming Audio Language Model for the Real World},
      author={Mini-Omni3 Team},
      year={2026},
      eprint={2605.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2605.XXXXX},
}
```

<a href="https://www.star-history.com/?repos=gpt-omni%2Fmini-omni%2Cmini-omni3%2FMini-Omni3&type=date&legend=bottom-right">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=gpt-omni/mini-omni%2Cmini-omni3/Mini-Omni3&type=date&theme=dark&legend=bottom-right" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=gpt-omni/mini-omni%2Cmini-omni3/Mini-Omni3&type=date&legend=bottom-right" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=gpt-omni/mini-omni%2Cmini-omni3/Mini-Omni3&type=date&legend=bottom-right" />
 </picture>
</a>