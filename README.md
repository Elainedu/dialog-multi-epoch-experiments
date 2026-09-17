# dialog-multi-epoch-experiments

Multi-turn Chinese dialogue fine-tuning experiments on BLOOM at different
epoch counts, wrapped in two Gradio chat UIs (standard and streaming) so
the resulting checkpoints can be compared interactively.

> **Academic context:** part of a series of LLM fine-tuning experiments
> exploring how training duration and base-model choice affect Chinese
> dialogue quality on the BLOOM / BLOOMZ family.

---

## Table of Contents

- [What This Does](#what-this-does)
- [Model & Dataset](#model--dataset)
- [Training Setup](#training-setup)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Key Files](#key-files)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Reproducing Training](#reproducing-training)
- [Running Inference](#running-inference)
- [Notes](#notes)
- [Files Not in Repo](#files-not-in-repo)
- [References](#references)
- [License](#license)

---

## What This Does

Three checkpoints are produced by full supervised fine-tuning of BLOOM-family
Chinese base models at different training durations (2, 3, and 5 + 5 epochs).
Two Gradio front-ends (`app.py`, `chat_streaming.py`) load a chosen
checkpoint and expose a multi-turn chat interface with Simplified/
Traditional Chinese conversion, letting the epoch-count effect be observed
directly in conversation.

The goal is empirical: hold the training recipe fixed and vary only the
epoch count / base model to see how coherence, repetition, and fluency
change.

---

## Model & Dataset

**Base models (three checkpoints):**

| Checkpoint directory                          | Epochs | Base model                | Notes                       |
| --------------------------------------------- | ------ | ------------------------- | --------------------------- |
| `my-pretrained-2epochs/`                      | 2      | `Langboat/bloom-389m-zh`  | Simplified Chinese          |
| `my-pretrained-3epochs-YeungNLP-zh-ch/`       | 3      | `YeungNLP/bloom-1b1-zh`   | Traditional-Chinese focus   |
| `my-pretrained-5+5epochs-Langboat-zh-cn/`     | 5 + 5  | `Langboat/bloom-389m-zh`  | Two-stage continued SFT     |

**Dataset:**

- YeungNLP simplified-Chinese human/assistant dialog corpus
- Bundled in the repo as `NLP/train_dataset_YeungNLP<...>.zip`
  (~1.29 MB compressed; ~16 MB train / ~241 KB val after extraction)
- Prompt template (multi-turn):

  ```
  Human: <turn 1 user>
  Assistant: <turn 1 bot>
  Human: <turn 2 user>
  Assistant:
  ```

**Task:** causal language modelling (autoregressive generation).

---

## Training Setup

Full supervised fine-tuning (all parameters updated) — no LoRA / no
quantisation. The training notebook lives under `NLP/` as
`w15-10-<full-finetune-YeungNLP-simplified>.ipynb` (Chinese filename on
disk) and is mirrored under `notebooks/`.

| Setting                    | Value                              |
| -------------------------- | ---------------------------------- |
| Method                     | Full SFT (causal LM)               |
| Epochs                     | 2 / 3 / 5 + 5 (three runs)         |
| Precision                  | fp16 on GPU (auto-detected)        |
| GPU used                   | single CUDA GPU (T4-class, ~12 GB) |
| Approx. train time / epoch | ~30–60 min on T4 for 389M model    |
| Tokenizer padding side     | **left** (BLOOM requirement)       |
| Data collator              | `DataCollatorForSeq2Seq` (pad → -100 on labels) |

Default **inference** generation parameters used in the demo:

```
max_new_tokens      = 200
temperature         = 1.0
top_p               = 0.95
top_k               = 200
repetition_penalty  = 1.2
```

Custom `StoppingCriteria` cuts generation at `Human:` / EOS to avoid
run-on turns.

---

## System Architecture

```
                    ┌──────────────────────────────┐
                    │ YeungNLP simplified dialog   │
                    │ corpus (~16 MB train)        │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ BloomTokenizerFast           │
                    │ (Langboat / YeungNLP variant)│
                    └──────────────┬───────────────┘
                                   │  Human/Assistant turns
                                   ▼
     ┌───────────────────────────────────────────────────────┐
     │  Training loop (full SFT, fp16)                       │
     │  - AutoModelForCausalLM(BLOOM base)                   │
     │  - DataCollatorForSeq2Seq (label pad = -100)          │
     │  - 2 / 3 / 5+5 epochs                                 │
     └──────────────┬────────────────────────────────────────┘
                    │  checkpoints
                    ▼
     ┌─────────────────────────────────────────────────────┐
     │ my-pretrained-2epochs/                              │
     │ my-pretrained-3epochs-YeungNLP-zh-ch/               │
     │ my-pretrained-5+5epochs-Langboat-zh-cn/             │
     └──────────────┬──────────────────────────────────────┘
                    │
                    ▼
     ┌─────────────────────────────────────────────────────┐
     │ Gradio serve                                        │
     │  app.py             → standard chat (blocking)      │
     │  chat_streaming.py  → TextIteratorStreamer + Thread │
     │                                                     │
     │  Multi-turn history → prompt template → generate    │
     │  OpenCC s2t / t2s for Trad/Simp conversion          │
     │  Custom StoppingCriteria on Human:/EOS              │
     └──────────────┬──────────────────────────────────────┘
                    │
                    ▼
              http://localhost:7860
```

---

## Repository Layout

```
dialog-multi-epoch-experiments/
├── app.py                                          # Gradio chat UI (blocking)
├── chat_streaming.py                               # Gradio chat UI (streaming)
├── requirements.txt
├── README.md
├── NLP/
│   ├── train_dataset_YeungNLP<simplified>.zip      # zipped train/val splits (CJK filename on disk)
│   └── w15-10-<full-finetune-YeungNLP>.ipynb       # full-SFT training notebook (CJK filename on disk)
├── notebooks/
│   └── w15-10-<full-finetune-YeungNLP>.ipynb       # mirror of the training notebook
├── my-pretrained-3epochs-YeungNLP-zh-ch-<...>.zip  # bundled 3-epoch checkpoint (~700 KB config-only)
├── my-pretrained-2epochs/                          # (weights excluded from git)
├── my-pretrained-3epochs-YeungNLP-zh-ch/           # (weights excluded from git)
└── my-pretrained-5+5epochs-Langboat-zh-cn/         # (weights excluded from git)
```

---

## Key Files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio Chatbot UI. Loads a local BLOOM checkpoint, builds the multi-turn prompt from history, applies OpenCC conversion, generates a full response, returns it to the UI. |
| `chat_streaming.py` | Same UI, but uses `TextIteratorStreamer` in a background `Thread` to stream tokens as they are generated (typewriter effect). |
| `NLP/w15-10-<full-finetune-YeungNLP>.ipynb` | Full training notebook (Chinese filename on disk). Loads dataset, tokenises, sets up `TrainingArguments`, runs full-parameter SFT, saves `my-pretrained-*` checkpoint. |
| `notebooks/w15-10-<full-finetune-YeungNLP>.ipynb` | Duplicate of the training notebook kept under `notebooks/`. |
| `NLP/train_dataset_YeungNLP<simplified>.zip` | Zipped Arrow-format dataset (train + val). Filename on disk is in Chinese. |
| `requirements.txt` | Broad course-wide dependency list (torch 2.0, transformers 4.28, peft, gradio, opencc, etc.). Only a subset is needed at inference time. |

> **Note on `gradio.py` → `app.py` rename.** The docstrings inside
> `app.py` and `chat_streaming.py` still refer to `python gradio.py` as the
> launch command from an earlier layout. The actual entry points are now
> `app.py` and `chat_streaming.py` — the docstrings are historical.

---

## Requirements

The `requirements.txt` in this repo is the course-wide superset
(TensorFlow, YOLO, Django, etc.). The minimum set actually needed by the
Gradio demos is:

```
python>=3.10
torch>=2.0
transformers>=4.28
accelerate
safetensors
sentencepiece
gradio
opencc-python-reimplemented
```

Hardware:

- **GPU (recommended):** any CUDA GPU with ~2–4 GB VRAM for FP16 inference
  of the 389M base; ~4–8 GB for the 1.1B base.
- **CPU:** works but is slow (a few tokens/second).
- For **training**, a T4-class GPU (~12 GB) is enough for the 389M model at
  micro-batch 4.

Install:

```bash
pip install -r requirements.txt
```

---

## Configuration

There are no environment variables or config files — settings live inline
at the top of `app.py` / `chat_streaming.py`:

```python
model_name_or_path = "my-pretrained-2epochs"   # ← point at your checkpoint
```

Change this string to switch between the 2 / 3 / 5+5 epoch checkpoints,
or to a HuggingFace model id such as `Langboat/bloom-389m-zh` /
`YeungNLP/bloom-1b1-zh` for a baseline.

The generation sliders in the Gradio UI expose `max_new_tokens`,
`temperature`, `top_p`, `top_k`, and `repetition_penalty` at runtime.

---

## Reproducing Training

```bash
# 1. Clone the repo
git clone https://github.com/Elainedu/dialog-multi-epoch-experiments.git
cd dialog-multi-epoch-experiments

# 2. Install deps
pip install -r requirements.txt

# 3. Unzip the dataset
unzip NLP/train_dataset_YeungNLP*.zip -d NLP/

# 4. Open the training notebook
jupyter notebook NLP/w15-10-*.ipynb

# 5. Adjust the epoch count / base model at the top of the notebook:
#      num_train_epochs = 2   # or 3, or 5 (then rerun for 5+5)
#      model_name       = "Langboat/bloom-389m-zh"   # or YeungNLP/bloom-1b1-zh
#    then Run All.

# 6. The trained checkpoint is written to `my-pretrained-<...>/`
#    Point the Gradio launcher at it and re-launch.
```

To reproduce the **5+5** run, load the 5-epoch checkpoint as the base and
continue for another 5 epochs on the same data.

---

## Running Inference

```bash
# Standard chat UI (waits for full response, then displays)
python app.py

# Streaming chat UI (typewriter effect via TextIteratorStreamer)
python chat_streaming.py
```

Both launch on `http://127.0.0.1:7860`. The UI provides:

- Multi-turn chat history (`[(user, bot), ...]`), re-serialised to the
  BLOOM prompt on every turn.
- Sliders for `max_new_tokens`, `temperature`, `top_p`, `top_k`,
  `repetition_penalty`.
- Preset example prompts (self-introduction, capital cities, recipe
  generation, ML explanations, etc.).
- Automatic Simplified ↔ Traditional Chinese conversion via OpenCC
  (`s2t`, `t2s`).

Empirical observations from the three checkpoints:

| Epochs | Behaviour |
| --- | --- |
| 2 | Basic conversational ability; occasional repetition. |
| 3 | Noticeably more fluent, fewer generation loops. |
| 5 + 5 | Most coherent; risk of over-training on repeated patterns. |

Tips: raise `repetition_penalty` if the model loops, lower `temperature`
for factual answers, and adjust `top_p` / `top_k` to trade off diversity
vs. determinism.

---

## Notes

- **CUDA is auto-detected.** With no GPU the model runs on CPU (much
  slower); on GPU the weights load in FP16 to save VRAM.
- **OpenCC** is required for the built-in Traditional/Simplified conversion;
  make sure `opencc-python-reimplemented` installs cleanly.
- **BLOOM padding side.** The BLOOM tokenizer defaults to **left** padding
  — do not flip it to right for causal-LM training.
- **Historical launch names.** Docstrings mention `python gradio.py`;
  the modern entry points are `app.py` / `chat_streaming.py`.

---

## Files Not in Repo

To keep the repository small, large binary artefacts are excluded:

| Excluded                                       | Why                | How to obtain / regenerate                     |
| ---------------------------------------------- | ------------------ | ---------------------------------------------- |
| `my-pretrained-2epochs/pytorch_model.bin`      | ~600 MB per file   | Rerun the training notebook with `epochs=2`.   |
| `my-pretrained-3epochs-YeungNLP-zh-ch/*.bin`   | ~600 MB per file   | Rerun with `epochs=3` on the YeungNLP base.    |
| `my-pretrained-5+5epochs-Langboat-zh-cn/*.bin` | ~600 MB per file   | Rerun with `epochs=5`, then continue for 5 more.|
| Extracted dataset Arrow files                  | already zipped     | `unzip NLP/train_dataset_YeungNLP*.zip -d NLP/`.|

Instead of retraining, you can also point `model_name_or_path` at the
public HuggingFace base models
([`Langboat/bloom-389m-zh`](https://huggingface.co/Langboat/bloom-389m-zh)
or [`YeungNLP/bloom-1b1-zh`](https://huggingface.co/YeungNLP/bloom-1b1-zh))
to sanity-check the UI without the fine-tuned weights.

---

## References

- Le Scao et al. **BLOOM: A 176B-Parameter Open-Access Multilingual
  Language Model.** arXiv:2211.05100. <https://arxiv.org/abs/2211.05100>
- Muennighoff et al. **Crosslingual Generalization through Multitask
  Finetuning (BLOOMZ / mT0).** arXiv:2211.01786.
  <https://arxiv.org/abs/2211.01786>
- Base models used:
  - <https://huggingface.co/Langboat/bloom-389m-zh>
  - <https://huggingface.co/YeungNLP/bloom-1b1-zh>
- Gradio streaming pattern (`TextIteratorStreamer`):
  <https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextIteratorStreamer>
- OpenCC (Simplified/Traditional conversion):
  <https://github.com/BYVoid/OpenCC>

---

## License

Educational use only. Base model weights follow the licenses of their
upstream publishers (BLOOM RAIL License for BLOOM-derived checkpoints).
Demo code, notebooks, and training scripts in this repository may be
freely used, modified, and redistributed for teaching and research.
