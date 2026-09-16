# dialog-multi-epoch-experiments

Multi-turn Chinese dialogue fine-tuning experiments across different training epochs, served through a Gradio chat UI.

## Overview

This repository benchmarks BLOOM-family Chinese language models fine-tuned for
multi-turn dialogue at different training durations (2, 3, and 5+5 epochs).
Two Gradio front-ends are provided so the resulting checkpoints can be compared
interactively side by side, with streaming token output and Simplified/
Traditional Chinese conversion.

## Model / Approach

Three checkpoints were produced from the same training pipeline, varying only
the base model and the number of epochs:

| Checkpoint directory                        | Epochs | Base model                       | Language               |
| ------------------------------------------- | ------ | -------------------------------- | ---------------------- |
| `my-pretrained-2epochs/`                    | 2      | Langboat BLOOM                   | Simplified Chinese     |
| `my-pretrained-3epochs-YeungNLP-zh-ch/`     | 3      | YeungNLP BLOOM                   | Traditional-optimised  |
| `my-pretrained-5+5epochs-Langboat-zh-cn/`   | 5 + 5  | Langboat BLOOM (two-stage)       | Simplified Chinese     |

- Task: multi-turn dialogue generation (causal LM)
- Prompt format: `Human: Q1\nAssistant: A1\nHuman: Q2\nAssistant: ...`
- Training data: YeungNLP Simplified Chinese conversational corpus
  (`NLP/train_dataset_YeungNLP*`) with ~16 MB train / ~241 KB validation splits
- Training notebook: `NLP/w15-10-full-finetune-YeungNLP.ipynb`
- Default generation parameters used in the demo:
  `max_new_tokens=200`, `temperature=1.0`, `top_p=0.95`, `top_k=200`,
  `repetition_penalty=1.2`

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.28+
- gradio (Gradio 3/4 compatible)
- accelerate, safetensors, sentencepiece
- opencc-python-reimplemented (Simplified/Traditional conversion)
- A CUDA GPU with ~2-4 GB VRAM is recommended; CPU inference works but is slow

Install with:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Elainedu/dialog-multi-epoch-experiments.git
cd dialog-multi-epoch-experiments

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download or place a checkpoint under the repo root, e.g.
#    ./my-pretrained-2epochs/
#    Adjust the `model_name_or_path` variable inside the launcher script.

# 4. Launch the Gradio demo
python gradio.py
# or the streaming variant
python 10-10-gradio-chat-streaming.py
```

The Gradio app defaults to `http://localhost:7860`.

## Project Structure

```
dialog-multi-epoch-experiments/
├── gradio.py                                # Chatbot UI with OpenCC + streaming
├── 10-10-gradio-chat-streaming.py           # Streaming-focused variant (TextIteratorStreamer)
├── requirements.txt
├── NLP/
│   ├── train_dataset_YeungNLP*/             # Train / val splits (Simplified Chinese)
│   └── w15-10-full-finetune-YeungNLP.ipynb  # Full fine-tuning notebook
├── my-pretrained-2epochs/                   # 2-epoch checkpoint (config only in git)
├── my-pretrained-3epochs-YeungNLP-zh-ch/    # 3-epoch YeungNLP checkpoint
├── my-pretrained-5+5epochs-Langboat-zh-cn/  # 5+5-epoch Langboat checkpoint
├── configs/  data/  notebooks/  results/  src/
└── README.md
```

## Gradio UI

The chat interface exposes:

- Conversation history as `[(user, bot), ...]` tuples, re-serialised into the
  BLOOM prompt template on every turn
- Streaming output via `TextIteratorStreamer` and a background generation
  thread
- Custom `StoppingCriteria` to cut generation at `Human:` / end-of-turn markers
- Automatic Simplified <-> Traditional conversion (`s2t`, `t2s`) using OpenCC
- Sliders for `max_new_tokens`, `temperature`, `top_p`, `top_k`,
  `repetition_penalty`
- Preset example prompts (Harry Potter intro, capitals quiz, pizza recipe,
  article generation, news headlines, ML algorithm explanation)

## Empirical Notes

Informal observations across the three checkpoints:

1. **2 epochs** - basic conversational ability, occasional repetition.
2. **3 epochs** - noticeably more fluent, fewer loops.
3. **5+5 epochs** - most coherent responses.

Tips: raise `repetition_penalty` if the model loops, lower `temperature` for
factual Q&A, and adjust `top_p` / `top_k` to trade off diversity vs.
determinism.

## Notes

- **Large files are not tracked in git.** The `my-pretrained-*/` weight files
  (`pytorch_model.bin` / `model.safetensors`, ~600 MB each) are excluded.
  To reproduce the demo you must either:
  1. Re-run the notebook in `NLP/` to train a checkpoint yourself, or
  2. Load a comparable base model from HuggingFace, e.g.
     [`Langboat/bloom-389m-zh`](https://huggingface.co/Langboat/bloom-389m-zh)
     or [`YeungNLP/bloom-1b1-zh`](https://huggingface.co/YeungNLP/bloom-1b1-zh),
     and point `model_name_or_path` at it.
- **Model path** in each launcher script must match the actual directory you
  place under the repo root.
- **CUDA is auto-detected.** With no GPU the demo falls back to CPU (much
  slower); on GPU the model is loaded in FP16 to save VRAM.
- **OpenCC** is required for the built-in Traditional/Simplified conversion -
  make sure `opencc-python-reimplemented` installs cleanly.

## License

Educational use only. Base model weights follow the license of their upstream
publishers (BLOOM RAIL License for BLOOM-derived checkpoints). The demo code
in this repository may be freely used, modified, and redistributed for
teaching and research.
