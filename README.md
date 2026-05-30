# AlmondBERT — End-to-End BERT Task Adaptation from Scratch

> *Small like an almond. Built to understand language, not to generate noise.*

AlmondBERT is a small BERT-style encoder model built from scratch to understand the full lifecycle of modern language representation learning: tokenizer, bidirectional Transformer encoder, Masked Language Modeling pre-training, and supervised task adaptation.

**This is not a fine-tuned version of an existing BERT model.** Every core component is implemented manually: BPE tokenizer, encoder-only Transformer architecture with modern design choices, MLM pre-training objective, and downstream NER fine-tuning head.

---

## Pipeline Overview

```
Wikipedia Simple English (raw text)
          ↓
sent_tokenize → per-sentence corpus
          ↓
BPE Tokenizer (from scratch)
          ↓
Pre-Training — AlmondBERT Base (MLM)
          ↓
Task Adaptation — NER Fine-Tuning
          ↓
AlmondBERT for Named Entity Recognition
```

---

## Architecture

AlmondBERT uses modern improvements over vanilla BERT 2018:

| Component | Vanilla BERT | AlmondBERT | Why |
|-----------|-------------|------------|-----|
| Positional Encoding | Learned PE | RoPE | Consistent relative distances, better extrapolation |
| Normalization | Post-LN LayerNorm | Pre-LN LayerNorm | Training stability |
| FFN Activation | GELU | GeGLU | Gated non-linearity, better gradient flow |
| Attention | Full MHA O(N²) | Alternating Local+Global | Compute efficiency |
| Padding | Standard padded | Unpadding | No wasted compute on PAD tokens |
| Attention kernel | Standard | Flash Attention | Memory efficient CUDA-level optimization |
| Pre-training | MLM + NSP | MLM only | NSP shown to be unhelpful (RoBERTa) |
| Tokenizer | WordPiece | BPE | Consistent with AlmondGPT ecosystem |

**Model size:** ~10-15M parameters
**Vocab size:** 10,887 (BPE early-stopped at low-frequency threshold)
**Context length:** 128 tokens

---

## Training Summary

### Pre-Training (MLM)
- **Dataset:** Wikipedia Simple English (~50k sentences)
- **Objective:** Masked Language Modeling — predict 15% randomly masked tokens
- **Masking:** 80% [MASK], 10% random token, 10% unchanged
- **Optimizer:** AdamW
- **Epochs:** 15
- **Final Eval Loss:** 6.0180

![Pre-Training Loss](docs/assets/pretrain_loss.png)

### NER Fine-Tuning
- **Dataset:** WikiANN-NER
- **Task:** Token-level classification — PER, ORG, LOC entities
- **Epochs:** 5
- **Learning Rate:** 2e-5
- **Final Eval Loss:** 0.6363

![NER Fine-Tuning Loss](docs/assets/ner_loss.png)

---

## Results — NER

```
Precision : 0.2912
Recall    : 0.4157
F1        : 0.3425
```

| Entity | Precision | Recall | F1 |
|--------|-----------|--------|----|
| LOC | 0.3349 | 0.4256 | 0.3748 |
| ORG | 0.2012 | 0.3014 | 0.2413 |
| PER | 0.3470 | 0.5253 | 0.4179 |

**Sample predictions:**

```
TOKENS: [..., 'India', ';', ..., 'Adyar', ...]
GOLD  : [..., 'B-LOC', ..., 'B-LOC', ...]
PRED  : [..., 'O', ...,     'B-LOC', ...]     ← missed "India", got "Adyar"

TOKENS: ['Blacktown', 'railway', 'station']
GOLD  : ['B-ORG', 'I-ORG', 'I-ORG']
PRED  : ['B-ORG', 'I-ORG', 'I-ORG']          ← correct ✓
```

---

## Limitations

- **Scale** — 10-15M parameters vs 110M BERT-base. Representations have limited capacity.
- **Pre-training corpus** — Wikipedia Simple English is encyclopedia-style. WikiANN is more entity-dense. Domain gap contributes to lower F1.
- **BPE subword fragmentation** — BPE can split entity names into non-meaningful fragments. WordPiece or Whole Word Masking would improve MLM quality.
- **No evaluation benchmark** — not evaluated on CoNLL-2003 or standard NER benchmarks.

**What this project proves:** the full encoder pre-training + task adaptation pipeline works end-to-end. Every component is implemented from scratch and functionally correct.

---

## Project Structure

```
end-to-end-bert-task-adaptation/
├── basemodel/
│   ├── src/
│   │   ├── model/          # Encoder, Block, Attention, GeGLU, RoPE, Unpadding
│   │   ├── tokenizer/      # BPE tokenizer from scratch
│   │   └── training/       # Pre-training loop
│   └── data/
│       ├── raw/            # Wikipedia raw + sent_tokenized corpus
│       └── processed/      # .jsonl per-sentence token IDs
├── downstream/
│   ├── src/
│   │   ├── model/          # AlmondBERTForNER — encoder + NER head
│   │   ├── data/           # NER dataset, collate_fn, dataloader
│   │   └── training/       # Fine-tuning loop + evaluation
│   └── data/
│       └── raw/            # WikiANN-NER train/val .jsonl
├── configs/
│   └── config.yaml
├── docs/
│   ├── pretrain.md
│   ├── ner.md
│   └── assets/
│       ├── pretrain_loss.png
│       └── ner_loss.png
├── utils/
└── plot_losses.py
```

---

## Setup

```bash
git clone https://github.com/adinfarel/end-to-end-bert-task-adaptation.git
cd end-to-end-bert-task-adaptation
pip install -e .
pip install torch datasets pyyaml tqdm matplotlib nltk python-box jsonlines
```

### Run Full Pre-Training Pipeline

```bash
# 1. Download and sent_tokenize Wikipedia corpus
python -m basemodel.src.data.sent_tokenize

# 2. Train BPE tokenizer
python -m basemodel.src.training.train_tokenizer

# 3. Tokenize corpus to .jsonl
python -m basemodel.src.data.processed --train-split true

# 4. Pre-train AlmondBERT (MLM)
python -m basemodel.src.training.train_model
```

### Run NER Fine-Tuning

```bash
python -m downstream.src.training.train_ner
```

### Generate Loss Plots

```bash
python plot_losses.py
```

---

## What I Learned

**BERT vs GPT is not about architecture, it's about objective.**

GPT always has an unambiguous ground truth — the next token. BERT's masked token can have multiple valid predictions — "The [MASK] Commission" could be "European" or "Federal" or "Trade". This inherent ambiguity means BERT loss stays higher than GPT loss even when training is going well. It's not a bug, it's the task.

**Pre-training quality is the ceiling for fine-tuning.**

You can't fine-tune your way out of bad representations. Better corpus, larger model, and longer pre-training raise that ceiling before fine-tuning even starts. F1 0.34 is not a fine-tuning failure — it's a pre-training scale limitation.

**Unpadding is more complex than it sounds.**

Removing padding tokens before attention means you lose the batch structure. Every token needs to know which sentence it came from, and attention must be masked to prevent cross-sentence information flow. The batch_id masking approach used here is the same technique used in production models.

**BPE and BERT have a fundamental tension.**

BPE splits by frequency, not linguistic boundaries. Masking a subword fragment is harder for the model than masking a full word. This is why BERT original used WordPiece, and why Whole Word Masking was introduced in later BERT variants. For downstream tasks, this matters less — the representations are still useful — but pre-training loss will always be higher than WordPiece-based models at the same scale.

---

*Built by Adinfarel — Semester 4, AI Engineering track*
*"The goal was never to build something impressive. The goal was to understand."*
