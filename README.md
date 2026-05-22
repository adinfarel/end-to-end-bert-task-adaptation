# AlmondBERT — End-to-End BERT Task Adaptation from Scratch

> *Small like an almond. Built to understand language, not to generate noise.*

AlmondBERT is a small BERT-style encoder model built from scratch to understand the full lifecycle of modern language representation learning: tokenizer, bidirectional Transformer encoder, Masked Language Modeling pre-training, and supervised task adaptation.

**This is not a fine-tuned version of an existing BERT model.** Every core component is implemented manually: tokenizer pipeline, encoder-only Transformer architecture, MLM objective, and downstream fine-tuning heads for language understanding tasks.
