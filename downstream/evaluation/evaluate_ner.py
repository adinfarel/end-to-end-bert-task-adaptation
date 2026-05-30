'''
downstream/evaluation/evaluate_ner.py

Evaluate how good model in terms numeric.
'''

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import torch
from box import Box
from tqdm import tqdm
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from configs.config import CONFIG
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT
from downstream.src.data.loader import load_jsonl, build_label_map
from downstream.src.model.ner import AlmondBERTForNER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def encode_words(
    tokens: list[str],
    tokenizer: AlmondTokenizerBERT,
    max_len: int,
) -> tuple[list[int], list[int], list[int]]:
    """Convert word-level token into BPE input ids for NER evaluation."""
    
    cls_id = tokenizer.special_token_to_id["[CLS]"]
    sep_id = tokenizer.special_token_to_id["[SEP]"]
    
    
    input_ids = [cls_id]
    first_subword_positions: list[int] = []
    
    max_body_len = max_len - 2
    for word in tokens:
        word_piece_ids = tokenizer.encode(word)
        
        if not word_piece_ids:
            continue
        
        if len(input_ids) - 1 + len(word_piece_ids) > max_body_len:
            break
        
        first_subword_positions.append(len(input_ids))
        input_ids.extend(word_piece_ids)
    
    input_ids.append(sep_id)
    
    attn_mask = [1] * len(input_ids)
    
    return input_ids, attn_mask, first_subword_positions

def pad_batch(
    input_ids_batch: list[list[int]],
    attn_mask_batch: list[list[int]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in input_ids_batch)
    
    padded_ids = []
    padded_masks = []
    
    for input_ids, attn_mask in zip(input_ids_batch, attn_mask_batch):
        pad_len = max_len - len(input_ids)
        
        padded_ids.append(input_ids + [pad_token_id] * pad_len)
        padded_masks.append(attn_mask + [0] * pad_len)
        
    return (
        torch.tensor(padded_ids, dtype=torch.long, device=DEVICE),
        torch.tensor(padded_masks, dtype=torch.long, device=DEVICE),
    )

@torch.no_grad()
def evaluate_ner(
    config: Box,
    split: str = "test",
    batch_size: int = 32
) -> None:
    """Evaluate NER Model using seqeval."""
    print("EVALUATE ALMOND BERT FOR NER")
    print(f"Device: {DEVICE}")
    print(f"Split: {split}")
    
    tokenizer = AlmondTokenizerBERT(config)
    tokenizer.load(
        vocab_path=config.tokenizer.vocab_path,
        merges_path=config.tokenizer.merges_path,
    )

    train_path = Path(config.downstream.raw_dataset_path) / "train.jsonl"
    eval_path = Path(config.downstream.raw_dataset_path) / f"{split}.jsonl"

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval split not found: {eval_path}")
    
    train_data = load_jsonl(train_path)
    eval_data = load_jsonl(eval_path)
    
    label_to_id, id_to_label = build_label_map(train_data)
    num_labels = len(label_to_id)

    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples : {len(eval_data)}")
    print(f"Labels       : {label_to_id}")
    
    base_ckpt_path = Path(config.models.model_save_path) / "best_model.pt"
    ner_ckpt_path = Path(config.downstream.save_ft_ner_path) / "best_model_ner.pt"
    
    if not base_ckpt_path.exists():
        raise FileNotFoundError(f"Base BERT checkpoint not found: {base_ckpt_path}")

    if not ner_ckpt_path.exists():
        raise FileNotFoundError(f"NER checkpoint not found: {ner_ckpt_path}")
    
    model = AlmondBERTForNER.from_pretrained(
        config=config,
        checkpoint_path=base_ckpt_path,
        num_labels=num_labels,
    )

    checkpoint = torch.load(ner_ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)

    model = model.to(DEVICE)
    model.eval()
    
    pad_token_id = tokenizer.special_token_to_id["[PAD]"]

    all_true_labels: list[list[str]] = []
    all_pred_labels: list[list[str]] = []
    
    for start in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
        batch = eval_data[start : start + batch_size]
        
        input_ids_batch: list[list[int]] = []
        attention_mask_batch: list[list[int]] = []
        first_positions_batch: list[list[int]] = []
        gold_labels_batch: list[list[str]] = []
        
        for item in batch:
            tokens = item['tokens']
            ner_tags = item['ner_tags']
            
            input_ids, attention_mask, first_positions = encode_words(
                tokens=tokens,
                tokenizer=tokenizer,
                max_len=config.models.max_len,
            )
            
            included_word_count = len(first_positions)
            gold_labels = ner_tags[:included_word_count]
            
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            first_positions_batch.append(first_positions)
            gold_labels_batch.append(gold_labels)
        
        input_ids_tensor, attn_mask_tensor = pad_batch(
            input_ids_batch=input_ids_batch,
            attn_mask_batch=attention_mask_batch,
            pad_token_id=pad_token_id
        )
        
        logits, _ = model(
            x=input_ids_tensor,
            attn_mask=attn_mask_tensor,
            targets=None
        )
        
        pred_ids = torch.argmax(logits, dim=-1)
        
        for sample_idx, first_positions in enumerate(first_positions_batch):
            sample_pred_labels: list[str] = []
            
            for pos in first_positions:
                label_id = int(pred_ids[sample_idx, pos].item())
                sample_pred_labels.append(id_to_label[label_id])
            
            all_pred_labels.append(sample_pred_labels)
            all_true_labels.append(gold_labels_batch[sample_idx])
    
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)
    
    print("\n==================== SUMMARY ====================")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    
    print("\n================ CLASSIFICATION REPORT ================")
    print(classification_report(all_true_labels, all_pred_labels, digits=4))
    
    print("\n================ SAMPLE PREDICTIONS ================")

    for i in range(min(5, len(eval_data))):
        tokens = eval_data[i]["tokens"]
        gold = all_true_labels[i]
        pred = all_pred_labels[i]

        print("-" * 80)
        print("TOKENS:", tokens[: len(gold)])
        print("GOLD  :", gold)
        print("PRED  :", pred)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AlmondBERT NER model.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )

    args = parser.parse_args()

    evaluate_ner(
        config=CONFIG,
        split=args.split,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()