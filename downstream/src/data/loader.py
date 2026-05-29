'''
src/data/loader.py

Create DataLoader, Dataset, collate_fn for datasets NER but with different approach from pre-training BERT
'''

from __future__ import annotations

import torch
import jsonlines
from typing import Any
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from functools import partial
from box import Box

from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT

# HELPER
def load_jsonl(file_path: Path) -> list[dict]:
    '''Load jsonl data.'''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File {str(file_path)} not found.")
    
    if file_path.suffix != ".jsonl":
        raise ValueError(f"Expected file .jsonl, got {str(file_path)}")
    
    with jsonlines.open(file_path, mode='r') as reader:
        data = list(reader)
    
    return data

def build_label_map(data: list[dict[str, Any]]) -> tuple[dict[str, int], dict[int, str]]:
    '''
    build label_to_id and id_to_label
    
    EXAMPLE LABELS:
    O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
    '''
    labels = set()
    
    for item in data:
        for tag in item['ner_tags']:
            labels.add(tag)
    
    # Keep "O" at index 0
    sorted_labels = ["O"] + sorted(label for label in labels if label != "O")
    
    labels_to_id = {label: i for i, label in enumerate(sorted_labels)}
    id_to_labels = {i: label for label, i in labels_to_id.items()}
    
    return labels_to_id, id_to_labels

def align_tokens_and_labels(
    tokens: list[str],
    ner_tags: list[str],
    tokenizer: AlmondTokenizerBERT,
    label_to_id: dict[str, int],
    max_len: int
) -> dict[str, list[int]]:
    """
    Convert word-level NER to BPE-token-level labels.
    
    Strategy:
    - Add [CLS] at beginning and [SEP] at end.
    - Encode each dataset token using custom BPE tokenizer.
    - Assign NER label only to the first BPE sub-token.
    - Assign -100 to continuation sub-tokens.
    - Assign -100 to [CLS] and [SEP].

    Why first-subword-only?
    Because original NER labels are word-level. If a word splits into multiple
    BPE tokens, counting all sub-tokens in the loss would overweight long words.
    """
    if len(tokens) != len(ner_tags):
        raise ValueError(
            f"Length tokens and ner_tags mismatch {len(tokens)} vs {len(ner_tags)}"
        )
    
    cls_id = tokenizer.special_token_to_id["[CLS]"]
    sep_id = tokenizer.special_token_to_id["[SEP]"]
    
    input_ids = [cls_id]
    labels = [-100]
    
    max_body_len = max_len - 2 # Cause 2 space for [CLS] and [SEP]
    
    for word, tag in zip(tokens, ner_tags):
        word_piece_ids = tokenizer.encode(word)
        
        if not word_piece_ids:
            continue
        
        # Prevent exceeding max length
        if len(input_ids) - 1 + len(word_piece_ids) > max_body_len:
            break
        
        tag_id = label_to_id[tag]
        
        input_ids.extend(word_piece_ids)
        
        labels.append(tag_id)
        labels.extend([-100] * (len(word_piece_ids) - 1))
    
    input_ids.append(sep_id)
    labels.append(-100)
    
    attn_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "labels": labels
    }

class BERTNERDataset(Dataset):
    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: AlmondTokenizerBERT,
        label_to_id: dict[str, int],
        max_len: int,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix) -> dict[str, list[int]]:
        item = self.data[ix]
        
        return align_tokens_and_labels(
            tokens=item['tokens'],
            ner_tags=item['ner_tags'],
            tokenizer=self.tokenizer,
            label_to_id=self.label_to_id,
            max_len=self.max_len,
        )

def collate_fn(
    batch: list[dict[str, list[int]]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PAD:
    
    input_ids: [PAD]
    attn_mask: 0
    labels   : -100
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    
    input_ids_batch = []
    attn_mask_batch = []
    labels_batch    = []
    
    for item in batch:
        input_ids = item["input_ids"]
        attn_mask = item["attn_mask"]
        labels    = item["labels"]
        
        pad_len = max_len - len(input_ids)
        
        input_ids = input_ids + [pad_token_id] * pad_len
        attn_mask = attn_mask + [0] * pad_len
        labels    = labels    + [-100] * pad_len
        
        input_ids_batch.append(torch.tensor(input_ids, dtype=torch.long))
        attn_mask_batch.append(torch.tensor(attn_mask, dtype=torch.long))
        labels_batch.append(torch.tensor(labels, dtype=torch.long))
    
    return (
        torch.stack(input_ids_batch),
        torch.stack(attn_mask_batch),
        torch.stack(labels_batch),
    )
    
def create_ner_dataloaders(
    data: list[dict[str, Any]],
    tokenizer: AlmondTokenizerBERT,
    label_to_id: dict[str, int],
    config: Box,
    shuffle: bool = True
) -> DataLoader:
    '''Create datasets into DataLoader'''
    dataset = BERTNERDataset(
        data=data,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_len=config.models.max_len,
    )
    
    pad_token_id = tokenizer.special_token_to_id["[PAD]"]
    
    custom_collate_fn = partial(
        collate_fn,
        pad_token_id=pad_token_id
    )
    
    return DataLoader(
        dataset=dataset,
        collate_fn=custom_collate_fn,
        batch_size=config.downstream.batch_size,
        shuffle=shuffle
    )