'''
src/data/loader.py

Create DataLoader, Dataset, and collate_fn for dataset pretraining
'''

import torch
import random
from torch.utils.data import Dataset, DataLoader
from functools import partial
from pathlib import Path
from box import Box

import jsonlines
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT
from configs.config import CONFIG

# HELPER
def load_jsonl(file_path: str) -> list[dict[str, list[int]]]:
    '''Load jsonlines file.'''
    if not file_path.endswith(".jsonl"):
        raise ValueError(f"File {file_path} not end with .jsonl. Extension mismatch")

    path = Path(file_path)
    
    if not path.exists():
        raise FileExistsError(f"File with {file_path} not exists.")
    
    with jsonlines.open(path) as reader:
        data = list(reader)
    
    return data
    
class BERTDatasetPretraining(Dataset):
    def __init__(self, data: list[dict[str, list[int]]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix) -> list[int]:
        return self.data[ix]["tokens"]

def collate_fn(
    batch: list[list[int]],
    tokenizer: AlmondTokenizerBERT,
    max_len: int,
    mask_prob: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Function that run by DataLoader'''
    
    CLS_ID = tokenizer.special_token_to_id["[CLS]"]
    PAD_ID = tokenizer.special_token_to_id["[PAD]"]
    MASK_ID = tokenizer.special_token_to_id["[MASK]"]
    SEP_ID = tokenizer.special_token_to_id["[SEP]"]
    
    vocab_ids = list(tokenizer.vocab.keys())
    
    input_ids_batch = []
    attn_mask_batch = []
    labels_batch = []
    apnd_input = input_ids_batch.append; apnd_attn = attn_mask_batch.append; apnd_lbls = labels_batch.append
    
    for token in batch:
        token_ids = token[:max_len - 2]
        
        tokens = [CLS_ID] + token_ids + [SEP_ID]
        
        labels = [-100] * len(tokens)
        
        for j in range(1, len(tokens) - 1):
            if random.random() < mask_prob:
                labels[j] = tokens[j]
                
                r = random.random()
                if r < 0.8:
                    tokens[j] = MASK_ID
                
                elif r < 0.9:
                    tokens[j] = random.choice(vocab_ids)
        
        pad_len     = max_len - len(tokens)
        attn_mask   = [1] * len(tokens) + [0] * pad_len
        tokens      = tokens + [PAD_ID] * pad_len
        labels      = labels + [-100] * pad_len
        
        apnd_input(torch.tensor(tokens, dtype=torch.long))
        apnd_attn(torch.tensor(attn_mask, dtype=torch.long))
        apnd_lbls(torch.tensor(labels, dtype=torch.long))
    
    return (
        torch.stack(input_ids_batch),
        torch.stack(attn_mask_batch),
        torch.stack(labels_batch),
    )

def create_dataloaders(
    data: list[dict[str, list[int]]],
    tokenizer: AlmondTokenizerBERT,
    config: Box,
    shuffle: bool = True,
) -> DataLoader:
    '''Create dataloader for any dataset'''
    dataset = BERTDatasetPretraining(data)
    
    custom_collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        max_len=config.models.max_len,
        mask_prob=float(config.models.mask_prob),
    )
    
    return DataLoader(
        dataset,
        batch_size=config.models.batch_size,
        collate_fn=custom_collate,
        shuffle=shuffle
    )

if __name__ == "__main__":
    train_data_path: str = CONFIG.datasets.dataset_processed_path + "train_data.jsonl"
    val_data_path: str = CONFIG.datasets.dataset_processed_path + "val_data.jsonl"
    
    print(f"Initialized tokenizer: {AlmondTokenizerBERT.__name__}")
    vocab_path: str = CONFIG.tokenizer.vocab_path
    merges_path: str = CONFIG.tokenizer.merges_path
    tokenizer = AlmondTokenizerBERT(CONFIG)
    tokenizer.load(
        vocab_path=vocab_path,
        merges_path=merges_path
    )
    
    print(f"Load jsonlines file from\nTrain data: {train_data_path}\nVal data: {val_data_path}")
    train_data: list[dict[str, list[int]]] = load_jsonl(train_data_path)
    val_data: list[dict[str, list[int]]] = load_jsonl(val_data_path)
    
    print("Creating DataLoader for Train data and Val data")
    train_loader = create_dataloaders(
        data=train_data,
        tokenizer=tokenizer,
        config=CONFIG,
        shuffle=False
    )
    val_loader = create_dataloaders(
        data=val_data,
        tokenizer=tokenizer,
        config=CONFIG,
        shuffle=False
    )
    
    print("===============SAMPLE================")
    data_iter = iter(train_loader)
    first_batch = next(data_iter)
    print("Tokens after:")
    print("Input ids: ",first_batch[0][0])
    print("Attn mask: ",first_batch[0][1])
    print("Label ids: ",first_batch[0][2])