'''
src/data/processed.py

Convert corpus text into corpus integer
'''

from configs.config import CONFIG
from utils.common import save_bin, load_txt
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT

import argparse
from box import Box
from tqdm import tqdm

def processed_dataset(config: Box, train_split: bool = True, corpus: bool = True, train_ratio: float = 0.95) -> None:
    print("Processing corpus text data into corpus integer data")
    
    print(f"Load raw corpus data text from {config.datasets.dataset_tokenize_path}")
    data = load_txt(config.datasets.dataset_tokenize_path)
    
    print(f"Initialized tokenizer BPE: {AlmondTokenizerBERT.__name__}")
    tokenizer = AlmondTokenizerBERT(config)
    tokenizer.load(
        vocab_path=config.tokenizer.vocab_path,
        merges_path=config.tokenizer.merges_path,
    )
    
    print(f"Starting process convert corpus text to corpus integer")
    all_token_ids = []
    extend = all_token_ids.extend
    lines = data.splitlines(keepends=True)
    
    print(f"Tokenizing {len(lines)} lines of text")
    
    if corpus:
        for i, line in tqdm(enumerate(lines), desc="Tokenizing text"):
            if not line.strip():
                continue
            
            token_ids = tokenizer.encode(line)
            if i % 100 == 0:
                print(
                    f"Steps: {i}"
                    f"\nLines text: {line[:10]}"
                    f"\nToken ids: {token_ids[:10]}"
                )
            extend(token_ids)
        print("Tokenizing corpus text complete")

        file_path = config.datasets.dataset_processed_path + "corpus.bin"
        save_bin(file_path=file_path, data=all_token_ids)
        print(f"Processed corpus integer save to path: {file_path}")
    
    if train_split:
        split_idx = int(train_ratio * len(lines))
        splits = [(lines[:split_idx], "train"), (lines[split_idx:], "val")]
        
        print(f"Tokenizing train data: {train_ratio * 100:.2f}% of corpus and "
              f"val data: {(1 - train_ratio) * 100:.2f}% of corpus.")
        
        for data, name in splits:
            all_tokens = []
            extend = all_tokens.extend
            
            for i, line in tqdm(enumerate(data), desc=f"Tokenizing {name} data"):
                if not line.strip():
                    continue
                
                token_ids = tokenizer.encode(line)
                if i % 100 == 0:
                    print(
                        f"Steps: {i}"
                        f"\nLines text: {line[:10]}"
                        f"\nToken ids: {token_ids[:10]}"
                    )
                extend(token_ids)
            
            file_path = config.datasets.dataset_processed_path + f"{name}_data.bin"
            save_bin(file_path=file_path, data=all_tokens)
            print(f"Tokenizing {name} data complete. Save to {file_path}")

def str2bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    
    raise argparse.ArgumentTypeError("Boolean value expected.")   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tokenizing corpus text into corpus bin")
    
    parser.add_argument(
        "--train-split",
        type=str2bool,
        default=True,
        help="Set true if wanna get split dataset binary -> [train.bin, val.bin]"
    )
    
    parser.add_argument(
        "--corpus",
        type=str2bool,
        default=False,
        help="Set true if wanna get corpus dataset binary -> corpus.bin"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Set ratio of train and val, better ratio better model."
    )
    
    args = parser.parse_args()
    
    processed_dataset(
        CONFIG,
        corpus=args.corpus,
        train_split=args.train_split,
        train_ratio=args.train_ratio,
    )