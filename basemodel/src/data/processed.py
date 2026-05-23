'''
src/data/processed.py

Convert corpus text into corpus integer
'''

from configs.config import CONFIG
from utils.common import save_bin, load_txt
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT

from box import Box
from tqdm import tqdm

def processed_dataset(config: Box) -> None:
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
    for i, line in tqdm(enumerate(lines), desc="Tokenizing text"):
        token_ids = tokenizer.encode(line)
        if i % 100 == 0:
            print(
                f"Steps: {i}"
                f"\nLines text: {line[:10]}"
                f"\nToken ids: {token_ids[:10]}"
            )
        extend(token_ids)
    print("Tokenizing corpus text complete")
    
    save_bin(file_path=config.datasets.dataset_processed_path, data=all_token_ids)
    print(f"Processed corpus integer save to path: {config.datasets.dataset_processed_path}")

if __name__ == "__main__":
    processed_dataset(CONFIG)