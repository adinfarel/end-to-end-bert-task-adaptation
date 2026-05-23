'''
src/training/train_tokenizer.py

Training pipeline for tokenizer to get vocab and merges
'''

from configs.config import CONFIG
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT
from utils.common import load_txt, save_yaml

from pathlib import Path
from box import Box

def main(config: Box) -> None:
    print(f"Training pipeline for tokenizer {AlmondTokenizerBERT.__name__}")
    
    data = load_txt(config.datasets.dataset_tokenize_path)
    print(f"Raw sentence-tokenize corpus load from {config.datasets.dataset_tokenize_path}. Ready for training")
    
    print(f"Initialized tokenizer BERT: {AlmondTokenizerBERT.__name__}. Vocab size before train tokenizer: {config.tokenizer.vocab_size}")
    tokenizer = AlmondTokenizerBERT(config)
    tokenizer.train(data)
    tokenizer.save()
    
    # REWRITE VOCAB_SIZE
    config.tokenizer.vocab_size = tokenizer.get_vocab_size
    save_yaml(file_path=Path("configs/config.yaml"), data=config.to_dict())
    
    print(f"Training pipeline complete. Vocab size after train tokenizer: {tokenizer.get_vocab_size}. All process done")

if __name__ == "__main__":
    main(CONFIG)