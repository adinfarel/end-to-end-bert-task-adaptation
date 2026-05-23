'''
tests/validate_tokenizer.py

Check whether tokenizer healthy or not
'''

from utils.common import load_bin, load_txt
from configs.config import CONFIG
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT

if __name__ == "__main__":
    ids = load_bin(CONFIG.datasets.dataset_processed_path)
    print(ids[:100])
    print(ids.shape)
    print(ids.max())
    print(ids.min())
    
    # VALIDATE TEXT ORIGINAL == ? TEXT DECODE TOKENIZER
    text_original = load_txt(CONFIG.datasets.dataset_tokenize_path)
    tokenizer = AlmondTokenizerBERT(CONFIG)
    tokenizer.load(
        vocab_path=CONFIG.tokenizer.vocab_path,
        merges_path=CONFIG.tokenizer.merges_path,
    )
    
    sample_ids = ids[:100].tolist()
    decoded = tokenizer.decode(sample_ids)
    
    print("="*30 + "DECODED" + "=" * 30)
    print("Text after decode through tokenizer:")
    print(decoded)
    print("="*30 + "ORIGINAL" + "=" * 30)
    print("Text before decode through tokenizer:")
    print(text_original[:len(decoded)])