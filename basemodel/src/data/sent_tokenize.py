'''
src/data/sent_tokenize.py

Tokenize sentence large plain text dataset Wikipedia
example:
"Hello, my friend how are you? What do you think...."
 ->
["Hello, my friend how are you?", "What do you think...."]
'''

from utils.common import load_txt, save_txt
from configs.config import CONFIG

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

def sentences_tokenize() -> None:
    print(f"Load dataset {CONFIG.datasets.dataset_name} .txt")
    text = load_txt(CONFIG.datasets.dataset_raw_path)
    
    print("Sentences-tokenize dataset text")
    sentences = sent_tokenize(text=text)
    
    print("Filter sentences that too short (Noise)")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip().split()) >= 5]
    sentences = sentences[:int(CONFIG.datasets.max_dataset)]
    
    print("Save dataset that has been sentences tokenized")
    save_txt(
        CONFIG.datasets.dataset_tokenize_path,
        "\n".join(sentences) + "\n",
        mode="w"
    )
    
    print(
        f"Sentence tokenization completed. "
        f"Saved {len(sentences)} sentences at {CONFIG.datasets.dataset_tokenize_path}"
    )
    print(f"Overview dataset:\n{sentences[:10]}")

if __name__ == "__main__":
    sentences_tokenize()