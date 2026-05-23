'''
src/tokenizer/bpe.py

Tokenize text to token id based on vocab and merges
'''

from utils.common import load_json, save_json

import regex as re
from collections import Counter
from box import Box
from tqdm import tqdm

class AlmondTokenizerBERT:
    
    SPECIAL_TOKENS = [
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[PAD]",  #TODO: if i'm use WordPiece add [UNK] token
    ]
    
    def __init__(self, config: Box) -> None:
        self.config: Box = config
        self.vocab_size: int = self.config.tokenizer.vocab_size
        self.single_byte_size: int = 256 # Byte-level 0-255 (Totally 256)
        self.vocab: dict[int, bytes] = {}
        self.merges: dict[tuple[int, int], int] = {}
        
        self.special_token_to_id: dict[str, int] = {
            token: self.single_byte_size + i
            for i, token in enumerate(self.SPECIAL_TOKENS)
        }
        
        self.special_id_to_token: dict[int, str] = {
            idx: token
            for token, idx in self.special_token_to_id.items()
        }
    
    @property
    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_stats(self, ids: list[int]) -> Counter[tuple[int, int]]:
        '''Get stats token based on frequency of appearance of 2 adjacent tokens.
        
        Example:
        text: "helolo"
        token: [10, 20, 30, 40, 30, 40]
        
        -> token: [10, 20, new_id, new_id], cause "l" & "o" most freq (freq=2) so merges into new token or new id
        '''
        #TODO: Uncomment code below if want use specific code
        # counts = {}
        
        # for pair in zip(ids, ids[1:]):
        #     counts[pair] = counts.get(pair, 0) + 1
        
        # return counts
        return Counter(zip(ids, ids[1:]))
    
    def merge_pair(self, ids: list[int], pair: tuple[int, int], new_idx: int) -> list[int]:
        '''Merge 2 token that always appear together.'''
        new_ids: list[int] = []
        append = new_ids.append # Prevent lookup function in loop
        a, b = pair # Prevent object creation in loop
        
        i = 0
        n = len(ids)
        
        while i < n:
            
            if i < n - 1 and ids[i] == a and ids[i+1] == b:
                append(new_idx)
                i += 2
            
            else:
                append(ids[i])
                i += 1
        
        return new_ids
    
    def encode(self, text: str) -> list[int]:
        '''Encode text to token integer'''
        # HANDLE SPECIAL TOKEN
        pattern = '(' + '|'.join(re.escape(t) for t in self.SPECIAL_TOKENS) + ')'
        parts = re.split(pattern, text)
        
        ids: list[int] = []
        append = ids.append
        extend = ids.extend
        
        for part in parts:
            if part in self.SPECIAL_TOKENS:
                append(self.special_token_to_id[part])
            
            else:
                if not part:
                    continue
                byte_ids = list(part.encode('utf-8'))
                while len(byte_ids) >= 2:
                    stats = self.get_stats(byte_ids)
                    best_pair = min(stats, key=lambda pair: self.merges.get(pair, float('inf')))
                    if best_pair not in self.merges:
                        break
                    byte_ids = self.merge_pair(byte_ids, best_pair, self.merges[best_pair])
                extend(byte_ids)
                
        return ids

    def decode(self, ids: list[int]) -> str:
        '''Decode token id into text.'''
        if not self.vocab:
            self.get_vocab()
            
        tokens = b''.join([self.vocab[id] for id in ids])
        text = tokens.decode('utf-8', errors='replace') # if errors replace into '?'
        return text
    
    def train(self, text: str) -> None:
        '''Train bpe to get merges.'''
        print("TRAIN BPE TO GET VOCAB AND MERGES...")
        
        tokens = text.encode('utf-8')
        print("Encode text with encoding utf-8\nOverview:", tokens[:100])
        num_merges = self.vocab_size - self.single_byte_size - len(self.SPECIAL_TOKENS)
        id_used = self.single_byte_size + len(self.SPECIAL_TOKENS)
        ids = list(tokens)
        
        for i in tqdm(range(0, num_merges), desc="Training BPE in process..."):
            stats = self.get_stats(ids)
            if not stats:
                break
            
            pair = max(stats, key=stats.get) #type: ignore
            freq = stats[pair]
            
            if freq < 5:
                print(f"Freq under 5, prefer stop because it's rarely tokens.")
                break
            
            idx = id_used + i
            if i % 100 == 0:
                print(f"Step: {i}: merging {pair} -> {idx}")
            
            ids = self.merge_pair(ids, pair, idx)
            self.merges[pair] = idx
        
        self.get_vocab()
        print(f"Training BPE complete. total merges: {len(self.merges)} and total vocab: {len(self.vocab)}/{self.vocab_size}")
    
    def get_vocab(self) -> None:
        '''Get vocab after train bpe.'''
        self.vocab = {i: bytes([i]) for i in range(self.single_byte_size)}
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.vocab[self.single_byte_size + i] = token.encode('utf-8')
        
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
    
    def save(self, vocab_path: str | None = None, merges_path: str | None = None) -> None:
        '''Save tokenizer (vocab & merges)'''
        if vocab_path is None:
            vocab_path = self.config.tokenizer.vocab_path
        if merges_path is None:
            merges_path = self.config.tokenizer.merges_path
        
        vocab_data = {str(k): v.decode('latin-1') for k, v in self.vocab.items()}
        merges_data = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        save_json(vocab_path, vocab_data) #type: ignore
        save_json(merges_path, merges_data) #type: ignore
        print(f"Vocabulary save to {vocab_path} and merges save to {merges_path}")
    
    def load(self, vocab_path: str, merges_path: str) -> None:
        '''Load tokenizer (vocab & merges)'''
        vocab_data = load_json(vocab_path)
        merges_data = load_json(merges_path)
        
        self.vocab = {int(k): v.encode('latin-1') for k, v in vocab_data.items()}
        self.merges = {tuple(map(int, k.split(','))): v for k, v in merges_data.items()} #type: ignore
        print(f"Vocabulary load from {vocab_path} and merges load from {merges_path}")