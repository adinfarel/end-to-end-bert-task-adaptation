'''
prototype/main.py

Prototype of BERT (Bidirectional-Encoder Representation Transformers)
based on paper: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
link paper: https://arxiv.org/pdf/1810.04805
'''

import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, n_seg: int ,embd_dim: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embd_dim)
        self.pos_emb = nn.Embedding(max_len, embd_dim)
        self.seg_emb = nn.Embedding(n_seg, embd_dim)
    
    def forward(self, seq, seg): #type: ignore
        '''
        x.shape = (B, T)
        
        returns: x.shape = (B, T, C), C is embedding dimension of token
        
        but for example, im use one single batch so it just sequence
        '''
        embed_val = self.tok_emb(seq) + self.pos_emb(torch.arange(len(seq))) + self.seg_emb(seg)
        return embed_val

class BERT(nn.Module):
    def __init__(self, vocab_size: int, n_seg: int, max_len: int, embd_dim: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, max_len, n_seg, embd_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embd_dim, n_heads, embd_dim * 4)
        self.transformer_block = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq, seg):
        out = self.embedding(seq, seg)
        out = self.transformer_block(out)
        return self.dropout(out)
    
if __name__ == "__main__":
    VOCAB_SIZE = 30_000
    N_SEGMENTS = 3
    MAX_LEN = 512
    EMBED_DIM = 768
    N_LAYERS = 12
    N_HEADS = 12
    DROPOUT = 0.1
    
    sample_seq = torch.randint(high=VOCAB_SIZE, size=(MAX_LEN,))
    sample_seg = torch.randint(high=N_SEGMENTS, size=(MAX_LEN,))
    
    print(f"Sample seq: {sample_seq}\nSample seg: {sample_seg}")
    
    embedding = BERTEmbedding(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        n_seg=N_SEGMENTS,
        embd_dim=EMBED_DIM,
    )
    
    x = embedding(sample_seq, sample_seg)
    print("After plugin into embedding: ", x.shape)
    
    # ------------------------------------------------
    bert = BERT(
        vocab_size=VOCAB_SIZE,
        n_seg=N_SEGMENTS,
        max_len=MAX_LEN,
        embd_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT
    )
    
    out = bert(sample_seq, sample_seg)
    print("After plugin into model BERT: ", out.shape)