'''
src/model/embedding.py

To get representation vector numeric each token in vector space
'''

from turtle import forward

import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x: torch.Tensor):
        return self.embedding(x)