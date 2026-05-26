'''
src/model/block.py

A single block transformer that contains all piece of modern Architecture BERT
'''

import torch
import torch.nn as nn

from basemodel.src.model import(
    ffwd,
    mha
)

class Block(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, layer_idx: int, dropout: float):
        super().__init__()
        self.attn = mha.MultiHeadAttention(
            embed_dim=n_embd,
            n_heads=n_heads,
            dropout=dropout,
            layer_idx=layer_idx,
            window_size=3
        )
        self.pre_ln1 = nn.LayerNorm(n_embd)
        self.ffwd = ffwd.GeGLU(embed_dim=n_embd, dropout=dropout)
        self.pre_ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        # Why x + ... ? it is method to prevent vanishing gradient, name method is ResNet 
        x = x + self.attn(self.pre_ln1(x), unpad_mask=attn_mask)
        x = x + self.ffwd(self.pre_ln2(x))
        return x