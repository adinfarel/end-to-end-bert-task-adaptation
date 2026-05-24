'''
src/model/mha.py

Main mechanism of transformers, Attention is way how token see each other and get affinity with other token how important between token
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from basemodel.src.model.pos_enc import RotaryPositionalEncoding

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        self.head_size = self.embed_dim // self.num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryPositionalEncoding(self.head_size)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        B, T, C = x.shape
        
        if attn_mask is not None:
            attn_mask = attn_mask.view(B, 1, 1, T).bool()
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        K = K.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        V = V.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        
        Q_rope = self.rope(Q)
        K_rope = self.rope(K)
        
        out = F.scaled_dot_product_attention(
            query=Q_rope,
            key=K_rope,
            value=V,
            attn_mask=attn_mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        out = out.transpose(1, 2).contiguous() # (B, T, num_heads, head_size)
        out = out.view(B, T, self.num_heads * self.head_size)
        
        return self.proj(out)