'''
src/model/mha.py

Main mechanism of transformers, Attention is way how token see each other and get affinity with other token how important between token
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from basemodel.src.model.pos_enc import RotaryPositionalEncoding

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, layer_idx: int, dropout: float = 0.1, window_size: int = 3):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        self.head_size = self.embed_dim // self.num_heads
        self.layer_idx = layer_idx
        self.window_size = window_size
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryPositionalEncoding(self.head_size)
    
    def forward(self, x: torch.Tensor, unpad_mask: torch.Tensor | None = None):
        Total_Tokens, C = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(1, Total_Tokens, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        K = K.view(1, Total_Tokens, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        V = V.view(1, Total_Tokens, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        
        Q_rope = self.rope(Q)
        K_rope = self.rope(K)
        
        final_mask = None if unpad_mask is None else unpad_mask.clone() #type: ignore
        
        # Alternating Attention
        if self.layer_idx % 2 != 0:
            coords = torch.arange(Total_Tokens, device=x.device)
            dist_matrix = torch.abs(coords.unsqueeze(0) - coords.unsqueeze(1)) #MANHATTAN
            
            sliding_mask = (dist_matrix > self.window_size).unsqueeze(0).unsqueeze(0) # (1, 1, Total_Tokens, TOtal_Tokens)
            
            if final_mask is None:
                final_mask = torch.zeros(
                    (1, 1, Total_Tokens, Total_Tokens),
                    device=x.device,
                    dtype=Q.dtype
                )
            
            final_mask = final_mask.masked_fill(sliding_mask, float('-inf'))
        
        out = F.scaled_dot_product_attention(
            query=Q_rope,
            key=K_rope,
            value=V,
            attn_mask=final_mask,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        
        out = out.transpose(1, 2).contiguous() # (B, T, num_heads, head_size)
        out = out.view(Total_Tokens, self.embed_dim)
        
        return self.proj(out)