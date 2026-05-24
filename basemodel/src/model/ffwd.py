'''
src/model/ffwd.py

Feed forward is the way model mapping linearity to non-linearity, this time i'm use GeGLU rather than FFWD Regular
Why? FFWD Regular not quite enough to capture non-linearity between token because FFWD ignore weight < 0 or if u use GELU the model tolerates a slight weight below zero
but if we use GeGLU we create 3 matrix: Gated, Value, and out, prefer model put in weight into gated and with GELU we now weight whatever is important
because if weight through gated and weight < 0 if we use ReLU and we pass weight that weight passed through gated if we multiple with weight not through gated, the weight will be zero cause multiple by weight gated that get (0)
and if weight through gated not be zero (0) then weight not through gated will be increase significantly cause multiple by weight not zero, so model can capture non-linearity better and better

#FIXME: if my explanation less understandable, feel free to tell me >.<
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeGLU(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        
        # Cause embed dim in FFWD * 4, to get same dim we divide embed dim with (8/3), why? cause we have 3 matrix rather than 2
        self.embd_dim = int(embed_dim * (8/3))
        self.gate = nn.Linear(embed_dim, self.embd_dim, bias=False)
        self.value = nn.Linear(embed_dim, self.embd_dim, bias=False)
        self.out = nn.Linear(self.embd_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        value = self.value(x)
        gate = F.gelu(self.gate(x))
        hidden = value * gate
        out = self.out(hidden)
        return self.dropout(out)