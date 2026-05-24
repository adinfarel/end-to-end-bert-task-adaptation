'''
src/model/pos_enc.py

Positional encoding with RoPE architecture, here why im use it:

LPE as known as regular PE in transformer architecture why this component not use?
because LPE cannot extrapolarate sequence length, if model meet new pos id model immediately confuse and model getting worse, hallucinate and catastrophic.
then LPE cannot smart to understand relatif positonal between token, like what if in this batch model get for example
token 'I'm' and token 'Better' with range difference 3 but if model get range diff between token in next batch like before we meet
token 1 and token 2 in early sentence and next batch we meet again token 1 and token 2 but in different position example last sentence, nah model must be learn again
range between token 1 and token 2 so it's make model learn, learn, and learn more until model can mapping position each token in each sentence which is it take longer model for converge

here it is im use RoPE, this is intuition behind RoPE in terms of my explanation:
RoPE mechanism use math trigonometry, so range between token 1 and token 2 be like inside a spinning wheel, if 2 tokens it often close together which mean there is attachment
between 2 token that, so if we put 2 token whether at the end or beginning of a sentence position each token just rotated as many times as the token is positioned from its original position

FIXME: if yall disagree with my intuition feel free to comment and discuss it with me (ignore grammar or language errors hehe >.<)

FIX:
This fix intuition from GEMINI
The Intuition behind RoPE (The "Spinning Wheel" Analogy):
RoPE implements rotation via trigonometry (Sine and Cosine) onto 2D chunks of token embeddings. 
Think of it like a spinning wheel: when two tokens are close to each other, their angular 
distance on the wheel remains tight, yielding a high attention score. 

No matter where this pair of tokens is located—whether at the very beginning or the 
deep end of a sentence—their absolute vectors are rotated further, BUT the relative angle 
between them remains identical. This gives the model a natural, mathematically hardcoded 
understanding of relative distances, leading to faster convergence and out-of-the-box length extrapolation.
'''

import torch 
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, base=10000.0):
        super().__init__()
        self.dim = embedding_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, seq_offset: int = 0):
        T, C = x.shape[-2], x.shape[-1]
        assert C == self.dim, "dimension of x mismatch with dimension of RoPE"
        l = torch.arange(T, dtype=self.inv_freq.dtype, device=x.device) + seq_offset #type: ignore
        theta = torch.einsum("i,j->ij", l, self.inv_freq)
        hat_theta = torch.cat([theta, theta], dim=-1)
        sin = torch.sin(hat_theta)
        cos = torch.cos(hat_theta)
        xu, xd = x[..., : C // 2], x[..., C // 2 :]
        hatx = torch.cat([-xd, xu], dim=-1)
        return x * cos + hatx * sin        