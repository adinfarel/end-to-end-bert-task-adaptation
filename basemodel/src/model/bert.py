'''
src/model/bert.py

This is cinema cools hehe, in this section i will creat BERT with piece that i'm build previous
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box

from basemodel.src.model.block import Block
from basemodel.src.model.embedding import Embedding
from configs.config import CONFIG

class AlmondBERTModel(nn.Module):
    def __init__(self, config: Box):
        super().__init__()
        self.config = config
        self.embedding = Embedding(self.config.tokenizer.vocab_size, self.config.models.embed_dim)
        self.blocks = nn.ModuleList(
            [Block(n_embd=self.config.models.embed_dim, n_heads=self.config.models.n_heads, layer_idx=i, dropout=self.config.models.dropout) 
             for i in range(self.config.models.n_blocks)]
        )
        self.mlm_transform = nn.Sequential(
            nn.Linear(self.config.models.embed_dim, self.config.models.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.models.embed_dim)
        )
        self.lm = nn.Linear(self.config.models.embed_dim, self.config.tokenizer.vocab_size)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Inputs:
            x: Token ID Tensor (B, T)
            attn_mask: Binary Mask (B, T) -> 1 for valid token, 0 -> [PAD
            targets: Label MLM (B, T)
        """
        B, T = x.shape
        non_pad_indices = torch.nonzero(attn_mask.flatten(), as_tuple=True)[0]
        
        # Unpadding (ModernBERT Efficiency)
        x_emb = self.embedding(x)
        x_flat = x_emb.view(B*T, -1)
        x_unpadded = x_flat[non_pad_indices]
        
        batch_ids = torch.arange(B, device=x.device).unsqueeze(1).expand(B, T).flatten()
        valid_batch_ids = batch_ids[non_pad_indices]
        
        same_batch_matrix = (valid_batch_ids.unsqueeze(0) == valid_batch_ids.unsqueeze(1))
        
        unpad_attn_mask = torch.zeros_like(same_batch_matrix, dtype=x_emb.dtype)
        unpad_attn_mask = unpad_attn_mask.masked_fill(~same_batch_matrix, float("-inf"))
        unpad_attn_mask = unpad_attn_mask.unsqueeze(0).unsqueeze(0)
        
        for block in self.blocks:
            x_unpadded = block(x_unpadded, attn_mask=unpad_attn_mask)
            
        x_repadded_flat = torch.zeros(B * T, self.config.models.embed_dim, device=x.device)
        x_repadded_flat[non_pad_indices] = x_unpadded
        x_repadded = x_repadded_flat.view(B, T, self.config.models.embed_dim) 
        
        logits = self.lm(self.mlm_transform(x_repadded))
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)
        
        return logits, loss
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor, mask_token_id: int, top_k: int = 5) -> dict:
        self.eval() # Turn off dropout
        
        attn_mask = torch.ones_like(x)
        logits, _ = self(x, attn_mask=attn_mask, targets=None)
        logits = logits.squeeze(0)
        mask_indices = torch.nonzero(x.squeeze(0) == mask_token_id, as_tuple=True)[0]
        
        results = {}
        for i, mask_id in enumerate(mask_indices):
            mask_logits = logits[mask_id]
            probabilities = F.softmax(mask_logits, dim=-1)
            
            topk_probs, topk_idx = torch.topk(probabilities, top_k)
            
            results[mask_id.item()] = {"topk_probs": topk_probs.tolist(), "topk_idx": topk_idx.tolist()}
        
        return results

    def encode(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode input token ids into contextual hidden states.
        
        Inputs:
            x: Token IDs, shape (B, T)
            attn_mask: 1 for valid token, 0 for PAD, shape (B, T)
        
        Returns:
            hidden_states: contextual token representation, shape (B, T)
        
        This is the reusable BERT backbone output for downstream task such as:
        - MLM
        - NER
        - POS Tagging
        - text classification
        """
        B, T = x.shape
        D = self.config.models.embed_dim
        
        non_pad_indices = torch.nonzero(
            attn_mask.reshape(-1),
            as_tuple=True
        )[0]
        
        x_emb = self.embedding(x)
        x_flat = x_emb.reshape(B * T, D)
        x_unpadded = x_flat[non_pad_indices]
        
        batch_ids = (
            torch.arange(B, device=x.device) # (B,)
            .unsqueeze(1) # (B, 1)
            .expand(B, T) # (B, T)
            .reshape(-1) # (B * T)
        )
        
        valid_batch_ids = batch_ids[non_pad_indices]
        
        same_batch_ids = (
            valid_batch_ids.unsqueeze(0) == valid_batch_ids.unsqueeze(1)
        )
        
        unpad_attn_mask = torch.zeros(
            same_batch_ids.shape,
            device=x.device,
            dtype=x_emb.dtype,
        )
        
        unpad_attn_mask = unpad_attn_mask.masked_fill(
            ~same_batch_ids, float("-inf")
        )
        
        unpad_attn_mask = unpad_attn_mask.unsqueeze(0).unsqueeze(0) # (1, 1, T, T)
        
        for block in self.blocks:
            x_unpadded = block(x_unpadded, attn_mask=unpad_attn_mask)
        
        x_repadded_flat = x_emb.new_zeros(B * T, D)
        x_repadded_flat[non_pad_indices] = x_unpadded
        
        hidden_states = x_repadded_flat.reshape(B, T, D)
        
        return hidden_states