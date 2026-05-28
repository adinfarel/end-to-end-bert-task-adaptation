'''
src/model/ner.py

In this section i will build NER for task adaptation from BERT Pre-train model >.<
'''

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box
from pathlib import Path

from basemodel.src.model.bert import AlmondBERTModel

# ENV
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HELPER
def load_pretrained_model(config: Box, file_path: str | Path) -> AlmondBERTModel:
    '''Load pre-train model.'''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File {str(file_path)} not found.")
    
    model = AlmondBERTModel(config)

    checkpoints = torch.load(file_path, map_location=DEVICE)
    model.load_state_dict(checkpoints['model'], strict=True) #type: ignore
    
    return model

class AlmondBERTForNER(nn.Module):
    def __init__(self, model: AlmondBERTModel, num_labels: int, dropout: float = 0.1, freeze_model: bool = False) -> None:
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(
            self.model.config.models.embed_dim,
            num_labels,
            bias=True
        )
        
        # Freeze model
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
        
    @classmethod
    def from_pretrained(
        cls,
        config: Box,
        checkpoint_path: str | Path,
        num_labels: int,
        dropout: float = 0.1,
        freeze_model: bool = False,
    ) -> "AlmondBERTForNER": #type: ignore
        """Build NER model from pre-trained model."""
        model = load_pretrained_model(
            config,
            checkpoint_path
        )
        
        return cls(
            model=model,
            num_labels=num_labels,
            dropout=dropout,
            freeze_model=freeze_model,
        )
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, targets: None | torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x_encoder = self.model.encode(x, attn_mask=attn_mask)
        logits = self.classifier(self.dropout(x_encoder))
        
        if targets is None:
            loss = None
        else:
            B, T, num_labels = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, num_labels),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def predict(self, 
                input_ids: torch.Tensor, 
                attn_mask: torch.Tensor,
                id_to_label: dict[int, str] | None = None,
                special_token_ids: set[int] | None = None) -> list[list[dict[str, Any]]]:
        self.eval()
        
        logits, _ = self(
            input_ids,
            attn_mask=attn_mask,
            targets=None
        )
        
        probs = F.softmax(logits, dim=-1)
        scores, pred_ids = torch.max(probs, dim=-1)
        
        batch_results: list[list[dict[str, Any]]] = []
        
        batch_size, seq_len = input_ids.shape
        
        for batch_ids in range(batch_size):
            sample_results = []
            
            for pos in range(seq_len):
                if attn_mask[batch_ids, pos].item() == 0:
                    continue
                
                token_id = int(input_ids[batch_ids, pos].item())
                
                if special_token_ids is not None and token_id in special_token_ids:
                    continue
                    
                label_id = int(pred_ids[batch_ids, pos].item())
                score = float(scores[batch_ids, pos].item())
                
                item: dict[str, Any] = {
                    "position": pos,
                    "token_id": token_id,
                    "label_id": label_id,
                    "score": score
                }
                
                if id_to_label is not None:
                    item["label"] = id_to_label[label_id]
                    
                sample_results.append(item)
            
            batch_results.append(sample_results)
        
        return batch_results