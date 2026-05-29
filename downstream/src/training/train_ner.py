'''
src/training/train_ner.py

Training pipeline for task adaptation NER from pre-trained model.
'''

import argparse
import sys
import os
from pathlib import Path

# SET ROOT PROJECT
ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from box import Box
from tqdm import tqdm

from configs.config import CONFIG
from downstream.src.model.ner import AlmondBERTForNER
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT
from downstream.src.data.loader import create_ner_dataloaders, build_label_map, load_jsonl
from basemodel.src.training.train_model import (
    save_model,
)

# ENV
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# UTILS
@torch.no_grad()
def eval_loss(model: AlmondBERTForNER, val_loader: DataLoader) -> float:
    '''Eval loss to see how good model.'''
    model.eval()
    losses = []
    for inputs, attn_mask, labels in val_loader:
        inputs, attn_mask, labels = inputs.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
        _, loss = model(inputs, attn_mask, labels)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) # Average eval loss

def main(config: Box, early_stopping: bool = True, freeze_model: bool = False) -> None:
    print("TRAINING PIPELINE NER")
    train_data_path = Path(config.downstream.raw_dataset_path) / "train.jsonl"
    val_data_path = Path(config.downstream.raw_dataset_path) / "val.jsonl"
    
    print("LOAD TOKENIZER")
    tokenizer = AlmondTokenizerBERT(config)
    tokenizer.load(
        vocab_path=config.tokenizer.vocab_path,
        merges_path=config.tokenizer.merges_path,
    )
    print(f"Load tokenizer {AlmondTokenizerBERT.__name__}")
    
    print("LOAD DATASET")
    train_data = load_jsonl(train_data_path)
    val_data = load_jsonl(val_data_path)
    print(
        f"Load dataset: train path: {train_data_path} with total data train: {len(train_data)} "
        f"val path: {val_data_path} with total data validation: {len(val_data)}."
    )
    
    print("BUILD LABEL MAP")
    label_to_id, id_to_label = build_label_map(train_data)
    num_labels = len(label_to_id)
    print(f"Label to id:\n{label_to_id}\nId to label:\n{id_to_label}\nTotal label: {num_labels}")
    
    print("CREATE DATALOADERS")
    train_loader = create_ner_dataloaders(
        data=train_data,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        config=config,
        shuffle=True
    )
    val_loader = create_ner_dataloaders(
        data=val_data,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        config=config,
        shuffle=False
    )
    
    
    print("LOAD PRETRAINED MODEL")
    model_path = Path(config.models.model_save_path) / "best_model.pt"
    model = AlmondBERTForNER.from_pretrained(config, checkpoint_path=model_path, num_labels=num_labels, freeze_model=freeze_model)
    model = model.to(DEVICE)
    print(f"Load model pretrained successfully {AlmondBERTForNER.__name__}")
    
    print("SETUP COMPONENT TRAINING")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.model.parameters(),
                "lr": float(config.downstream.encoder_lr)
            },
            {
                "params": model.classifier.parameters(),
                "lr": float(config.downstream.head_lr)
            },
        ],
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config.downstream.epochs * len(train_loader))
    )
    scaler = torch.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda" and DTYPE == torch.float16))
    early_stopping_counter = 0
    early_stopping_patience = config.downstream.early_stopping_patience
    
    # Checkpoints
    start_iter = 0
    best_val_loss = float('inf')
    resume_path = Path(config.downstream.save_ft_ner_path) / "best_model_ner.pt"
    
    print("CHECK WHETHER CHECKPOINT EXISTS OR NOT")
    if resume_path.exists():
        print(f"Checkpoint found at: {str(resume_path)}. Resuming training...")
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_iter = checkpoint.get('iter', start_iter)
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        print(f"Resuming training for iteration {start_iter} with best validation loss {best_val_loss}")
    else:
        print("Checkpoint not found. Start training from scratch...")
    
    for epoch in range(start_iter + 1, config.downstream.epochs + 1):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.downstream.epochs}")
        for input_ids, attn_mask, labels in progress_bar:
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == 'cuda')):
                _, loss = model(
                    input_ids,
                    attn_mask,
                    labels
                )
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        if epoch % config.downstream.eval_interval == 0:
            losses = eval_loss(model, val_loader)
            print(f"Step {epoch} | Eval loss: {losses:.4f}")
            
            if losses < best_val_loss:
                early_stopping_counter = 0
                best_val_loss = losses
                checkpoints = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iter': epoch,
                    'best_val_loss': best_val_loss
                }
                save_model(checkpoints, file_path=resume_path)
                print(f"--> New best model save at step {epoch} with loss {best_val_loss:.4f}")

            else:
                early_stopping_counter += 1
                if early_stopping and early_stopping_counter >= early_stopping_patience:
                    print(f"--> Early stopping at step {epoch}")
                    break
        
    final_iter = config.downstream.epochs if early_stopping_counter < early_stopping_patience else epoch
    
    save_model(
        checkpoints={
            'model':model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iter': final_iter,
            'best_val_loss': best_val_loss,
        },
        file_path=Path(os.path.join(config.downstream.save_ft_ner_path, 'ckpt_latest.pt'))
    )
    
    print("TRAINING NER MODEL FINISH >.<")

def str2bool(value: str) -> bool:
    '''Helper for detect parser argument.'''
    if value.lower() in {'true', 'y', 'yes'}:
        return True
    
    if value.lower() in {'false', 'n', 'no'}:
        return False
    
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training NER with pretrained model")
    parser.add_argument('--early-stopping', type=str2bool, default=True)
    parser.add_argument('--freeze-model', type=str2bool, default=False)
    args = parser.parse_args()
    
    # -----------------------
    # TRAIN
    # -----------------------
    
    main(
        CONFIG,
        early_stopping=args.early_stopping,
        freeze_model=args.freeze_model
    )