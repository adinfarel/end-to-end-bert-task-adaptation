'''
src/training/train_model.py

This is pipeline training for bert, im choose Training Scheduler, like saving checkpoints if found best loss,
Gradient Clipping for prevent Exploding Gradient, Warmup + Learning Rate Scheduler (Cosine Annealing)
'''

import sys
from pathlib import Path

# ADDING PROJECT ROOT
ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import torch
from torch.utils.data import DataLoader
from box import Box
from tqdm import tqdm

from configs.config import CONFIG
from basemodel.src.data.loader import create_dataloaders, load_jsonl
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT
from basemodel.src.model.bert import AlmondBERTModel

    
# ENVIRONTMENT
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# HELPER
def load_tokenizer_and_model() -> tuple[AlmondBERTModel, AlmondTokenizerBERT]:
    tokenizer = AlmondTokenizerBERT(CONFIG)
    tokenizer.load(
        vocab_path=CONFIG.tokenizer.vocab_path,
        merges_path=CONFIG.tokenizer.merges_path
    )
    model = AlmondBERTModel(CONFIG)
    model.train(); model.to(DEVICE)
    
    return model, tokenizer

@torch.no_grad()
def eval_loss(model: AlmondBERTModel, val_loader: DataLoader) -> float:
    '''Eval loss to see how good model.'''
    model.eval()
    losses = []
    for inputs, attn_mask, labels in val_loader:
        inputs, attn_mask, labels = inputs.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
        _, loss = model(inputs, attn_mask, labels)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) # Average eval loss
    
def save_model(checkpoints: dict, file_path: str | Path) -> None:
    '''Save checkpoint model.'''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoints, file_path)

def main(config: Box) -> None:
    print("TRAINING PIPELINE BERT")
    train_data_path: str = config.datasets.dataset_processed_path + "train_data.jsonl"
    val_data_path: str = config.datasets.dataset_processed_path + "val_data.jsonl"
    
    print("LOAD TOKENIZER AND INITIALIZED MODEL")
    model, tokenizer = load_tokenizer_and_model()
    
    print("LOAD DATASET PROCESSED")
    train_data = load_jsonl(train_data_path)
    val_data = load_jsonl(val_data_path)
    print(f"Successful load. Total train data: {len(train_data)} and val data: {len(val_data)}")
    
    print("CREATE DATALOADERS")
    train_loader = create_dataloaders(
        train_data,
        tokenizer=tokenizer,
        config=config,
        shuffle=True
    )
    val_loader = create_dataloaders(
        val_data,
        tokenizer=tokenizer,
        config=config,
        shuffle=False
    )

    print("SETUP COMPONENT TRAINING")
    total_steps = config.models.epochs * len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.models.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps,
        eta_min=1e-5,
    )
    scaler = torch.GradScaler(device="cuda", enabled=(DTYPE == torch.float16))
    early_stopping_counter = 0
    early_stopping_patience = config.models.early_stopping_patience
    
    best_val_loss = float('inf')
    start_iter = 0
    
    print("CHECK CHECKPOINT")
    resume_path = Path(config.models.model_save_path) / 'best_model.pt'
    
    if resume_path.exists():
        print(f'Checkpoint found at: {str(resume_path)}. Resuming training...')
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_iter = checkpoint.get('iter', start_iter)
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        print(f"Resuming training for iteration {start_iter} with best validation loss {best_val_loss}")
    
    else:
        print("Checkpoint not found. Starting training from scratch.")
    
    for iter in range(start_iter, config.models.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {iter + 1}/{config.models.epochs}")
        if iter % config.models.eval_interval == 0:
            losses = eval_loss(model=model, val_loader=val_loader)
            print(f"Step {iter} | Eval loss {losses}")
            
            if losses < best_val_loss:
                early_stopping_counter = 0
                best_val_loss = losses
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iter': iter,
                    'best_val_loss': best_val_loss
                }

                save_model(checkpoints=checkpoint,
                           file_path=Path(config.models.model_save_path) / 'best_model.pt')
                print(f"--> New best model saved at step {iter} with loss {losses:.4f}")

            else:
                early_stopping_counter += 1
                if config.models.early_stopping and early_stopping_counter >= early_stopping_patience:
                    print(f"--> Early stopping at step {iter}")
                    break
        
        for inputs, attn_mask, labels in progress_bar:
            inputs, attn_mask, labels = inputs.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
            
            with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == 'cuda')):
                _, loss = model(inputs, attn_mask, labels)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    final_iter = config.models.epochs if early_stopping_counter < early_stopping_patience else iter
    
    save_model(
        checkpoints={
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iter': final_iter,
            'best_val_loss': best_val_loss
        },
        file_path=Path(config.models.model_save_path) / 'ckpt_latest.pt'
    )

if __name__ == "__main__":
    main(CONFIG)