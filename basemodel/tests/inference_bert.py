'''
basemodel/tests/inference_bert.py

Test bert pre-training, how good model prediction word behind [MASK] token
'''

import torch
from pathlib import Path

from configs.config import CONFIG
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT
from basemodel.src.model.bert import AlmondBERTModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HELPER
def token_from_id(tokenizer: AlmondTokenizerBERT, token_id: int) -> str:
    if token_id in tokenizer.special_id_to_token:
        return tokenizer.special_id_to_token[token_id]

    return tokenizer.decode([token_id])

def load_tokenizer_and_model() -> tuple[AlmondBERTModel, AlmondTokenizerBERT]:
    tokenizer = AlmondTokenizerBERT(CONFIG)
    tokenizer.load(
        vocab_path=CONFIG.tokenizer.vocab_path,
        merges_path=CONFIG.tokenizer.merges_path,
    )
    
    model_path = Path(CONFIG.models.model_save_path) / 'best_model.pt'
    model = AlmondBERTModel(CONFIG).to(DEVICE)
    checkpoints = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoints['model'])
    model.eval()
    
    return model, tokenizer

def main() -> None:
    model, tokenizer = load_tokenizer_and_model()
    
    MASK_ID = tokenizer.special_token_to_id["[MASK]"]
    
    text = f"Paris is the capital of [MASK][MASK]."
    ids = tokenizer.encode(text)
    
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    
    result = model.predict(
        x=x,
        mask_token_id=MASK_ID,
        top_k=5
    )
    
    print("Text             : ", text)
    print("Ids              : ", ids)
    print("Mask Token Ids   : ", MASK_ID)
    print("Result           : ", result)
    
    for mask_pos, pred in result.items():
        print(f"\n[MASK] position: {mask_pos}")
        
        for prob, token_id in zip(pred['topk_probs'], pred['topk_idx']):
            token = token_from_id(tokenizer, token_id)
            print(f"{int(token_id):5d} | {token:20s} | {prob:.6f}")

if __name__ == "__main__":
    main()