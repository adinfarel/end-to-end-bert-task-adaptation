'''
downstream/tests/test_ner_model.py

Testing how good model in Token Classification NER >.< (I don't expect too much tho =(.)
'''
import os

import torch
from pathlib import Path

from downstream.src.model.ner import AlmondBERTForNER
from downstream.src.data.loader import load_jsonl, build_label_map
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT
from configs.config import CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def test_model() -> None:
    # Load model and tokenizer
    tokenizer = AlmondTokenizerBERT(CONFIG)
    tokenizer.load(
        vocab_path=CONFIG.tokenizer.vocab_path,
        merges_path=CONFIG.tokenizer.merges_path,
    )
    cls_id = tokenizer.special_token_to_id["[CLS]"]
    pad_id = tokenizer.special_token_to_id["[PAD]"]
    sep_id = tokenizer.special_token_to_id["[SEP]"]
    
    special_token_ids = {cls_id, pad_id, sep_id,}
    
    # FOR REPRODUCEBILITY
    train_path = Path(CONFIG.downstream.raw_dataset_path) / "train.jsonl"
    train_data = load_jsonl(
        file_path=train_path
    )
    
    label_to_id, id_to_label = build_label_map(
        data=train_data
    )
    
    # Load pre-training model
    model = AlmondBERTForNER.from_pretrained(
        config=CONFIG,
        checkpoint_path=Path(CONFIG.models.model_save_path) / "best_model.pt",
        num_labels=len(label_to_id)
    )
    
    # Load fine-tuned NER model
    ner_model_path = Path(CONFIG.downstream.save_ft_ner_path) / "best_model_ner.pt"
    checkpoints = torch.load(ner_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoints['model'], strict=True)
    
    model = model.to(DEVICE)
    model.eval()
    
    # Create text
    text = ["It", "is", "found", "in", "Peru", "."]
    input_ids = [cls_id]
    for word in text:
        input_ids.extend(tokenizer.encode(word))
    input_ids.append(sep_id)
    attn_mask = [1] * len(input_ids)
    
    print(f"Input IDS: {input_ids}")
    print(f"Attn Mask: {attn_mask}")
    
    results = model.predict(
        input_ids=torch.tensor([input_ids], device=DEVICE),
        attn_mask=torch.tensor([attn_mask], device=DEVICE),
        id_to_label=id_to_label,
        special_token_ids=special_token_ids
    )
    
    for res in results:
        for i, item in enumerate(res, start=1):
            token = tokenizer.decode([item['token_id']])
            
            print(f"Token       : {i}")
            print(f"Position    : {item['position']}")
            print(f"Token Id    : {item['token_id']}")
            print(f"Token Text  : {token}")
            print(f"Label Id    : {item['label_id']}")
            print(f"Label       : {item['label']}")
            print(f"Score       : {item['score']}")

if __name__ == "__main__":
    test_model()