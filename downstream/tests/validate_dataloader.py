'''
downstream/tests/validate_datalaoder.py

Test dataloader 
'''

from pathlib import Path

from configs.config import CONFIG
from basemodel.src.tokenizer.bpe import AlmondTokenizerBERT
from downstream.src.data.loader import (
    load_jsonl,
    build_label_map,
    create_ner_dataloaders
)

def main() -> None:
    print("TEST DATALOADER")
    
    path_file_train = Path(CONFIG.downstream.raw_dataset_path) / "train.jsonl"
    train_data = load_jsonl(file_path=path_file_train)
    print(f"TRAIN DATA SUCCESSFULLY LOAD, Total train data: {len(train_data)}")
    
    print("LOAD TOKENIZER")
    tokenizer = AlmondTokenizerBERT(CONFIG)
    tokenizer.load(
        vocab_path=CONFIG.tokenizer.vocab_path,
        merges_path=CONFIG.tokenizer.merges_path,
    )
    
    print("BUILD LABEL MAP")
    label_to_id, id_to_label = build_label_map(
        data=train_data
    )
    print("Label to id : ", label_to_id)
    print("Id to label : ", id_to_label)
    
    print("CREATE DATALOADER")
    train_loader = create_ner_dataloaders(
        data=train_data,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        config=CONFIG,
        shuffle=True
    )
    
    print("="*60)
    name = "TRAIN DATALOADER"
    print(f"{name:^60}")
    print("="*60)
    
    input_ids, attn_mask, labels = next(iter(train_loader))

    print(f"input ids shape     : {input_ids.shape}")
    print(f"attn mask shape     : {attn_mask.shape}")
    print(f"labels shape        : {labels.shape}")
    
    print(f"first input ids      : {input_ids[0][:50]}")
    print(f"first attention_mask :", attn_mask[0][:50])
    print(f"first labels         :", labels[0][:50])
    
    supervised_count = (labels != -100).sum().item()
    print("Supervised count : ", supervised_count)
    
if __name__ == "__main__":
    main()