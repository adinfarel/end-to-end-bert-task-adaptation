'''
src/data/datasets.py

Load dataset NER from huggingface for task adaptation BERT
'''

# import jsonlines
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

from configs.config import CONFIG

def load_datasets_from_hf() -> None:
    print("DOWNLOAD DATASET FROM HUGGINGFACE...")
    dataset = load_dataset(CONFIG.downstream.dataset_name)
    split_name = ["train", "validation", "test"]
    
    for name in split_name:
        file_path = Path(CONFIG.downstream.raw_dataset_path) / f"{name}.jsonl"
        
        if not file_path.parent.exists():
            print(f"CREATE FOLDER {str(file_path.parent)} FOR DATASET")
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # NOTE: This not optimal code, this code just for learn manual loop
        # I'm prefer use built-function from HuggingFace
         
        # with jsonlines.open(file_path, mode='w') as f:
        #     for rows in tqdm(dataset[name], desc=f"Create {name} dataset .jsonl"):
        #         f.write({"tokens": rows['tokens'], "ner_tags": rows['ner_tags']})
        
        split_dataset = dataset[name].select_columns(['tokens', 'ner_tags'])
        split_dataset.to_json(file_path, orient="records", lines=True)
        
        print(f"{name} datasets complete extract to .jsonl")
    
if __name__ == "__main__":
    load_datasets_from_hf()