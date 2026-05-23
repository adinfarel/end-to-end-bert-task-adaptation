'''
utils/common.py

utilities that useful during build BERT
'''

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Any
from box import Box

def ensure_parent_dir(file_path: str | Path) -> None:
    '''Create present directory if it exists in the given file path.'''
    path = Path(file_path)
    
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

def save_txt(file_path: str | Path, data: str, mode: str = 'w') -> None:
    """Save data into .txt file."""
    ensure_parent_dir(file_path)
    
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(data)
    
def load_txt(file_path: str | Path) -> str:
    """Load data .txt"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    return data

def save_json(file_path: str | Path, data: dict) -> None:
    """Save dictionary to JSON file."""
    ensure_parent_dir(file_path)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str | Path) -> dict:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def save_yaml(file_path: str | Path, data: dict) -> None:
    """Save dictionary to YAML file."""
    ensure_parent_dir(file_path)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)

def load_yaml(file_path: str | Path, use_box: bool = True) -> Box:
    """Load YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if use_box:
        return Box(data)
    
    return data

def save_bin(file_path: str | Path, data: list | np.ndarray) -> None:
    """Save data to numpy array binary file."""
    ensure_parent_dir(file_path)
    
    if isinstance(data, list):
        data = np.array(data, dtype=np.uint16)
    
    data.tofile(file_path)
    
def load_bin(file_path: str | Path) -> np.ndarray:
    """Load binary file."""
    data = np.memmap(file_path, dtype=np.uint16, mode='r')
    return data