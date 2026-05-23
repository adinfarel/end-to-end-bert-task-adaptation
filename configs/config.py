'''
configs/config.py

Load config for all file in this project
'''

from utils.common import load_yaml
from box import Box

CONFIG_PATH = "configs/config.yaml"
CONFIG: Box = load_yaml(CONFIG_PATH, use_box=True)