import yaml 
import pandas as pd
from pathlib import Path
from classes import Paths

# Read config file
config_dict = None
config_path: Path = Path().cwd() / "config" / "config.yaml"

if config_path.exists():
    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

if config_dict is None:
    raise FileNotFoundError("Missing config/struct.yaml file")

paths = Paths(**config_dict['path_structure'])
home_dir = Path().cwd()
input_folder = home_dir / 'input'
sample_submission = pd.read_csv(input_folder / 'sample_submission.csv')
sample_submission.head(10)
