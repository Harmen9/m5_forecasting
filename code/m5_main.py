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

paths: Path = Paths(**config_dict['path_structure'])
home_dir: Path = Path().cwd()
input_folder: Path = home_dir / 'input'
train_val: pd.DataFrame = pd.read_csv(input_folder / 'sales_train_validation.csv')


# Transformation
pd.melt(
    train_val,
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'state_id'],
        value_vars=['B'],
        var_name='myVarname',
        value_name='myValname')
train_val.melt()