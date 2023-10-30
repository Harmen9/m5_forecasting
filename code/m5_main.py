import yaml 
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from classes import Paths

# Read config file
config_dict = None
config_path: Path = Path().cwd() / "config" / "config.yaml"

if config_path.exists():
    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

if config_dict is None:
    raise FileNotFoundError("Missing config.yaml file")

paths: Path = Paths(**config_dict['path_structure'])
home_dir: Path = Path().cwd()
input_folder: Path = home_dir / 'input'
train_val: pd.DataFrame = pd.read_csv(input_folder / 'sales_train_validation.csv')


# Transformation

# To speed up the training of the data during data exploration only store CA_1 is taken 
# into account
store_mask: pd.Series = train_val['store_id'] == 'CA_1'
train_val: pd.DataFrame = train_val[store_mask]
del store_mask
index_columns: list = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

train_val: pd.DataFrame = pd.melt(
    train_val,
    id_vars=index_columns,
    var_name='d',
    value_name='sales'
    )
#train_val: pd.DataFrame = train_val.set_index(index_columns)
print("test")

# Get lag features
lags: list = list(range(1, 15))
for lag in lags:
    train_val[f'sales_lag_{lag}'] = train_val.groupby(['id'])['sales'].shift(lag)

