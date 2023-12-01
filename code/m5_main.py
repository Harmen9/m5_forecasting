import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from classes import TsPreproc
from config import *

# Read config file
paths = Paths()
home_dir: Path = Path().cwd()
train_val: pd.DataFrame = pd.read_csv(paths.INPUT / 'sales_train_validation.csv')
run_params = globals()['Baseline']() 

# Transformation

# To speed up the training of the data during data exploration only store CA_1 is taken 
# into account
store_mask: pd.Series = train_val['store_id'].isin(run_params.stores)
train_val: pd.DataFrame = train_val[store_mask]
del store_mask
train_val: TsPreproc = TsPreproc(train_val)

# Pivot table
train_val.melt(
    id_vars=run_params.index_columns,
    var_name='d',
    value_name=run_params.TARGET
)

train_val.data['d'] = train_val.data['d'].str.replace('d_', '').astype(int)

# Get lag features
train_val.generate_lags(
    lags=run_params.lags,
    group_by=['id'],
    lag_column=run_params.TARGET
    )

MAX_D = max(train_val.data['d'])
END_TRAIN   = MAX_D - run_params.TRAIN_SPLIT
X = train_val.data.drop(columns=['id', 'store_id', 'state_id', run_params.TARGET])

# Split data 
mask_train = train_val.data['d']<=END_TRAIN
valid_mask = mask_train&(train_val.data['d']>(END_TRAIN-run_params.P_HORIZON))

y = train_val.data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)


max_column_number = max(int(col.split("_")[1]) for col in df.columns)
n = 100
start_column_number = max_column_number - n

mask_val = [col for col in df.columns if col not in selected_columns]


n = 100

# Create test/train set
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'subsample': 0.5,
    'subsample_freq': 1,
    'learning_rate': 0.015,
    'num_leaves': 2**11-1,
    'min_data_in_leaf': 2**12-1,
    'feature_fraction': 0.5,
    'max_bin': 100,
    'n_estimators': 3000,
    'boost_from_average': False,
    'verbose': -1,
} 
