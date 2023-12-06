import json
from datetime import datetime
import pandas as pd
from pathlib import Path
import lightgbm as lgb

from classes import TsPreproc
from config import *

# Get the current date and time
current_datetime = datetime.now().strftime("%Y%m%d%H%M")
selected_run_id = 'Baseline'

# Read config file
paths = Paths()
train_val: pd.DataFrame = pd.read_csv(paths.INPUT / 'sales_train_validation.csv')
run_params = globals()[selected_run_id]() 

# Create output folder:
output_folder = paths.OUTPUT / f"{selected_run_id}_{current_datetime}"
output_folder.mkdir(parents=True, exist_ok=True)

# Store output parameters
with open(output_folder / "run_params.json", 'w', encoding="utf-8") as json_file:
    json.dump(run_params.to_dict(), json_file, indent=4)

# Transformation

# To speed up the training of the data during data exploration only store CA_1 is taken 
# into account
store_mask: pd.Series = train_val['store_id'].isin(run_params.SELECTED_STORES)
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
    lags=run_params.LAGS,
    group_by=['id'],
    lag_column=run_params.TARGET
    )

categorical_features = ['item_id', 'dept_id', 'cat_id']
train_val.label_encode(
    categorical_columns=categorical_features
    )


# Split data
MAX_D = max(train_val.data['d'])
END_TRAIN = MAX_D - run_params.TRAIN_SPLIT
mask_train = train_val.data['d']<=END_TRAIN

train_data = lgb.Dataset(
    data=train_val.data.drop(columns=['id', 'store_id', 'state_id', run_params.TARGET])[mask_train],
    label=train_val.data['sales'][mask_train]
    )
val_data = lgb.Dataset(
    data=train_val.data.drop(columns=['id', 'store_id', 'state_id', run_params.TARGET])[~mask_train],
    label=train_val.data['sales'][~mask_train]
    )
del train_val

evals={}
bst = lgb.train(
    run_params.lgb_params,
    train_data,
    categorical_feature=categorical_features,
    valid_sets=[val_data],
    callbacks = [lgb.record_evaluation(evals)]
    )
bst.save_model(output_folder / f'{selected_run_id}.txt')
lgb.plot_metric(evals)
with open(output_folder / "evals.json", 'w', encoding="utf-8") as json_file:
    json.dump(evals, json_file)

print("test")
