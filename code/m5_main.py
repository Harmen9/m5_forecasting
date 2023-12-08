'''This functions runs the m5 forecasting of the wallmart data.
The input data can be downloaded from Kaggle and should be stored in the
input folder. The different configurations can be set in the code/config folder'''

import json
from datetime import datetime
import pandas as pd
import lightgbm as lgb

from classes import TsPreproc
from config import *

import wandb
from wandb.lightgbm import wandb_callback, log_summary


# Get the current date and time
current_datetime = datetime.now().strftime("%Y%m%d%H%M")
selected_run_id = 'Baseline'

# Read config file
paths = Paths()
data_grid: pd.DataFrame = pd.read_csv(paths.INPUT / 'sales_train_validation.csv')
run_params = globals()[selected_run_id]() 

wandb.init(
    # set the wandb project where this run will be logged
    project="m5_forecasting",
    
    # track hyperparameters and run metadata
    config=run_params.to_dict()
)
# Create output folder:
output_folder = paths.OUTPUT / f"{selected_run_id}_{current_datetime}"
output_folder.mkdir(parents=True, exist_ok=True)

# Store output parameters
with open(output_folder / "run_params.json", 'w', encoding="utf-8") as json_file:
    json.dump(run_params.to_dict(), json_file, indent=4)

# Transformation

# To speed up the training of the data during data exploration only store CA_1 is taken 
# into account
store_mask: pd.Series = data_grid['store_id'].isin(run_params.SELECTED_STORES)
data_grid: pd.DataFrame = data_grid[store_mask]
del store_mask
data_grid: TsPreproc = TsPreproc(data_grid)

# Pivot table
data_grid.melt(
    id_vars=run_params.index_columns,
    var_name='d',
    value_name=run_params.TARGET
)

# Add calendar, only use the first event
calendar_columns: list[str] = ['wday', 'month', 'year', 'd', 'event_name_1', 'event_type_1']
calendar_df: pd.DataFrame = pd.read_csv(paths.INPUT / 'calendar.csv')
data_grid.data = data_grid.data.merge(
    right=calendar_df[calendar_columns],
    how='left',
    left_on='d',
    right_on='d'
    )

data_grid.data['d'] = data_grid.data['d'].str.replace('d_', '').astype(int)

# Get lag features
data_grid.generate_lags(
    lags=run_params.LAGS,
    group_by=['id'],
    lag_column=run_params.TARGET
    )


# Use encoding to encode categorical features.
categorical_features = [
    'item_id',
    'dept_id',
    'cat_id',
    'event_name_1',
    'event_type_1'
    ]
data_grid.label_encode(
    categorical_columns=categorical_features
    )



# Split data
MAX_D = max(data_grid.data['d'])
END_TRAIN = MAX_D - run_params.TRAIN_SPLIT
mask_train = data_grid.data['d']<=END_TRAIN

# Create train and val datasets
train_data = lgb.Dataset(
    data=data_grid.data.drop(
        columns=[
            'id',
            'store_id',
            'state_id',
            run_params.TARGET
            ]
        )[mask_train],
    label=data_grid.data['sales'][mask_train]
    )

val_data = lgb.Dataset(
    data=data_grid.data.drop(
        columns=[
            'id',
            'store_id',
            'state_id',
            run_params.TARGET
            ]
        )[~mask_train],
    label=data_grid.data['sales'][~mask_train]
    )
del data_grid
# Initialize the callback to record evaluation metrics

evals={}
callbacks = [
    lgb.record_evaluation(evals),  # `evals` dictionary to store evaluation results
    wandb_callback()  # W&B callback
]
bst = lgb.train(
    run_params.lgb_params,
    train_data,
    categorical_feature=categorical_features,
    valid_sets=[val_data],
    callbacks=callbacks
    )
log_summary(bst, save_model_checkpoint=True)

bst.save_model(output_folder / f'{selected_run_id}.txt')
with open(output_folder / "evals.json", 'w', encoding="utf-8") as json_file:
    json.dump(evals, json_file)

wandb.finish()
lgb.plot_metric(evals)

print("test")
