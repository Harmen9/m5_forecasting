class Baseline:
    '''Defines the baseline parameters for preprocessing and training.'''
    NAME: str = 'baseline'
    TARGET: str = 'sales'
    LAGS: list = list(range(1, 15))
    TRAIN_SPLIT: int = 28
    P_HORIZON: int = 28
    SELECTED_STORES: list = ['CA_1']
    index_columns: list = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    lgb_params: dict = {
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

    def to_dict(self):
        return {
            "NAME": self.NAME,
            "TARGET": self.TARGET,
            "LAGS": self.LAGS,
            "TRAIN_SPLIT": self.TRAIN_SPLIT,
            "P_HORIZON": self.P_HORIZON,
            "SELECTED_STORES": self.SELECTED_STORES,
            "index_columns": self.index_columns,
            "lgb_params": self.lgb_params
        }

