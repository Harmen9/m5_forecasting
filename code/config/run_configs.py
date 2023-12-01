class Baseline:
    '''Defines the baseline parameters for preprocessing and training.'''
    NAME = 'baseline'
    TARGET = 'sales'
    lags: list = list(range(1, 15))
    TRAIN_SPLIT = 28
    P_HORIZON = 28
    stores = ['CA_1']
    index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

