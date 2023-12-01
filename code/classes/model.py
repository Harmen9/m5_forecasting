import pandas as pd
from typing import List


class TsPreproc:
    data: pd.DataFrame

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def melt(self, id_vars: List[str], var_name: str, value_name: str):
        self.data: pd.DataFrame = pd.melt(
            self.data,
            id_vars=id_vars,
            var_name=var_name,
            value_name=value_name
        )

    def generate_lags(self, lags: List[int], group_by: List[str], lag_column: str):
        for lag in lags:
            self.data[f'{lag_column}_{lag}'] = self.data.groupby(group_by)[lag_column].shift(lag)
        # Drop rows which have Nan in the lags
        lag_columns = [f'{lag_column}_{lag}' for lag in lags]
        self.data.dropna(subset=lag_columns, inplace=True)

