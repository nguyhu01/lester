import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class DataCleaner:
    def __init__(self):
        pass

    def remove_duplicates(self, data):
        return data.drop_duplicates()

    def handle_missing_values(self, data, strategy='mean', **kwargs):
        if strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'mode':
            return data.fillna(data.mode().iloc[0])
        elif strategy == 'knn':
            imputer = KNNImputer(**kwargs)
            return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif strategy == 'mice':
            imputer = IterativeImputer(**kwargs)
            return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")