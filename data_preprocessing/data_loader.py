import pandas as pd

class DataLoader:
    def __init__(self):
        pass

    def load_csv(self, file_path):
        return pd.read_csv(file_path)

    def load_excel(self, file_path):
        return pd.read_excel(file_path)

    def load_sql(self, connection, query):
        return pd.read_sql(query, connection)

    def load_json(self, file_path):
        return pd.read_json(file_path)

    def load_parquet(self, file_path):
        return pd.read_parquet(file_path)