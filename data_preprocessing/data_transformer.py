from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce

class DataTransformer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.target_encoder = ce.TargetEncoder()

    def encode_categorical_variables(self, data, columns, encoding='label', target_column=None):
        if encoding == 'label':
            for column in columns:
                data[column] = self.label_encoder.fit_transform(data[column])
        elif encoding == 'onehot':
            ct = ColumnTransformer(
                [('onehot', self.onehot_encoder, columns)],
                remainder='passthrough'
            )
            data = ct.fit_transform(data)
        elif encoding == 'target':
            if target_column is None:
                raise ValueError("Target column must be provided for target encoding.")
            else:
                data[columns] = self.target_encoder.fit_transform(data[columns], data[target_column])
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        return data

    def scale_numerical_features(self, data, columns, scaler='standard'):
        if scaler == 'standard':
            scaler = StandardScaler()
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
        elif scaler == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler}")

        data[columns] = scaler.fit_transform(data[columns])
        return data

    def split_data(self, data, target_column, test_size=0.2, random_state=42, stratify=None):
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)