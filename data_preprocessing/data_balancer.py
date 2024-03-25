from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

class DataBalancer:
    def __init__(self):
        pass

    def oversample(self, X, y, strategy='random'):
        if strategy == 'random':
            oversampler = RandomOverSampler()
        elif strategy == 'smote':
            oversampler = SMOTE()
        else:
            raise ValueError(f"Unknown oversampling strategy: {strategy}")

        X_resampled, y_resampled = oversampler.fit_resample(X, y)
        return X_resampled, y_resampled

    def undersample(self, X, y):
        undersampler = RandomUnderSampler()
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        return X_resampled, y_resampled