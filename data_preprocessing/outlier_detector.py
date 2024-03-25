from sklearn.ensemble import IsolationForest
from scipy import stats

class OutlierDetector:
    def __init__(self):
        pass

    def detect_outliers_zscore(self, data, threshold=3):
        z_scores = stats.zscore(data)
        outliers = (abs(z_scores) > threshold).any(axis=1)
        return data[~outliers]

    def detect_outliers_iqr(self, data, factor=1.5):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data < (Q1 - factor * IQR)) | (data > (Q3 + factor * IQR))).any(axis=1)
        return data[~outliers]

    def detect_outliers_isolation_forest(self, data, contamination=0.1):
        isolation_forest = IsolationForest(contamination=contamination)
        outliers = isolation_forest.fit_predict(data) == -1
        return data[~outliers]