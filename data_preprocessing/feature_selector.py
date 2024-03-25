from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class FeatureSelector:
    def __init__(self):
        pass

    def variance_threshold(self, data, threshold=0.1):
        selector = VarianceThreshold(threshold)
        selector.fit(data)
        selected_features = data.columns[selector.get_support()]
        return data[selected_features]

    def correlation_analysis(self, data, target_column, top_k=10):
        corr_matrix = data.corr()
        top_corr_features = corr_matrix[target_column].abs().sort_values(ascending=False).head(top_k+1).index[1:]
        return data[top_corr_features]

    def univariate_selection(self, X, y, top_k=10, scoring='f_classif'):
        selector = SelectKBest(score_func=f_classif, k=top_k)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
        return X[selected_features]

    def recursive_feature_elimination(self, X, y, n_features_to_select=10, estimator=LogisticRegression()):
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
        return X[selected_features]

    def feature_importance(self, X, y, top_k=10, estimator=RandomForestClassifier()):
        estimator.fit(X, y)
        importances = pd.Series(estimator.feature_importances_, index=X.columns)
        top_features = importances.nlargest(top_k).index
        return X[top_features]