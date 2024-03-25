from data_preprocessing import DataLoader, DataCleaner, DataTransformer, FeatureSelector, OutlierDetector, DataBalancer
from data_preprocessing.utils import print_data_summary, plot_feature_distribution, plot_correlation_matrix

# Load data
loader = DataLoader()
data = loader.load_csv('example.csv')

# Clean data
cleaner = DataCleaner()
data = cleaner.remove_duplicates(data)
data = cleaner.handle_missing_values(data, strategy='knn', n_neighbors=5)

# Transform data
transformer = DataTransformer()
data = transformer.encode_categorical_variables(data, ['category'], encoding='onehot')
data = transformer.scale_numerical_features(data, ['feature1', 'feature2'], scaler='robust')

# Feature selection
selector = FeatureSelector()
data = selector.variance_threshold(data, threshold=0.2)
data = selector.correlation_analysis(data, 'target', top_k=5)

# Outlier detection
detector = OutlierDetector()
data = detector.detect_outliers_iqr(data, factor=1.5)

# Data balancing
balancer = DataBalancer()
X = data.drop('target', axis=1)
y = data['target']
X_resampled, y_resampled = balancer.oversample(X, y, strategy='smote')

# Split data
X_train, X_test, y_train, y_test = transformer.split_data(data, 'target', test_size=0.2, stratify=y)

# Print data summary and visualizations
print_data_summary(data)
plot_feature_distribution(data, 'feature1')
plot_correlation_matrix(data)