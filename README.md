# lester
A framework for automating and accelerating the data cleaning and preprocessing process. The framework provides reusable and scalable components for loading, cleaning, transforming, and preparing data for machine learning tasks.

## Features
- Load data from various sources (CSV, Excel, SQL databases, JSON, parquet)
- Handle missing values with advanced techniques (KNN imputation, MICE)
- Encode categorical variables (label encoding, one-hot encoding, target encoding)
- Scale and normalize numerical features (StandardScaler, MinMaxScaler, RobustScaler)
- Perform feature selection (variance threshold, correlation analysis, recursive feature elimination)
- Detect and handle outliers (Z-score, IQR, isolation forest)
- Balance imbalanced datasets (oversampling, undersampling, SMOTE)
- Split data into training and testing sets (stratified split, time-based split)
- Visualize data distributions and correlations
- Extensible architecture for adding custom preprocessing steps

## Installation
1. Clone the repository: `git clone https://github.com/nguyhu01/lester.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage
See the `examples/example_usage.py` file for a sample usage of the framework.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any bug fixes, enhancements, or new features.

## License
This project is licensed under the MIT License.