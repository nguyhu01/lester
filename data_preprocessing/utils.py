import matplotlib.pyplot as plt
import seaborn as sns

def print_data_summary(data):
    print("Data Summary:")
    print(data.head())
    print("\nData Shape:", data.shape)
    print("\nData Types:")
    print(data.dtypes)
    print("\nMissing Values:")
    print(data.isnull().sum())

def plot_feature_distribution(data, feature):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.show()