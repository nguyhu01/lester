from setuptools import setup, find_packages

setup(
    name='lester',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'seaborn',
        'imbalanced-learn',
    ],
)