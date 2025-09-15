import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.signal import convolve2d

from pathlib import Path
import pandas as pd


print("Loading data")
if not Path('mnist_data.csv').exists():
    mnist = fetch_openml('mnist_784', version=1, parser="auto")
    mnist.data.to_csv("mnist_data.csv",index=False)
    mnist.target.to_csv("mnist_target.csv",index=False)
    X, y = mnist.data, mnist.target.astype(int)
else:
    X, y = pd.read_csv('mnist_data.csv'), pd.read_csv('mnist_target.csv').astype(int)
# Load the MNIST dataset
#mnist = fetch_openml('mnist_784', version=1)
#X, y = mnist.data, mnist.target.astype(int)

print("Reshaping X")
X = X.values.reshape(-1, 28, 28, 1)
print("Finished reshaping X")
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a custom transformer for the CNN feature extraction
class CNNFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.filters = [np.random.randn(3, 3) for _ in range(32)]
        self.pool_size = (2, 2)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feature_maps = np.array([self._apply_filters(x) for x in X])
        pooled_maps = np.array([self._apply_pooling(fm) for fm in feature_maps])
        return pooled_maps.reshape(pooled_maps.shape[0], -1)

    def _apply_filters(self, x):
        return np.array([convolve2d(x[:, :, 0], f, mode='valid') for f in self.filters])

    def _apply_pooling(self, feature_map):
        pooled_map = np.zeros((feature_map.shape[0], feature_map.shape[1] // self.pool_size[0], feature_map.shape[2] // self.pool_size[1]))
        for i in range(0, feature_map.shape[1], self.pool_size[0]):
            for j in range(0, feature_map.shape[2], self.pool_size[1]):
                pooled_map[:, i // self.pool_size[0], j // self.pool_size[1]] = np.max(feature_map[:, i:i + self.pool_size[0], j:j + self.pool_size[1]], axis=(1, 2))
        return pooled_map

# Create the pipeline
pipeline = Pipeline([
    ('cnn', CNNFeatureExtractor()),
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, random_state=42))
])

print("Starting pipeline")

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the pipeline
score = pipeline.score(X_test, y_test)
print(f'Accuracy: {score:.2%}')