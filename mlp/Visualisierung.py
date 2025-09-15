import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# MNIST-Daten laden
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)  # Normalisierung (0-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP-Modell definieren
mlp = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=10, random_state=42)
mlp.fit(X_train, y_train)

# Gewichte der ersten Schicht extrahieren
weights = mlp.coefs_[0]  # Erste Gewichtsmatrix (784 x 128)

# Visualisierung der ersten 64 Neuronen
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < 64:  # Nur 64 Neuronen anzeigen
        ax.imshow(weights[:, i].reshape(28, 28), cmap='gray')
        ax.axis('off')

plt.show()