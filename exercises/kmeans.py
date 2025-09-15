import numpy as np

def kmeans(X, k, max_iters=100):
    # Zufällige Initialisierung der Cluster-Zentren
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Berechnung der Abstände und Zuordnung der Punkte zu den nächsten Zentren
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Berechnung der neuen Zentren
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Überprüfung auf Konvergenz
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels



# Beispiel-Daten
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
k = 2

# Anwendung des K-Means-Algorithmus
centroids, labels = kmeans(X, k)
print("Zentren:", centroids)
print("Labels:", labels)