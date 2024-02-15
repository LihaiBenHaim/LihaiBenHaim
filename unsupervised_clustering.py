import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(data, n_clusters=5):
    """
    Implements unsupervised K-Means clustering for data classification.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(scaled_data)
    
    return clusters, kmeans.cluster_centers_

if __name__ == "__main__":
    X = np.random.rand(100, 10)
    labels, centers = perform_clustering(X)
    print(f"Clustering complete. Identified {len(np.unique(labels))} groups.")