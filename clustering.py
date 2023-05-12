from sklearn.cluster import KMeans
import numpy as np

# Generate some random data
X = np.random.rand(100, 2)  # 100 samples with 2 features each

# Create a KMeans object with the desired number of clusters
kmeans = KMeans(n_clusters=3)

# Fit the data to the KMeans algorithm
kmeans.fit(X)

# Get the cluster labels assigned to each sample
labels = kmeans.labels_

# Get the coordinates of the cluster centers
centroids = kmeans.cluster_centers_

# Print the cluster labels and centroids
print("Cluster Labels:")
print(labels)
print("Centroids:")
print(centroids)
