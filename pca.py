from sklearn.decomposition import PCA
import numpy as np

# Generate some random data
X = np.random.rand(100, 5)  # 100 samples with 5 features each

# Create a PCA object with the desired number of components
pca = PCA(n_components=3)

# Fit the data to the PCA model
pca.fit(X)

# Transform the data to the lower-dimensional representation
X_transformed = pca.transform(X)

# Print the transformed data
print("Transformed Data:")
print(X_transformed)
