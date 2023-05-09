import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path

import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.datasets import make_swiss_roll

class Isomap:
    def __init__(self, n_components, n_neighbors):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
    
    def fit_transform(self, X):
        n_samples, n_features = X.shape
        
        # Compute pairwise distances between data points
        dist_matrix = cdist(X, X)
        
        # Find nearest neighbors of each data point
        neighbors = np.zeros((n_samples, self.n_neighbors))
        for i in range(n_samples):
            # Find distances to all other points
            distances = dist_matrix[i, :]
            # Sort distances and get indices of the k-nearest neighbors
            nn_indices = np.argsort(distances)[1:self.n_neighbors+1]
            # Store indices of the k-nearest neighbors
            neighbors[i, :] = nn_indices
        
        # Compute shortest path distance matrix using Floyd-Warshall algorithm
        path_matrix = shortest_path(dist_matrix, method='FW', directed=False)
        
        # Compute centering matrix
        J = np.eye(n_samples) - np.ones((n_samples, n_samples))/n_samples
        
        # Compute Gram matrix
        B = -0.5 * J.dot(path_matrix ** 2).dot(J)
        
        # Compute eigenvalues and eigenvectors of Gram matrix
        eigvals, eigvecs = np.linalg.eigh(B)
        
        # Select top k eigenvectors corresponding to largest eigenvalues
        eigvals = eigvals[::-1][:self.n_components]
        eigvecs = eigvecs[:, ::-1][:, :self.n_components]
        
        # Transform data to new low-dimensional space
        Y = eigvecs.dot(np.diag(np.sqrt(eigvals)))
        
        return Y

# Example implementation
# Generate toy data with three concentric circles
X, color = make_swiss_roll(n_samples = 3000)
plt.style.use('classic')
plt.rcParams['figure.facecolor'] = 'white'
fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(1, 2, 1, projection = '3d')
ax.scatter(X[(color < 10), 0], X[(color < 10), 1], X[(color < 10), 2], c='blue')
ax.scatter(X[(color >= 10), 0], X[(color >= 10), 1], X[(color >= 10), 2], c='red')
ax.set_title('Swiss Roll', size=16)

ax = fig.add_subplot(1, 2, 2, projection = '3d')
ax.scatter(X[(color < 10), 0], X[(color < 10), 1], X[(color < 10), 2], c='blue')
ax.scatter(X[(color >= 10), 0], X[(color >= 10), 1], X[(color >= 10), 2], c='red')
ax.set_title('Alternate View', size=16)
ax.view_init(4, -80);
plt.show();

# Apply Isomap algorithm to the data
n_components = 2
n_neighbors = 10
isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
X_reduced = isomap.fit_transform(X)
print(X_reduced)


fx = fig.add_subplot(1, 2, 1)
fx.scatter(X_reduced[(color < 10), 0], X_reduced[(color < 10), 1], c='blue')
fx.scatter(X_reduced[(color >= 10), 0], X_reduced[(color >= 10), 1], c='red')
fx.set_title('PCA from scratch', size=14)
plt.show()
