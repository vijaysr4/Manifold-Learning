
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler


from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csgraph


class CustomIsomap:
    def __init__(self, n_components, n_neighbors=5, eps=1e-8):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.eps = eps
    
    def fit_transform(self, X):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # Find k-nearest neighbors and create the graph
        knn = NearestNeighbors(n_neighbors=self.n_neighbors+1, algorithm='ball_tree')
        knn.fit(X)
        dist, neighbors = knn.kneighbors(X)
        print("Neighbors: ", neighbors)
        # Convert graph to adjacency matrix
        G = np.zeros((len(X), len(X)))
        for i, nbrs in enumerate(neighbors):
            G[i, nbrs] = dist[i, 1:]
            G[nbrs, i] = dist[i, 1:]
        
        # Calculate shortest paths and add regularization term
        G_shortest = csgraph.shortest_path(G)
        G_shortest[np.isinf(G_shortest)] = np.max(G_shortest[~np.isinf(G_shortest)]) * self.eps
        np.fill_diagonal(G_shortest, 0)
        
        # Compute centering matrix
        n = X.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        
        # Compute the low-dimensional embedding
        S = -0.5 * H @ G_shortest**2 @ H
        eigvals, eigvecs = np.linalg.eigh(S)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx][:self.n_components]
        eigvecs = eigvecs[:, idx][:, :self.n_components]
        Y = eigvecs @ np.diag(np.sqrt(eigvals))
        
        return Y



'''
class CustomIsomap:
    def __init__(self, n_components, n_neighbors=5, eps=1e-8):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.eps = eps
    
    def fit_transform(self, X):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # Find k-nearest neighbors and create the graph
        knn = NearestNeighbors(n_neighbors=self.n_neighbors+1)
        knn.fit(X)
        neighbors = knn.kneighbors_graph(X, mode='distance')
        print("Neighbors: ", neighbors)
        # Convert graph to adjacency matrix
        G = neighbors.toarray()
        G[G == 0] = np.inf
        
        # Calculate shortest paths and add regularization term
        G_shortest = shortest_path(G)
        G_shortest[np.isinf(G_shortest)] = np.max(G_shortest[~np.isinf(G_shortest)]) * self.eps
        np.fill_diagonal(G_shortest, 0)
        
        # Compute centering matrix
        n = X.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        
        # Compute the low-dimensional embedding
        S = -0.5 * H @ G_shortest**2 @ H
        eigvals, eigvecs = np.linalg.eigh(S)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx][:self.n_components]
        eigvecs = eigvecs[:, idx][:, :self.n_components]
        Y = eigvecs @ np.diag(np.sqrt(eigvals))
        
        return Y
    '''
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap

# Generate Swiss Roll data
X, _ = make_swiss_roll(n_samples=1000, random_state=42)
n_c = 2
n_n = 5

# Fit and transform using scikit-learn
isomap_sk = Isomap(n_components=n_c, n_neighbors=n_n)
Y_sk = isomap_sk.fit_transform(X)
print("Scikit-learn:\n", Y_sk)
'''
# Fit and transform using our implementation
isomap_custom = Isomap(n_components=n_c, n_neighbors=n_n)
Y_custom = isomap_custom.fit_transform(X)
'''
# Fit and transform using our custom implementation
isomap_custom = CustomIsomap(n_components=2, n_neighbors=5)
Y_custom = isomap_custom.fit_transform(X)
print("Custom:\n", Y_custom)
# Compare the outputs
print(np.allclose(Y_sk, Y_custom))  # should output True

import matplotlib.pyplot as plt

# Plot the Swiss Roll dataset
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap=plt.cm.Spectral)
ax.set_title("Swiss Roll dataset")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Plot the embeddings obtained using scikit-learn and our custom implementation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.scatter(Y_sk[:, 0], Y_sk[:, 1], c=X[:, 2], cmap=plt.cm.Spectral)
ax1.set_title("Isomap (scikit-learn)")
ax1.set_xlabel("Component 1")
ax1.set_ylabel("Component 2")

ax2.scatter(Y_custom[:, 0], Y_custom[:, 1], c=X[:, 2], cmap=plt.cm.Spectral)
ax2.set_title("Isomap (custom implementation)")
ax2.set_xlabel("Component 1")
ax2.set_ylabel("Component 2")

plt.show()
