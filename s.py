import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
#from sklearn.utils.graph import graph_shortest_path

def make_swissroll(n_samples):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    x = t * np.cos(t)
    y = np.random.rand(1, n_samples)
    z = t * np.sin(t)
    return np.concatenate((x, y, z), axis=0).T

def pairwise_distances(X):
    dists = cdist(X, X)
    return dists

def shortest_path_distances(X, n_neighbors):
    dists = pairwise_distances(X)
    n_samples = X.shape[0]
    indices = np.argsort(dists, axis=1)[:, 1:n_neighbors+1]
    rows = np.repeat(np.arange(n_samples), n_neighbors)
    cols = indices.reshape(-1)
    vals = dists[rows, cols]
    W = np.zeros((n_samples, n_samples))
    W[rows, cols] = vals
    W[cols, rows] = vals
    for k in range(n_samples):
        for i in range(n_samples):
            for j in range(n_samples):
                if W[i, k] > 0 and W[k, j] > 0:
                    if W[i, j] == 0 or W[i, j] > W[i, k] + W[k, j]:
                        W[i, j] = W[i, k] + W[k, j]
    return W

def isomap(X, n_neighbors, n_components):
    G = shortest_path_distances(X, n_neighbors)
    G = (G + G.T) / 2
    H = np.eye(G.shape[0]) - np.ones((G.shape[0], G.shape[0])) / G.shape[0]
    B = -H @ G @ H / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    indices = np.argsort(eigvals)[:n_components]
    embedding = eigvecs[:, indices]
    return embedding

X = make_swissroll(500)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap='jet')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
'''
embedding = isomap(X, n_neighbors=10, n_components=2)
plt.scatter(embedding[:, 0], embedding[:, 1], c=X[:, 2], cmap='jet')
plt.colorbar()
plt.show()
'''
from sklearn import manifold

def isomap_(X, n_components, n_neighbors):
    model = (manifold.Isomap(n_components=n_components, n_neighbors=n_neighbors))
    Y = model.fit_transform(X)
    return Y

# Apply Isomap algorithm
Y = isomap(X, n_components=2, n_neighbors=3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.scatter(Y[:, 0], Y[:, 1], c=X[:, 2], cmap=plt.cm.Spectral)
ax.set_title("Isomap Embedding")
plt.show()

import networkx as nx
from scipy.spatial import cKDTree
# Compute nearest neighbor graph using k-d tree
tree = cKDTree(Y)
distances, indices = tree.query(Y, k=10)

# Create graph from edges
edges = [(i, indices[i, j]) for i in range(indices.shape[0]) for j in range(1, indices.shape[1])]
G = nx.Graph(edges)

# Plot graph with node positions
pos = dict(enumerate(Y))
nx.draw_networkx(G, pos=pos, node_size=10, node_color=X[:, 2], cmap=plt.cm.Spectral)
plt.title("Nearest Neighbor Graph from Isomap on Swiss Roll Dataset")
plt.show()