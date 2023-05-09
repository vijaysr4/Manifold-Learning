import matplotlib.pyplot as plt
from sklearn import manifold, datasets

from sklearn.datasets import make_swiss_roll

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


from sklearn.decomposition import PCA
fig = plt.figure(figsize=(12,6))
data = X
k = 8

pca = PCA(n_components=2)

pca.fit(data)
print(pca.components_)
X_pca = pca.transform(data)

print(X_pca)
'''
ax = fig.add_subplot(1, 2, 1)
ax.scatter(X_pca[(color < 10), 0], X_pca[(color < 10), 1], c='blue')
ax.scatter(X_pca[(color >= 10), 0], X_pca[(color >= 10), 1], c='red')
ax.set_title('PCA with two components', size=14)
'''


'''
import numpy as np

def eigenvectors(A):
    # Get the eigenvalues of the matrix
    eigenvalues = np.linalg.eigvals(A)
    eigenvectors = []
    for value in eigenvalues:
        # Find the eigenvector corresponding to the eigenvalue
        eigenvector = np.linalg.solve(A - value*np.eye(A.shape[0]), np.ones(A.shape[0]))
        # Normalize the eigenvector
        eigenvector /= np.linalg.norm(eigenvector)
        eigenvectors.append(eigenvector)
    return eigenvalues, eigenvectors

# Use the function to find the eigenvectors of a matrix
#eigenvalues, eigenvectors = eigenvectors(A)

# X is the data matrix
def covariance_matrix(X):
    # Get the number of observations
    n = X.shape[0]
    # Mean center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    # Initialize the covariance matrix with zeros
    cov_matrix = np.zeros((X.shape[1], X.shape[1]))
    # Fill the covariance matrix
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            cov_matrix[i, j] = (X_centered[:, i] * X_centered[:, j]).sum() / (n - 1)
    return cov_matrix

# Use the function to compute the covariance matrix
#cov_matrix = covariance_matrix(X)

def covariance_matrix_np(X):
    # Mean center the data
    X = X - np.mean(X, axis=0)
    # Compute the covariance matrix
    cov_matrix = np.dot(X.T, X) / (X.shape[0] - 1)
    return cov_matrix
#cov_matrix1 = covariance_matrix(X)
#print("function",cov_matrix1)



# X is the data matrix, n_components is the number of principal components to keep
def PCA(X, n_components):
    # Mean center the data
    #X = X - np.mean(X, axis=0)
    # Compute the covariance matrix
    #cov_matrix = np.cov(X.T)
    
    cov_matrix = covariance_matrix(X)

    print(cov_matrix)
    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Take the first n_components eigenvectors
    eigenvectors = eigenvectors[:, :n_components]
    # Project the data onto the new feature space
    X_reduced = np.dot(X, eigenvectors)
    return X_reduced, eigenvectors, eigenvalues

# Use the function to reduce the dimensionality of the data
X_reduced, eigenvectors, eigenvalues = PCA(X, 2)

print("X-reduced: ", X_reduced)
print("EigenVectors: ", eigenvectors)


ax = fig.add_subplot(1, 2, 1)
ax.scatter(X_reduced[(color < 10), 0], X_reduced[(color < 10), 1], c='blue')
ax.scatter(X_reduced[(color >= 10), 0], X_reduced[(color >= 10), 1], c='red')
ax.set_title('PCA with two components', size=14)



sr_points, sr_color = datasets.make_swiss_roll(n_samples=1500, random_state=0)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, s=50, alpha=0.8
)
ax.set_title("Swiss Roll in Ambient Space")
ax.view_init(azim=-66, elev=12)
_ = ax.text2D(0.8, 0.05, s="n_samples=1500", transform=ax.transAxes)

sr_lle, sr_err = manifold.locally_linear_embedding(
    sr_points, n_neighbors=12, n_components=2
)


fig, axs = plt.subplots(figsize=(8, 8), nrows=2)
axs[0].scatter(sr_lle[:, 0], sr_lle[:, 1], c=sr_color)
axs[0].set_title("LLE Embedding of Swiss Roll")


import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh

# X is the high-dimensional data, k is the number of nearest neighbors
def LLE(X, k):
    n_samples, n_features = X.shape
    # Compute pairwise distances
    distances = pairwise_distances(X)
    # Find the k-nearest neighbors for each data point
    knn = np.argsort(distances, axis=1)[:, 1:k+1]
    # Construct the weight matrix
    W = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        z = X[knn[i]] - X[i] # difference between the i-th point and its k-nearest neighbors
        C = np.dot(z, z.T)   # local covariance
        C = C + np.eye(k) * 1e-3 * np.trace(C) # regularization to avoid singularity
        w = np.linalg.solve(C, np.ones(k))    # solve Cw = 1
        W[i, knn[i]] = w / np.sum(w)          # normalize
    # Construct the matrix M
    M = (np.eye(n_samples) - W).T.dot(np.eye(n_samples) - W)
    # Find the eigenvectors of M that correspond to the k smallest eigenvalues
    eigenvalues, eigenvectors = eigsh(M, k+1, sigma=1e-3, which='LM')
    return eigenvectors[:, 1:], eigenvalues[1:]

# Use the function to reduce the dimensionality of the data
X_reduced, eigenvalues = LLE(sr_points, 12)
print(X_reduced, eigenvalues )

'''