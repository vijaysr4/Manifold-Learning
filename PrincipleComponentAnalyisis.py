import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
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

print("X shape: ", len(X))
#X2 = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

def PCA_scikit(X):
    fig = plt.figure(figsize=(12,6))
    data = X
    k = 8

    pca = PCA(n_components=2)

    pca.fit(data)
    print("EigenVectors-scikit:\n", pca.components_, "\n")
    print("Covariance-scikit:\n", pca.get_covariance(), "\n")
    X_pca = pca.transform(data)

    print("X-reduced: \n",X_pca)

    
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(X_pca[(color < 10), 0], X_pca[(color < 10), 1], c='blue')
    ax.scatter(X_pca[(color >= 10), 0], X_pca[(color >= 10), 1], c='red')
    ax.set_title('PCA with two components', size=14)

PCA_scikit(X)


def covariance_matrix(X):
    # Mean center the data
    #X = X - np.mean(X, axis=0)
    # Compute the covariance matrix
    cov_matrix = np.dot(X.T, X) / (X.shape[0] - 1)
    return cov_matrix
#covmatrix = covariance_matrix(X)
#print("fun cov",covmatrix)

import numpy as np

# A is the matrix for which you want to find the eigenvectors
def eigen_vectors(A):
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


# X is the data matrix, n_components is the number of principal components to keep
def PCA(X, n_components):
    # Mean center the data
    X = X - np.mean(X, axis=0)
    # Compute the covariance matrix
    cov_matrix = covariance_matrix(X)
    print("Covarience Matrix\n", cov_matrix)
    
    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print("Eigen val: \n", eigenvalues, "\n Eigen V: \n", eigenvectors)
    #eigenvalues1, eigenvectors1 = eigen_vectors(cov_matrix)
    #print("Eigen val 1: \n", eigenvalues1, "\n Eigen V 1: \n", eigenvectors1)
    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Take the first n_components eigenvectors
    eigenvectors = eigenvectors[:, :n_components]
    # Project the data onto the new feature space
    X_reduced = np.dot(X, eigenvectors)
    
    fx = fig.add_subplot(1, 2, 1)
    fx.scatter(X_reduced[(color < 10), 0], X_reduced[(color < 10), 1], c='blue')
    fx.scatter(X_reduced[(color >= 10), 0], X_reduced[(color >= 10), 1], c='red')
    fx.set_title('PCA from scratch', size=14)
    plt.show()
    return X_reduced, eigenvectors, eigenvalues

# Use the function to reduce the dimensionality of the data
X_reduced, eigenvectors, eigenvalues = PCA(X, 2)

print("Eigen Vectors PCA Without Scikit:\n ", eigenvectors.T)
print("X_reduced PCA Without Scikit:\n ", X_reduced)


def PCA_svd(X, n_components):
    # Mean center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Perform SVD on the centered data
    U, s, VT = np.linalg.svd(X_centered)
    print("\nU",U , "\ns",s, "\nVT", VT)
    
    # Project the data onto the first two principal components
    components = VT[:n_components].T
    X_projected = np.dot(X_centered, components)
    return X_projected

Pca_svd_x = PCA_svd(X, 2)
print("\nReduced X from SVD PCA: \n", Pca_svd_x)