from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap
from sklearn.neighbors import BallTree
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.offline as pyo
import heapq

# Generate Swiss Roll data
X, _ = make_swiss_roll(n_samples=1000, random_state=42)

# Construct BallTree
tree = BallTree(X)

# Compute the nearest neighbors
distances, indices = tree.query(X, k=5)
print(distances.shape, indices.shape)

# Convert indices to list of tuples
indices_list = [tuple(row) for row in indices]

# Create a graph representation as an adjacency list
graph = {i: [] for i in range(len(X))}

# Populate graph with neighbors
for i, neighbors in enumerate(indices_list):
    graph[i] = neighbors

# Function to compute shortest path using Dijkstra's algorithm
def dijkstra(graph, start):
    # Initialize distances with large values except for the starting node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # Priority queue to store nodes to visit
    queue = [(0, start)]
    
    # Set to keep track of visited nodes
    visited = set()
    
    while queue:
        # Pop node with minimum distance from the priority queue
        (dist, node) = heapq.heappop(queue)
        
        # Skip if node has already been visited
        if node in visited:
            continue
        
        # Add node to visited set
        visited.add(node)
        
        # Update distances to neighboring nodes
        for neighbor in graph[node]:
            weight = distances[node] + np.linalg.norm(X[node] - X[neighbor])
            
            # Update distance if it's smaller than current distance
            if weight < distances[neighbor]:
                distances[neighbor] = weight
                heapq.heappush(queue, (weight, neighbor))
    
    return distances

# Compute shortest path from node 0 using Dijkstra's algorithm
start_node = 0
shortest_distances = dijkstra(graph, start_node)

# Print shortest distances
print("Shortest distances from node", start_node, "to all other nodes:")
for i, distance in shortest_distances.items():
    print("Node", i, ": Distance", distance)


import numpy as np
from sklearn.decomposition import TruncatedSVD

# Compute the shortest distances using Dijkstra's algorithm
# (assuming 'shortest_distances' contains the distances)
# ...

# Perform partial eigenvalue decomposition (t-SVD) on the distance matrix
n_components = 2  # Number of components to reduce to
tsvd = TruncatedSVD(n_components=n_components)
X_embedded = tsvd.fit_transform(distances)

# Access the reduced dimensional data
x = X_embedded[:, 0]  # x-coordinates in reduced space
y = X_embedded[:, 1]  # y-coordinates in reduced space

# Plot the reduced dimensional data
plt.scatter(x, y)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Partial Eigenvalue Decomposition')
plt.show()

