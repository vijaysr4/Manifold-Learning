from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import BallTree
import plotly.graph_objs as go
import plotly.offline as pyo

# Generate Swiss Roll data
X, _ = make_swiss_roll(n_samples=1000, random_state=42)

# Separate the X, Y, and Z coordinates
x_coords = X[:, 0]
y_coords = X[:, 1]
z_coords = X[:, 2]

# Normalize the x_coords values to a range of [0, 1]
normalized_x_coords = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())

# Create a dictionary to store data points and their corresponding colors
points_colors = {}
for i, (x, y, z) in enumerate(X):
    points_colors[i] = (x, y, z, normalized_x_coords[i])

# Swiss Roll plot
scatter = go.Scatter3d(
    x=x_coords,
    y=y_coords,
    z=z_coords,
    mode='markers',
    marker=dict(size=5, color=normalized_x_coords, colorscale='Plasma', showscale=True)
)

layout_swiss_roll = go.Layout(
    title='Swiss Roll 3D Interactive Plot',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    )
)

fig_swiss_roll = go.Figure(data=[scatter], layout=layout_swiss_roll)
pyo.plot(fig_swiss_roll, filename='swiss_roll_fixed.html')

# BallTree plot
tree = BallTree(X)
distances, indices = tree.query(X, k=5)

traces = []
for i in range(X.shape[0]):
    x, y, z, color = points_colors[i]
    trace = go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker=dict(size=5, color=color, colorscale='Plasma', showscale=False))
    traces.append(trace)

for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:
        x1, y1, z1, _ = points_colors[i]
        x2, y2, z2, _ = points_colors[j]
        trace = go.Scatter3d(x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines', line=dict(width=1, color='blue'))
        traces.append(trace)

layout_balltree = go.Layout(title='Interactive BallTree Visualization - No of neighbor as threshold', scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')))
fig_balltree = go.Figure(data=traces, layout=layout_balltree)
pyo.plot(fig_balltree, filename='balltree_colored_nodes_plasma_fixed.html')
