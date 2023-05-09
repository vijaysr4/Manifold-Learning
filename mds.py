import plotly.io as pio
pio.renderers.default = 'notebook_connected'

import plotly.graph_objs as go
from sklearn.datasets import make_s_curve
from sklearn.manifold import MDS

# Generate S-curve data in 3D
X, color = make_s_curve(n_samples=1000, random_state=0)

# Apply MDS to project data onto a 2D plane
model = MDS(n_components=2, random_state=2)
out = model.fit_transform(X)

# Visualize the data using an interactive 3D plot
fig = go.Figure(data=[go.Scatter3d(x=X[:, 0],
                                   y=X[:, 1],
                                   z=X[:, 2],
                                   mode='markers',
                                   marker=dict(color=color,
                                               colorscale='Jet'))])
fig.update_layout(scene=dict(xaxis_title='Dimension 1',
                             yaxis_title='Dimension 2',
                             zaxis_title='Dimension 3'))
fig.show()
