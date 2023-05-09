'''
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

graph = [
[0, 1, 2, 0],
[0, 0, 0, 1],
[2, 0, 0, 3],
[0, 0, 0, 0]
]
graph = csr_matrix(graph)
print(graph)
dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=0, return_predecessors=True)
print(dist_matrix)
print(predecessors)
'''

