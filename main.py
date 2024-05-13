import torch

"""
Aggregate computing through tensor computation
The idea in this work is to map the abstraction of aggregate computing through tensor computation.
In this way, we can use the power of tensor computation to perform aggregate computing.

# Model
The model is a graph, where each node has:
- a feature tensor that comprises the node's state
- a sensor tensor that comprises the node's sensor readings
- an edge tensor that comprises the node's edges
- a weight tensor that comprises the node's weights

Given N nodes, the feature tensor is a tensor of shape [N, F], where F is the number of features.
The sensor tensor is a tensor of shape [N, S], where S is the number of sensors.
The edge tensor is a tensor of shape [2, E], where E is the number of edges.
The weight tensor is a tensor of shape [E, W], where W is the number of weights.

With this information, we have to compute the matrix of neighbors, which is a tensor of shape [N, N, F].

"""

# Set GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.set_default_device(device)
class Graph():
    def __init__(self, sensor, nodes, edges, weights):
        self.sensor = sensor
        self.nodes = nodes
        self.edges = edges
        self.weights = weights
        ## self.neighbors should be created as follow
        ## consider a nodes as tensor [f_0, f_1, f_2, f_3, f_4]
        ## consider a edges as tensor [[0, 1, 2, 3], [1, 2, 3, 4]]
        ## the self.tensor should be [[f0_1, f0_2, f0_3, f0_4], [f1_0, f1_2, f1_3, f1_4], [f2_0, f2_1, f2_3, f2_4], [f3_0, f3_1, f3_2, f3_4], [f4_1, f4_2, f4_3, f4_0]]
        ## where f0_1 is f1 if edges has 0, 1, otherwise 0
        self.neighbors = self.__compute_neighbors__()
        self.weight_matrix = self.__compute_weight_matrix__()
    def update(self, f):
        self.nodes = f(self)
        self.neighbors = self.__compute_neighbors__()
    def __repr__(self):
        return str(self.neighbors)

    def __compute_neighbors__(self):
        num_nodes = self.nodes.size(0)
        feature_size = self.nodes.size(1)
        # Create an empty adjacency matrix
        adjacency_matrix = torch.full((num_nodes, num_nodes, feature_size), float('inf'), dtype=self.nodes.dtype)
        # Use edges to fill in the adjacency matrix
        # Assuming edges is a tensor of shape [2, num_edges] where the first row is the source and the second is the target
        adjacency_matrix[self.edges[0], self.edges[1]] = 1
        adjacency_matrix[self.edges[1], self.edges[0]] = 1  # For undirected graph; remove if directed
        # Generate the neighbors tensor using matrix multiplication
        # This will sum up the node features for each neighbor
        return adjacency_matrix

    def node_neighbors(self):
        return self.nodes * self.neighbors

    def sensor_neighbors(self):
        return self.sensor * self.neighbors

    def __compute_weight_matrix__(self):
        num_nodes = self.nodes.size(0)
        feature_size = self.nodes.size(1)
        # Create an empty tensor for the weighted adjacency matrix with the appropriate shape
        weighted_adjacency_matrix = torch.zeros((num_nodes, num_nodes, feature_size), dtype=self.nodes.dtype)

        # Apply weights to the connections defined in self.edges
        weighted_adjacency_matrix[self.edges[0], self.edges[1]] = self.weights
        weighted_adjacency_matrix[
            self.edges[1], self.edges[0]] = self.weights  # For undirected graph; remove if directed

        # Expand the weighted adjacency matrix to have the same third dimension as features
        # This is necessary to multiply it with the nodes' features
        expanded_weighted_adjacency_matrix = weighted_adjacency_matrix.expand(-1, -1, feature_size)
        return expanded_weighted_adjacency_matrix

zeros = torch.tensor([[0.0], [1000.0], [1000.0], [1000.0], [1000.0]]) # 5 nodes
sources = torch.tensor([[1], [0], [0], [0], [0]])
edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]) # 5 nodes, 4 edges
weight = torch.tensor([[1.0], [2.0], [3.0], [4.0]]) # 4 weights
simpleGraph = Graph(sources, zeros, edges, weight)
simpleGraph.__compute_weight_matrix__()
def update(graph):
    print(torch.min(graph.node_neighbors(), dim=1).values)
    return graph.nodes + 1

for i in range(10):
    simpleGraph.update(update)

print(simpleGraph.nodes)
