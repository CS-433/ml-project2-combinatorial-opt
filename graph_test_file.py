import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import operator as op

def random_connected_graph(min_nbr_nodes: int,  max_nbr_nodes: int, seed = None) -> nx.Graph:
    G = nx.Graph()

    nbr_of_nodes = np.random.randint(min_nbr_nodes, max_nbr_nodes + 1)
    G.add_nodes_from(range(nbr_of_nodes))

    nbr_of_edges = np.random.randint(ncr(nbr_of_nodes, 2))

    for _ in range(nbr_of_edges):
        edge = [np.random.randint(nbr_of_nodes), np.random.randint(nbr_of_nodes)]
        while G.has_edge(edge[0], edge[1]):
            edge = [np.random.randint(nbr_of_nodes), np.random.randint(nbr_of_nodes)]
        G.add_edge(edge[0], edge[1])
                
    if not nx.is_connected(G):
        G = nx.complement(G)

    return G

def is_hamiltonian(G: nx.Graph) -> bool:
    """
    Checks if graph G is Hamiltonian (contain a Hamilton cycle).
    """
    if (not nx.is_connected(G)) or (G.order() < 3):
        return False
    degrees = list(G.degree())
    order = G.order()
    degrees.sort(key=lambda x: x[1])
    return degrees[0][1] + degrees[1][1] >= order

def random_hamiltonian_graph(G: nx.Graph) -> nx.Graph:
    """
    Takes in a connected graph G, and returns a Hamiltonian graph
    
    For connected graphs with p >= 3, if for each pair of vertices u, v one has:

            deg(u) + deg(v) >= p

    then G is Hamiltonian
    """
    if is_hamiltonian(G):
        return G
    degrees = list(G.degree())
    order = G.order()

    for vertex in degrees:
        while G.degree(vertex[0]) < order/2:
            edge = [vertex[0], np.random.randint(order)]
            idx = 0
            while G.has_edge(edge[0], edge[1]) and vertex[0] != idx and idx < order:
                edge = [vertex[0], idx]
                idx += 1
            G.add_edge(edge[0], edge[1])

    return G

def assign_random_weights(G: nx.Graph, max_edge_weight: int) -> nx.Graph:
    edges = list(nx.edges(G))

    for edge in edges:
        weight = np.random.randint(1, max_edge_weight)
        nx.set_edge_attributes(G, {edge: {'weight': weight}})

    return G


def ncr(n, r):
    """
    The combination n over r = n! / ((n-r)!r!)
    """
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

# G = random_connected_graph(3, 8)
# G = random_hamiltonian_graph(G)
# G.name = "Hamiltonian Graph"


G = nx.Graph()
G.add_nodes_from(range(5))
G.add_edge(0, 1)
G.add_edge(0, 2)
G.add_edge(0, 3)
G.add_edge(0, 4)

G = assign_random_weights(G, 10)

print(G.get_edge_data(1, 2, 'weight'))

# a = G.get_edge_data(0, 1)['weight']
# print(a)
# print(type(a))

# a = nx.get_edge_attributes(G, 'weight')
# print(a)
# print(type(a))

# b = sum(a.values())
# print("Edge weight sum:", b)

# deg = list(nx.edges(G))
# print(deg)
# print(type(deg))

# seed = 31
# pos = nx.spring_layout(G, seed=seed)
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx(G, pos=pos)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()
