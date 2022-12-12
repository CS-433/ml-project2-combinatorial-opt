import torch
import numpy as np
from numpy import random
import networkx as nx

def create_J_tensor(G: nx.Graph, A: float, B: float, C: float) -> torch.Tensor:
    """
    Creates the tensor J in the Hamiltonian for Travelling Salesman Problem (Ising Model)
    """
    n = G.order()
    J = torch.zeros(size = [n, n, n, n], requires_grad=True)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if ((i == j + 1) or 
                    (j == i + 1) or 
                    (i == 0 and j == n-1) or 
                    (j == 1 and i == n-1)):
                        if k == l:
                            continue
                        W = G.get_edge_data(k, l)['weight'] 
                        J[i, j, k, l] = (A / 8) * W
                    
                    elif i == j and k != l:
                        J[i, j, k, l] = B / 4
                    elif k == l and i != j:
                        J[i, j, k, l] = C / 4
                    elif i == j and k == l:
                        J[i, j, k, l] = (B + C) / 4
    return J

def create_h_matrix(G: nx.Graph, A: float, B: float, C: float) -> torch.Tensor:
    """
    Creates the matrix h in the Hamiltonian for the Travelling Salesman Problem (Ising Model)
    """
    n = G.order()
    weights = nx.get_edge_attributes(G, 'weight').values()
    W = sum(weights)

    h = (A / 2) * W + (n-2) * B / 2 + (n-2) * C / 2
    h = torch.tensor(h, requires_grad=True)
    return h

def tsp_hamiltonian(sample: torch.Tensor, G: nx.Graph, A: float, B: float, C: float):
    """
    Returns the Hamiltonian for the Travelling Salesman Problem given a sample
    
    Params:
        sample: torch.Tensor,   the sample to calculate the hamiltonian for
        G: nx.Graph,            the graph
        A: float                hyperparam
        B: float                hyperparam
        C: float                hyperparam
    Returns:
        output: float,          the energy of the state (sample)
    """
    output = 0
    n = G.order()
    J = create_J_tensor(G, A, B, C)
    h = create_h_matrix(G, A, B, C)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    output += J[i, j, k, l]*sample[i, k]*sample[j, l] + h[i, k]*sample[i, k]

    return output
