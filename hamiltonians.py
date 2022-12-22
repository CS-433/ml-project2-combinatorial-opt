import torch
import numpy as np
from numpy import random
import networkx as nx

                        
def create_J_tensor(G: nx.Graph, A: float, B: float, C: float, D: float) -> torch.Tensor:
    """
    Creates the tensor J in the Hamiltonian for Travelling Salesman Problem (Ising Model)
    """
    n = G.order()
    J = torch.zeros(size = [n, n, n, n])
    #J.to_sparse()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if ((i == j + 1) or (j == i + 1) or 
                        (i == 0 and j == n-1) or 
                        (j == 0 and i == n-1)):
                        if k != l:
                            if G.has_edge(k, l):
                                W = G.get_edge_data(k, l)['weight']
                            else:
                                W = D
                            J[i, j, k, l] = (A / 8) * W
                    if i == j and k != l:
                        J[i, j, k, l] += B / 4
                    if i == j and k == l:
                        J[i, j, k, l] += (B + C) / 4
                    if k == l and i != j:
                        J[i, j, k, l] += C / 4
    return J

def create_h_matrix(G: nx.Graph, A: float, B: float, C: float) -> torch.Tensor:
    """
    Creates the matrix h in the Hamiltonian for the Travelling Salesman Problem (Ising Model)
    """
    n = G.order()
    h = torch.zeros(size=[n, n])

    weights = nx.get_edge_attributes(G, 'weight').values()
    W = sum(weights)
    h = h + (A / 2) * W + (n-2) * B / 2 + (n-2) * C / 2
    return h

def tsp_hamiltonian(sample: torch.Tensor, J: torch.Tensor, h: torch.Tensor):
    """
    Returns the Hamiltonian for the Travelling Salesman Problem given a sample
    
    Params:
        sample: torch.Tensor,   the sample to calculate the hamiltonian for
        J: torch.Tensor,        J tensor in the Hamiltonian
        h: torch.Tensor,        h matrix in the Hamiltonian
    Returns:
        output: float,          the energy of the state (sample)
    """
    output = 0
    n = J.size(dim=0)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    output += J[i, j, k, l]*sample[:, :, i, k]*sample[:, :, j, l] + h[i, k]*sample[:, :, i, k]

    return output