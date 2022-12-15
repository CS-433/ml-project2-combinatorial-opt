import torch
from hamiltonians import *
from graphs import *

#Hyperparams and kwargs
A = 8
B = 1000
C = 1000
D = 1000

seed = 0

# Create Graph 
G = random_connected_graph(8, 9, seed)
G = random_hamiltonian_graph(G)
G = assign_random_weights(G, 40)

# Create Matrices
J = create_J_tensor(G, A, B, C, D)
h = create_h_matrix(G, A, B, C)

print(J[1, 2, :, :])
print(J[1, 1, :, :])
