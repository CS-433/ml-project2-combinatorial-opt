import torch
from hamiltonians import *
from graphs import *

#Hyperparams and kwargs
A = 8
B = 1000
C = 3000
D = 1000

seed = 0

# Create Graph 
G = random_connected_graph(2, 2, seed)
G = random_hamiltonian_graph(G)
G = assign_random_weights(G, 10)

# Create Matrices
J_alt, h_alt = create_J_and_h(G, A, B, C, D)
J = create_J_tensor(G, A, B, C, D)
h = create_h_matrix(G, A, B, C)

#Create sample
sample = torch.tensor([[[[-1,-1],[1,-1]]]])
print("sample:", sample)
print("J:", J)
energy = tsp_hamiltonian(sample, J, h)

print("Energy:", energy)
