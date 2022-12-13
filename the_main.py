import torch
from conv_net_implementation import AutoRegressiveCNN
from hamiltonians import *
from graphs import *

#Hyperparams and kwargs
A = 8
B = 1000
C = 1000
D = 1000

max_iters = 500
batch_size = 1
beta = 1

seed = 0

# Create Graph 
G = random_connected_graph(8, 9, seed)
G = random_hamiltonian_graph(G)
G = assign_random_weights(G, 40)

# Create Matrices
J = create_J_tensor(G, A, B, C, D)
h = create_h_matrix(G, A, B, C)

# Create Neural Net
vars = {'L': G.order(), 'net_depth': 5, 'net_width': 3, 'half_kernel_size': 3, 'bias': 0.5, 'epsilon': 0.00001, 'device': 'cpu'}

net = AutoRegressiveCNN(**vars)
optimizer = torch.optim.Adam(net.parameters())

for count in range(max_iters):
    optimizer.zero_grad()

    with torch.no_grad():
        sample, s_hat = net.sample(batch_size)

    log_prob = net.log_prob(sample)

    with torch.no_grad():
        energy = tsp_hamiltonian(sample, J, h)
    
    loss = log_prob + beta*energy
    loss.backward()

    optimizer.step()

    if count % 50 == 0:
        print(f"Training loop {count} / {max_iters}\n")


s_hat, sample = net.sample(4)

print("Final Samples:", sample)
print("Final s_hat", s_hat)