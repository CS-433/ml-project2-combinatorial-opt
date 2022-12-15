import torch
from conv_net_implementation import AutoRegressiveCNN
from neural_net_implementation import SimpleAutoregressiveModel
from hamiltonians import *
from graphs import *
import matplotlib.pyplot as plt

#Hyperparams and kwargs
A = 0.1
B = 5000
C = 1000
D = 1000

max_iters = 100
batch_size = 100
beta = 1000

seed = 0

# Create Graph 
G = random_connected_graph(4, 4, seed)
G = random_hamiltonian_graph(G)
G = assign_random_weights(G, 40)

# Create Tensors/Matrices
J = create_J_tensor(G, A, B, C, D)
h = create_h_matrix(G, A, B, C)

# Create Neural Net
net = 'linear'

if net == 'linear':
    vars = {'L': G.order(), 'n': batch_size, 'net_width': G.order()**2, 'net_depth': 5, 'epsilon': 0.00001, 'device': 'cpu'}
    net = SimpleAutoregressiveModel(**vars)

elif net == 'conv':
    vars = {'L': G.order(), 'net_depth': 7, 'net_width': 5, 'half_kernel_size': 3, 'bias': 0.5, 'epsilon': 0.00001, 'device': 'cpu'}
    net = AutoRegressiveCNN(**vars)

optimizer = torch.optim.Adam(net.parameters())
losses = []
for count in range(max_iters):
    optimizer.zero_grad()

    with torch.no_grad():
        sample, s_hat = net.sample(batch_size)
    assert not sample.requires_grad
    assert not s_hat.requires_grad

    log_prob = net.log_prob(sample)

    with torch.no_grad():
        energy = tsp_hamiltonian(sample, J, h)
        loss = log_prob + beta*energy
    assert not energy.requires_grad
    assert not loss.requires_grad

    loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
    loss_reinforce.backward()
    losses.append(loss_reinforce.data)
    optimizer.step()

    if count % 50 == 0:
        print(f"Training loop {count} / {max_iters}")
        print('Loss:', loss_reinforce, "\n")
        print('Log-prob:', log_prob)


s_hat, sample = net.sample(100)
r = torch.randint(0, 99, size=[1])
sample = sample[r, :, :, :]
s_hat = s_hat[r, :, :, :]
print("Final Samples:", sample)
print("Final s_hat", s_hat)

plt.plot(range(max_iters), losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss at Training step i')
plt.show() 