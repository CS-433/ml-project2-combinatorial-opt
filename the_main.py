import torch
from conv_net_implementation import AutoRegressiveCNN
from neural_net_implementation import SimpleAutoregressiveModel
from linear_autoreg_nn import MADE
from hamiltonians import *
from graphs import *
from test import *
import matplotlib.pyplot as plt

#Hyperparams and kwargs
A = 0.1   # Constant to multiply edge weights with
B = 10    # Punishes multiple cities for one step 
C = 30    # Punishes City visited multiple times
D = 10

ham = 'tsp' # or 'simple_ising'

max_iters = 12000
batch_size = 32
beta_final = 10
beta_anneal = 0.998
dim = 2
seed = 0
kernel_height = 5

# Plots
plot_G = False
plot_loss = True

# Create Graph 
G = random_connected_graph(dim, dim, seed)
G = random_hamiltonian_graph(G)
G = assign_random_weights(G, 40)

if plot_G:
    plot_graph(G, seed)

# Create Tensors/Matrices
J = create_J_tensor(G, A, B, C, D)
h = create_h_matrix(G, A, B, C)

J_test = torch.rand(dim,dim)
h_test = torch.zeros(dim)

# Create Neural Net
net = 'made'
kernel_size = (kernel_height, dim)

if net == 'conv':
    vars = {'L': G.order(), 'net_depth': 2, 'net_width': 32, 'kernel_size': 3, 'bias': 0.5, 'epsilon': 0.00001, 'device': 'cpu'}
    net = AutoRegressiveCNN(**vars)
    print(net)

elif net == 'made':
    vars = {'L': G.order(), 'net_depth': 2, 'net_width': 4, 'bias': 0.5, 'z2': False, 'res_block': True, 'x_hat_clip': False, 'device': 'cpu', 'epsilon': 0.00001}
    net = MADE(**vars)

optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)

re_losses = []
losses = []

for count in range(max_iters):
    optimizer.zero_grad()

    with torch.no_grad():
        s_hat, sample = net.sample(batch_size)
    assert not sample.requires_grad
    assert not s_hat.requires_grad

    log_prob = net.log_prob(sample)

    with torch.no_grad():
        if ham == 'tsp':
            energy = tsp_hamiltonian(sample, J, h).squeeze()

        elif ham == 'simple_ising':
            # Simple Ising
            x1 = torch.roll(sample, 1, 2)
            x2 = torch.roll(sample, 1, 3)
            energy = sample * (x1 + x2)
            energy = energy.sum(dim=(2, 3)).squeeze(dim=1)

        else:
            print("Unknown Hamiltonian")

        beta = beta_final#*(1 - beta_anneal)**(count+1)
        loss = beta*energy + log_prob
    assert not energy.requires_grad
    assert not loss.requires_grad

    loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
    loss_reinforce.backward()

    re_losses.append(loss_reinforce.data)
    losses.append(loss.mean())
    optimizer.step()

    if count % 100 == 0:
        print(f"Training loop {count} / {max_iters}")
        print('Loss:', loss_reinforce, "\n")

    if loss_reinforce.data < -1000:
        print("Sample:", sample)
        break

s_hat, sample = net.sample(batch_size)

r = torch.randint(0, batch_size, size=[3])
sample = sample[r, :, :, :]
s_hat = s_hat[r, :, :, :]
print("Final Samples:", sample)
print("Final s_hat", s_hat)

if plot_loss:
    plt.plot(range(max_iters), re_losses)
    plt.plot(range(max_iters), losses)
    plt.legend(['loss_reinforce', 'loss'])
    plt.xlabel('Step')
    plt.ylabel('Loss')
    if ham == 'tsp':
        if net == 'conv':
            plt.title('Loss at Training step i for AutoRegressiveCNN on TSP')
        else:
            plt.title('Loss at Training step i for MADE on TSP')

    else:
        if net == 'conv':
            plt.title('Loss at Training step i for AutoRegressiveCNN on Simple Ising')
        else:
            plt.title('Loss at Training step i for MADE on Simple Ising')
    plt.show()