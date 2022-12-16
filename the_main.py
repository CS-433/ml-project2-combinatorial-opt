import torch
from conv_net_implementation import AutoRegressiveCNN
from neural_net_implementation import SimpleAutoregressiveModel
from linear_autoreg_nn import MADE
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
beta = 100
dim = 4
seed = 0
kernel_height = 5

# Create Graph 
G = random_connected_graph(dim, dim, seed)
G = random_hamiltonian_graph(G)
G = assign_random_weights(G, 40)

# Create Tensors/Matrices
J = create_J_tensor(G, A, B, C, D)
h = create_h_matrix(G, A, B, C)

# Create Neural Net
net = 'made'
kernel_size = (kernel_height, dim)

if net == 'linear':
    vars = {'L': G.order(), 'n': batch_size, 'net_width': G.order()**2, 'net_depth': 5, 'epsilon': 0.00001, 'device': 'cpu'}
    net = SimpleAutoregressiveModel(**vars)

elif net == 'conv':
    vars = {'L': G.order(), 'net_depth': 5, 'net_width': 16, 'kernel_size': 3, 'bias': 0.5, 'epsilon': 0.00001, 'device': 'cpu'}
    net = AutoRegressiveCNN(**vars)
    print(net)

elif net == 'made':
    vars = {'L': G.order(), 'net_depth': 5, 'net_width': 16, 'bias': 0.5, 'z2': False, 'res_block': True, 'x_hat_clip': False, 'device': 'cpu', 'epsilon': 0.00001}
    net = MADE(**vars)

optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.92, patience=100, threshold=1e-4, min_lr=1e-7)
losses = []
grads = []
for count in range(max_iters):
    optimizer.zero_grad()

    with torch.no_grad():
        s_hat, sample = net.sample(batch_size)
    assert not sample.requires_grad
    assert not s_hat.requires_grad

    log_prob = net.log_prob(sample)

    with torch.no_grad():
        # energy = tsp_hamiltonian(sample, J, h)

        # Simple Ising
        x1 = torch.roll(sample, 1, 2)
        x2 = torch.roll(sample, 1, 3)
        energy = -sample * (x1 + x2)
        energy = energy.sum(dim=(2, 3)).squeeze(dim=1)
        # print("sample", sample.shape, sample)
        # print("log_prob", log_prob.shape, log_prob)
        # print("energy", energy.shape, energy)

        loss = log_prob + beta*energy
    assert not energy.requires_grad
    assert not loss.requires_grad

    loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
    loss_reinforce.backward()
    if count % 100 == 0:
        pass
      #  grads.append(torch.norm(loss_reinforce.grad))
    losses.append(loss_reinforce.data)
    optimizer.step()

    if count % 50 == 0:
        print(f"Training loop {count} / {max_iters}")
        print('Loss:', loss_reinforce, "\n")
        # print('Log-prob:', log_prob)


s_hat, sample = net.sample(batch_size)

# exit()
r = torch.randint(0, batch_size, size=[1])
sample = sample[r, :, :, :]
s_hat = s_hat[r, :, :, :]
print("Final Samples:", sample)
print("Final s_hat", s_hat)
#print(grads)
plt.plot(range(max_iters), losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss at Training step i')
plt.show() 