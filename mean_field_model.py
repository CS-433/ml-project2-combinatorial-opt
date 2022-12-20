import torch
from hamiltonians import *
from graphs import *
import matplotlib.pyplot as plt

class MeanFieldModel:
    def __init__(self, dim: int, hamiltonian: str, max_edge_weight: int, seed: int, A: float, B: float, C: float, D: float) -> None:
        self.dim = dim
        self.seed = seed
        self.hamiltonian = hamiltonian

        self.max_edge_weight = max_edge_weight
        self.A = A
        self.B = B
        self.B = B
        self.C = C
        self.D = D
        self.G = random_connected_graph(self.dim, self.dim, seed)
        self.G = random_hamiltonian_graph(self.G)
        self.G = assign_random_weights(self.G, self.max_edge_weight)

        if self.hamiltonian == 'tsp':
            self.J = create_J_tensor(self.G, self.A, self.B, self.C, self.D)
            self.h = create_h_matrix(self.G, self.A, self.B, self.C)

    def prob(self, thetas, sample):
        with torch.no_grad():
            distribution = torch.zeros(size=sample.size())
            distribution[(sample == 1).nonzero(as_tuple=True)] = torch.cos(thetas[(sample == 1).nonzero(as_tuple=True)]) ** 2
            distribution[(sample == -1).nonzero(as_tuple=True)] = 1 - torch.cos(thetas[(sample == -1).nonzero(as_tuple=True)]) ** 2
            prob = torch.prod(torch.prod(distribution, dim=3), dim=2).squeeze()

        assert prob.requires_grad == False
        assert distribution.requires_grad == False
        return prob, distribution

    def sample(self, distribution):
        return torch.bernoulli(distribution) * 2 - 1

    def compute_gradient(self, thetas, sample):
        prob, _ = self.prob(thetas=thetas, sample=sample)
        grad = torch.zeros(size=thetas.flatten().size())
        thetas = thetas.flatten()

        for i in range(sample.flatten().size(dim=0)):
            if sample.flatten()[i] == 1:
                grad[i] = torch.tan(thetas[i])
            else:
                grad[i] = 1 / torch.tan(thetas[i])

        grad = grad.reshape(sample.size())
        for i in range(prob.size(dim=0)):
            grad[i, :, :] = grad[i, :, :]* prob[i]

        return grad

    def train_model_with_backwards(self, max_iters, batch_size, lr):
        # Initial random sample
        initial_sample = torch.bernoulli(torch.rand(size = [batch_size, 1, self.dim, self.dim])) * 2 - 1

        # Initial theta
        thetas = torch.rand(size=[batch_size, 1, self.dim, self.dim], requires_grad=True)
        sample = initial_sample
        losses = []
        # Optimizer
        optimizer = torch.optim.SGD([thetas], lr)
        for iter in range(max_iters):
            optimizer.zero_grad()
            with torch.no_grad():
                prob, distribution = self.prob(thetas, sample)
            if self.hamiltonian == 'simple_ising':
                with torch.no_grad():
                    energy = simple_ising_energy(sample)
            elif self.hamiltonian == 'tsp':
                with torch.no_grad():
                    energy = tsp_hamiltonian(sample, self.J, self.h)

            loss = torch.mean(prob * energy)#.requires_grad_(True)
            losses.append(loss.data)
            loss.backward()
            if iter % 100 == 0:
                print(f"Training loop {iter} / {max_iters}")
                print("Loss:", loss, "\n")

            optimizer.step()
        
        return losses, prob, distribution

    def train_model(self, max_iters, batch_size, lr):
        # Initial random sample
        initial_sample = torch.bernoulli(torch.rand(size=[batch_size, 1, self.dim, self.dim])) * 2 - 1

        # Initial theta
        thetas = torch.ones(size=[batch_size, 1, self.dim, self.dim]) / self.dim
        sample = initial_sample
        losses = []

        for iter in range(max_iters):
            with torch.no_grad():
                prob, distribution = self.prob(thetas, sample)
            if self.hamiltonian == 'simple_ising':
                with torch.no_grad():
                    energy = simple_ising_energy(sample)
            elif self.hamiltonian == 'tsp':
                with torch.no_grad():
                    energy = tsp_hamiltonian(sample, self.J, self.h)
            loss = torch.mean(prob * energy)
            losses.append(loss)
            
            # Gradient Descent
            grad = self.compute_gradient(thetas, sample)
            thetas = thetas - lr*grad
            sample = self.sample(distribution)
            if iter % 100 == 0:
                print(f"Training loop {iter} / {max_iters}")
                print("Loss:", loss, "\n")

        return losses, thetas, sample


mfm = MeanFieldModel(
    dim=2,
    hamiltonian='simple_ising',
    max_edge_weight=4,
    seed=0,
    A=0.1,
    B=1,
    C=1,
    D=1
)
plot_loss = False
dim = 3
max_iters = 5000
batch_size = 100
lr = 1e-4
sample = torch.tensor([[[[1, 1], [1, -1]]], [[[-1, 1], [-1, -1]]]])
thetas = torch.ones(size=sample.size(), requires_grad=True) * 1 / mfm.dim

losses, thetas, sample = mfm.train_model(max_iters, batch_size, lr)

if plot_loss:
    plt.plot(range(max_iters), losses)
    plt.show()
print("Sample:", sample)