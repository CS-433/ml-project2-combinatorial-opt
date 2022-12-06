import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
import jax as jax

class Methods:
    
    def __init__(self, dim, seed):
        """
        params:
            dim: dimnsiom of the problem (dim v_mat = (dim, dim), dim thetas = (dim, 1)
            seed: random seed
        """
        self.dim = 5
        self.seed = np.random.seed(seed)
        self.v_mat = np.random.rand(dim, dim)*10
        for i in range(dim): 
            for j in range(i,dim):
                self.v_mat[i,j] = 0

    def gen_samples(self, thetas: np.ndarray, size: int) -> np.ndarray:
        """
        Generates sample states 
        params: 
            thetas: ndarray, dim thetas = (self.dim, 1)
            size: int, sample size
        
        returns:
            samples: ndarray dim samples = (size, dim)
        """
        samples = np.zeros((size, len(thetas)))
        for idx, theta in enumerate(thetas):
            samples[:, idx] =  np.random.binomial(1,np.cos(theta)**2,size)

        return samples

    def energy(self, s) -> float:
        """
        Returns the energy of the state, s

        params:
            s: sample state, dim s = (self.dim, 1)
        returns:
            the energy of the state: int
        """
        return s.T @ self.v_mat @ s

    def loss3(self, thetas):
        """
        Loss function

        params:
            thetas: ndarray, dim thetas = (dim, 1) the parameters to optimize
        returns:
            the value of the loss function
        """
        energies = np.zeros(100)
        tot_energy = 0
        samples = self.gen_samples2(thetas,100)
        for idx in range(len(energies)):
            tot_energy += self.energy(samples[idx, :])
        
        return tot_energy / samples.shape[0]

    def loss4(self, thetas, samples):
        """
        Loss function

        params:
            thetas: ndarray, dim thetas = (self.dim, 1) the parameters to optimize
            samples: ndarray, dim samples = (size, self.dim). samples generated from gen_samples
        returns:
            the value of the loss function
        """
        energies = np.zeros((samples.shape[0], 1))

        for idx in range(len(energies)):
            energies[idx] = self.energy(samples[idx, :])*P(samples[idx, :])
        
        return np.sum(energies)

    def probability(self, samples: np.ndarray, thetas: np.ndarray):
        """
        The probability of a given state

        params:
            samples: ndarray, dim samples = (size, self.dim)
            thetas: ndarray, dim thetas = (self.dim, 1)
        returns:
            probability of the state: int
        """
        return np.prod(np.cos(thetas[np.where(samples==1)])**2
        ) * np.prod(1 - np.cos(thetas[np.where(samples==-1)])**2)
        """ones = np.where(S == 1)
        m_ones = np.where(S == -1)
        phis = np.zeros(S.shape)
        
        phis[ones] = np.cos(theta)**2
        phis[m_ones] = 1 - np.cos(theta)**2
        return np.prod(phis)"""

    def grad_loss(self, thetas):
        """
        Returns the gradient of the loss function

        params:
            thetas: ndarray dim thetas = (self.dim, 1)
        returns: 
            the gradient of the loss function: ndarray dim gradient = (self.dim, 1)
        """
        samples = self.gen_samples(thetas,100)
        def g(s_j: int, theta_j: float) -> float: 
            -np.tan(theta_j) if s_j==1 else 1/np.tan(theta_j)
        test = g(samples[0][0], thetas[0]) * self.energy(samples[0])
        return np.array([np.sum( [2 * s * g(s[i],thetas[i]) * self.energy(s) for s in samples]) for i in range(dim)]) / samples.shape[0]
        """
        energies = np.zeros(100)
        tot_energy = 0
        grad_vector = np.zeros(len(thetas))
        
        for dimension in range(self.dim):
            for sample_nbr in range(len(energies)):
                if samples[sample_nbr,dimension] == 1:
                    tant = - np.tan(thetas[dimension])
                else:    
                    tant = 1 / np.tan(thetas[dimension])

                tot_energy += tant* self.energy(samples[sample_nbr, :])
            grad_vector[dimension] = tot_energy

            
            return (2 * grad_vector / samples.shape[0]).T
        """
    def gradient_descent(self, param, nbr_steps, learning_rate):
        for i in range(nbr_steps):
            print(self.grad_loss(param).shape)
            param -= learning_rate*(self.grad_loss(param)).T
            print(param.shape)
            if i % 10 == 0:
                print(f'Theta values: {param}')

        return param

#grad_loss = grad(loss3,0)
#grad_loss2 = jax.vjp.vgrad(loss3)(thetas)
#print(grad_loss(thetas+100))

dim = 5
seed = 12
thetas = 2*np.pi*np.random.randint(0,2,(dim,1))
leaernig_rate = 0.1
nbr_steps = 100
print(thetas)
sample_test = 2 * np.random.randint(0,2,(100,dim)) - 1


m = Methods(dim, seed)
m.gradient_descent(thetas, nbr_steps, 0.1)

