import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import jax.numpy as jnp
from jax import grad
import jax as jax

dim = 5
np.random.seed(12)
#v_mat is the weight matrix, distance between cities
v_mat = np.random.rand(dim, dim)*10
for i in range(dim): 
    for j in range(i,dim):
        v_mat[i,j] = 0


rng_key = jax.random.PRNGKey(42)

thetas = jax.random.uniform(rng_key,(dim, 1))*2*np.pi
print(thetas)
sample_test = 2 * np.random.randint(0,2,(100,dim)) - 1

def gen_samples(thetas: np.ndarray, size: int) -> np.ndarray:
    samples = np.zeros((size, len(thetas)))
    for idx, theta in enumerate(thetas):
       samples[:, idx] =  np.random.binomial(1,np.cos(theta)**2,size)
    return samples

def gen_samples2(thetas: np.ndarray,size: int):
   # print(rng_key)
    samples = np.zeros((size, len(thetas)))
    for idx, theta in enumerate(thetas):
       samples[:, idx] =  jax.random.bernoulli(rng_key,jnp.cos(theta)**2,(size,))
    return samples

def energy(s: np.ndarray((dim,1))) -> int:
    return s.T @ v_mat @ s

#s_gen is the 
def loss(s_gen, shots: int) -> int:
    return jax.sum([energy(s_gen) for _ in range(shots)])

def loss2(s_gen: np.ndarray)-> int:
    energies = np.zeros((s_gen.shape[0], 1))
    for idx in range(len(energies)):
        energies[idx] = energy(s_gen[idx, :])
    
    return np.sum(energies) / energies.shape[0]

def loss3(s, shots: int) -> int:
    s_batch = s_gen(shots)
    return jax.sum([s.T@v_mat@s for s in s_batch])

def s_gen(shots: int) -> np.ndarray((shots, dim)):
    

samples = gen_samples(thetas,10)
print(loss2(samples))

def loss3(thetas):
    energies = np.zeros(100)
    tot_energy = 0
    samples = gen_samples2(thetas,100)
    for idx in range(len(energies)):
        tot_energy += energy(samples[idx, :])
    
    return tot_energy / samples.shape[0]

def loss4(thetas,samples):
    energies = np.zeros((samples.shape[0], 1))
   # P = [np.cos(thetas[idx])**2 for idx in range(len(thetas)) if samples[idx] == 1 else 1 - np.cos(theta)**2]

    for idx in range(len(energies)):
        energies[idx] = energy(samples[idx, :])*P(samples[idx, :])
    
    return np.sum(energies)

def P(S: np.array, thetas):
    ones = np.where(S == 1)
    m_ones = np.where(S == -1)
    phis = np.zeros(S.shape)
    
    phis[ones] = np.cos(theta)**2
    phis[m_ones] = 1 - np.cos(theta)**2
    return np.prod(phis)


grad_loss = grad(loss3,0)
#grad_loss2 = jax.vjp.vgrad(loss3)(thetas)
print(grad_loss)
print(grad_loss(thetas+100))

