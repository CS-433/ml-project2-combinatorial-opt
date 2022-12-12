import numpy as np
import jax.numpy as jnp
from jax import grad
import jax

dim = 5
global V
V = np.random.random_integers(1, 10, dim)
# for i in range(dim): 
#     for j in range(i,dim): 
#         V[i][j]=0

theta = jax.random.uniform(jax.random.PRNGKey(42),(dim, 1))*2*np.pi

def energy(s):
    return s.T @ V @ s

def loss_function(theta):
    rng_key = jax.random.PRNGKey(42)
    S = jax.random.bernoulli(rng_key, p=0.5, shape=(100, dim))

    def P(S: np.array):
        phis = np.zeros(S.shape)
        for idx in range(len(theta)):
            if S[idx] == 1:
                phis[idx] = jnp.cos(theta[idx])**2
            else:
                phis[idx] = 1 - jnp.cos(theta[idx])**2
        return jnp.prod(phis)

    loss = 0
    for s in S:
        prob = P(s)
        print(s.shape)
        E = energy(s)
        loss += prob*E
    
    return loss
    
    


    def p(s: np.ndarray):
        np.prod([np.cos(theta[i])**2 if s[i] == 1 else (1-np.cos(theta[i])**2) for i in range(len(s))])

    a = P(np.array([1]))
    return a
    

print(loss_function(theta))