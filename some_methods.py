import numpy as np
from consts import * 
# import scipy as sp
# from matplotlib import pyplot as plt
# import jax.numpy as jnp
# from jax import grad

def energy(s: np.ndarray(dim)) -> int:
    s.T @ v_mat @ s

#s_gen is the 
def loss(s_gen, shots: int) -> int:
    return sum([energy(s_gen) for _ in range(shots)])

