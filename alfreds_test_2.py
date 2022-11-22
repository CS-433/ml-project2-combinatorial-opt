import numpy as np
#import scipy as sp
#from matplotlib import pyplot as plt
#import jax.numpy as jnp
#from jax import grad
from some_methods import *

num_cities = 5
num_paths = int(num_cities*(num_cities-1)/2)
#v is the weight matrix, distance between cities
v_mat = np.random.triangular(0,1,5,(num_cities,num_cities))
for i in range(num_cities): v_mat[i,i] = 0
global theta
theta = np.zeros(num_paths)

def gen_s() -> np.ndarray:
    s = -np.ones(num_cities)
    


