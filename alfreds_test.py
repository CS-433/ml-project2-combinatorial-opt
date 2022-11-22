import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

num_cities = 5
#J is the weight matrix, distance between cities
v_mat = np.random.triangular(num_cities)
for i in range(num_cities): v_mat[i,i] = 0

def energy(s: np.ndarray((num_cities,1))) -> int:
    return s.T @ v_mat @ s

#s_gen is the 
def loss(s_gen, shots: int) -> int:
    return sum([energy(s_gen) for _ in range(shots)])


