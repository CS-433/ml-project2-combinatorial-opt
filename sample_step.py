import numpy as np

def sample_step(dist: np.array) -> np.array(dtype=int):
    assert(abs(sum(dist)-1) <= 1e5) #Assure distribution is normalized
    dim = len(dist)
    num = np.random.rand()
    sum = 0
    for i in range(dim):
        if sum >= num: 
            return one_step_vec(i) 
        else: sum += dist[i]
    #Exception("Sum of dist less than rand~[0,1)")

    def one_step_vec(indx: int) -> np.array(dim, int):
        vec = np.ones(dim, int)
        vec[indx] = -1
        return -vec