import numpy as np
import torch

def sample_step(dist: torch.Tensor):
    assert(abs(torch.sum(dist)-1) <= 1e5) #Assure distribution is normalized
    dim = len(dist)
    num = torch.rand(1)
    sum = 0
    for i in range(dim):
        print(dim)
        if sum >= num: 
            return one_step_vec(i) 
        else: sum += dist[i]
    #Exception("Sum of dist less than rand~[0,1)")

    def one_step_vec(indx: int) -> torch.Tensor(dim, int):
        vec = torch.ones(dim, dtype=int)
        vec[indx] = -1
        return -vec