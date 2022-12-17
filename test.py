import torch

def energy_func(J: torch.tensor, h: torch.tensor, s: torch.tensor) -> float:
    rows, cols = J.shape
    energy = 0
    for row in range(rows):
        for col in range(cols):
            energy += J[row,col]*s[row]*s[col]
        energy += h[row]*s[row]
    return energy

def gen_states(dim):
    def gen_flat_states(dim2):
        if dim2 > 0:
            gen_flat_states


