import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

class SimpleAutoregressiveModel(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super(SimpleAutoregressiveModel, self).__init__()
        self.layers = []
        self.L = kwargs['L']
        self.width = kwargs['width']
        self.depth = kwargs['depth']
        self.device = kwargs['device']
        self.n = self.L**2

        self.layers.append(MaskedLinear(
            in_features=1,
            out_features=1 if self.depth == 1 else self.width,
            n=self.n
        ))
        for count in range(self.depth - 2):
            self.layers.append(self.build_block(self.width, self.width))
        if self.depth >= 2:
            self.layers.append(MaskedLinear(self.width, 1))
        self.layers.append(torch.nn.Sigmoid())

        self.network = torch.nn.Sequential(*self.layers)

    def build_block(self, in_features, out_features):
        """
        Builds a simple block of a ReLU activation and 
        a MaskedLinear layer.

        args: 
            in_features: int
            out_features: int

        returns:
            block: torch.nn.Sequential
        """
        layers = []
        layers.append(torch.nn.ReLU())
        layers.append(MaskedLinear(
            in_features=in_features,
            out_features=out_features,
            n = self.n,
        ))
        block = torch.nn.Sequential(*layers)
        return block

    def forward(self, x):
        return self.network(x)

    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, 1, self.L, self.L],
            device=self.device
        )
        for i in range(self.L):
            for j in range(self.L):
                s_hat = self.forward(sample)
                sample[:,:,i,j] = torch.bernoulli(s_hat[:, :, i, j])
        return sample, s_hat
    

class MaskedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, n: int, bias: bool = True, device=None, dtype=None) -> None:
        super(MaskedLinear, self).__init__(in_features * n, out_features * n, bias, device, dtype)

        self.in_features = in_features
        self.out_features = out_features
        self.n = n

        self.register_buffer('mask', torch.ones([self.n] * 2))
        self.mask = torch.tril(self.mask)
        self.mask = torch.cat([self.mask] * in_features, dim=1)
        self.mask = torch.cat([self.mask] * out_features, dim=0)

        self.weight.data *= self.mask

    def forward(self, x):
        return torch.nn.functional.linear(x, self.mask * self.weight, self.bias)



        

    