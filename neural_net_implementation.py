import torch
from torch import nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

class SimpleAutoregressiveModel(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(SimpleAutoregressiveModel, self).__init__()
        self.layers = []
        self.L = kwargs['L']
        self.width = kwargs['net_width']
        self.depth = kwargs['net_depth']
        self.device = kwargs['device']
        self.n = kwargs['n']
        self.epsilon = kwargs['epsilon']
        self.width *= self.n

        self.layers.append(MaskedLinear(
            in_features=self.width,
            out_features=1 if self.depth == 1 else self.width,
            device=self.device,
        ))
        for count in range(self.depth - 2):
            self.layers.append(self.build_block(self.width, self.width))
        if self.depth >= 2:
            self.layers.append(MaskedLinear(self.width, self.width))
        self.layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.layers)

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
        layers.append(nn.ReLU())
        layers.append(MaskedLinear(
            in_features=in_features,
            out_features=out_features,
            device=self.device,
        ))
        block = nn.Sequential(*layers)
        return block

    def forward(self, x):
        s_hat = self.net(x)
        return s_hat

    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, 1, self.L, self.L],
            device=self.device
        )
        for i in range(self.L):
            for j in range(self.L):
                sample = sample.flatten()
                s_hat = self.forward(sample)
                s_hat = torch.reshape(s_hat, shape=[batch_size, 1, self.L, self.L])
                sample = torch.zeros(size=[batch_size, 1, self.L, self.L])
                sample[:, :, i, j] = torch.bernoulli(s_hat[:, :, i, j]).to(torch.float64) * 2 - 1
        return s_hat, sample

    def _log_prob(self, sample, s_hat):
        mask = (1 + sample) / 2
        log_prob = (torch.log(s_hat + self.epsilon) * mask +
                    torch.log(1 - s_hat + self.epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        sample = sample.flatten()
        s_hat = self.forward(sample)
        s_hat = torch.reshape(s_hat, shape=[self.n, 1, self.L, self.L])
        sample = torch.reshape(sample, shape=[self.n, 1, self.L, self.L])
        log_prob = self._log_prob(sample, s_hat)

        return log_prob
    

class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super(MaskedLinear, self).__init__(in_features, out_features, bias, device, dtype)

        self.in_features = in_features
        self.out_features = in_features
        self.device = device
        self.mask = self.create_mask()
        self.weight.data *= self.mask

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

    def create_mask(self) -> torch.Tensor:
        """
        Creates a in_features*in_features mask (exclusive)
        """
        h, w = self.weight.data.size()
        mask = torch.ones(size=[self.in_features, self.in_features])
        mask = torch.tril(mask)
        return 1 - mask    