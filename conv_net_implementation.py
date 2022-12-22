import torch
from torch import nn
import numpy as np

class MaskedConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.exclusive = kwargs.pop('exclusive')
        super().__init__(*args, **kwargs)

        _, _, kh, kw = self.weight.shape
        self.register_buffer('mask', torch.ones([kh, kw]))
        self.mask[kh // 2, kw // 2 + (not self.exclusive):] = 0
        self.mask[kh // 2 + 1:] = 0
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())
    
    def forward(self, x):
        y = nn.functional.conv2d(x, self.mask * self.weight, self.bias,
                                    self.stride, self.padding, self.dilation,
                                    self.groups)

        return y

    def extra_repr(self):
        return (super(MaskedConv, self).extra_repr() +
                ', exclusive={exclusive}'.format(**self.__dict__))

    
class AutoRegressiveCNN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(AutoRegressiveCNN, self).__init__()
        self.L = kwargs['L']
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.kernel_size = kwargs['kernel_size']
        self.bias = kwargs['bias']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']

        self.padding = (self.kernel_size - 1) // 2

        layers = []
        layers.append(
            MaskedConv(
                1,
                1 if self.net_depth == 1 else self.net_width,
                exclusive=True,
                kernel_size=self.kernel_size,
                padding=self.padding
            )
        )
        for _ in range(self.net_depth-2):
            layers.append(self.build_block())
        layers.append(nn.ReLU())
        layers.append(MaskedConv(
            self.net_width,
            1,
            exclusive=False,
            kernel_size=self.kernel_size,
            padding=self.padding
        ))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

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
                s_hat = self.forward(sample)
                sample[:, :, i, j] = torch.bernoulli(
                    s_hat[:, :, i, j]).to(torch.float64) * 2 - 1
        return s_hat, sample
        
    def _log_prob(self, sample, s_hat):
        mask = (1 + sample) / 2
        log_prob = (torch.log(s_hat + self.epsilon) * mask +
                    torch.log(1 - s_hat + self.epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        s_hat = self.forward(sample)
        log_prob = self._log_prob(sample, s_hat)

        return log_prob

    def build_block(self):
        layers = []
        layers.append(
            nn.ReLU()
        )
        layers.append(
            MaskedConv(self.net_width, self.net_width, exclusive=False, kernel_size=self.kernel_size,
                padding=self.padding)
        )
        block = nn.Sequential(*layers)
        return block


        
