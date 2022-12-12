import torch
from torch import nn
import numpy as np

class MaskedConv(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, exclusive: bool, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(MaskedConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.exclusive = exclusive
        self.mask = self.create_mask(self.exclusive)
        self.weight.data *= self.mask
    
    def create_mask(self, exclusive: bool) -> torch.Tensor:
        _, _, h, w = self.weight.shape 
        mask = torch.ones(size=[h, w])
        if exclusive:
            mask = torch.triu(mask)
        else:
            mask = torch.triu(mask, diagonal=1)

        return mask
    
    def forward(self, x):
        return nn.functional.conv2d(x, self.mask * self.weight, self.bias,
                                    self.stride, self.padding, self.dilation,
                                    self.groups)

    
class AutoRegressiveCNN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(AutoRegressiveCNN, self).__init__()
        self.L = kwargs['L']
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.half_kernel_size = kwargs['half_kernel_size']
        self.bias = kwargs['bias']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']

        layers = []
        layers.append(
            MaskedConv(
                1,
                1 if self.net_depth == 1 else self.net_width,
                exclusive=True,
                kernel_size=3
            )
        )
        for _ in range(self.net_depth-2):
            layers.append(self.build_block())
        layers.append(nn.ReLU())
        layers.append(MaskedConv(
            self.net_width,
            self.net_width,
            exclusive=True,
            kernel_size=3
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
                print("sample size:", sample.size())
                s_hat = self.forward(sample)
                print("s_hat size:", s_hat.size())
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

    def build_exclusive_block(self):
        layers = []
        layers.append(nn.ReLU())
        layers.append(
            MaskedConv(self.net_width, self.net_width, exclusive=True, kernel_size=3)
        )
        block = nn.Sequential(*layers)
        return block

    def build_block(self):
        layers = []
        layers.append(
            nn.ReLU()
        )
        layers.append(
            MaskedConv(self.net_width, self.net_width, exclusive=False, kernel_size=3)
        )
        block = nn.Sequential(*layers)
        return block


        
