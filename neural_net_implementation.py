import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

class SimpleAutoregressiveModel(torch.nn.Module):
    
    def __init__(self, in_features) -> None:
        super().__init__()
        self.input_layer = torch.nn.Linear(
            in_features=in_features,
            out_features=in_features,
        )

    def forward(self, inputs, state=None):
        """
        
        """
        pass

    def construct_mask(features):
        mask = torch.zeros(size=(features, features))
        

    