# This code is based on: 
# https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_FINER_Flexible_Spectral-bias_Tuning_in_Implicit_NEural_Representation_by_Variable-periodic_CVPR_2024_paper.pdf

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

from ..base_model import BaseModel


class VariablePeriodicLayer(nn.Module):
    '''
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if init_weights:
            self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1/torch.sqrt(2), 1/torch.sqrt(2))      
            else:
                self.linear.weight.uniform_(-1/torch.sqrt(2), 1/torch.sqrt(2))
        
    def forward(self, input):
        y = self.linear(input)
        return torch.sin(self.omega_0 * (torch.abs(y) + 1) * y)
    
class Finer(BaseModel):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = VariablePeriodicLayer
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
            
            with torch.no_grad():
                const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        output = self.net(coords)
                    
        return output
  