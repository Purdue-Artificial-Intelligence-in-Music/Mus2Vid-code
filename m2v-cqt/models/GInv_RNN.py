import torch
import torch.nn as nn
import numpy as np
import math
from models.GInv_Linear import *
from models.GInv_structures import subspace

class GInvariantRecurrentLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=None, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        R'''
        Init for a G-invariant RNN. Modeled after Torch documentation except with an invariant MLP for the new input layer.
        Assume batch_first = True.'''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.hidden_state = None

        self.inv_layer = GInvariantLayer(input_dim=input_dim, output_dim=hidden_dim, device=self.device)

        self.hidden_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=self.device, dtype=torch.double)

        if activation is None:
            self.activation = torch.nn.ReLU()
        else:
            self.activation = activation
        

    def forward(self, x, h_0=None):
        R'''
        Forward pass for the GInvariantRNN.
        
        Inputs: 
        x: Input tensor of size (N, L, input_dim)
        h_0: Initial hidden state of size (N, hidden_dim)
        
        Returns:
        out: Output tensor of size (N, L, hidden_dim)'''

        if x.device != self.device:
            x = x.to(self.device)
        if x.dtype != torch.double:
            x = x.double()
        if h_0 is None:
            h_0 = torch.zeros((x.shape[0], self.hidden_dim), dtype=x.dtype, device=self.device)
        elif h_0.device != self.device:
            h_0 = h_0.to(self.device)
        
        self.hidden_state = h_0

        x = torch.swapdims(x, 0, 1)

        out = torch.empty((x.shape[0], x.shape[1], self.hidden_dim), device=self.device)

        for i in range(x.shape[0]):
            hidden_state = self.inv_layer.forward(x[i]) + self.hidden_layer.forward(self.hidden_state)
            hidden_state = self.activation(hidden_state)
            out[i] = hidden_state
            self.hidden_state = hidden_state
        
        return torch.swapdims(out, 0, 1)

class GInvariantRNN_Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=None, num_layers=1, dropout=0.0, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        R'''
        Init for a G-invariant RNN. Modeled after Torch documentation except with an invariant MLP for the new input layer.'''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.first_layer = GInvariantRecurrentLayer(input_dim, hidden_dim, activation, device=device)
        self.dropout_func = None
        self.RNN_layers = []
        if num_layers > 1:
            self.dropout_func = torch.nn.Dropout(p=dropout)
            for i in range(num_layers - 1):
                self.RNN_layers.append(GInvariantRecurrentLayer(hidden_dim, hidden_dim, activation, device=device))
        

    def forward(self, x, h_0=None):
        R'''
        Forward pass for the GInvariantRNN.
        
        Inputs: 
        x: Input tensor of size (L, N, input_dim)
        h_0: Initial hidden state of size (N, hidden_dim)
        
        Returns:
        out: Output tensor of size (L, N, hidden_dim)'''

        if x.device != self.device:
            x = x.to(self.device)
        if x.dtype != torch.double:
            x = x.double()
        if h_0 is None:
            h_0 = torch.zeros((x.shape[0], self.hidden_dim), dtype=x.dtype, device=self.device)
        elif h_0.device != self.device:
            h_0 = h_0.to(self.device)
        
        curr_input = self.first_layer.forward(x, h_0)
        for layer in self.RNN_layers:
            curr_input = layer.forward(self.dropout_func(curr_input), h_0)
        
        return curr_input
        

