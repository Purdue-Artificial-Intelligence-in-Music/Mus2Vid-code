import torch
import torch.nn as nn
import numpy as np
import math
from models.GInv_Linear import *
from models.GInv_structures import subspace

class GInvariantLSTMLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.i_inv_layer = GInvariantLayer(input_dim=input_dim, output_dim=hidden_dim, device=self.device)
        self.f_inv_layer = GInvariantLayer(input_dim=input_dim, output_dim=hidden_dim, device=self.device)
        self.g_inv_layer = GInvariantLayer(input_dim=input_dim, output_dim=hidden_dim, device=self.device)
        self.o_inv_layer = GInvariantLayer(input_dim=input_dim, output_dim=hidden_dim, device=self.device)

        self.i_lin_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=self.device, dtype=torch.double)
        self.f_lin_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=self.device, dtype=torch.double)
        self.g_lin_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=self.device, dtype=torch.double)
        self.o_lin_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=self.device, dtype=torch.double)

        self.i_act_func = torch.nn.Sigmoid()
        self.f_act_func = torch.nn.Sigmoid()
        self.g_act_func = torch.nn.Tanh()
        self.o_act_func = torch.nn.Sigmoid()
        self.c_act_func = torch.nn.Tanh()

        self.cell = torch.zeros(hidden_dim, dtype=torch.double, device=self.device)
        self.hidden_state = torch.zeros(hidden_dim, dtype=torch.double, device=self.device)
        
    def to(self, device):
        self.i_inv_layer = self.i_inv_layer.to(device)
        self.f_inv_layer = self.f_inv_layer.to(device)
        self.g_inv_layer = self.g_inv_layer.to(device)
        self.o_inv_layer = self.o_inv_layer.to(device)
        self.i_lin_layer = self.i_lin_layer.to(device)
        self.f_lin_layer = self.f_lin_layer.to(device)
        self.g_lin_layer = self.g_lin_layer.to(device)
        self.o_lin_layer = self.o_lin_layer.to(device)
        
    def forward(self, X, H=None):
        '''
        Forward pass for the GInvariantLSTM.
        
        Inputs: 
        X: Input tensor of size (N, L, input_dim)
        H: Initial hidden state of size (N, hidden_dim)
        
        Returns:
        out: Output tensor of size (N, L, hidden_dim)'''
        if X.device != self.device:
            X = X.to(self.device)
        if X.dtype != torch.double:
            X = X.double()
        if H == None:
            H = torch.zeros(X.shape[0], self.hidden_dim, dtype=torch.double, device=self.device)
        elif H.dtype != torch.double:
            H = H.double()
        if H.device != self.device:
            H = H.to(self.device)

        self.hidden_state = H

        X = torch.swapdims(X, 0, 1)

        for iter in range(X.shape[0]):
            i = self.i_act_func(self.i_inv_layer.forward(X[iter]) + self.i_lin_layer.forward(self.hidden_state))
            f = self.f_act_func(self.f_inv_layer.forward(X[iter]) + self.f_lin_layer.forward(self.hidden_state))
            g = self.g_act_func(self.g_inv_layer.forward(X[iter]) + self.g_lin_layer.forward(self.hidden_state))
            o = self.o_act_func(self.o_inv_layer.forward(X[iter]) + self.o_lin_layer.forward(self.hidden_state))
            self.cell = torch.mul(f, self.cell) + torch.mul(i, g)
            self.hidden_state = torch.mul(o, self.c_act_func(self.cell))

        return torch.swapdims(self.hidden_state, 0, 1)

class GInvariantLSTM_Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=None, num_layers=1, dropout=0.0, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        R'''
        Init for a G-invariant LSTM. Modeled after Torch documentation except with an invariant MLP for the new input layer.
        Assume batch_first = True.'''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.first_layer = GInvariantLSTMLayer(input_dim, hidden_dim, device=self.device)
        self.dropout_func = None
        self.RNN_layers = []
        if num_layers > 1:
            self.dropout_func = torch.nn.Dropout(p=dropout)
            for i in range(num_layers - 1):
                self.RNN_layers.append(GInvariantLSTMLayer(hidden_dim, hidden_dim, device=device))
        

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
        if h_0 is None:
            h_0 = torch.zeros((x.shape[0], self.hidden_dim), dtype=x.dtype, device=self.device)
        elif h_0.device != self.device:
            h_0 = h_0.to(self.device)
        
        curr_input = self.first_layer.forward(x, h_0)
        for layer in self.RNN_layers:
            curr_input = layer.forward(self.dropout_func(self.curr_input), h_0)
        
        return curr_input


