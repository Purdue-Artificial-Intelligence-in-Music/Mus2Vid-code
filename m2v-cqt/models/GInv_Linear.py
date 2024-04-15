#
import torch
import numpy as np
from typing import List, cast
import torch.nn as nn
import math
from models.GInv_structures import subspace
import sys


class GInvariantLayer(nn.Module):
    R"""
    GInvariantLayer is Invariant Linear Layer
    """

    def __init__(self, input_dim, output_dim, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        R"""
        input_dim: int, size of the input, For MNIST it is 28*28
        output_dim: int, size of the output layer, For MNIST it is 10(equal to number of classes)
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        # subspace is where the eigenvectors
        # calculated and stored
        subspace(self.input_dim)
        with open("rf-{:d}.npy".format(self.input_dim), "rb") as file:
            eigenvectors = np.load(file)
            #print(eigenvectors.shape)
        self.register_buffer(
            "basis",
            torch.from_numpy(eigenvectors).to(torch.get_default_dtype()),
        )
        assert self.basis.shape[1] == self.input_dim

        ###
        # Task: initialize the learnable parameters
        # for the coefficients and bias
        ###
        self.alphas = nn.Parameter(torch.Tensor(output_dim,
                                            self.basis.shape[0], 1)).double()
        self.bias = nn.Parameter(torch.Tensor(output_dim)).double()

        stdv = 1.0 / math.sqrt(output_dim)
        self.alphas.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

        self.to(self.device)

    def to(self, device):
        self.basis = self.basis.to(device)
        self.alphas = self.alphas.to(device)
        self.bias = self.bias.to(device)
        return super().to(device)

    def forward(self, X):
        # Input shape: torch.Size([minibatch, input_dim])

        ####
        # Task: implement the forward pass
        # for equivarinat MLP
        # return the output of the layer 
        ####
        if X.dtype != torch.double:
            X = X.double()
        weights = torch.mul(self.alphas, self.basis)
        weights = weights.sum(dim=-2)
        return nn.functional.linear(X, weights, self.bias)

        pass


class GInvariantMLP(nn.Module):
    def __init__(self, input_size, layer_sizes, output_dim, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        R"""
        image_size: int, size of the input image, For MNIST it is 28
        layer_sizes: List[int], size of the hidden layers
        output_dim: int, size of the output layer, For MNIST it is 10(equal to number of classes)
        """
        super().__init__()

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim
        self.device = device

        self.first_layer = GInvariantLayer(input_size, layer_sizes[0], device=self.device)

        #### 
        # Task: initialize other linear layer 
        # with regular linear layer from torch.nn
        #####
        layers = []
        layers.append(torch.nn.ReLU())
        for i in range(0, len(layer_sizes)):
            if i == len(layer_sizes) - 1:
                layers.append(torch.nn.Linear(layer_sizes[i], output_dim, device=self.device).double())
            else:
                layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], device=self.device).double())
                layers.append(torch.nn.ReLU())
        layers = [self.first_layer] + layers
        self.mlp = torch.nn.Sequential(*layers)

    def get_hidden_rep(self, X):
        R"""
        takes an image X and return the hidden representation
        at each layer of the network
        """
        X = X.reshape(-1, self.input_size)
        hidden_rep = []
        for layer in self.mlp:
            X = layer(X)
            hidden_rep.append(X.clone())
        return hidden_rep

    def forward(self, X):
        X = X.reshape(-1, self.input_size)
        ###
        # Task: implement the forward pass
        ###
        if X.device != self.device:
            X = X.to(self.device)
        curr = X
        # out = ...
        return self.mlp.forward(X)


class MLP(nn.Module):
    R"""
    image_size: int, size of the input image, For MNIST it is 28
    layer_sizes: List[int], size of the hidden layers
    output_dim: int, size of the output layer, For MNIST it is 10(equal to number of classes)
    """

    def __init__(self, input_size, layer_sizes, output_dim, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim
        self.device = device

        
        ####
        # Task: initialize the linear layers
        ###
        layers = [torch.nn.Linear(input_size, layer_sizes[0])]
        for i in range(0, len(layer_sizes)):
            if i == len(layer_sizes) - 1:
                layers.append(torch.nn.Linear(layer_sizes[i], output_dim))
            else:
                layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                layers.append(torch.nn.ReLU())
        self.mlp = torch.nn.Sequential(*layers)

    def get_hidden_rep(self, X):
        X = X.reshape(-1, self.input_size)
        hidden_rep = []
        for layer in self.mlp:
            X = layer(X)
            hidden_rep.append(X.clone())
        return hidden_rep

    def forward(self, X):
        X = X.reshape(-1, self.input_size)
        if X.device != self.device:
            X = X.to(self.device)
        ####
        # Task: implement the forward pass
        ###
        # out = ...
        return self.mlp.forward(X)
