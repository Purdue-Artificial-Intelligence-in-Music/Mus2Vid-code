import argparse
import numpy as np
import numpy.typing as npt
import multiprocessing as mp
import torch
from typing import Callable, List
from functools import partial


def subspace(size: int):
    R"""
    Compute the Reynolds Operator’s Eigenvector
    and save the necessary Eigen vectors in a file.
    Which will be loaded in the GinvariantLayer.

    size: int, size of the input image, For MNIST it is 28
    """
    # TASK:
    # Following the class, compute and save the required eigenvectors of the Reynolds Operator
    # of rotation and refection group (flip around x and y axis)

    one_hots = torch.nn.functional.one_hot(torch.arange(size), size)
    one_hots = one_hots.reshape([-1] + [size])
    # one_hots_transforms = []
    T_rots = torch.empty(size, *(one_hots.shape))
    for d in range(size):
        T_rots[d] = torch.roll(one_hots, d, dims=0)
    # T_rots = torch.cat(one_hots_transforms).float()
    # print(T_rots)

    # Compute Reynolds operator
    T_bar = torch.mean(T_rots, 0).double()
    # print(T_bar)
    # Compute the left-eigenvectors with lambda=1
    L, V = torch.linalg.eigh(T_bar.T)
    # Select the lambda=1
    basis = V.T[L > 0.99]

    eigenvectors = basis

    with open("rf-{:d}.npy".format(size), "wb") as file:
        np.save(file, eigenvectors)
