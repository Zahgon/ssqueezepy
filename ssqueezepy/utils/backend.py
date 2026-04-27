# -*- coding: utf-8 -*-
import numpy as np
# torch & cupy imported at bottom


def allclose(a, b, device='cuda'):
    """`numpy.allclose` or `torch.allclose`, latter if input(s) are Tensor."""
    pass


def astype(x, dtype, device='cuda'):
    pass


def array(x, dtype=None, device='cuda'):
    pass


def asarray(x, dtype=None, device='cuda'):
    pass


def zeros(shape, dtype=None, device='cuda'):
    pass


def ones(shape, dtype=None, device='cuda'):
    pass


def is_tensor(*args, mode='all'):
    pass


def is_array_or_tensor(*args, mode='all'):
    pass


def is_dtype(x, str_dtype):
    pass


def atleast_1d(x, dtype=None, device='cuda'):
    pass


def asnumpy(x):
    pass


def arange(a, b=None, dtype=None, device='cuda'):
    pass


def vstack(x):
    pass


#### misc + dummies ##########################################################
def warn_if_tensor_and_par(x, parallel):
    pass


def _torch_dtype(dtype):
    pass


class _TensorDummy():
    pass


class TorchDummy():
    """Dummy class with dummy attributes."""
    def __init__(self):
        self.Tensor = _TensorDummy
        self.dtype = _TensorDummy


class CupyDummy():
    """Dummy class with dummy attributes."""
    def memoize(self, *args, **kwargs):
        def wrap(fn):
            pass
        pass


class _Q():
    """Class for accessing `numpy` or `torch` attributes according to `USE_GPU()`.
    """
    def __getattr__(self, name):
        if USE_GPU():
            return getattr(torch, name)
        return getattr(np, name)


##############################################################################
Q = _Q()

try:
    import torch
    import cupy as cp
except:
    torch = TorchDummy()
    cp    = CupyDummy()

from ..configs import USE_GPU
