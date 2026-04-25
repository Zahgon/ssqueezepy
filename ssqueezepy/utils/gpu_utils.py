# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple
from string import Template
from .backend import torch, cp


Stream = namedtuple('Stream', ['ptr'])

def _run_on_gpu(kernel, grid, block, *args, **kwargs):
    pass


@cp.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    pass


def _get_kernel_params(x, dim=1, threadsperblock=None):
    pass
