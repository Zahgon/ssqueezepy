# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft, fftshift
from numba import jit, prange
from scipy import integrate
from .gpu_utils import _run_on_gpu, _get_kernel_params
from ..configs import IS_PARALLEL
from .backend import torch
from . import backend as S

__all__ = [
    "buffer",
    "unbuffer",
    "window_norm",
    "window_resolution",
    "window_area",
]


def buffer(x, seg_len, n_overlap, modulated=False, parallel=None):
    """Build 2D array where each column is a successive slice of `x` of length
    `seg_len` and overlapping by `n_overlap` (or equivalently incrementing
    starting index of each slice by `hop_len = seg_len - n_overlap`).

    Mimics MATLAB's `buffer`, with less functionality.

    Supports batched input with samples along dim 0, i.e. `(n_inputs, input_len)`.
    See `help(stft)` on `modulated`.

    Ex:
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        xb = buffer(x, seg_len=5, n_overlap=3)
        xb == [[0, 1, 2, 3, 4],
               [2, 3, 4, 5, 6],
               [4, 5, 6, 7, 8]].T
    """
    pass


@jit(nopython=True, cache=True)
def _buffer(x, out, seg_len, n_segs, hop_len, s20, s21, modulated=False):
    pass


@jit(nopython=True, cache=True, parallel=True)
def _buffer_par(x, out, seg_len, n_segs, hop_len, s20, s21, modulated=False):
    pass


def _buffer_gpu(x, seg_len, n_segs, hop_len, s20, s21, modulated=False, out=None):
    pass


def unbuffer(xbuf, window, hop_len, n_fft, N, win_exp=1):
    """Undoes `buffer` (minus unpadding), per padding logic used in `stft`:
        (N, n_fft) : logic
         even, even: left = right + 1
             (N, n_fft, len(xp), pl, pr) -> (128, 120, 247, 60, 59)
          odd,  odd: left = right
             (N, n_fft, len(xp), pl, pr) -> (129, 121, 249, 60, 60)
         even,  odd: left = right
             (N, n_fft, len(xp), pl, pr) -> (128, 121, 248, 60, 60)
          odd, even: left = right + 1
             (N, n_fft, len(xp), pl, pr) -> (129, 120, 248, 60, 59)
    """
    pass


def window_norm(window, hop_len, n_fft, N, win_exp=1):
    """Computes window modulation array for use in `stft` and `istft`."""
    pass


@jit(nopython=True, cache=True)
def _overlap_add(x, xbuf, window, hop_len, n_fft):
    pass


@jit(nopython=True, cache=True)
def _window_norm(wn, window, hop_len, n_fft, win_exp=1):
    pass


def window_resolution(window):
    """Minimal function to compute a window's time & frequency widths, assuming
    Fourier spectrum centered about dc (else use `ssqueezepy.wavelets` methods).

    Returns std_w, std_t, harea. `window` must be np.ndarray and >=0.
    """
    pass


def window_area(window, time=True, frequency=False):
    """Minimal function to compute a window's time or frequency 'area' as area
    under curve of `abs(window)**2`. `window` must be np.ndarray.
    """
    pass
