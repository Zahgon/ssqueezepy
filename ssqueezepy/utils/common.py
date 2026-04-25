# -*- coding: utf-8 -*-
import numpy as np
import logging
from textwrap import wrap
from .fft_utils import fft, ifft


logging.basicConfig(format='')
WARN = lambda msg: logging.warning("WARNING: %s" % msg)
NOTE = lambda msg: logging.warning("NOTE: %s" % msg)  # else it's mostly ignored
pi = np.pi
EPS32 = np.finfo(np.float32).eps  # machine epsilon
EPS64 = np.finfo(np.float64).eps

__all__ = [
    "WARN",
    "NOTE",
    "pi",
    "EPS32",
    "EPS64",
    "p2up",
    "padsignal",
    "trigdiff",
    "mad",
    "est_riskshrink_thresh",
    "find_closest_parallel_is_faster",
    "assert_is_one_of",
    "_textwrap",
]


def p2up(n):
    """Calculates next power of 2, and left/right padding to center
    the original `n` locations.

    # Arguments:
        n: int
            Length of original (unpadded) signal.

    # Returns:
        n_up: int
            Next power of 2.
        n1: int
            Left  pad length.
        n2: int
            Right pad length.
    """
    pass


def padsignal(x, padtype='reflect', padlength=None, get_params=False):
    """Pads signal and returns trim indices to recover original.

    # Arguments:
        x: np.ndarray / torch.Tensor
            Input vector, 1D or 2D. 2D has time in dim1, e.g. `(n_inputs, time)`.

        padtype: str
            Pad scheme to apply on input. One of:
                ('reflect', 'symmetric', 'replicate', 'wrap', 'zero').
            'zero' is most naive, while 'reflect' (default) partly mitigates
            boundary effects. See [1] & [2].

            Torch doesn't support all padding schemes, but `cwt` will still
            pad it via NumPy.

        padlength: int / None
            Number of samples to pad input to (i.e. len(x_padded) == padlength).
            Even: left = right, Odd: left = right + 1.
            Defaults to next highest power of 2 w.r.t. `len(x)`.

    # Returns:
        xp: np.ndarray
            Padded signal.
        n_up: int
            Next power of 2, or `padlength` if provided.
        n1: int
            Left  pad length.
        n2: int
            Right pad length.

    # References:
        1. Signal extension modes. PyWavelets contributors
        https://pywavelets.readthedocs.io/en/latest/ref/
        signal-extension-modes.html

        2. Wavelet Bases and Lifting Wavelets. H. Xiong.
        http://min.sjtu.edu.cn/files/wavelet/
        6-lifting%20wavelet%20and%20filterbank.pdf
    """
    def _process_args(x, padtype):
        pass
    pass


def trigdiff(A, fs=1., padtype=None, rpadded=None, N=None, n1=None, window=None,
             transform='cwt'):
    """Trigonometric / frequency-domain differentiation; see `difftype` in
    `help(ssq_cwt)`. Used internally by `ssq_cwt` with `order > 0`.

    Un-transforms `A`, then transforms differentiated.

    # Arguments:
        A: np.ndarray
            2D array to differentiate (or 3D, batched).

        fs: float
            Sampling frequency, used to scale derivative to physical units.

        padtype: str / None
            Whether to pad `A` (along dim1) before differentiating.

        rpadded: bool (default None)
            Whether `A` is already padded. Defaults to True if `padtype` is None.
            Must pass `N` if True.

        N: int
            Length of unpadded signal (i.e. `A.shape[1]`).

        n1: int
            Will trim differentiated array as `A_diff[:, n1:n1+N]` (un-padding).

        transform: str['cwt', 'stft']
            Whether `A` stems from CWT or STFT, which changes how differentiation
            is done. `'stft'` currently not supported.

    """
    def _process_args(A, rpadded, padtype, N, transform, window):
        pass
    pass


def est_riskshrink_thresh(Wx, nv):
    """Estimate the RiskShrink hard thresholding level, based on [1].
    This has a denoising effect, but risks losing much of the signal; it's larger
    the more high-frequency content there is, even if not noise.

    # Arguments:
        Wx: np.ndarray
            CWT of a signal (see `cwt`).
        nv: int
            Number of voices used in CWT (see `cwt`).

    # Returns:
        gamma: float
            The RiskShrink hard thresholding estimate.

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        est_riskshrink_thresh.m
    """
    pass


def find_closest_parallel_is_faster(shape, dtype='float32', trials=7, verbose=1):
    """Returns True if `find_closest(, parallel=True)` is faster, as averaged
    over `trials` trials on dummy data.
    """
    pass


def mad(data, axis=None):
    """Mean absolute deviation"""
    pass


def assert_is_one_of(x, name, supported, e=ValueError):
    pass


def _textwrap(txt, wrap_len=50):
    """Preserves line breaks and includes `'\n'.join()` step."""
    pass
