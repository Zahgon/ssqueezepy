# -*- coding: utf-8 -*-
import warnings
import numpy as np
from .wavelets import Wavelet, center_frequency
from .utils import backend as S, cwt_scalebounds, p2up
from .utils.common import EPS32, EPS64, trigdiff
from .ssqueezing import ssqueeze
from ._ssq_cwt import phase_cwt, phase_cwt_num
from ._ssq_stft import phase_stft, _make_Sfs


__all__ = ['freq_to_scale', 'scale_to_freq', 'phase_ssqueeze', 'phase_transform']


def freq_to_scale(freqs, wavelet, N, fs=1, n_search_scales=None, kind='peak',
                  base=2):
    """Convert frequencies to scales.

    # Arguments:
        freqs: np.ndarray
            1D array of frequencies. Must range between 0 and `N/fs/2` (Nyquist).

        wavelet: wavelets.Wavelet / str / tupe[str, dict]
            Wavelet.

        N: int
            `len(x)` of interest.

        fs: int
            Sampling rate in Hz.

        n_search_scales: int / None
            This method approximates the conversion of scales. Higher = better
            accuracy, but takes longer. Defaults to `10 * len(freqs)`.

        kind: str
            Mapping to use, one of: 'peak', 'energy', 'peak-ct'.
            See `help(ssqueezepy.center_frequency)`.

        base: int
            Base of exponent of `freqs`. Defaults to 2.
            `freqs` can be any distributed in any way, including mix of log
            and linear, so the base only helps improve the search if it matches
            that of `freqs`. If `freqs` is purely exponential, then
            `base = np.diff(np.log(freqs))[0] * 2.718281828`.

    # Returns:
        scales: np.ndarray
            1D arrays of scales.
    """
    pass


def scale_to_freq(scales, wavelet, N, fs=1, padtype='reflect'):
    """Convert scales to frequencies.

    # Arguments:
        freqs: np.ndarray
            1D array of frequencies. Must range between 0 and `N/fs/2` (Nyquist).

        wavelet: wavelets.Wavelet / str / tupe[str, dict]
            Wavelet.

        N: int
            `len(x)` of interest.

        fs: int
            Sampling rate in Hz.

        padtype: str / None
            `padtype` used in the transform. Used to determine the length
            of wavelets used in the transform: `None` uses half the length
            relative to `not None`.
            The exact value doesn't matter, only whether it's `None` or not.

    # Returns:
        freqs: np.ndarray
            1D arrays of frequencies.
    """
    pass


def phase_ssqueeze(Wx, dWx=None, ssq_freqs=None, scales=None, Sfs=None, fs=1.,
                   t=None, squeezing='sum', maprange=None, wavelet=None,
                   gamma=None, was_padded=True, flipud=False,
                   rpadded=False, padtype=None, N=None, n1=None,
                   difftype=None, difforder=None,
                   get_w=False, get_dWx=False, transform='cwt'):
    """Take `phase_transform` then `ssqueeze`. Can be used on an arbitrary
    CWT/STFT-like time-frequency transform `Wx`.
    Experimental; prefer `ssq_cwt` & `ssq_stft`.
    # Arguments:
        Wx, dWx (see w), ssq_freqs, scales, Sfs, fs, t, squeezing, maprange,
        wavelet, gamma, was_padded, flipud:
            See `help(ssqueezing.ssqueeze)`.
        rpadded: bool (default None) / None
            Whether `Wx` (and `dWx`) is passed in padded. `True` will unpad
            `Wx` and `dWx`  before SSQ. Also, if `dWx` is None:
                - `rpadded==False`: will pad `Wx` in computing `dWx` if
                `padtype!=None`, then unpad both before SSQ
                - `rpadded==True`: won't pad `Wx` regardless of `padtype`
        padtype: str / None
            Used if `rpadded==False`. See `help(utils.padsignal)`. Note that
            padding `Wx` isn't same as passing padded `Wx` from `cwt`, but it
            can get close.
        N, n1: int / None
            Needed if `rpadded==True` to unpad `Wx` & `dWx` as `Wx[:, n1:n1 + N]`.
        difftype, difforder: str
            Used if `dWx = None` and `transform == 'cwt'`; see `help(ssq_cwt)`.
        get_w, get_dWx: bool
            See `help(ssq_cwt)`.
    # Returns:
        Tx, Wx, ssq_freqs, scales, Sfs, w, dWx
    """
    pass


def phase_transform(Wx, dWx=None, difftype='trig', difforder=4, gamma=None,
                    fs=1., Sfs=None, rpadded=False, padtype='reflect', N=None,
                    n1=None, get_w=False, transform='cwt'):
    """Unified method for CWT & STFT SSQ phase transforms.
    See `help(_ssq_cwt.phase_cwt)` and `help(_ssq_stft.phase_stft)`.
    """
    pass
