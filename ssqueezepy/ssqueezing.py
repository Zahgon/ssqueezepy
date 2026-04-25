# -*- coding: utf-8 -*-
import numpy as np
from types import FunctionType
from .algos import indexed_sum_onfly, ssqueeze_fast
from .utils import p2up, process_scales, infer_scaletype, _process_fs_and_t
from .utils import NOTE, pi, logscale_transition_idx, assert_is_one_of
from .utils.backend import Q
from .utils.common import WARN
from .utils import backend as S
from .wavelets import center_frequency


def ssqueeze(Wx, w=None, ssq_freqs=None, scales=None, Sfs=None, fs=None, t=None,
             squeezing='sum', maprange='maximal', wavelet=None, gamma=None,
             was_padded=True, flipud=False, dWx=None, transform='cwt'):
    """Synchrosqueezes the CWT or STFT of `x`.

    # Arguments:
        Wx or Sx: np.ndarray
            CWT or STFT of `x`. CWT is assumed L1-normed, and STFT with
            `modulated=True`. If 3D, will treat elements along dim0 as independent
            inputs, synchrosqueezing one-by-one (but memory-efficiently).

        w: np.ndarray / None
            Phase transform of `Wx` or `Sx`. Must be >=0.
            If None, `gamma` & `dWx` must be supplied (and `Sfs` for SSQ_STFT).

        ssq_freqs: str['log', 'log-piecewise', 'linear'] / np.ndarray / None
            Frequencies to synchrosqueeze CWT scales onto. Scale-frequency
            mapping is only approximate and wavelet-dependent.
            If None, will infer from and set to same distribution as `scales`.
            See `help(cwt)` on `'log-piecewise'`.

        scales: str['log', 'log-piecewise', 'linear', ...] / np.ndarray
            See `help(cwt)`.

        Sfs: np.ndarray
            Needed if `transform='stft'` and `dWx=None`. See `help(ssq_stft)`.

        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which makes ssq
            frequencies range from 1/dT to 0.5*fs, i.e. as fraction of reference
            sampling rate up to Nyquist limit; dT = total duration (N/fs).
            Overridden by `t`, if provided.
            Relevant on `t` and `dT`: https://dsp.stackexchange.com/a/71580/50076

        t: np.ndarray / None
            Vector of times at which samples are taken (eg np.linspace(0, 1, n)).
            Must be uniformly-spaced.
            Defaults to `np.linspace(0, len(x)/fs, len(x), endpoint=False)`.
            Overrides `fs` if not None.

        squeezing: str['sum', 'lebesgue'] / function
            - 'sum': summing `Wx` according to `w`. Standard synchrosqueezing.
            Invertible.
            - 'lebesgue': as in [3], summing `Wx=ones()/len(Wx)`. Effectively,
            raw `Wx` phase is synchrosqueezed, independent of `Wx` values. Not
            recommended with CWT or STFT with `modulated=True`. Not invertible.
            For `modulated=False`, provides a more stable and accurate
            representation.
            - 'abs': summing `abs(Wx)` according to `w`. Not invertible
            (but theoretically possible to get close with least-squares estimate,
             so much "more invertible" than 'lebesgue'). Alt to 'lebesgue',
            providing same benefits while losing much less information.

            Custom function can be used to transform `Wx` arbitrarily for
            summation, e.g. `Wx**2` via `lambda x: x**2`. Output shape
            must match `Wx.shape`.

        maprange: str['maximal', 'peak', 'energy'] / tuple(float, float)
            See `help(ssq_cwt)`. Only `'maximal'` supported with STFT.

        wavelet: wavelets.Wavelet
            Only used if maprange != 'maximal' to compute center frequencies.
            See `help(cwt)`.

        gamma: float
            See `help(ssq_cwt)`.

        was_padded: bool (default `rpadded`)
            Whether `x` was padded to next power of 2 in `cwt`, in which case
            `maprange` is computed differently.
              - Used only with `transform=='cwt'`.
              - Ignored if `maprange` is tuple.

        flipud: bool (default False)
            Whether to fill `Tx` equivalently to `flipud(Tx)` (faster & less
            memory than calling `Tx = np.flipud(Tx)` afterwards).

        dWx: np.ndarray,
            Used internally by `ssq_cwt` / `ssq_stft`; must pass when `w` is None.

        transform: str['cwt', 'stft']
            Whether `Wx` is from CWT or STFT (`Sx`).

    # Returns:
        Tx: np.ndarray [nf x n]
            Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
            (nf = len(ssq_freqs); n = len(x))
            `nf = na` by default, where `na = len(scales)`.
        ssq_freqs: np.ndarray [nf]
            Frequencies associated with rows of `Tx`.

    # References:
        1. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        3. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        4. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_squeeze.m
    """
    pass
