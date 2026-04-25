# -*- coding: utf-8 -*-
import numpy as np
from .utils import fft, ifft, ifftshift, FFT_GLOBAL
from .utils import WARN, adm_cwt, adm_ssq, _process_fs_and_t, is_array_or_tensor
from .utils import padsignal, process_scales, logscale_transition_idx
from .utils import backend as S
from .utils.backend import Q
from .algos import replace_at_inf_or_nan
from .wavelets import Wavelet


def cwt(x, wavelet='gmw', scales='log-piecewise', fs=None, t=None, nv=32,
        l1_norm=True, derivative=False, padtype='reflect', rpadded=False,
        vectorized=True, astensor=True, cache_wavelet=None, order=0, average=None,
        nan_checks=None, patience=0):
    """Continuous Wavelet Transform. Uses FFT convolution via frequency-domain
    wavelets matching (padded) input's length.

    Differentiability enabled via `os.environ['SSQ_GPU'] = '1'`.

    Uses `Wavelet.dtype` precision.

    # Arguments:
        x: np.ndarray / torch.Tensor
            Input vector(s), 1D / 2D.

            2D: does *not* do 2D CWT. Instead, treats dim0 as separate inputs,
            e.g. `(n_channels, time)`, improving speed & memory w.r.t. looping.

        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain.
                - str: name of builtin wavelet. See `ssqueezepy.wavs()`
                  or `Wavelet.SUPPORTED`.
                - tuple: name of builtin wavelet and its configs.
                  E.g. `('morlet', {'mu': 5})`.
                - `wavelets.Wavelet` instance. Can use for custom wavelet.
                  See `help(wavelets.Wavelet)`.

        scales: str['log', 'log-piecewise', 'linear', 'log:maximal', ...]
                / np.ndarray
            CWT scales.
                - 'log': exponentially distributed scales, as pow of 2:
                         `[2^(1/nv), 2^(2/nv), ...]`
                - 'log-piecewise': 'log' except very high `scales` are downsampled
                  to prevent redundancy. This is recommended. See
                  https://github.com/OverLordGoldDragon/ssqueezepy/issues/
                  29#issuecomment-776792726
                - 'linear': linearly distributed scales.
                  !!! this scheme is not recommended; use with caution

            str assumes default `preset` of `'minimal'` for low scales and
            `'maximal'` for high, which can be changed via e.g. 'log:maximal'.
            See `preset` in `help(utils.cwt_scalebounds)`.

        nv: int
            Number of voices (wavelets per octave). Suggested >= 16.

        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which for
            `maprange='maximal'` makes ssq frequencies range from 1/dT to 0.5*fs,
            i.e. as fraction of reference sampling rate up to Nyquist limit;
            dT = total duration (N/fs).
            Used to compute `dt`, which is only used if `derivative=True`.
            Overridden by `t`, if provided.
            Relevant on `t` and `dT`: https://dsp.stackexchange.com/a/71580/50076

        t: np.ndarray / None
            Vector of times at which samples are taken (eg np.linspace(0, 1, n)).
            Must be uniformly-spaced.
            Defaults to `np.linspace(0, len(x)/fs, len(x), endpoint=False)`.
            Used to compute `dt`, which is only used if `derivative=True`.
            Overrides `fs` if not None.

        l1_norm: bool (default True)
            Whether to L1-normalize the CWT, which yields a more representative
            distribution of energies and component amplitudes than L2 (see [3]).
            If False (default True), uses L2 norm.

        derivative: bool (default False)
            Whether to compute and return `dWx`. Requires `fs` or `t`.

        padtype: str / None
            Pad scheme to apply on input. See `help(utils.padsignal)`.
            `None` -> no padding.

        rpadded: bool (default False)
             Whether to return padded Wx and dWx.
             `False` drops the added padding per `padtype` to return Wx and dWx
             of .shape[1] == len(x).

        vectorized: bool (default True)
            Whether to compute quantities for all scales at once, which is
            faster but uses more memory.

        astensor: bool (default True)
            If `'SSQ_GPU' == '1'`, whether to return arrays as on-GPU tensors
            or move them back to CPU & convert to Numpy arrays.

        cache_wavelet: bool (default None) / None
            If True, will store `wavelet` computations for all `scales` in
            `wavelet._Psih` (only if `vectorized`).
                - Defaults to True if `wavelet` is passed that's a `Wavelet`,
                throws warning if True with non-`Wavelet` `wavelet` and sets self
                to False (since the array's discarded at `return` anyway).
                - Ignored with `order > 2`, defaults to False.

        order: int (default 0) / tuple[int] / range
            > 0 computes `cwt` with higher-order GMWs. If tuple, computes
            `cwt` at each specified order. See `help(_cwt.cwt_higher_order)`.

            NOTE: implementation may be not entirely correct. Specifically,
            alignment by center frequency rather than scales may be optimal.

        average: bool / None
            Only used for tuple `order`; see `help(_cwt.cwt_higher_order)`.

        nan_checks: bool / None
            Checks whether input has `nan` or `inf` values, and zeros them.
            `False` saves compute. Doesn't support torch inputs.

            Defaults to `True` for NumPy inputs, else `False`.

        patience: int / tuple[int, int]
            pyFFTW parameter for faster FFT on CPU; see `help(ssqueezepy.FFT)`.

    # Returns:
        Wx: [na x n] np.ndarray (na = number of scales; n = len(x))
            CWT of `x`. (rows=scales, cols=timeshifts)
        scales: [na] np.ndarray
            Scales at which CWT was computed.
        dWx: [na x n] np.ndarray  (if `derivative=True`)
            Time-derivative of the CWT of `x`, computed via frequency-domain
            differentiation (effectively, derivative of trigonometric
            interpolation; see [4]). Implements as described in Sec IIIB of [2].

    # Note:
        CWT is cross-correlation of wavelets with input. For zero-phase wavelets
        (real-valued in Fourier), this is equivalent to convolution. All
        ssqueezepy wavelets are zero-phase. If a custom general wavelet is
        used, it must be conjugated in frequency, and it should *not* be used
        with synchrosqueezing (see one-integral inverse References in `icwt`).

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications. G. Thakur,
        E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. How to validate a wavelet filterbank (CWT)? John Muradeli.
        https://dsp.stackexchange.com/a/86069/50076

        3. Wavelet "center frequency" explanation? Relation to CWT scales?
        John Muradeli.
        https://dsp.stackexchange.com/a/76371/50076

        4. The Exponential Accuracy of Fourier and Chebyshev Differencing Methods.
        E. Tadmor.
        http://webhome.auburn.edu/~jzl0097/teaching/math_8970/Tadmor_86.pdf

        5. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf

        6. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        cwt_fw.m
    """
    def _vectorized(xh, scales, wavelet, derivative, cache_wavelet):
        pass
    def _for_loop(xh, scales, wavelet, derivative, is_2D):
        pass
    def _process_args(x, scales, nv, fs, t, nan_checks, wavelet, cache_wavelet):
        pass
    pass


def icwt(Wx, wavelet='gmw', scales='log-piecewise', nv=None, one_int=True,
         x_len=None, x_mean=0, padtype='reflect', rpadded=False, l1_norm=True):
    """The inverse Continuous Wavelet Transform of `Wx`, via double or
    single integral.

    # Arguments:
        Wx: np.ndarray
            CWT computed via `ssqueezepy.cwt`.

              - 2D: (n_scales, n_times)
              - 3D: (n_inputs, n_scales, n_times).
                Doesn't support `one_int=False`.

        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain.
                - str: name of builtin wavelet. `ssqueezepy.wavs()`
                - tuple[str, dict]: name of builtin wavelet and its configs.
                  E.g. `('morlet', {'mu': 5})`.
                - `wavelets.Wavelet` instance. Can use for custom wavelet.

        scales: str['log', 'linear', 'log:maximal', ...] / np.ndarray
            See help(cwt).

        nv: int / None
            Number of voices. Suggested >= 32. Needed if `scales` isn't array
            (will default to `cwt`'s).

        one_int: bool (default True)
            Whether to use one-integral iCWT or double.
            Current one-integral implementation performs best.
                - True: Eq 2.6, modified, of [6]. Explained in [1].
                - False: Eq 4.67 of [3]. Explained in [2].

        x_len: int / None. Length of `x` used in forward CWT, if different
            from Wx.shape[1] (default if None).

        x_mean: float. mean of original `x` (not picked up in CWT since it's an
            infinite scale component). Default 0.
            Note: if `Wx` is 3D, `x_mean` should be 1D (`x.mean()` along samples
            axis).

        padtype: str
            Pad scheme to apply on input, in case of `one_int=False`.
            See `help(utils.padsignal)`.

        rpadded: bool (default False)
            True if Wx is padded (e.g. if used `cwt(, rpadded=True)`).

        l1_norm: bool (default True)
            True if Wx was obtained via `cwt(, l1_norm=True)`.

    # Returns:
        x: np.ndarray
            The signal(s), as reconstructed from Wx.

            If `Wx` is 3D, `x` has shape `(n_inputs, n_times)`.

    # References:
        1. One integral inverse CWT. John Muradeli.
        https://dsp.stackexchange.com/a/76239/50076

        2. Inverse CWT derivation. John Muradeli.
        https://dsp.stackexchange.com/a/71148/50076

        3. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf

        4. Why iCWT may be inexact. John Muradeli.
        https://dsp.stackexchange.com/a/87104/50076

        5. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications. G. Thakur,
        E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        6. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf

        7. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_cwt_iw.m
    """
    pass


def _icwt_2int(Wx, scales, scaletype, l1_norm, wavelet, x_len,
               padtype='zero', rpadded=False):
    """Double-integral iCWT; works with any(?) wavelet."""
    pass


def _icwt_1int(Wx, scales, scaletype, l1_norm):
    """One-integral iCWT; assumes analytic wavelet."""
    pass


def _icwt_norm(scaletype, l1_norm):
    pass


def _process_gmw_wavelet(wavelet, l1_norm):
    """Ensure `norm` for GMW is consistent with `l1_norm`."""
    pass


def cwt_higher_order(x, wavelet='gmw', order=1, average=None, astensor=True,
                     **kw):
    """Compute `cwt` with GMW wavelets of order 0 to `order`. See `help(cwt)`.

    Yields lower variance and more noise robust representation. See VI in ref[1].

    # Arguments:
        x: np.ndarray
            Input, 1D/2D. See `help(cwt)`.

        wavelet: str / wavelets.Wavelet
            CWT wavelet.

        order: int / tuple[int] / range
            Order of GMW to use for CWT. If tuple, will compute for each
            order specified in tuple, subject to `average`.

        average: bool (default True if `order` is tuple)
            If True, will take arithmetic mean of resulting `Wx` (and `dWx`
            if `derivative=True`), else return as list. Note for phase transform,
            one should compute derivative of averaged `Wx` rather than take
            average of individual `dWx`s.
            Ignored with non-tuple `order.

        kw: dict / kwargs
            Arguments to `cwt`.
            If `scales` is string, will reuse zeroth-order's; zeroth order
            isn't included in `order`, will set from wavelet at `order=0`.

    # References
        [1] Generalized Morse Wavelets. S. C. Olhede, A. T. Walden. 2002.
        https://spiral.imperial.ac.uk/bitstream/10044/1/1150/1/
        OlhedeWaldenGenMorse.pdf
    """
    def _process_wavelet(wavelet, order):
        pass
    def _process_args(wavelet, order, average, kw):
        pass
    pass
