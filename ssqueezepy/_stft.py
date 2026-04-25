# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal as sig
from .utils import WARN, padsignal, buffer, unbuffer, window_norm
from .utils import _process_fs_and_t
from .utils.fft_utils import fft, ifft, rfft, irfft, fftshift, ifftshift
from .utils.backend import torch, is_tensor
from .algos import zero_denormals
from .wavelets import _xifn, _process_params_dtype
from .configs import gdefaults, USE_GPU


def stft(x, window=None, n_fft=None, win_len=None, hop_len=1, fs=None, t=None,
         padtype='reflect', modulated=True, derivative=False, dtype=None):
    """Short-Time Fourier Transform.

    `modulated=True` computes "modified" variant from [1] which is advantageous
    to reconstruction & synchrosqueezing (see "Modulation" below).

    # Arguments:
        x: np.ndarray
            Input vector(s), 1D or 2D. See `help(cwt)`.

        window: str / np.ndarray / None
            STFT windowing kernel. If string, will fetch per
            `scipy.signal.get_window(window, win_len, fftbins=True)`.
            Defaults to `scipy.signal.windows.dpss(win_len, win_len//8)`;
            the DPSS window provides the best time-frequency resolution.

            Always padded to `n_fft`, so for accurate filter characteristics
            (side lobe decay, etc), best to pass in pre-designed `window`
            with `win_len == n_fft`.

        n_fft: int >= 0 / None
            FFT length, or `(STFT column length) // 2 + 1`.
            If `win_len < n_fft`, will pad `window`. Every STFT column is
            `fft(window * x_slice)`.
            Defaults to `len(x)//hop_len`, up to 512.

        win_len: int >= 0 / None
            Length of `window` to use. Used to generate a window if `window`
            is string, and ignored if it's np.ndarray.
            Defaults to `n_fft//8` or `len(window)` (if `window` is np.ndarray).

        hop_len: int > 0
            STFT stride, or number of samples to skip/hop over between subsequent
            windowings. Relates to 'overlap' as `overlap = n_fft - hop_len`.
            Must be 1 for invertible synchrosqueezed STFT.

        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which makes ssq frequencies
            range from 0 to 0.5*fs, i.e. as fraction of reference sampling rate
            up to Nyquist limit. Used to compute `dSx` and `ssq_freqs`.

        t: np.ndarray / None
            Vector of times at which samples are taken (eg np.linspace(0, 1, n)).
            Must be uniformly-spaced.
            Defaults to `np.linspace(0, len(x)/fs, len(x), endpoint=False)`.
            Overrides `fs` if not None.

        padtype: str
            Pad scheme to apply on input. See `help(utils.padsignal)`.

        modulated: bool (default True)
            Whether to use "modified" variant as in [1], which centers DFT
            cisoids at the window for each shift `u`. `False` will not invert
            once synchrosqueezed.
            Recommended `True`. See "Modulation" and [2] below.

        derivative: bool (default False)
            Whether to compute and return `dSx`. Uses `fs`.

        dtype: str['float32', 'float64'] / None
            Compute precision; use 'float32` for speed & memory at expense of
            accuracy (negligible for most purposes).
            If None, uses value from `configs.ini`.

            To be safe with `'float32'`, time-localized `window`, and large
            `hop_len`, use

                from ssqueezepy._stft import _check_NOLA
                _check_NOLA(window, hop_len, 'float32', imprecision_strict=True)

    **Modulation**
        `True` will center DFT cisoids at the window for each shift `u`:
            Sm[u, k] = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*(n - u)/N)
        as opposed to usual STFT:
            S[u, k]  = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*n/N)

        Most implementations (including `scipy`, `librosa`) compute *neither*,
        but rather center the window for each slice, thus shifting DFT bases
        relative to n=0 (t=0). These create spectra that, viewed as signals, are
        of high frequency, making inversion and synchrosqueezing very unstable.
        Details & visuals: https://dsp.stackexchange.com/a/72590/50076

        Better explanation in ref [2].

    # Returns:
        Sx: [(n_fft//2 + 1) x n_hops] np.ndarray
            STFT of `x`. Positive frequencies only (+dc), via `rfft`.
            (n_hops = (len(x) - 1)//hop_len + 1)
            (rows=scales, cols=timeshifts)

        dWx: [(n_fft//2 + 1) x n_hops] np.ndarray
            Returned only if `derivative=True`.
            Time-derivative of the STFT of `x`, computed via STFT done with
            time-differentiated `window`, as in [1]. This differs from CWT's,
            where its (and Sx's) DFTs are taken along columns rather than rows.
            d/dt(window) obtained via freq-domain differentiation (help(cwt)).

    # References:
        1. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        2. Equivalence between "windowed Fourier transform" and STFT as
        convolutions/filtering. John Muradeli.
        https://dsp.stackexchange.com/a/86938/50076

        3. STFT: why overlapping the window? John Muradeli.
        https://dsp.stackexchange.com/a/88124/50076

        4. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        stft_fw.m
    """
    def _stft(xp, window, diff_window, n_fft, hop_len, fs, modulated, derivative):
        pass
    pass


def istft(Sx, window=None, n_fft=None, win_len=None, hop_len=1, N=None,
          modulated=True, win_exp=1):
    """Inverse Short-Time Fourier transform. Computed with least-squares
    estimate for `win_exp`=1 per Griffin-Lim [1], recommended for STFT with
    modifications, else simple inversion with `win_exp`=0:

        x[n] = sum(y_t[n] * w^a[n - tH]) / sum(w^{a+1}[n - tH]),
        y_t = ifft(Sx), H = hop_len, a = win_exp, t = hop index, n = sample index

    Warns if `window` NOLA constraint isn't met (see [2]), invalidating inversion.
    Nice visuals and explanations on istft:
        https://www.mathworks.com/help/signal/ref/istft.html

    # Arguments:
        Sx: np.ndarray
            STFT of 1D `x`.

        window, n_fft, win_len, hop_len, modulated
            Should be same as used in forward STFT. See `help(stft)`.

        N: int > 0 / None
            `len(x)` of original `x`, used in inversion padding and windowing.
            If None, assumes longest possible `x` given `hop_len`, `Sx.shape[1]`.

        win_exp: int >= 0
            Window power used in inversion (see [1], [2], or equation above).

    # Returns:
        x: np.ndarray, 1D
            Signal as reconstructed from `Sx`.

    # References:
        1. Signal Estimation from Modified Short-Time Fourier Transform.
        D. W. Griffin, J. S. Lim.
        https://citeseerx.ist.psu.edu/viewdoc/
        download?doi=10.1.1.306.7858&rep=rep1&type=pdf

        2. Invertibility of overlap-add processing. B. Sharpe.
        https://gauss256.github.io/blog/cola.html

        3. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        stft_iw.m
    """
    pass


def get_window(window, win_len, n_fft=None, derivative=False, dtype=None):
    """See `window` in `help(stft)`. Will return window of length `n_fft`,
    regardless of `win_len` (will pad if needed).
    """
    pass


def _check_NOLA(window, hop_len, dtype=None, imprecision_strict=False):
    """https://gauss256.github.io/blog/cola.html"""
    pass
