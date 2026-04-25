# -*- coding: utf-8 -*-
import numpy as np
import gc
from numba import jit
from types import FunctionType
from scipy import integrate
from .algos import find_maximum
from .configs import gdefaults, USE_GPU, IS_PARALLEL
from .utils import backend as S
from .utils.fft_utils import ifft, fftshift, ifftshift
from .utils.backend import torch, Q, atleast_1d


class Wavelet():
    """Central wavelet class. `__call__` computes Fourier frequency-domain
    wavelet, `psih`, `.psifn` computes time-domain wavelet, `psi`.

    `Wavelet.SUPPORTED` for names of built-in wavelets passable to `__init__()`;
    `Wavelet.VISUALS`   for names of visualizations    passable to `viz()`.
    `viz()` to run visuals, `info()` to print relevant wavelet info.

    # Arguments:
        wavelet: str / tuple[str, dict] /FunctionType
            Name of supported wavelet (must be one of `Wavelet.SUPPORTED`)
            or custom function. Or tuple, name of wavelet and its configs,
            e.g. `('morlet', {'mu': 5})`.

        N: int
            Default length of wavelet.

        dtype: str / type (np.dtype) / None
            dtype at which wavelets are generated; can't change after __init__.
            Must be one of `Wavelet.DTYPES`. If None, uses value from
            `configs.ini`, global (if set) or wavelet-specific.

            'float32' is unsupported for GMW's `norm='energy'` and will be
            overridden by 'float64' (with a warning if it was passed to __init__).

    # Example:
        wavelet = Wavelet(('morlet', {'mu': 7}), N=1024)
        plt.plot(wavelet(scale=8))
    """
    SUPPORTED = {'gmw', 'morlet', 'bump', 'cmhat', 'hhhat'}
    VISUALS = {'time-frequency', 'heatmap', 'waveforms', 'filterbank',
               'harea', 'std_t', 'std_w', 'anim:time-frequency'}
    DTYPES = {'float32', 'float64'}
    # TODO ensure everything is accounted
    # Attributes whose data is stored on GPU (if env flag 'SSQ_GPU' == '1')
    ON_GPU = {'xi', '_Psih', '_Psih_scale'}
    # Time-frequency attributes
    TF_PROPS = {'wc', 'wc_ct', 'scalec_ct', 'std_t', 'std_w',
                'std_t_d', 'std_w_d'}

    def __init__(self, wavelet='gmw', N=1024, dtype=None):
        pass

    #### Main methods / properties ###########################################
    def __call__(self, w=None, *, scale=None, N=None, nohalf=True, imag_th=1e-8):
        """wavelet(w) if called with positional argument, w = float or array, else
           wavelet(scale * xi), where `xi` is recomputed if `N` is not None.

        `nohalf=False` (default=True) halves the Nyquist bin for even-length
        psih to ensure proper time-domain wavelet decay and analyticity:
            https://github.com/jonathanlilly/jLab/issues/13

        If evaluated wavelet's imaginary component is less than `imag_th`*(sum of
        real), will drop it; set to None to disable.
        """
        if w is not None:
            psih = self.fn(S.asarray(w, self.dtype))
        else:
            psih = self.fn(self.xifn(scale, N))

        if not nohalf:
            psih = self._halve_nyquist(psih)
        if (S.is_dtype(psih, ('complex64', 'complex128')) and
                (imag_th is not None) and
                (psih.imag.sum() / psih.real.sum() < imag_th)):
            psih = psih.real
        return psih

    @staticmethod
    def _halve_nyquist(psih):
        """https://github.com/jonathanlilly/jLab/issues/13"""
        pass

    def psifn(self, w=None, *, scale=None, N=None):
        """Compute time-domain wavelet; simply `ifft(psih)` with appropriate
        extra steps.
        """
        pass

    def xifn(self, scale=None, N=None):
        """Computes `xi`, radian frequencies at which `wavelet` is sampled,
        as fraction of sampling frequency: 0 to pi & -pi to 0, scaled by
        `scale` - or more precisely:

            N=128: [0, 1, 2, ..., 64, -63, -62, ..., -1] * (2*pi / N) * scale
            N=129: [0, 1, 2, ..., 64, -64, -63, ..., -1] * (2*pi / N) * scale
        """
        pass

    def Psih(self, scale=None, N=None, nohalf=True):
        """Return pre-computed `psih` at scale(s) `scale` of length `N` if
        same `scale` & `N` were passed previously, else compute anew.

        `dtype` will override `self.dtype` if not None.

        If both `scale` & `N` are None, will return previously computed `Psih`.
        """
        pass

    @property
    def N(self):
        """Default value used when `N` is not passed to a `Wavelet` method."""
        pass

    @N.setter
    def N(self, value):
        """Ensure `xi` always matches `N`."""
        pass

    @property
    def xi(self):
        """`xi` computed at `scale=1` and `N=self.N`. See `help(Wavelet.xifn)`."""
        pass

    @property
    def dtype(self):
        """dtype at which psih and psi are generated; can't change post-init."""
        pass

    #### Properties ##########################################################
    @property
    def name(self):
        """Name of underlying freq-domain function, processed by
        `wavelets._fn_to_name`.
        """
        pass

    @property
    def config_str(self):
        """`self.config` formatted into a nice string."""
        pass

    @property
    def wc(self):
        """Energy center frequency at scale=scalec_ct [(radians*cycles)/samples]

        Ideally we'd compute at `scale=1`, but that's trouble for 'energy' center
        frequency; see `help(wavelets.center_frequency)`. Away from scale
        extrema, 'energy' and 'peak' are same for bell-like |wavelet(w)|^2.

        Reported as "dimensional" in `info()` since it's tied to same `scale`
        used for computing `std_t_d` & `std_t_w`
        """
        pass

    @property
    def wc_ct(self):
        """'True' radian peak center frequency, i.e. `w` which maximizes the
        underlying continuous-time function. Can be used to find `scale`
        that centers the wavelet anywhere from 0 to pi in discrete space.

        Reported as "nondimensional" in `info()` since it's scale-decoupled.
        """
        pass

    @property
    def scalec_ct(self):
        """'Center scale' in sense of `wc_ct`, making wavelet peak at pi/4.
        See `help(Wavelet.wc_ct)`.
        """
        pass

    @property
    def std_t(self):
        """Non-dimensional time resolution"""
        pass

    @property
    def std_w(self):
        """Non-dimensional frequency resolution (radian)"""
        pass

    @property
    def std_f(self):
        """Non-dimensional frequency resolution (cyclic)"""
        pass

    @property
    def harea(self):
        """Heisenberg area: std_t * std_w >= 0.5"""
        pass

    @property
    def std_t_d(self):
        """Dimensional time resolution [samples/(cycles*radians)]"""
        pass

    @property
    def std_w_d(self):
        """Dimensional frequency resolution [(cycles*radians)/samples]"""
        pass

    @property
    def std_f_d(self):
        """Dimensional frequency resolution [cycles/samples]"""
        pass

    #### Misc ################################################################
    def info(self, nondim=True, reset=False):
        """Prints time & frequency resolution quantities. Refer to pertinent
        methods' docstrings on how each quantity is computed, and to
        tests/props_test.py on various dependences (e.g. `std_t` on `N`).
        If `reset`, will recompute all quantities (can be used with e.g. new `N`).

        See `help(Wavelet.x)`, x: `std_t, std_w, wc, wc_ct, scalec_ct`.

        Detailed overview: https://dsp.stackexchange.com/q/72042/50076
        """
        pass

    def reset_properties(self):
        """Reset time-frequency properties (`Wavelet.TF_PROPS`), i.e.
        recompute for current `self.N`.
        """
        pass

    def viz(self, name='overview', **kw):
        """`Wavelet.VISUALS` for list of supported `name`s."""
        pass

    def _viz(self, name, **kw):
        pass

    def _desc(self, N=None, scale=None, show_N=True):
        """Nicely-formatted parameter summary, used in other methods"""
        pass

    @classmethod
    def _process_dtype(self, dtype, as_str=None):
        """Ensures `dtype` is supported, and converts per `as_str` (if True,
        numpy/torch -> str, else vice versa; if None, returns as-is).
        """
        pass

    #### Init ################################################################
    @classmethod
    def _init_if_not_isinstance(self, wavelet, **kw):
        """Circumvents type change from IPython's super-/auto-reload,
        but first checks with usual isinstance."""
        pass

    def _validate_and_set_wavelet(self, wavelet):
        pass


@jit(nopython=True, cache=True)
def _xifn(scale, N, dtype=np.float64):
    """N=128: [0, 1, 2, ..., 64, -63, -62, ..., -1] * (2*pi / N) * scale
       N=129: [0, 1, 2, ..., 64, -64, -63, ..., -1] * (2*pi / N) * scale
    """
    pass

def _process_params_dtype(*params, dtype, auto_gpu=True):
    pass

#### Wavelet functions ######################################################
def morlet(mu=None, dtype=None):
    """Higher `mu` -> greater frequency, lesser time resolution.
    Recommended range: 4 to 16. For `mu > 6` the wavelet is almsot exactly
    Gaussian for most scales, providing maximum joint resolution.

    `mu=13.4` matches Generalized Morse Wavelets' `(beta, gamma) = (3, 60)`.
    For full correspondence see `help(_gmw.gmw)`.

    https://en.wikipedia.org/wiki/Morlet_wavelet#Definition
    https://www.desmos.com/calculator/0nslu0qivv
    """
    pass

@jit(nopython=True, cache=True)
def _morlet(w, mu, ks, C):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _morlet_par(w, mu, ks, C):
    pass

def _morlet_gpu(w, mu, ks, C):
    pass


def bump(mu=None, s=None, om=None, dtype=None):
    """Bump wavelet.
    https://www.mathworks.com/help/wavelet/gs/choose-a-wavelet.html
    """
    pass

@jit(nopython=True, cache=True)
def _bump(w, _w, s, C, C0):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _bump_par(w, _w, s, C, C0):
    pass

def _bump_gpu(w, _w, s, C, C0):
    pass


def cmhat(mu=None, s=None, dtype=None):
    """Complex Mexican Hat wavelet.
    https://en.wikipedia.org/wiki/Complex_mexican_hat_wavelet
    """
    pass

@jit(nopython=True, cache=True)
def _cmhat(_w, s, C):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _cmhat_par(_w, s, C):
    pass

def _cmhat_gpu(_w, s, C):
    pass


def hhhat(mu=None, dtype=None):
    """Hilbert analytic function of Hermitian Hat."""
    pass

@jit(nopython=True, cache=True)
def _hhhat(_w, C):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _hhhat_par(_w, C):
    pass

def _hhhat_gpu(_w, C):
    pass


#### Wavelet properties ######################################################
def center_frequency(wavelet, scale=None, N=1024, kind='energy', force_int=None,
                     viz=False):
    """Center frequency (radian) of `wavelet`, either 'energy', 'peak',
    or 'peak-ct'.

    Detailed overviews:
        (1) https://dsp.stackexchange.com/a/76371/50076
        (2) https://dsp.stackexchange.com/q/72042/50076

    **Note**: implementations of `center_frequency`, `time_resolution`, and
    `freq_resolution` are discretized approximations of underlying
    continuous-time parameters. This is a flawed approach (see (1)).
      - Caution is advised for scales near minimum and maximim (obtained via
        `cwt_scalebounds(..., preset='maximal')`), where inaccuracies may be
        significant.
      - For intermediate scales and sufficiently large N (>=1024), the methods
        are reliable. May improve in the future

    # Arguments
        wavelet: wavelets.Wavelet

        scale: float / None
            Scale at which to compute `wc`; ignored if `kind='peak-ct'`.

        N: int
            Length of wavelet.

        kind: str['energy', 'peak', 'peak-ct']
            - 'energy': weighted mean of wavelet energy, or energy expectation;
              Eq 4.52 of [1]:
                wc_1     = int w |wavelet(w)|^2 dw  0..inf
                wc_scale = int (scale*w) |wavelet(scale*w)|^2 dw 0..inf
                         = wc_1 / scale
            - 'peak': value of `w` at which `wavelet` at `scale` peaks
              (is maximum) in discrete time, i.e. constrained 0 to pi.
            - 'peak-ct': value of `w` at which `wavelet` peaks (without `scale`,
              i.e. `scale=1`), i.e. peak location of the continuous-time function.
              Can be used to find `scale` at which `wavelet` is most well-behaved,
              e.g. at eighth of sampling frequency (centered between 0 and fs/4).
            - 'energy' == 'peak' for wavelets exactly even-symmetric about mode
              (peak location)

        force_int: bool / None
            Relevant only if `kind='energy'`, then defaulting to True. Set to
            False to compute via formula - i.e. first integrate at a
            "well-behaved" scale, then rescale. For intermediate scales, this
            won't yield much difference. For extremes, it matches the
            continuous-time results closer - but this isn't recommended, as it
            overlooks limitations imposed by discretization (trimmed/undersampled
            freq-domain bell).

        viz: bool (default False)
            Whether to visualize obtained center frequency.

    **Misc**

    For very high scales, 'energy' w/ `force_int=True` will match 'peak'; for
    very low scales, 'energy' will always be less than 'peak'.

    To convert to Hz:
        wc [(cycles*radians)/samples] / (2pi [radians]) * fs [samples/second]
        = fc [cycles/second]

    See tests/props_test.py for further info.

    # References
        1. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
    """
    def _viz(wc, params):
        pass
    def _params(wavelet, scale, N):
        pass
    def _energy_wc(wavelet, scale, N, force_int):
        pass
    def _peak_wc(wavelet, scale, N):
        pass
    def _peak_ct_wc(wavelet, N):
        pass
    pass


def freq_resolution(wavelet, scale=10, N=1024, nondim=True, force_int=True,
                    viz=False):
    """Compute wavelet frequency width (std_w) for a given scale and N; larger N
    -> less discretization error, but same N as in application works best
    (larger will be "too accurate" and misrepresent true discretized values).

    `nondim` will divide by peak center frequency and return unitless quantity.

    Eq 22 in [1], Sec 4.3.2 in [2].
    Detailed overview: https://dsp.stackexchange.com/q/72042/50076
    See tests/props_test.py for further info.

    # References
        1. Higher-Order Properties of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede.
        https://sci-hub.st/10.1109/TSP.2008.2007607

        2. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
    """
    def _viz():
        pass
    pass


def time_resolution(wavelet, scale=10, N=1024, min_decay=1e3, max_mult=2,
                    min_mult=2, force_int=True, nondim=True, viz=False):
    """Compute wavelet time resolution for a given scale and N; larger N
    -> less discretization error, but same N as in application should suffice.

    Eq 21 in [1], Sec 4.3.2 in [2].
    Detailed overview: https://dsp.stackexchange.com/q/72042/50076

    `nondim` will multiply by peak center frequency and return unitless quantity.
    ______________________________________________________________________________

    **Interpretation**

    Measures time-span of 68% of wavelet's energy (1 stdev for Gauss-shaped
    |psi(t)|^2). Inversely-proportional with `N`, i.e. same `scale` spans half
    the fraction of sequence that's twice long. Is actually *half* the span
    per unilateral (radius) std.

        std_t ~ scale (T / N)
    ______________________________________________________________________________

    **Implementation details**

    `t` may be defined from `min_mult` up to `max_mult` times the original span
    for computing stdev since wavelet may not decay to zero within target frame.
    For any mult > 1, this is biased if we are convolving by sliding windows of
    length `N` in CWT, but we're not (see `cwt`); our scheme captures full wavelet
    characteristics, i.e. as if conv/full decayed length (but only up to mult=2).

    `min_decay` controls decay criterion of time-wavelet domain in integrating,
    i.e. ratio of max to endpoints of |psi(t)|^2 must exceed this. Will search
    up to `max_mult * N`-long `t`.

    For small `scale` (<~3) results are harder to interpret and defy expected
    behavior per discretization complications (call with `viz=True`). Workaround
    via computing at stable scale and calculating via formula shouldn't work as
    both-domain behaviors deviate from continuous, complete counterparts.
    ______________________________________________________________________________

    See tests/props_test.py for further info.

    # References
        1. Higher-Order Properties of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede.
        https://sci-hub.st/10.1109/TSP.2008.2007607

        2. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
    """
    def _viz():
        pass
    def _make_integration_t(wavelet, scale, N, min_decay, max_mult, min_mult):
        """Ensure `psi` decays sufficiently at integration bounds"""
        pass
    pass


#### Misc ####################################################################
def afftshift(xh):
    """Needed since analytic wavelets keep Nyquist bin at N//2 positive bin
    whereas FFT convention is to file it under negative (see `_xi`).
    Moves right N//2 + 1 bins to left.
    """
    pass

@jit(nopython=True, cache=True)
def _afftshift_even(xh, xhs):
    pass


def aifftshift(xh):
    """Inversion also different; moves left N//2+1 bins to right."""
    pass

@jit(nopython=True, cache=True)
def _aifftshift_even(xh, xhs):
    pass


def _fn_to_name(fn):
    """`_` to ` `, removes `<lambda>` & `.`, handles `SPECIALS`."""
    pass


def isinstance_by_name(obj, ref):
    """IPython reload can make isinstance(Obj(), Obj) fail; won't work if
    Obj has __str__ overridden."""
    def _class_name(obj):
        pass
    pass


##############################################################################
from ._gmw import gmw
from . import visuals
from .visuals import plot, _viz_cwt_scalebounds
from .utils.common import WARN, NOTE, pi, assert_is_one_of
from .utils.backend import asnumpy
