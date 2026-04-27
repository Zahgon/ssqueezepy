# -*- coding: utf-8 -*-
"""Generalized Morse Wavelets.

For complete functionality, utility functions have been ported from jLab, and
largely validated to match jLab's behavior. jLab tests not ported.
"""
import numpy as np
from numpy.fft import ifft
from numba import jit
from scipy.special import (gamma   as gamma_fn,
                           gammaln as gammaln_fn)
from .algos import nCk
from .wavelets import _xifn, _process_params_dtype
from .configs import gdefaults, USE_GPU, IS_PARALLEL
from .utils.backend import torch
from .utils import backend as S

pi = np.pi


#### Base wavelets (`K=1`) ###################################################
def gmw(gamma=None, beta=None, norm=None, order=None, centered_scale=None,
        dtype=None):
    """Generalized Morse Wavelets. Returns function which computes GMW in the
    frequency domain.

    Assumes `beta != 0`; for full functionality use `_gmw.morsewave`.
    Unlike `morsewave`, works with scales rather than frequencies.

    Note that function for `norm='energy'` does *not* rescale freq-domain wavelet
    per `sqrt(scale)`, for consistency with `ssqueezepy.wavelets`.
    See `_gmw.compute_gmw` for code computing freq- and time-domain wavelets
    as arrays with proper scaling in.

    An overview: https://overlordgolddragon.github.io/generalized-morse-wavelets/
    Interactive: https://www.desmos.com/calculator/4gcaeqidxd (bandpass)
                 https://www.desmos.com/calculator/zfxnblqh8p (energy)

    # Arguments
        gamma, beta: float > 0, float > 0
            GMW parameters. See `help(_gmw.morsewave)`.

        norm: str['energy', 'bandpass']
            Normalization to use:
                'energy': L2 norm, keeps time-domain wavelet's energy at unity
                for all `freqs`, i.e. `sum(abs(psi)**2) == 1`.
                'bandpass': L1 norm, keeps freq-domain wavelet's peak value at 2
                for all `freqs`, i.e. `max(psih) == 2`, `w[argmax(psih)] == wc`.

            Additionally see `help(_gmw.morsewave)`.

        order: int (default 1)
            Order of the wavelet. `k+1`-th order wavelet is orthogonal to `k`-th.
            `k=0` will call a different but equivalent function for simpler code
            and compute efficiency.

        centered_scale: bool (default False)
            Unlike other `ssqueezepy.wavelets`, by default `scale=1` in
            `morsewave` (i.e. `freqs=1`) computes the wavelet at (peak) center
            frequency. This ensures exact equality between `scale` and
            `1 / center_frequency`, by multiplying input radians `w` by peak
            center freq.

            False by default for consistency with other `ssqueezepy` wavelets.

        dtype: str / type (np.dtype) / None
            See `help(wavelets.Wavelet)`.

    # Returns
        psihfn: function
            Function that computes GMWs, taking `w` (radian frequency)
            as argument.

    # Usage
        wavelet = gmw(3, 60)
        wavelet = Wavelet('gmw')
        wavelet = Wavelet(('gmw', {'beta': 60}))
        Wx, *_  = cwt(x, 'gmw')

    # Correspondence with Morlet
        Following pairs yield ~same frequency resolution, which is ~same
        time-frequency resolution for `mu > 5`, assuming `gamma=3` for all:
            `mu`, `beta`
           (1.70, 1.00),
           (3.00, 3.00),
           (4.00, 5.15),
           (6.00, 11.5),
           (8.00, 21.5),
           (10.0, 33.5),
           (12.0, 48.5),
           (13.4, 60.0),
        The default `beta=12` is hence to closely match Morlet's default `mu=6.`.

    # vs Morlet
        Differences grow significant when seeking excellent time localization
        (low `mu`, <4), where Morlet's approximate analyticity breaks down and
        negative frequencies are leaked, whereas GMW remains exactly analytic,
        with vanishing moments toward dc bin. Else, the two don't behave
        noticeably different for `gamma=3`.

    # References
        [1] Generalized Morse Wavelets. S. C. Olhede, A. T. Walden. 2002.
        https://spiral.imperial.ac.uk/bitstream/10044/1/1150/1/
        OlhedeWaldenGenMorse.pdf

        [2] Generalized Morse Wavelets as a Superfamily of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede. 2012.
        https://sci-hub.st/10.1109/TSP.2012.2210890

        [3] Higher-Order Properties of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede. 2009.
        https://sci-hub.st/10.1109/TSP.2008.2007607

        [4] (c) Lilly, J. M. (2021), jLab: A data analysis package for Matlab,
        v1.6.9, http://www.jmlilly.net/jmlsoft.html
        https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsewave.m
    """
    pass


def compute_gmw(N, scale, gamma=3, beta=60, time=False, norm='bandpass',
                order=0, centered_scale=False, norm_scale=True, dtype=None):
    """Evaluates GMWs, returning as arrays. See `help(_gmw.gmw)` for full docs.

    # Arguments
        N: int > 0
            Number of samples to compute.

        scale: float > 0
            Scale at which to sample the freq-domain wavelet: `psih(s * w)`.

        gamma, beta, norm, order:
            See `help(_gmw.gmw)`.

        time: bool (default False)
            Whether to compute the time-domain wavelet, `psi`.

        centered_scale: bool (default False)
            See `help(_gmw.gmw)`.

        norm_scale: bool (default True)
            Whether to rescale as `sqrt(s) * psih(s * w)` for the `norm='energy'`
            case (no effect with `norm='bandpass'`).

    # Returns
        psih: np.ndarray [N]
            Frequency-domain wavelet.
        psi: np.ndarray [N]
            Time-domain wavelet, returned if `time=True`.
    """
    pass


def gmw_l1(gamma=3., beta=60., centered_scale=False, dtype='float64'):
    """Generalized Morse Wavelets, first order, L1(bandpass)-normalized.
    See `help(_gmw.gmw)`.
    """
    pass

@jit(nopython=True, cache=True)
def _gmw_l1(w, gamma, beta, wc, wcl):
    # NOTE: numba.jit, unlike numpy & torch, will promote to float64 with
    # array float32 and scalar float64
    pass

@jit(nopython=True, cache=True, parallel=True)
def _gmw_l1_par(w, gamma, beta, wc, wcl):
    # NOTE: numba.jit, unlike numpy & torch, will promote to float64 with
    # array float32 and scalar float64
    pass

def _gmw_l1_gpu(w, gamma, beta, wc, wcl):
    pass


def gmw_l2(gamma=3., beta=60., centered_scale=False, dtype='float64'):
    """Generalized Morse Wavelets, first order, L2(energy)-normalized.
    See `help(_gmw.gmw)`.
    """
    pass

@jit(nopython=True, cache=True)
def _gmw_l2(w, gamma, beta, wc, r, rgamma):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _gmw_l2_par(w, gamma, beta, wc, r, rgamma):
    pass

def _gmw_l2_gpu(w, gamma, beta, wc, r, rgamma):
    pass


def gmw_l1_k(gamma=3., beta=60., k=1, centered_scale=False, dtype='float64'):
    """Generalized Morse Wavelets, `k`-th order, L1(bandpass)-normalized.
    See `help(_gmw.gmw)`.
    """
    pass

@jit(nopython=True, cache=True)
def _gmw_l1_k(w, gamma, beta, wc, k_consts):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _gmw_l1_k_par(w, gamma, beta, wc, k_consts):
    pass

def _gmw_l1_k_gpu(w, gamma, beta, wc, k_consts):
    pass


def gmw_l2_k(gamma=3., beta=60., k=1, centered_scale=False, dtype='float64'):
    """Generalized Morse Wavelets, `k`-th order, L2(energy)-normalized.
    See `help(_gmw.gmw)`.
    """
    pass

@jit(nopython=True, cache=True)
def _gmw_l2_k(w, gamma, beta, wc, k_consts):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _gmw_l2_k_par(w, gamma, beta, wc, k_consts):
    pass

def _gmw_l2_k_gpu(w, gamma, beta, wc, k_consts):
    pass


def _gmw_k_constants(gamma, beta, k, norm='bandpass', dtype='float64'):
    """Laguerre polynomial constants & `coeff` term.

    Higher-order GMWs are coded such that constants are pre-computed and reused
    for any `w` input, since they remain fixed for said order.
    """
    pass


#### General order wavelets (any `K`) ########################################
def morsewave(N, freqs, gamma=3, beta=60, K=1, norm='bandpass'):
    """Generalized Morse wavelets of Olhede and Walden (2002).

    # Arguments:
        N: int > 0
            Number of samples / wavelet length

        freqs: float / list / np.ndarray
            (peak) center frequencies at which to generate wavelets,
            in *radians* (i.e. `w` in `w = 2*pi*f`).

        gamma, beta: float, float
            GMW parameters; `(gamma, beta) = (3, 60)` yields optimal
            time-frequency localization, and a good default for natural signals.
              - smaller `beta`: greater time resolution, lower freq resolution.
              - `gamma`: structurally alters the wavelet; 2 and 1 provide
                superior time localization but poor joint localization.
            See refs [2], [3].

        K: int > 0
            Will compute first `K` orthogonal GMWs, characterized by
            orders 0 through `K - 1`.
            Note this `K` is 1 greater than in original paper and than `order`
            throughout `ssqueezepy`, but is consistent with jLAB.

        norm: str['energy', 'bandpass']
            Normalization to use. See `help(_gmw.gmw)`, and below.

    # Returns:
        psih: np.ndarray [N x len(freqs) x K]
            Frequency-domain GMW, generated by sampling continuous-time function.
            Will collapse dims of length 1 (e.g. if `K=0` or `freqs` is integer).
        psi: np.ndarray [N x len(freqs) x K]
            Time-domain GMW, centered, generated via inverse DFT of `psih`.

    # References
        See `help(_gmw.gmw)`.
    __________________________________________________________________________

    **`beta==0` case**

    For BETA equal to zero, the generalized Morse wavelets describe
    a non-zero-mean function which is not in fact a wavelet. Only 'bandpass'
    normalization is supported for this case.

    In this case the frequency speficies the half-power point of the
    analytic lowpass filter.

    The frequency-domain definition of MORSEWAVE is not necessarily
    a good way to compute the zero-beta functions, however.  You will
    probably need to take a very small DT.

    **Multiple orthogonal wavelets**

    MORSEWAVE can compute multiple orthogonal versions of the generalized
    Morse wavelets, characterized by the order K.

    PSI=MORSEWAVE(N,K,GAMMA,BETA,FS) with a fifth numerical argument K
    returns an N x LENGTH(FS) x K array PSI which contains time-domain
    versions of the first K orthogonal generalized Morse wavelets.

    These K different orthogonal wavelets have been employed in
    multiwavelet polarization analysis, see Olhede and Walden (2003a,b).

    Again either bandpass or energy normalization can be applied. With
    bandpass normalization, all wavelets are divided by a constant, setting
    the peak value of the first frequency-domain wavelet equal to 2.
    """
    pass


def _morsewave1(N, f, gamma, beta, K, norm):
    """See `help(_gmw.morsewave)`."""
    pass


def _morsewave_first_family(fact, N, K, gamma, beta, w, psizero, norm):
    """See `help(_gmw.morsewave)`.

    See Olhede and Walden, "Noise reduction in directional signals using
    multiple Morse wavelets", IEEE Trans. Bio. Eng., v50, 51--57.
    The equation at the top right of page 56 is equivalent to the
    used expressions. Morse wavelets are defined in the frequency
    domain, and so not interpolated in the time domain in the same way
    as other continuous wavelets.
    """
    pass


def morseafun(gamma, beta, k=1, norm='bandpass'):
    """GMW amplitude or a-function (evaluated). Used internally by other funcs.

    # Arguments
        k: int >= 1
            Order of the wavelet; see `help(_gmw.morsewave)`.

        gamma, beta: float, float
            Wavelet parameters. See `help(_gmw.morsewave)`.

        norm: str['energy', 'bandpass']
            Wavelet normalization. See `help(_gmw.morsewave)`.

    # Returns
        A: float
            GMW amplitude (freq-domain peak value).
    ______________________________________________________________________
    Lilly, J. M. (2021), jLab: A data analysis package for Matlab, v1.6.9,
    http://www.jmlilly.net/jmlsoft.html
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morseafun.m
    """
    pass


def laguerre(x, k, c):
    """Generalized Laguerre polynomials. See `help(_gmw.morsewave)`.

    LAGUERRE is used in the computation of the generalized Morse
    wavelets and uses the expression given by Olhede and Walden (2002),
    "Generalized Morse Wavelets", Section III D.
    """
    pass


def morsefreq(gamma, beta, n_out=1):
    """Frequency measures for GMWs (with F. Rekibi).

    `n_out` controls how many parameters are computed and returned, in the
    following order: `wm, we, wi, cwi`, where:

        wm: modal / peak frequency
        we: energy frequency
        wi: instantaneous frequency at time-domain wavelet's center
        cwi: curvature of instantaneous frequency at time-domain wavelet's center

    All frequency quantities are *radian*, opposed to linear cyclic (i.e. `w`
    in `w = 2*pi*f`).

    For BETA=0, the "wavelet" becomes an analytic lowpass filter, and `wm`
    is not defined in the usual way. Instead, `wm` is defined as the point
    at which the filter has decayed to one-half of its peak power.

    # References
        [1] Higher-Order Properties of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede. 2009.
        https://sci-hub.st/10.1109/TSP.2008.2007607

        [2] (c) Lilly, J. M. (2021), jLab: A data analysis package for Matlab,
        v1.6.9, http://www.jmlilly.net/jmlsoft.html
        https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsefreq.m
    """
    pass


def _morsemom(p, gamma, beta, n_out=4):
    """Frequency-domain `p`-th order moments of the first order GMW.
    Used internally by other funcs.

    `n_out` controls how many parameters are coMputed and returned, in the
    following order: `Mp, Np, Kp, Lp`, where:

        Mp: p-th order moment
        Np: p-th order energy moment
        Kp: p-th order cumulant
        Lp: p-th order energy cumulant

    The p-th order moment and energy moment are defined as
        Mp = 1/(2 pi) int omegamma^p  psi(omegamma)     d omegamma
        Np = 1/(2 pi) int omegamma^p |psi(omegamma)|.^2 d omegamma
    respectively, where omegamma is the radian frequency. These are evaluated
    using the 'bandpass' normalization, which has `max(abs(psih(omegamma)))=2`.

    # References
        [1] Higher-Order Properties of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede. 2009.
        https://sci-hub.st/10.1109/TSP.2008.2007607

        [2] (c) Lilly, J. M. (2021), jLab: A data analysis package for Matlab,
        v1.6.9, http://www.jmlilly.net/jmlsoft.html
        https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsemom.m
    """
    def morsemom1(p, gamma, beta):
        pass
    def morsef(gamma, beta):
        pass
    pass


def _moments_to_cumulants(moments):
    """Convert moments to cumulants. Used internally by other funcs.

    Converts the first N moments   `moments  =[M0,M1,...M{N-1}]`
        into the first N cumulants `cumulants=[K0,K1,...K{N-1}]`.

    Note for a probability density function, M0=1 and K0=0.
    ______________________________________________________________________
    Lilly, J. M. (2021), jLab: A data analysis package for Matlab, v1.6.9,
    http://www.jmlilly.net/jmlsoft.html
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/moms
    """
    pass


def _check_args(gamma=None, beta=None, norm=None, order=None, scale=None,
                allow_zerobeta=True):
    """Only checks those that are passed in."""
    pass
