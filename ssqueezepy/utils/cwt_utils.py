# -*- coding: utf-8 -*-
import numpy as np
from scipy import integrate
from .common import WARN, assert_is_one_of, p2up
from .backend import torch, asnumpy
from ..configs import gdefaults

pi = np.pi

__all__ = [
    'adm_ssq',
    'adm_cwt',
    'cwt_scalebounds',
    'process_scales',
    'infer_scaletype',
    'make_scales',
    'logscale_transition_idx',
    'nv_from_scales',
    'find_min_scale',
    'find_max_scale',
    'find_downsampling_scale',
    'integrate_analytic',
    'find_max_scale_alt',
    '_process_fs_and_t',
]


def adm_ssq(wavelet):
    """Calculates the synchrosqueezing admissibility constant, the term
    R_psi in Eq 15 of [1] (also see Eq 2.5 of [2]). Uses numeric intergration.

        integral(conj(wavelet(w)) / w, w=0..inf)

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications. G. Thakur,
        E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf
    """
    pass


def adm_cwt(wavelet):
    """Calculates the cwt admissibility constant as per Eq. (4.67) of [1].
    Uses numeric integration.

        integral(wavelet(w) * conj(wavelet(w)) / w, w=0..inf)

    1. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
    https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
    """
    pass


def cwt_scalebounds(wavelet, N, preset=None, min_cutoff=None, max_cutoff=None,
                    cutoff=None, bin_loc=None, bin_amp=None, use_padded_N=True,
                    viz=False):
    """Finds range of scales for which `wavelet` is "well-behaved", as
    determined by `preset`. Assumes `wavelet` is uni-modal (one peak in freq
    domain); may be inaccurate otherwise.

    `min_scale`: found such that freq-domain wavelet takes on `cutoff` of its max
    value on the greatest bin.
      - Lesser `cutoff` -> lesser `min_scale`, always

    `max_scale`: search determined by `preset`:
        - 'maximal': found such that freq-domain takes `bin_amp` of its max value
          on the `bin_loc`-th (non-dc) bin
          - Greater `bin_loc` or lesser `bin_amp` -> lesser `max_scale`, always

        - 'minimal': found more intricately independent of precise bin location,
          but is likely to omit first several bins entirely; see
          `help(utils.find_max_scale_alt)`.
          - Greater `min_cutoff` -> lesser `max_scale`, generally

    `viz==2` for more visuals, `viz==3` for even more.

    # Arguments:
        wavelet: `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain. See `help(cwt)`.

        N: int
            Length of wavelet to use.

        min_cutoff, max_cutoff: float > 0 / None
            Used to find max scale with `preset='minimal'`.
            See `help(utils.find_max_scale_alt)`

        cutoff: float / None
            Used to find min scale. See `help(utils.find_min_scale)`

        preset: str['maximal', 'minimal', 'naive'] / None
            - 'maximal': yields a larger max and smaller min.
            - 'minimal': strives to keep wavelet in "well-behaved" range of std_t
            and std_w, but very high or very low frequencies' energies will be
            under-represented. Is closer to MATLAB's default `cwtfreqbounds`.
            - 'naive': returns (1, N), which is per original MATLAB Toolbox,
            but a poor choice for most wavelet options.
            - None: will use `min_cutoff, max_cutoff, cutoff` values, else
            override `min_cutoff, max_cutoff` with those of `preset='minimal'`,
            and of `cutoff` with that of `preset='maximal'`:
                (min_cutoff, max_cutoff, cutoff) = (0.6, 0.8, -.5)

        use_padded_N: bool (default True)
            Whether to use `N=p2up(N)` in computations. Typically `N == len(x)`,
            but CWT pads to next power of 2, which is the actual wavelet length
            used, which typically behaves significantly differently at scale
            extrema, thus recommended default True. Differs from passing
            `N=p2up(N)[0]` and False only for first visual if `viz`, see code.

    # Returns:
        min_scale, max_scale: float, float
            Minimum & maximum scales.
    """
    def _process_args(preset, min_cutoff, max_cutoff, cutoff, bin_loc, bin_amp):
        pass
    def _viz():
        pass
    pass


def _assert_positive_integer(g, name=''):
    pass


def process_scales(scales, N, wavelet=None, nv=None, get_params=False,
                   use_padded_N=True):
    """Makes scales if `scales` is a string, else validates the array,
    and returns relevant parameters if requested.

        - Ensures, if array,  `scales` is 1D, or 2D with last dim == 1
        - Ensures, if string, `scales` is one of ('log', 'linear')
        - If `get_params`, also returns (`scaletype`, `nv`, `na`)
           - `scaletype`: inferred from `scales` ('linear' or 'log') if array
           - `nv`, `na`: computed newly only if not already passed
    """
    def _process_args(scales, nv, wavelet):
        pass
    pass


def infer_scaletype(scales):
    """Infer whether `scales` is linearly or exponentially distributed (if latter,
    also infers `nv`). Used internally on `scales` and `ssq_freqs`.

    Returns one of: 'linear', 'log', 'log-piecewise'
    """
    pass


def make_scales(N, min_scale=None, max_scale=None, nv=32, scaletype='log',
                wavelet=None, downsample=None):
    """Recommended to first work out `min_scale` & `max_scale` with
    `cwt_scalebounds`.

    # Arguments:
        N: int
            `len(x)` or `len(x_padded)`.

        min_scale, max_scale: float, float
            Set scale range. Obtained e.g. from `utils.cwt_scalebounds`.

        nv: int
            Number of voices (wavelets) per octave.

        scaletype: str['log', 'log-piecewise', 'linear']
            Scaling kind to make.
            `'log-piecewise'` uses `utils.find_downsampling_scale`.

        wavelet: wavelets.Wavelet
            Used only for `scaletype='log-piecewise'`.

        downsample: int
            Downsampling factor. Used only for `scaletype='log-piecewise'`.

    # Returns:
        scales: np.ndarray
    """
    pass


def logscale_transition_idx(scales):
    """Returns `idx` that splits `scales` as `[scales[:idx], scales[idx:]]`.
    """
    pass


def nv_from_scales(scales):
    """Infers `nv` from `scales` assuming `2**` scales; returns array
    of length `len(scales)` if `scaletype = 'log-piecewise'`.
    """
    pass


def find_min_scale(wavelet, cutoff=1):
    """Design the wavelet in frequency domain. `scale` is found to yield
    `scale * xi(scale=1)` such that its last (largest) positive value evaluates
    `wavelet` to `cutoff * max(psih)`. If cutoff > 0, it lands to right of peak,
    else to left (i.e. peak excluded).
    """
    pass


def find_max_scale(wavelet, N, bin_loc=1, bin_amp=1):
    """Finds `scale` such that freq-domain wavelet's amplitude is `bin_amp`
    of maximum at `bin_loc` bin. Set `bin_loc=1` to ensure no lower frequencies
    are lost, but likewise mind redundancy (see `make_scales`).
    """
    pass


def find_downsampling_scale(wavelet, scales, span=5, tol=3, method='sum',
                            nonzero_th=.02, nonzero_tol=4., N=None, viz=False,
                            viz_last=False):
    """Find `scale` past which freq-domain wavelets are "excessively redundant",
    redundancy determined by `span, tol, method, nonzero_th, nonzero_tol`.

    # Arguments
        wavelet: np.ndarray / wavelets.Wavelet
            CWT wavelet.

        scales: np.ndarray
            CWT scales.

        span: int
            Number of wavelets to cross-correlate at each comparison.

        tol: int
            Tolerance value, works with `method`.

        method: str['any', 'all', 'sum']
            Condition relating `span` and `tol` to determine whether wavelets
            are packed "too densely" at a given cross-correlation, relative
            to "joint peak".

                'any': at least one of wavelet peaks lie `tol` or more bins away
                'all': all wavelet peaks lie `tol` or more bins away
                'sum': sum(distances between wavelet peaks and joint peak) > `tol`

        nonzero_th: float
            Wavelet points as a fraction of respective maxima to consider
            nonzero (i.e. `np.where(psih > psih.max()*nonzero_th)`).

        nonzero_tol: float
            Average number of nonzero points in a `span` group of wavelets above
            which testing is exempted. (e.g. if 5 wavelets have 25 nonzero points,
            average is 5, so if `nonzero_tol=4`, the `scale` is skipped/passed).

        N: int / None
            Length of wavelet to use. Defaults to 2048, which generalizes well
            along other defaults, since those params (`span`, `tol`, etc) would
            need to be scaled alongside `N`.

        viz: bool (default False)
            Visualize every test for debug purposes.

        viz_last: bool (default True)
            Visualize the failing scale (recommended if trying by hand);
            ignored if `viz=True`.
    """
    def check_group(psihs_peaks, joint_peak, method, tol):
        pass
    def _viz(psihs, psihs_peaks, joint_peak, psihs_nonzeros, i):
        pass
    pass


def integrate_analytic(int_fn, nowarn=False):
    """Assumes function that's zero for negative inputs (e.g. analytic wavelet),
    decays toward right, and is unimodal: int_fn(t<0)=0, int_fn(t->inf)->0.
    Integrates using trapezoidal rule, from 0 to inf (equivalently).

    Integrates near zero separately in log space (useful for e.g. 1/x).
    """
    def _est_arr(mxlim, N):
        pass
    def _find_convergent_array():
        pass
    def _integrate_near_zero():
        pass
    pass


def find_max_scale_alt(wavelet, N, min_cutoff=.1, max_cutoff=.8):
    """
    Design the wavelet in frequency domain. `scale` is found to yield
    `scale * xi(scale=1)` such that two of its consecutive values land
    symmetrically about the peak of `psih` (i.e. none *at* peak), while
    still yielding `wavelet(w)` to fall between `min_cutoff`* and `max_cutoff`*
    `max(psih)`. `scale` is selected such that the symmetry is attained
    using smallest possible bins (closest to dc). Steps:

        1. Find `w` (input value to `wavelet`) for which `wavelet` is maximized
        (i.e. peak of `psih`).
        2. Find two `w` such that `wavelet` attains `min_cutoff` and `max_cutoff`
        times its maximum value, using `w` in previous step as upper bound.
        3. Find `div_size` such that `xi` lands at both points of symmetry;
        `div_size` == increment between successive values of
        `xi = scale * xi(scale=1)`.
            - `xi` begins at zero; along the cutoff bounds, and us selecting
            the smallest number of divisions/increments to reach points of
            symmetry, we guarantee a unique `scale`.

    This yields a max `scale` that'll generally lie in 'nicely-behaved' region
    of std_t; value can be used to fine-tune further.
    See `visuals.sweep_std_t`.
    """
    pass


def _process_fs_and_t(fs, t, N):
    """Ensures `t` is uniformly-spaced and of same length as `x` (==N)
    and returns `fs` and `dt` based on it, or from defaults if `t` is None.
    """
    pass


#############################################################################
from ..algos import _min_neglect_idx, find_maximum, find_first_occurrence
from ..wavelets import Wavelet, center_frequency
from ..visuals import plot, scat, _viz_cwt_scalebounds, wavelet_waveforms
from ..visuals import sweep_harea
