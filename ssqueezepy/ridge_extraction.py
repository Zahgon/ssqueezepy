# -*- coding: utf-8 -*-
"""Authors: David Bondesson, John Muradeli

Ridge extraction from time-frequency representations (STFT, CWT, synchrosqueezed).
"""
import numpy as np
from numba import jit, prange
from .utils import EPS32, EPS64


def extract_ridges(Tf, scales, penalty=2., n_ridges=1, bw=15, transform='cwt',
                   get_params=False, parallel=True):
    """Tracks time-frequency ridges by performing forward-backward ridge tracking
    algorithm, based on ref [1] (a version of Eq. III.4).

    Also see: https://www.mathworks.com/help/signal/ref/tfridge.html

    # Arguments:
        Tf: np.ndarray
            Complex time-frequency representation.

        scales:
            Frequency scales to calculate distance penalty term.

        penalty: float
            Value to penalize frequency jumps; multiplies the square of change
            in frequency. Trialworthy values: 0.5, 2, 5, 20, 40. Higher reduces
            odds of a ridge derailing to noise, but makes harder to track fast
            frequency changes.

        n_ridges: int
            Number of ridges to be calculated.

        bw: int
            Decides how many bins will be subtracted around max energy frequency
            bins when extracting multiple ridges (2 is standard for ssq'd).
            See "bw selection".

        transform: str['cwt', 'stft']
            Treats `scales` logarithmically if 'cwt', else linearly.
            `ssq_cwt` & `ssq_stft` are still 'cwt' & 'stft'.

        get_params: bool (default False)
            Whether to also compute and return `ridge_f` & `ridge_f`.

        parallel: bool (default True)
            Whether to use parallelized JIT code; runs faster on some input sizes.

    # Returns
        ridge_idxs: np.ndarray [n_timeshifts x n_ridges]
            Indices for maximum frequency ridge(s).
        ridge_f: np.ndarray [n_timeshifts x n_ridges]
            Quantities corresponding to extracted ridges:
                - STFT: frequencies
                - CWT: scales
        ridge_e: np.ndarray [n_timeshifts x n_ridges]
            Energies corresponding to extracted ridges.

    **bw selection**

    When a component is extracted, a region around it (a number of bins above
    and below the ridge) is zeroed and no longer affects next ridge's extraction.
        - higher: more bins subtracted, lesser chance of selecting the same
        component as the ridge.
        - lower:  less bins subtracted, lesser chance of dropping an unrelated
        component before the component is considered.
        - In general, set higher if more `scales` (or greater `nv`), or lower
        frequency resolution:
            - cwt:  `wavelets.freq_resolution(wavelet, N, nondim=False)`
            - stft: `utils.window_resolution(window)`
            - `N = utils.p2up(len(x))[0]`

    # References
        1. On the extraction of instantaneous frequencies from ridges in
        time-frequency representations of signals.
        D. Iatsenko, P. V. E. McClintock, A. Stefanovska.
        https://arxiv.org/pdf/1310.7276.pdf
    """
    def generate_penalty_matrix(scales, penalty):
        """Penalty matrix describes all potential penalties of  jumping from
        current frequency (first axis) to one or several new frequencies (second
        axis)

        `scales`: frequency scale vector from time-freq transform
        `penalty`: user-set penalty for freqency jumps (standard = 1.0)
        """
        pass
    def fw_bw_ridge_tracking(energy_to_track, penalty_matrix, eps):
        """Calculates acummulated penalty in forward (t=end...0) followed by
        backward (t=end...0) direction

        `energy`: squared abs time-frequency transform
        `penalty_matrix`: pre calculated penalty for all potential jumps between
                          two frequencies

        Returns: `ridge_idxs_fw_bw`: estimated forward backward frequency
                                     ridge indices
        """
        pass
    pass


def _accumulated_penalty_energy_fw(energy_to_track, penalty_matrix, parallel):
    """Calculates acummulated penalty in forward direction (t=0...end).

    `energy_to_track`: squared abs time-frequency transform
    `penalty_matrix`: pre-calculated penalty for all potential jumps between
                      two frequencies

    # Returns:
        `penalized_energy`: new energy with added forward penalty
        `ridge_idxs`: calculated initial ridge with only forward penalty
    """
    pass


@jit(nopython=True, cache=True)
def __accumulated_penalty_energy_fw(penalized_energy, penalty_matrix):
    pass

@jit(nopython=True, cache=True, parallel=True)
def __accumulated_penalty_energy_fwp(penalized_energy, penalty_matrix):
    pass


def _accumulated_penalty_energy_bw(energy_to_track, penalty_matrix,
                                   penalized_energy_fw, ridge_idxs_fw,
                                   eps, parallel):
    """Calculates acummulated penalty in backward direction (t=end...0)

    `energy_to_track`: squared abs time-frequency transform
    `penalty_matrix`: pre calculated penalty for all potential jumps between
                      two frequencies
    `ridge_idxs_fw`: calculated forward ridge

    Returns: `ridge_idxs_fw`: new ridge with added backward penalty, int array
    """
    pass


@jit(nopython=True, cache=True)
def __accumulated_penalty_energy_bw(e, penalty_matrix, pen_e, ridge_idxs_fw, eps):
    pass

@jit(nopython=True, cache=True, parallel=True)
def __accumulated_penalty_energy_bwp(e, penalty_matrix, pen_e, ridge_idxs_fw,
                                     eps):
    # adding `prange` to `tidx` makes whole computation much faster (x3-4),
    # but breaks it on *some* inputs (unpredictably)
    pass
