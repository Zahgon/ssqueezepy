# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing
from scipy.fft import fftshift as sfftshift, ifftshift as sifftshift
from scipy.fft import fft as sfft, rfft as srfft, ifft as sifft, irfft as sirfft
from pathlib import Path
from . import backend as S
from ..configs import IS_PARALLEL

try:
    from torch.fft import (fft as tfft, rfft as trfft,
                           ifft as tifft, irfft as tirfft,
                           fftshift as tfftshift, ifftshift as tifftshift)
except ImportError:
    pass

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(600)
except ImportError:
    pyfftw = None

UTILS_DIR = Path(__file__).parent

__all__ = [
    'fft',
    'rfft',
    'ifft',
    'irfft',
    'fftshift',
    'ifftshift',
    'FFT',
    'FFT_GLOBAL',
]

#############################################################################


class FFT():
    """Global class for ssqueezepy FFT methods.

    Will use GPU via PyTorch if environment flag `'SSQ_GPU'` is set to `'1'`.
    Will use `scipy.fft` or `pyfftw` depending on `patience` argument (and
    whether `pyfftw` is installed).
    Both will use `threads` CPUs to accelerate computing.

    In a nutshell, if you plan on re-running FFT on input of same shape and dtype,
    prefer `patience=1`, which introduces a lengthy first-time overhead but may
    compute significantly faster afterwards.

    # Arguments (`fft`, `rfft`, `ifft`, `irfft`):
        x: np.ndarray
            1D or 2D.

        axis: int
            FFT axis. One of `0, 1, -1`.

        patience: int / tuple[int, int]
            If int:
                0: will use `scipy.fft`
                1: `pyfftw` with flag `'FFTW_PATIENT'`
                2: `pyfftw` with flag `'FFTW_EXHAUSTIVE'`
            Else, if tuple, second element specifies `planning_timelimit`
            passed to `pyfftw.FFTW` (so tuple requires `patience[0] != 0`).

            Set `planning_timelimit = None` to allow planning to finish,
            but beware; `patience = 1` can take hours for large inputs, and `2`
            even longer.

        astensor: bool (default False)
            If computing on GPU, whether to return as `torch.Tensor` (if False,
            will move to CPU and convert to `numpy.ndarray`).

        n: int / None
            Only for `irfft`; length of original input. If None, will default to
            `2*(x.shape[axis] - 1)`.

    __________________________________________________________________________
    # Arguments (`__init__`):
        planning_timelimit: int
            Default.

        wisdom_dir: str
            Where to save wisdom to or load from. Empty string means
            `ssqueezepy/utils/`.

        threads: int
            Number of CPU threads to use. -1 = maximum.

        patience: int
            Default `patience`.

        cache_fft_objects: bool (default False)
            If True, `pyfftw` objects generated throughout session are stored in
            `FFT._input_history`, and retrieved if all of below match:
                `(x.shape, x.dtype, real, patience, n)`
            where `patience` includes `planning_timelimit` as a tuple.
            Default False since loading from wisdom is very fast anyway.

        verbose: bool (default True)
            Controls whether a message is printed upon `patience >= 1`.
    __________________________________________________________________________
    **Wisdom**

    `pyfftw` uses "wisdom", basically storing and reusing generated FFT plans
    if input attributes match:
        (`x.shape`, `x.dtype`, `axis`, `flags`, `planning_timelimit`)
    `flags` and `planning_timelimit` are set via `patience`.

    With each `pyfftw` use, `save_wisdom()` is called, writing to `wisdom32` and
    `wisdom64` bytes files in `ssqueezepy/utils`. Each time ssqueezepy runs in a
    new session, `load_wisdom()` is called to load these values, so wisdom is
    only expansive.
    """
    def __init__(self, planning_timelimit=120, wisdom_dir=UTILS_DIR, threads=None,
                 patience=0, cache_fft_objects=False, verbose=1):
        pass

    @property
    def threads(self):
        """Set dynamically if `threads` wasn't passed in __init__."""
        pass

    @property
    def patience(self):
        """Setter will also set `planning_timelimit` if setting to tuple."""
        pass

    @patience.setter
    def patience(self, value):
        pass

    #### Main methods #########################################################
    def fft(self, x, axis=-1, patience=None, astensor=False):
        """See `help(ssqueezepy.utils.FFT)`."""
        pass

    def rfft(self, x, axis=-1, patience=None, astensor=False):
        """See `help(ssqueezepy.utils.FFT)`."""
        pass

    def ifft(self, x, axis=-1, patience=None, astensor=False):
        """See `help(ssqueezepy.utils.FFT)`."""
        pass

    def irfft(self, x, axis=-1, patience=None, astensor=False, n=None):
        """See `help(ssqueezepy.utils.FFT)`."""
        pass

    def fftshift(self, x, axes=-1, astensor=False):
        pass

    def ifftshift(self, x, axes=-1, astensor=False):
        pass

    def _maybe_gpu(self, name, x, astensor=False, **kw):
        pass

    #### FFT makers ###########################################################
    def _get_save_fill(self, x, axis, patience, real, inverse=False, n=None):
        pass

    def get_fft_object(self, x, axis, patience=1, real=False, inverse=False,
                       n=None):
        pass

    def _get_fft_object(self, x, axis, patience, real, inverse, n):
        pass

    def _process_input(self, x, axis, patience, real, inverse, n):
        pass

    def _get_output_shape(self, x, axis, real=False, inverse=False, n=None):
        pass

    #### Misc #################################################################
    def load_wisdom(self):
        pass

    def save_wisdom(self):
        """Will overwrite."""
        pass

    def _validate_input(self, x, axis, real, patience, inverse):
        """Assert is single/double precision and is 1D/2D."""
        pass

    def _validate_patience(self, patience):
        pass

    def _process_patience(self, patience):
        pass


FFT_GLOBAL = FFT()

fft   = FFT_GLOBAL.fft
rfft  = FFT_GLOBAL.rfft
ifft  = FFT_GLOBAL.ifft
irfft = FFT_GLOBAL.irfft
fftshift  = FFT_GLOBAL.fftshift
ifftshift = FFT_GLOBAL.ifftshift
