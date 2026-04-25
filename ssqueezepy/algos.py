# -*- coding: utf-8 -*-
"""CPU- & GPU-accelerated routines, and few neat algorithms.
"""
import numpy as np
from numba import jit, prange
from functools import reduce
from .utils.backend import asnumpy, cp, torch
from .utils.gpu_utils import _run_on_gpu, _get_kernel_params
from .utils import backend as S
from .configs import IS_PARALLEL


def nCk(n, k):
    """n-Choose-k"""
    pass

#### `indexed_sum` ###########################################################
def indexed_sum(a, k, parallel=None):
    """Sum `a` into rows of 2D array according to indices given by 2D `k`."""
    pass

@jit(nopython=True, cache=True)
def _indexed_sum(a, k, out):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_par(a, k, out):
    pass


def _process_ssq_params(Wx, w_or_dWx, ssq_freqs, const, logscale, flipud, out,
                        gamma, parallel, complex_out=True, Sfs=None):
    pass


def ssqueeze_fast(Wx, dWx, ssq_freqs, const, logscale=False, flipud=False,
                  gamma=None, out=None, Sfs=None, parallel=None):
    """`indexed_sum`, `find_closest`, and `phase_transform` within same loop,
    sparing two arrays and intermediate elementwise conditionals; see
    `help(algos.find_closest)` on how `k` is computed.
    """
    pass

def indexed_sum_onfly(Wx, w, ssq_freqs, const=1, logscale=False, flipud=False,
                      out=None, parallel=None):
    """`indexed_sum` and `find_closest` within same loop, sparing an array;
    see `help(algos.find_closest)` on how `k` is computed.
    """
    pass


@jit(nopython=True, cache=True)
def _indexed_sum_log(Wx, w, out, const, vlmin, dvl, omax, flipud=False):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_log_par(Wx, w, out, const, vlmin, dvl, omax, flipud=False):
    pass


@jit(nopython=True, cache=True)
def _indexed_sum_log_piecewise(Wx, w, out, const, vlmin0, vlmin1, dvl0, dvl1,
                               idx1, omax, flipud=False):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_log_piecewise_par(Wx, w, out, const, vlmin0, vlmin1, dvl0, dvl1,
                                   idx1, omax, flipud=False):
    # it's also possible to construct the if-else logic in terms of mappables
    # of `vlmin`, `dvl`, and `idx`, which generalizes to any number of transitions
    pass


@jit(nopython=True, cache=True)
def _indexed_sum_lin(Wx, w, out, const, vmin, dv, omax, flipud=False):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_lin_par(Wx, w, out, const, vmin, dv, omax, flipud=False):
    pass


#### `find_closest` algorithms ###############################################
def find_closest(a, v, logscale=False, parallel=None, smart=None):
    """`argmin(abs(a[i, j] - v)) for all `i, j`; `a` is 2D, `v` is 1D.

    # Arguments:
        a: np.ndarray
            2D array.

        v: np.ndarray
            1D array.

        logscale: bool (default False)
            Whether "closest" is taken in linear or logarithmic space.

        parallel: bool (default True) / None
            Whether to use algorithms with `numba.jit(parallel=True)`

        smart: bool (default False) / None
            Whether to use a very fast smart algorithm (but still the slowest
            for ssqueezing; see usage guide below).
            Credit: Divakar -- https://stackoverflow.com/a/64526158/10133797
    ____________________________________________________________________________
    **Default behavior**

    If only `a` & `v` are passed, `find_closest_smart` is called.
    ____________________________________________________________________________
    **Usage guide**

    If 100% accuracy is desired, or `v` is not linearly or logarithmically
    distributed, use `find_closest_smart` (`smart=True`) or `find_closest_brute`
    (not callable from here).
        `_smart` is faster on single CPU thread, but `_brute` can win
        via parallelism.

    Else, `find_closest_lin` and `find_closest_log` do the trick (the special
    case of log-piecewise is handled), and are much faster.
        - Relative to "exact", they differ only by 0% to 0.0001%, purely per
        float precision limitations, and never by more than one index in `out`
        (where  whether e.g. `w=0.500000001` belongs to 0 or 1 isn't statistically
        meaningful to begin with).

    ____________________________________________________________________________
    **How it works:** `find_closest_log`, `find_closest_lin`

    The root assumption is that `v` is uniformly (in linear or log space)
    distributed, and we calculate analytically in which bin `w` will land as:
        `(w - bin_min) / bin_step_size`
    Above is forced to bound in [0, len(v) - 1].
    """
    pass


@jit(nopython=True, cache=True, parallel=True)
def find_closest_brute(a, v):
    """Computes exactly but exhaustively."""
    pass


def find_closest_smart(a, v):
    """Equivalent to argmin(abs(a[i, j] - v)) for all i, j; a is 2D, v is 1D.
    Credit: Divakar -- https://stackoverflow.com/a/64526158/10133797
    """
    pass


def _ensure_nonzero_nonnegative(name, x, silent=False):
    pass


def _get_params_find_closest_log(v):
    pass

def find_closest_log(a, v, parallel=True):
    pass

@jit(nopython=True, cache=True)
def _find_closest_log(a, out, vlmin, dvl, omax):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _find_closest_log_par(a, out, vlmin, dvl, omax):
    pass


@jit(nopython=True, cache=True)
def _find_closest_log_piecewise(a, out, vlmin0, vlmin1, dvl0, dvl1, idx1,
                                omax):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _find_closest_log_piecewise_par(a, out, vlmin0, vlmin1, dvl0, dvl1, idx1,
                                    omax):
    # it's also possible to construct the if-else logic in terms of mappables
    # of `vlmin`, `dvl`, and `idx`, which generalizes to any number of transitions
    pass


def find_closest_lin(a, v, parallel=True):
    pass

@jit(nopython=True, cache=True)
def _find_closest_lin(a, out, vmin, dv, omax):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _find_closest_lin_par(a, out, vmin, dv, omax):
    pass

#### Replacers ###############################################################
def _process_replace_fn_args(x, ref):
    pass


def replace_at_inf_or_nan(x, ref=None, replacement=0.):
    pass

def replace_at_inf(x, ref=None, replacement=0.):
    pass

def replace_at_nan(x, ref=None, replacement=0.):
    pass

def replace_at_value(x, ref=None, value=0., replacement=0.):
    """Note: `value=np.nan` won't work (but np.inf will, separate from -np.inf)"""
    pass

def replace_under_abs(x, ref=None, value=0., replacement=0., parallel=None):
    pass


# TODO return None?
@jit(nopython=True, cache=True)
def _replace_at_inf_or_nan(x, ref, replacement=0.):
    pass

@jit(nopython=True, cache=True)
def _replace_at_inf(x, ref, replacement=0.):
    pass

@jit(nopython=True, cache=True)
def _replace_at_nan(x, ref, replacement=0.):
    pass

@jit(nopython=True, cache=True)
def _replace_at_value(x, ref, value=0., replacement=0.):
    pass


@jit(nopython=True, cache=True)
def _replace_under_abs(x, ref, value=0., replacement=0.):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _replace_under_abs_par(x, ref, value=0., replacement=0.):
    pass


def _replace_under_abs_gpu(w, Wx, value=0., replacement=0.):
    """Not as general as CPU variants (namely `w` must be real and `Wx`
    must be complex).
    """
    pass


def zero_denormals(x, parallel=None):
    """Denormals are very small non-zero numbers that can significantly slow CPU
    execution (e.g. FFT). See https://github.com/scipy/scipy/issues/13764
    """
    pass

@jit(nopython=True, cache=True)
def _zero_denormals(x, tiny):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _zero_denormals_par(x, tiny):
    pass

#### misc (short) ############################################################
@jit(nopython=True, cache=True)
def _min_neglect_idx(arr, th=1e-12):
    """Used in utils.integrate_analytic and ._integrate_bounded."""
    pass

#### misc (long) #############################################################
def find_maximum(fn, step_size=1e-3, steps_per_search=1e4, step_start=0,
                 step_limit=1000, min_value=-1):
    """Finds max of any function with a single maximum, and input value
    at which the maximum occurs. Inputs and outputs must be 1D.

    Must be strictly non-decreasing from step_start up to maximum of interest.
    Takes absolute value of fn's outputs.
    """
    pass


def find_first_occurrence(fn, value, step_size=1e-3, steps_per_search=1e4,
                          step_start=0, step_limit=1000):
    """Finds earliest input value for which `fn(input_value) == value`, searching
    from `step_start` to `step_limit` in `step_size` increments.
    Takes absolute value of fn's outputs.
    """
    pass


def phase_cwt_cpu(Wx, dWx, gamma, parallel=None):
    """Computes only the imaginary part of `dWx / Wx` while dividing by 2*pi
    in same operation; doesn't compute division at all if `abs(Wx) < gamma`.
    Less memory & less computation than `(dWx / Wx).imag / (2*pi)`, same result.
    """
    pass

@jit(nopython=True, cache=True)
def _phase_cwt(Wx, dWx, out, gamma):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _phase_cwt_par(Wx, dWx, out, gamma):
    pass


def phase_cwt_gpu(Wx, dWx, gamma):
    """Same as `phase_cwt_cpu`, but on GPU."""
    pass


def phase_stft_cpu(Wx, dWx, Sfs, gamma, parallel=None):
    pass

@jit(nopython=True, cache=True)
def _phase_stft(Wx, dWx, Sfs, out, gamma):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _phase_stft_par(Wx, dWx, Sfs, out, gamma):
    pass

def phase_stft_gpu(Wx, dWx, Sfs, gamma):
    pass


@jit(nopython=True, cache=True)
def _ssq_cwt_log_piecewise(Wx, dWx, out, const, gamma, vlmin0, vlmin1,
                           dvl0, dvl1, idx1, omax, flipud=False):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _ssq_cwt_log_piecewise_par(Wx, dWx, out, const, gamma, vlmin0, vlmin1,
                               dvl0, dvl1, idx1, omax, flipud=False):
    pass


@jit(nopython=True, cache=True)
def _ssq_cwt_log(Wx, dWx, out, const, gamma, vlmin, dvl, omax, flipud=False):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _ssq_cwt_log_par(Wx, dWx, out, const, gamma, vlmin, dvl, omax, flipud=False):
    pass


@jit(nopython=True, cache=True)
def _ssq_cwt_lin(Wx, dWx, out, const, gamma, vmin, dv, omax, flipud=False):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _ssq_cwt_lin_par(Wx, dWx, out, const, gamma, vmin, dv, omax, flipud=False):
    pass


@jit(nopython=True, cache=True)
def _ssq_stft(Wx, dWx, Sfs, out, const, gamma, vmin, dv, omax, flipud=False):
    pass

@jit(nopython=True, cache=True, parallel=True)
def _ssq_stft_par(Wx, dWx, Sfs, out, const, gamma, vmin, dv, omax, flipud=False):
    pass


#### CPU funcs & GPU kernel codes ############################################
_cpu_fns = {
    'ssq_cwt_log_piecewise':     _ssq_cwt_log_piecewise,
    'ssq_cwt_log_piecewise_par': _ssq_cwt_log_piecewise_par,
    'ssq_cwt_log':               _ssq_cwt_log,
    'ssq_cwt_log_par':           _ssq_cwt_log_par,
    'ssq_cwt_lin':               _ssq_cwt_lin,
    'ssq_cwt_lin_par':           _ssq_cwt_lin_par,

    'ssq_stft':     _ssq_stft,
    'ssq_stft_par': _ssq_stft_par,

    'indexed_sum_log_piecewise':     _indexed_sum_log_piecewise,
    'indexed_sum_log_piecewise_par': _indexed_sum_log_piecewise_par,
    'indexed_sum_log':               _indexed_sum_log,
    'indexed_sum_log_par':           _indexed_sum_log_par,
    'indexed_sum_lin':               _indexed_sum_lin,
    'indexed_sum_lin_par':           _indexed_sum_lin_par,
}

_kernel_codes = dict(
    ssq_cwt_log_piecewise='''
    extern "C" __global__
    void ssq_cwt_log_piecewise(${dtype} Wx[${M}][${N}][2],
                               ${dtype} dWx[${M}][${N}][2],
                               ${dtype} out[${M}][${N}][2],
                               ${dtype} const_arr[${M}],
                               ${dtype} *gamma,
                               double vlmin0, double vlmin1,
                               double dvl0, double dvl1,
                               int idx1) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= ${N})
            return;

        int k;
        double wl;
        ${dtype} w_ij, A, B, C, D;

        for (int i=0; i < ${M}; ++i){
          if (norm${f}(2, Wx[i][j]) > *gamma){

            A = dWx[i][j][0];
            B = dWx[i][j][1];
            C = Wx[i][j][0];
            D = Wx[i][j][1];
            w_ij = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));

            wl = log2${f}(w_ij);
            if (wl > vlmin1){
                k = (int)round((wl - vlmin1) / dvl1) + idx1;
                if (k >= ${M})
                    k = ${M} - 1;
            } else {
                k = (int)round((wl - vlmin0) / dvl0);
                if (k < 0)
                    k = 0;
            }
            ${extra}

            out[k][j][0] += Wx[i][j][0] * const_arr[i];
            out[k][j][1] += Wx[i][j][1] * const_arr[i];
          }
        }
    }
    ''',

    ssq_cwt_log='''
    extern "C" __global__
    void ssq_cwt_log(${dtype} Wx[${M}][${N}][2],
                     ${dtype} dWx[${M}][${N}][2],
                     ${dtype} out[${M}][${N}][2],
                     ${dtype} const_arr[${M}],
                     ${dtype} *gamma,
                     double vlmin, double dvl) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= ${N})
            return;

        int k;
        ${dtype} w_ij, A, B, C, D;

        for (int i=0; i < ${M}; ++i){
          if (norm${f}(2, Wx[i][j]) > *gamma){

            A = dWx[i][j][0];
            B = dWx[i][j][1];
            C = Wx[i][j][0];
            D = Wx[i][j][1];
            w_ij = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));

            k = (int)round(((double)log2${f}(w_ij) - vlmin) / dvl);
            if (k >= ${M})
                k = ${M} - 1;
            else if (k < 0)
                k = 0;
            ${extra}

            out[k][j][0] += Wx[i][j][0] * const_arr[i];
            out[k][j][1] += Wx[i][j][1] * const_arr[i];
          }
        }
    }
    ''',

    ssq_cwt_lin='''
    extern "C" __global__
    void ssq_cwt_lin(${dtype} Wx[${M}][${N}][2],
                     ${dtype} dWx[${M}][${N}][2],
                     ${dtype} out[${M}][${N}][2],
                     ${dtype} const_arr[${M}],
                     ${dtype} *gamma,
                     double vmin, double dv) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= ${N})
            return;

        int k;
        ${dtype} w_ij, A, B, C, D;

        for (int i=0; i < ${M}; ++i){
          if (norm${f}(2, Wx[i][j]) > *gamma){

            A = dWx[i][j][0];
            B = dWx[i][j][1];
            C = Wx[i][j][0];
            D = Wx[i][j][1];
            w_ij = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));

            k = (int)round(((double)w_ij - vmin) / dv);
            if (k >= ${M})
                k = ${M} - 1;
            else if (k < 0)
                k = 0;
            ${extra}

            out[k][j][0] += Wx[i][j][0] * const_arr[i];
            out[k][j][1] += Wx[i][j][1] * const_arr[i];
          }
        }
    }
    ''',

    ssq_stft='''
    extern "C" __global__
    void ssq_stft(${dtype} Wx[${M}][${N}][2],
                  ${dtype} dWx[${M}][${N}][2],
                  ${dtype} Sfs[${M}],
                  ${dtype} out[${M}][${N}][2],
                  ${dtype} const_arr[${M}],
                  ${dtype} *gamma,
                  double vmin, double dv) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= ${N})
            return;

        int k;
        ${dtype} w_ij, A, B, C, D;

        for (int i=0; i < ${M}; ++i){
          if (norm${f}(2, Wx[i][j]) > *gamma){

            A = dWx[i][j][0];
            B = dWx[i][j][1];
            C = Wx[i][j][0];
            D = Wx[i][j][1];
            w_ij = abs(Sfs[i] - (B*C - A*D) / ((C*C + D*D) * 6.283185307179586));

            k = (int)round(((double)w_ij - vmin) / dv);
            if (k >= ${M})
                k = ${M} - 1;
            else if (k < 0)
                k = 0;
            ${extra}

            out[k][j][0] += Wx[i][j][0] * const_arr[i];
            out[k][j][1] += Wx[i][j][1] * const_arr[i];
          }
        }
    }
    ''',

    indexed_sum_log_piecewise='''
    extern "C" __global__
    void indexed_sum_log_piecewise(${dtype} Wx[${M}][${N}][2],
                                   ${dtype} w[${M}][${N}],
                                   ${dtype} out[${M}][${N}][2],
                                   ${dtype} const_arr[${M}],
                                   double vlmin0, double vlmin1,
                                   double dvl0, double dvl1,
                                   int idx1)
    {
      int j = blockIdx.x * blockDim.x + threadIdx.x;

      if (j >= ${N})
        return;

      int k;
      double wl;
      for (int i=0; i < ${M}; ++i){
        if (!isinf(w[i][j])){
          wl = (double)log2${f}(w[i][j]);

          if (wl > vlmin1){
              k = (int)round((wl - vlmin1) / dvl1) + idx1;
              if (k >= ${M})
                  k = ${M} - 1;
          } else {
              k = (int)round((wl - vlmin0) / dvl0);
              if (k < 0)
                  k = 0;
          }
          ${extra}

          out[k][j][0] += Wx[i][j][0] * const_arr[i];
          out[k][j][1] += Wx[i][j][1] * const_arr[i];
        }
      }
    }
    ''',

    indexed_sum_log='''
    extern "C" __global__
    void indexed_sum_log(${dtype} Wx[${M}][${N}][2],
                         ${dtype} w[${M}][${N}],
                         ${dtype} out[${M}][${N}][2],
                         ${dtype} const_arr[${M}],
                         double vlmin, double dvl)
    {
      int j = blockIdx.x * blockDim.x + threadIdx.x;

      if (j >= ${N})
        return;

      int k;
      for (int i=0; i < ${M}; ++i){
        if (!isinf(w[i][j])){
          k = (int)round(((double)log2${f}(w[i][j]) - vlmin) / dvl);

          if (k >= ${M})
              k = ${M} - 1;
          else if (k < 0)
              k = 0;
          ${extra}

          out[k][j][0] += Wx[i][j][0] * const_arr[i];
          out[k][j][1] += Wx[i][j][1] * const_arr[i];
        }
      }
    }
    ''',

    indexed_sum_lin='''
    extern "C" __global__
    void indexed_sum_lin(${dtype} Wx[${M}][${N}][2],
                         ${dtype} w[${M}][${N}],
                         ${dtype} out[${M}][${N}][2],
                         ${dtype} const_arr[${M}],
                         double vmin, double dv)
    {
      int j = blockIdx.x * blockDim.x + threadIdx.x;

      if (j >= ${N})
        return;

      int k;
      for (int i=0; i < ${M}; ++i){
        if (!isinf(w[i][j])){
          k = (int)round(((double)(w[i][j]) - vmin) / dv);

          if (k >= ${M})
              k = ${M} - 1;
          else if (k < 0)
              k = 0;
          ${extra}

          out[k][j][0] += Wx[i][j][0] * const_arr[i];
          out[k][j][1] += Wx[i][j][1] * const_arr[i];
        }
      }
    }
    ''',

    phase_cwt='''
    extern "C" __global__
    void phase_cwt(${dtype} Wx[${M}][${N}][2],
                   ${dtype} dWx[${M}][${N}][2],
                   ${dtype} out[${M}][${N}],
                   ${dtype} *gamma) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= ${M} || j >= ${N})
            return;

        if (norm${f}(2, Wx[i][j]) < *gamma){
          out[i][j] = 1.0/0.0;
          return;
        }

        ${dtype} A = dWx[i][j][0];
        ${dtype} B = dWx[i][j][1];
        ${dtype} C = Wx[i][j][0];
        ${dtype} D = Wx[i][j][1];

        out[i][j] = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));
    }
    ''',
)

###############################################################################
from .utils.common import WARN, EPS64
from .utils.cwt_utils import logscale_transition_idx
