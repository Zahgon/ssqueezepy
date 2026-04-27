# -*- coding: utf-8 -*-
"""Convenience visual methods"""

import numpy as np
from pathlib import Path
from .algos import find_closest, find_maximum
from .configs import gdefaults
from . import plt


#### Visualizations ##########################################################
def wavelet_tf(wavelet, N=2048, scale=None, notext=False, width=1.1, height=1):
    """Visualize `wavelet` joint time-frequency resolution. Plots frequency-domain
    wavelet (psih) along y-axis, and time-domain wavelet (psi) along x-axis.

    Orthogonal units (e.g. y-axis for psi) are meaningless; function values
    aren't to scale, but *widths* are, so time-frequency uncertainties are
    accurately captured.

    `wavelet` is instance of `wavelets.Wavelet` or its valid `wavelet` argument.
    See also: https://www.desmos.com/calculator/0nslu0qivv
    """

    def pick_scale(wavelet, N):
        pass

    pass


def wavelet_tf_anim(
    wavelet,
    N=2048,
    scales=None,
    width=1.1,
    height=1,
    savepath="wavanim.gif",
    testing=False,
):
    """This method computes same as `wavelet_tf` but for all scales at once,
    and animates 'intelligently'. See help(wavelet_tf).

    `scales=None` will default to 'log:minimal' with (.9*min_scale,
    0.25*max_scale). These are selected to show the wavelet a little outside of
    "well-behaved" range (without slashing max_scale, it's a lot outside such
    range). May not work for every wavelet or all of their configs.
    """

    def animate(i):
        pass

    def unique_savepath(savepath):
        pass

    def _make_anim_scales(scales, wavelet, N):
        pass

    pass


def wavelet_heatmap(wavelet, scales="log", N=2048):
    pass


def sweep_std_t(wavelet, N, scales="log", get=False, **kw):
    def _process_kw(kw):
        pass

    pass


def sweep_std_w(wavelet, N, scales="log", get=False, **kw):
    def _process_kw(kw):
        pass

    pass


def sweep_harea(wavelet, N, scales="log", get=False, kw_w=None, kw_t=None):
    """Sub-.5 and near-0 areas will occur for very high scales as a result of
    discretization limitations. Zero-areas have one non-zero frequency-domain,
    and std_t==N/2, with latter more accurately set to infinity (which we don't).

    Sub-.5 are per freq-domain assymetries degrading time-domain decay,
    and limited bin discretization integrating unreliably (yet largely
    meaningfully; the unreliable-ness appears emergent from discretization).
    """
    pass


def wavelet_waveforms(wavelet, N, scale, zoom=True):
    pass


def _viz_cwt_scalebounds(
    wavelet, N, min_scale=None, max_scale=None, std_t=None, cutoff=1, stdevs=2, Nt=None
):
    """Can be used to visualize time & freq domains separately, where
    `min_scale` refers to scale at which to show the freq-domain wavelet, and
    `max_scale` the time-domain one.
    """

    def _viz_max(wavelet, N, max_scale, std_t, stdevs, Nt):
        pass

    def _viz_min(wavelet, N, min_scale, cutoff):
        pass

    pass


def wavelet_filterbank(
    wavelet,
    N=1024,
    scales="log",
    skips=0,
    title_append=None,
    positives=False,
    show=True,
    get=False,
):
    """Plot all frequency-domain wavelets, superposed.

    `skips=1` will plot every *other* wavelet, `=2` will skip 2, etc.
    `=0` shows all.

    `title_append`: will `title += title_append` if not None. Must be string.
    Can use to display additional info.

    `positives=True` will show full wavelets as opposed to trimmed at Nyquist.

    `get=True` to return the filter bank (ignores `skip`).
    """

    def _title():
        pass

    pass


def viz_cwt_higher_order(Wx_k, scales=None, wavelet=None, **imshow_kw):
    pass


def viz_gmw_orders(N=1024, n_orders=3, scale=5, gamma=3, beta=60, norm="bandpass"):
    pass


#### Visual tools ## messy code ##############################################
def imshow(
    data,
    title=None,
    show=1,
    cmap=None,
    norm=None,
    complex=None,
    abs=0,
    w=None,
    h=None,
    ridge=0,
    ticks=1,
    borders=1,
    aspect="auto",
    ax=None,
    fig=None,
    yticks=None,
    xticks=None,
    xlabel=None,
    ylabel=None,
    norm_scaling=1,
    **kw,
):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    ticks: False to not plot x & y ticks
    borders: False to not display plot borders
    w, h: rescale width & height
    norm_scaling: multiplies `norm`, even if `norm` is None (multiplies default)
    kw: passed to `plt.imshow()`

    others
    """
    pass


def plot(
    x,
    y=None,
    title=None,
    show=0,
    ax_equal=False,
    complex=0,
    abs=0,
    c_annot=False,
    w=None,
    h=None,
    dx1=False,
    xlims=None,
    ylims=None,
    vert=False,
    vlines=None,
    hlines=None,
    xlabel=None,
    ylabel=None,
    xticks=None,
    yticks=None,
    ax=None,
    fig=None,
    ticks=True,
    squeeze=True,
    auto_xlims=True,
    **kw,
):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    complex: plot `x.real` & `x.imag`; `2` to also plot `abs(x)`
    ticks: False to not plot x & y ticks
    w, h: rescale width & height
    kw: passed to `plt.imshow()`

    others
    """
    pass


def plots(
    X,
    Y=None,
    nrows=None,
    ncols=None,
    tight=True,
    sharex=False,
    sharey=False,
    skw=None,
    pkw=None,
    _scat=0,
    show=1,
    **kw,
):
    """Example:
    X = [[None, np.arange(xc, xc + wl)],
         [None, np.arange(xc + hop, xc + hop + wl)],
         None,
         None]
    Y = [[x, window],
         [x, window],
         xbuf[:, xbc],
         xbuf[:, xbc + 1]]
    pkw = [[{}]*2, [{}]*2, *[{'color': 'tab:green'}]*2]
    plots(X, Y, nrows=2, ncols=2, sharey='row', tight=tight, pkw=pkw)
    """

    def _process_args(X, Y, nrows, ncols, tight, skw, pkw, kw):
        pass

    pass


def scat(
    x,
    y=None,
    title=None,
    show=0,
    ax_equal=False,
    s=18,
    w=None,
    h=None,
    xlims=None,
    ylims=None,
    dx1=False,
    vlines=None,
    hlines=None,
    ticks=1,
    complex=False,
    abs=False,
    xlabel=None,
    ylabel=None,
    ax=None,
    fig=None,
    auto_xlims=True,
    **kw,
):
    pass


def plotscat(*args, **kw):
    pass


def hist(
    x,
    bins=500,
    title=None,
    show=0,
    stats=0,
    ax=None,
    fig=None,
    w=1,
    h=1,
    xlims=None,
    ylims=None,
    xlabel=None,
    ylabel=None,
):
    """Histogram. `stats=True` to print mean, std, min, max of `x`."""

    def _fmt(g):
        pass

    pass


def _vhlines(lines, kind="v", ax=None):
    pass


def _fmt(*nums):
    pass


def _ticks(xticks, yticks, ax):
    def fmt(ticks):
        pass

    pass


def _maybe_title(title, ax=None):
    pass


def _scale_plot(
    fig,
    ax,
    show=False,
    ax_equal=False,
    w=None,
    h=None,
    xlims=None,
    ylims=None,
    dx1=False,
    xlabel=None,
    ylabel=None,
    auto_xlims=True,
):
    pass


def _annotate(txt, xy=(0.85, 0.9), weight="bold", fontsize=16):
    pass


#############################################################################
from .wavelets import Wavelet, _xifn
from .wavelets import center_frequency, freq_resolution, time_resolution
from .utils.common import NOTE, _textwrap, p2up
from .utils.cwt_utils import process_scales, cwt_scalebounds, make_scales
from .utils.cwt_utils import infer_scaletype
from .utils.backend import asnumpy
