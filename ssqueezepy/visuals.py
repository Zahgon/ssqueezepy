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
    pass


def wavelet_tf_anim(wavelet, N=2048, scales=None, width=1.1, height=1,
                    savepath='wavanim.gif', testing=False):
    """This method computes same as `wavelet_tf` but for all scales at once,
    and animates 'intelligently'. See help(wavelet_tf).

    `scales=None` will default to 'log:minimal' with (.9*min_scale,
    0.25*max_scale). These are selected to show the wavelet a little outside of
    "well-behaved" range (without slashing max_scale, it's a lot outside such
    range). May not work for every wavelet or all of their configs.
    """
    pass


def wavelet_heatmap(wavelet, scales='log', N=2048):
    pass


def sweep_std_t(wavelet, N, scales='log', get=False, **kw):
    def _process_kw(kw):
        kw = kw.copy()  # don't change external dict
        defaults = dict(min_decay=1, max_mult=2, min_mult=2,
                        nondim=False, force_int=True)
        for k, v in kw.items():
            if k not in defaults:
                raise ValueError(f"unsupported kwarg '{k}'; must be one of: "
                                 + ', '.join(defaults))

        for k, v in defaults.items():
            kw[k] = kw.get(k, v)
        return kw

    kw = _process_kw(kw)
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    scales = process_scales(scales, N, wavelet)

    std_ts = np.zeros(scales.size)
    for i, scale in enumerate(scales):
        std_ts[i] = time_resolution(wavelet, scale=scale, N=N, **kw)

    title = "std_t [{}] vs log2(scales) | {} wavelet, {}".format(
        "nondim" if kw['nondim'] else "s/c-rad", wavelet.name, wavelet.config_str)
    hlines = ([N/2, N/4], dict(color='k', linestyle='--'))
    plot(np.log2(scales), std_ts, title=title, hlines=hlines, show=1)

    if get:
        return std_ts


def sweep_std_w(wavelet, N, scales='log', get=False, **kw):
    def _process_kw(kw):
        kw = kw.copy()  # don't change external dict
        defaults = dict(nondim=False, force_int=True)
        for k, v in kw.items():
            if k not in defaults:
                raise ValueError(f"unsupported kwarg '{k}'; must be one of: "
                                 + ', '.join(defaults))

        for k, v in defaults.items():
            kw[k] = kw.get(k, v)
        return kw

    kw = _process_kw(kw)
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    scales = process_scales(scales, N, wavelet)

    std_ws = np.zeros(scales.size)
    for i, scale in enumerate(scales):
        std_ws[i] = freq_resolution(wavelet, scale=scale, N=N, **kw)

    title = "std_w [{}] vs log2(scales) | {} wavelet, {}".format(
        "nondim" if kw['nondim'] else "s/c-rad", wavelet.name, wavelet.config_str)
    plot(np.log2(scales), std_ws, title=title, show=1)

    if get:
        return std_ws


def sweep_harea(wavelet, N, scales='log', get=False, kw_w=None, kw_t=None):
    """Sub-.5 and near-0 areas will occur for very high scales as a result of
    discretization limitations. Zero-areas have one non-zero frequency-domain,
    and std_t==N/2, with latter more accurately set to infinity (which we don't).

    Sub-.5 are per freq-domain assymetries degrading time-domain decay,
    and limited bin discretization integrating unreliably (yet largely
    meaningfully; the unreliable-ness appears emergent from discretization).
    """
    kw_w, kw_t = (kw_w or {}), (kw_t or {})
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    scales = process_scales(scales, N, wavelet)

    std_ws = sweep_std_w(wavelet, N, scales, get=True, **kw_w)
    plt.show()
    std_ts = sweep_std_t(wavelet, N, scales, get=True, **kw_t)
    plt.show()
    hareas = std_ws * std_ts

    hline = (.5, dict(color='tab:red', linestyle='--'))
    title = "(std_w * std_t) vs log2(scales) | {} wavelet, {}".format(
        wavelet.name, wavelet.config_str)
    plot(np.log2(scales), hareas, color='k', hlines=hline, title=title)
    plt.show()

    if get:
        return hareas, std_ws, std_ts


def wavelet_waveforms(wavelet, N, scale, zoom=True):
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    ## Freq-domain sampled #######################
    w_peak, _ = find_maximum(wavelet.fn)

    w_ct = np.linspace(0, w_peak*2, max(4096, p2up(N)[0]))  # 'continuous-time'
    w_dt = np.linspace(0, np.pi, N//2) * scale  # sampling pts at `scale`
    psih_ct = asnumpy(wavelet(w_ct))
    psih_dt = asnumpy(wavelet(w_dt))

    title = ("wavelet(w) sampled by xi at scale={:.2f}, N={} | {} wavelet, {}"
             ).format(scale, N, wavelet.name, wavelet.config_str)
    plot(w_ct, psih_ct, title=title, xlabel="radians")
    scat(w_dt, psih_dt, color='tab:red')

    plt.legend(["psih at scale=1", "sampled at scale=%.2f" % scale], fontsize=13)
    plt.axvline(w_peak, color='tab:red', linestyle='--')
    plt.show()

    ## Freq-domain #######################
    # if peak not near left, don't zoom; same as `if .. (w_peak >= w_dt.max())`
    if not zoom or (np.argmax(psih_dt) > .05 * N/2):
        end = None
    else:
        peak_idx = np.argmax(psih_dt)
        end = np.where(psih_dt[peak_idx:] < 1e-4*psih_dt.max())[0][0]
        end += peak_idx + 3  # +3: give few more indices for visual

    w_dtn = w_dt * (np.pi / w_dt.max())  # norm to span true w
    plot(w_dtn[:end], psih_dt[:end], xlabel="radians",
         title="Freq-domain waveform (psih)" + ", zoomed" * (end is not None))
    scat(w_dtn[:end], psih_dt[:end], color='tab:red', show=1)

    ## Time-domain #######################
    psi = asnumpy(wavelet.psifn(scale=scale, N=N))
    apsi = np.abs(psi)
    t = np.arange(-N/2, N/2, step=1)

    # don't zoom unless there's fast decay
    peak_idx = np.argmax(apsi)
    if not zoom or (apsi.max() / apsi[peak_idx:].min() <= 1e3):
        start, end = 0, None
    else:
        dt = np.where(apsi[peak_idx:] < 1e-3*apsi.max())[0][0]
        start, end = (N//2 - dt, N//2 + dt + 1)

    plot(t[start:end], psi[start:end], complex=1, xlabel="samples",
         title="Time-domain waveform (psi)" + ", zoomed" * (end is not None))
    plot(t[start:end], apsi[start:end], color='k', linestyle='--', show=1)


def _viz_cwt_scalebounds(wavelet, N, min_scale=None, max_scale=None,
                         std_t=None, cutoff=1, stdevs=2, Nt=None):
    """Can be used to visualize time & freq domains separately, where
    `min_scale` refers to scale at which to show the freq-domain wavelet, and
    `max_scale` the time-domain one.
    """
    def _viz_max(wavelet, N, max_scale, std_t, stdevs, Nt):
        if Nt is None:
            Nt = p2up(N)[0]
        if std_t is None:
            # permissive max_mult to not crash visual
            std_t = time_resolution(wavelet, max_scale, N, nondim=False,
                                    min_mult=2, max_mult=2, min_decay=1)

        t = np.arange(-Nt/2, Nt/2, step=1)
        t -= t.mean()
        psi = asnumpy(wavelet.psifn(scale=max_scale, N=len(t)))

        plot(t, np.abs(psi)**2, ylims=(0, None),
             title="|Time-domain wavelet|^2, extended (outside dashed)")

        plt.axvline(std_t,          color='tab:red')
        plt.axvline(std_t * stdevs, color='tab:green')
        # mark target (non-extended) frame
        _ = [plt.axvline(v, color='k', linestyle='--') for v in (-N/2, N/2-1)]

        _kw = dict(fontsize=16, xycoords='axes fraction', weight='bold')
        plt.annotate("1 stdev",
                     xy=(.88, .95), color='tab:red',   **_kw)
        plt.annotate("%s stdevs" % stdevs,
                     xy=(.88, .90), color='tab:green', **_kw)
        plt.show()

    def _viz_min(wavelet, N, min_scale, cutoff):
        w = _xifn(1, N)[:N//2 + 1]  # drop negative freqs
        psih = asnumpy(wavelet(min_scale * w, nohalf=True))
        _, mx = find_maximum(wavelet)

        plot(w, psih, title=("Frequency-domain wavelet, positive half "
                             "(cutoff=%s, peak=%.3f)" % (cutoff, mx)))
        plt.axhline(mx * abs(cutoff), color='tab:red')
        plt.show()

    if min_scale is not None:
        _viz_min(wavelet, N, min_scale, cutoff)
    if max_scale is not None:
        _viz_max(wavelet, N, max_scale, std_t, stdevs, Nt)
    if not (min_scale or max_scale):
        raise ValueError("Must set at least one of `min_scale`, `max_scale`")


def wavelet_filterbank(wavelet, N=1024, scales='log', skips=0, title_append=None,
                       positives=False, show=True, get=False):
    """Plot all frequency-domain wavelets, superposed.

    `skips=1` will plot every *other* wavelet, `=2` will skip 2, etc.
    `=0` shows all.

    `title_append`: will `title += title_append` if not None. Must be string.
    Can use to display additional info.

    `positives=True` will show full wavelets as opposed to trimmed at Nyquist.

    `get=True` to return the filter bank (ignores `skip`).
    """
    pass


def viz_cwt_higher_order(Wx_k, scales=None, wavelet=None, **imshow_kw):
    pass


def viz_gmw_orders(N=1024, n_orders=3, scale=5, gamma=3, beta=60,
                   norm='bandpass'):
    pass


#### Visual tools ## messy code ##############################################
def imshow(data, title=None, show=1, cmap=None, norm=None, complex=None, abs=0,
           w=None, h=None, ridge=0, ticks=1, borders=1, aspect='auto', ax=None,
           fig=None, yticks=None, xticks=None, xlabel=None, ylabel=None,
           norm_scaling=1, **kw):
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
    # axes
    if (ax or fig) and complex:
        NOTE("`ax` and `fig` ignored if `complex`")
    if complex:
        fig, ax = plt.subplots(1, 2)
    else:
        ax  = ax  or plt.gca()
        fig = fig or plt.gcf()

    # norm
    if norm is None:
        mx = np.max(np.abs(data))
        vmin, vmax = ((-mx, mx) if not abs else
                      (0, mx))
    else:
        vmin, vmax = norm
    vmin *= norm_scaling
    vmax *= norm_scaling

    # colormap
    import matplotlib as mpl
    mpl33 = tuple(map(int, mpl.__version__.split('.')[:2])) >= (3, 3)
    if cmap is None:
        cmap = (('turbo' if mpl33 else 'jet') if abs else
                'bwr')
    elif cmap == 'turbo':
        if not mpl33:
            from .utils import WARN
            WARN("'turbo' colormap requires matplotlib>=3.3; using 'jet' instead")
            cmap = 'jet'

    _kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect, **kw)

    if abs:
        ax.imshow(np.abs(data), **_kw)
    elif complex:
        ax[0].imshow(data.real, **_kw)
        ax[1].imshow(data.imag, **_kw)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            wspace=0, hspace=0)
    else:
        ax.imshow(data.real, **_kw)

    if w or h:
        fig.set_size_inches(12 * (w or 1), 12 * (h or 1))

    if ridge:
        data_mx = np.where(np.abs(data) == np.abs(data).max(axis=0))
        ax.scatter(data_mx[1], data_mx[0], color='r', s=4)

    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks, ax)
    if not borders:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
    if xlabel is not None:
        ax.set_xlabel(xlabel, weight='bold', fontsize=15)
    if ylabel is not None:
        ax.set_ylabel(ylabel, weight='bold', fontsize=15)

    _maybe_title(title, ax=ax)
    if show:
        plt.show()


def plot(x, y=None, title=None, show=0, ax_equal=False, complex=0, abs=0,
         c_annot=False, w=None, h=None, dx1=False, xlims=None, ylims=None,
         vert=False, vlines=None, hlines=None, xlabel=None, ylabel=None,
         xticks=None, yticks=None, ax=None, fig=None, ticks=True, squeeze=True,
         auto_xlims=True, **kw):
    """
    norm: color norm, tuple of (vmin, vmax)
    abs: take abs(data) before plotting
    complex: plot `x.real` & `x.imag`; `2` to also plot `abs(x)`
    ticks: False to not plot x & y ticks
    w, h: rescale width & height
    kw: passed to `plt.imshow()`

    others
    """
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if auto_xlims is None:
        auto_xlims = bool((x is not None and len(x) != 0) or
                          (y is not None and len(y) != 0))

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        y = y if isinstance(y, list) or not squeeze else y.squeeze()
        x = np.arange(len(y))
    elif y is None:
        x = x if isinstance(x, list) or not squeeze else x.squeeze()
        y = x
        x = np.arange(len(x))
    x = x if isinstance(x, list) or not squeeze else x.squeeze()
    y = y if isinstance(y, list) or not squeeze else y.squeeze()

    if vert:
        x, y = y, x
    if complex:
        ax.plot(x, y.real, color='tab:blue', **kw)
        ax.plot(x, y.imag, color='tab:orange', **kw)
        if complex == 2:
            ax.plot(x, np.abs(y), color='k', linestyle='--', **kw)
        if c_annot:
            _kw = dict(fontsize=15, xycoords='axes fraction', weight='bold')
            ax.annotate("real", xy=(.93, .95), color='tab:blue', **_kw)
            ax.annotate("imag", xy=(.93, .90), color='tab:orange', **_kw)
    else:
        if abs:
            y = np.abs(y)
        ax.plot(x, y, **kw)
    if dx1:
        ax.set_xticks(np.arange(len(x)))

    # styling
    if vlines:
        _vhlines(vlines, kind='v', ax=ax)
    if hlines:
        _vhlines(hlines, kind='h', ax=ax)
    if abs and ylims is None:
        ylims = (0, None)

    ticks = ticks if isinstance(ticks, (list, tuple)) else (ticks, ticks)
    if not ticks[0]:
        ax.set_xticks([])
    if not ticks[1]:
        ax.set_yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks, ax)
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks, ax)

    _maybe_title(title, ax=ax)
    _scale_plot(fig, ax, show=show, ax_equal=ax_equal, w=w, h=h,
                xlims=xlims, ylims=ylims, dx1=(len(x) if dx1 else 0),
                xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims)


def plots(X, Y=None, nrows=None, ncols=None, tight=True, sharex=False,
          sharey=False, skw=None, pkw=None, _scat=0, show=1, **kw):
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
        X = X if isinstance(X, list) else [X]
        Y = Y if isinstance(Y, list) else [Y]
        skw = skw or {}
        pkw = pkw or [{}] * len(X)

        if nrows is None and ncols is None:
            nrows, ncols = len(X), 1
        elif nrows is None:
            nrows = max(len(X) // ncols, 1)
        elif ncols is None:
            ncols = max(len(X) // nrows, 1)

        default = dict(left=0, right=1, bottom=0, top=1, hspace=.1, wspace=.05)
        if tight:
            if not isinstance(tight, dict):
                tight = default.copy()
            else:
                for name in default:
                    if name not in tight:
                        tight[name] = default[name]

        kw['w'] = kw.get('w', .8)
        kw['h'] = kw.get('h', .8)  # default 'tight' enlarges plot
        return X, Y, nrows, ncols, tight, skw, pkw, kw

    X, Y, nrows, ncols, tight, skw, pkw, kw = _process_args(
        X, Y, nrows, ncols, tight, skw, pkw, kw)

    _, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey, **skw)
    for ax, x, y, _pkw in zip(axes.flat, X, Y, pkw):
        if isinstance(x, list):
            for _x, _y, __pkw in zip(x, y, _pkw):
                plot(_x, _y, ax=ax, **__pkw, **kw)
                if _scat:
                    scat(_x, _y, ax=ax, **__pkw, **kw)
        else:
            plot(x, y, ax=ax, **_pkw, **kw)
            if _scat:
                scat(x, y, ax=ax, **_pkw, **kw)

    if tight:
        plt.subplots_adjust(**tight)
    if show:
        plt.show()


def scat(x, y=None, title=None, show=0, ax_equal=False, s=18, w=None, h=None,
         xlims=None, ylims=None, dx1=False, vlines=None, hlines=None, ticks=1,
         complex=False, abs=False, xlabel=None, ylabel=None, ax=None, fig=None,
         auto_xlims=True, **kw):
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if auto_xlims is None:
        auto_xlims = bool((x is not None and len(x) != 0) or
                          (y is not None and len(y) != 0))

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        x = np.arange(len(y))
    elif y is None:
        y = x
        x = np.arange(len(x))

    if complex:
        ax.scatter(x, y.real, s=s, **kw)
        ax.scatter(x, y.imag, s=s, **kw)
    else:
        if abs:
            y = np.abs(y)
        ax.scatter(x, y, s=s, **kw)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    _maybe_title(title, ax=ax)
    if vlines:
        _vhlines(vlines, kind='v', ax=ax)
    if hlines:
        _vhlines(hlines, kind='h', ax=ax)
    _scale_plot(fig, ax, show=show, ax_equal=ax_equal, w=w, h=h,
                xlims=xlims, ylims=ylims, dx1=(len(x) if dx1 else 0),
                xlabel=xlabel, ylabel=ylabel, auto_xlims=auto_xlims)


def plotscat(*args, **kw):
    pass


def hist(x, bins=500, title=None, show=0, stats=0, ax=None, fig=None,
         w=1, h=1, xlims=None, ylims=None, xlabel=None, ylabel=None):
    """Histogram. `stats=True` to print mean, std, min, max of `x`."""
    pass


def _vhlines(lines, kind='v', ax=None):
    lfn = getattr(plt if ax is None else ax, f'ax{kind}line')

    if not isinstance(lines, (list, tuple)):
        lines, lkw = [lines], {}
    elif isinstance(lines, (list, np.ndarray)):
        lkw = {}
    elif isinstance(lines, tuple):
        lines, lkw = lines
        lines = lines if isinstance(lines, (list, np.ndarray)) else [lines]
    else:
        raise ValueError("`lines` must be list or (list, dict) "
                         "(got %s)" % lines)

    for line in lines:
        lfn(line, **lkw)


def _fmt(*nums):
    pass

def _ticks(xticks, yticks, ax):
    def fmt(ticks):
        if all(isinstance(h, str) for h in ticks):
            return "%s"
        return ("%.d" if all(float(h).is_integer() for h in ticks) else
                "%.2f")

    if yticks is not None:
        if not hasattr(yticks, '__len__') and not yticks:
            ax.set_yticks([])
        else:
            idxs = np.linspace(0, len(yticks) - 1, 8).astype('int32')
            yt = [fmt(yticks) % h for h in np.asarray(yticks)[idxs]]
            ax.set_yticks(idxs)
            ax.set_yticklabels(yt)
    if xticks is not None:
        if not hasattr(xticks, '__len__') and not xticks:
            ax.set_xticks([])
        else:
            idxs = np.linspace(0, len(xticks) - 1, 8).astype('int32')
            xt = [fmt(xticks) % h for h in np.asarray(xticks)[idxs]]
            ax.set_xticks(idxs)
            ax.set_xticklabels(xt)

def _maybe_title(title, ax=None):
    if title is None:
        return

    title, kw = (title if isinstance(title, tuple) else
                 (title, {}))
    defaults = gdefaults('visuals._maybe_title', get_all=True, as_dict=True)
    for name in defaults:
        kw[name] = kw.get(name, defaults[name])

    if ax:
        ax.set_title(str(title), **kw)
    else:
        plt.title(str(title), **kw)


def _scale_plot(fig, ax, show=False, ax_equal=False, w=None, h=None,
                xlims=None, ylims=None, dx1=False, xlabel=None, ylabel=None,
                auto_xlims=True):
    if xlims:
        ax.set_xlim(*xlims)
    elif auto_xlims:
        xmin, xmax = ax.get_xlim()
        rng = xmax - xmin
        ax.set_xlim(xmin + .018 * rng, xmax - .018 * rng)

    if ax_equal:
        yabsmax = max(np.abs([*ax.get_ylim()]))
        mx = max(yabsmax, max(np.abs([xmin, xmax])))
        ax.set_xlim(-mx, mx)
        ax.set_ylim(-mx, mx)
        fig.set_size_inches(8*(w or 1), 8*(h or 1))
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)
    if dx1:
        plt.xticks(np.arange(dx1))
    if w or h:
        fig.set_size_inches(14*(w or 1), 8*(h or 1))
    if xlabel is not None:
        plt.xlabel(xlabel, weight='bold', fontsize=15)
    if ylabel is not None:
        plt.ylabel(ylabel, weight='bold', fontsize=15)
    if show:
        plt.show()


def _annotate(txt, xy=(.85, .9), weight='bold', fontsize=16):
    pass


#############################################################################
from .wavelets import Wavelet, _xifn
from .wavelets import center_frequency, freq_resolution, time_resolution
from .utils.common import NOTE, _textwrap, p2up
from .utils.cwt_utils import process_scales, cwt_scalebounds, make_scales
from .utils.cwt_utils import infer_scaletype
from .utils.backend import asnumpy
