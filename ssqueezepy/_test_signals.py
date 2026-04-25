# -*- coding: utf-8 -*-
"""
Signals for testing effectiveness of time-frequency transforms against
variety of localization characteristics.

1. **sine**: pure sine or cosine at one frequency, `cos(2pi f t)`
    a. sine
    b. cosine
    c. phase-shifted
    d. trimmed (others complete exactly one cycle) (not implemented but is
       trivial; do e.g. `x = x[20:-50]`)

2. **<name>:am**: <name> with amplitude modulation, i.e. `A(t) * fn(t)`
    a. |sine|
    b. |cosine|
    c. exp
    d. gauss

3. **#<name>**: superimpose reflected <name> onto itself, i.e. `x += x[::-1]`

4. **lchirp**: linear chirp, `cos(2pi t**2/2)`, spanning `fmin` to `fmax`

5. **echirp**: exponential chirp, `cos(2pi exp(t))`, spanning `fmin` to `fmax`

6. **hchirp**: hyperbolic chirp, `cos(2pi a/(b - t))`, spanning `fmin` to `fmax`

7, 8, 9: **par_lchirp, par_echirp, par_hchirp**: linear, exponential, hyperbolic
         chirps, superposed, with frequency modulation in parallel,
         spanning `fmin1` to `fmax1` and `fmin2` to `fmax2`.

10. **jumps**: large instant frequency transitions, `cos(2pi f*t), f=2 -> f=100`

11. **packed**: closely-spaced bands of sinusoids with majority overlap, e.g.
                `cos(w*t[No:]) + cos((w+1)*t[-No:]) + cos((w+3)*t[No:]) + ...`,
                `No = .8*len(t)`.

12. **packed_poly**: closely-packed polynomial frequency modulations
                (non-configurable)
                Generates https://www.desmos.com/calculator/swbhgezpjk with A.M.

13. **poly_cubic**: cubic polynomial frequency variation + pure tone
               (non-configurable)
"""
import inspect
import numpy as np
import scipy.signal as sig
from numpy.fft import rfft

from . import plt
from ._ssq_cwt import ssq_cwt
from ._ssq_stft import ssq_stft
from .utils import WARN, _textwrap
from .wavelets import Wavelet
from .visuals import plot, plots, imshow
from .ridge_extraction import extract_ridges


pi = np.pi
DEFAULT_N = 512
DEFAULT_SNR = None
DEFAULT_SEED = None
DEFAULT_ARGS = {
    'cosine': dict(f=64, phi0=0),
    'sine':   dict(f=64, phi0=0),
    'lchirp': dict(tmin=0, tmax=1, fmin=0, fmax=None),
    'echirp': dict(tmin=0, tmax=1, fmin=1, fmax=None),
    'hchirp': dict(tmin=0, tmax=1, fmin=1, fmax=None),
    'jumps':  dict(),
    'low':    dict(),
    'am-cosine': dict(amin=.1),
    'am-sine':   dict(amin=.1),
    'am-exp':    dict(amin=.1),
    'am-gauss':  dict(amin=.01),
    'sine:am-cosine': (dict(f=16), dict(amin=.5)),
}
DEFAULT_TKW = dict(tmin=0, tmax=1, endpoint=True)


#### Test signals ############################################################
class TestSignals():
    """Signals of varying time-frequency characteristics. Convenience methods
    to plot multiple signals and their transforms under varying wavelet / window
    parameters.

    `.demo(signals)` to visualize `signals`, `test_transforms(fn)` to apply `fn`
    to and visualize output.

    See `examples/` on Github, and
    https://overlordgolddragon.github.io/test-signals/

    Also see `help(ssqueezepy._test_signals)`, `TestSignals.SUPPORTED`,
    `TestSignals.DEMO`.

    **Sweep functions**
        For `lchirp`, `echirp`, & `hchirp`, `N` will be determined automatically
        if `tmin`, `tmax`, `fmin`, and `fmax` are provided, minimally such that
        no aliasing occurs.

    **Demo signals**
        `TestSignals.DEMO` holds list of `signals` names invoked when passing
        `signals='all'`, which can be changed.

    # Arguments
        N: int
            Will use this as default `N` anytime `N` is left unspecified.

        snr: float / None
            If not None, will add random normal (White Gaussian) noise to
            signal of SNR `snr` - computed as:
                SNR = 10*log10(xo_var / noise_var)
                noise_var = xo_var / 10^(SNR/10)
            where `xo_var` = unnoised signal variance.

        default_args: dict
            `{<signal_name>: {'param_name': value}}` pairs, where `signal_name`
            is one of `SUPPORTED`. See `test_signals.DEFAULT_ARGS`.

        default_tkw: dict
            Example with all key-value pairs: `dict(tmin=0, tmax=1)`.

        warn_alias: bool (default True)
            Whether to print warning if generated signal aliases (f > fs/2);
            to disable, pass `warn_alias=False` to `__init__()`, or set directly
            on instance (`TestSignals().warn_alias=False`).

        seed: int / None
            If not None, will `np.random.seed(seed)` before applying `snr` noise.
    """
    SUPPORTED = ['cosine', 'sine', 'lchirp', 'echirp', 'echirp_pc', 'hchirp',
                 'par-lchirp', 'par-echirp', 'par-hchirp', 'jumps', 'packed',
                 'packed-poly', 'poly-cubic',
                 'am-sine', 'am-cosine', 'am-exp', 'am-gauss']
    # what to show with `signal='all'`, and in what order
    DEMO = ['cosine', 'sine',
            'lchirp', 'echirp', 'hchirp',
            '#lchirp', '#echirp', '#hchirp',
            'par-lchirp', 'par-echirp', 'par-hchirp', '#par-lchirp',
            'jumps', 'packed', 'packed-poly', 'poly-cubic',
            'am-sine', 'am-cosine', 'am-exp', 'am-gauss']

    def __init__(self, N=None, snr=None, default_args=None, default_tkw=None,
                 warn_alias=True, seed=None):
        self.N = N    or DEFAULT_N
        self.snr = snr or DEFAULT_SNR
        self.default_args = default_args or DEFAULT_ARGS
        self.default_tkw  = default_tkw  or DEFAULT_TKW
        self.warn_alias   = warn_alias
        self.seed = seed or DEFAULT_SEED

        # set defaults on unspecified
        for k, v in DEFAULT_ARGS.items():
            self.default_args[k] = self.default_args.get(k, v)
        for k, v in DEFAULT_TKW.items():
            self.default_tkw[k] = self.default_tkw.get(k, v)

    #### test signals ########################################################
    def _maybe_warn_alias(self, phi, tol=.02):
        # allow non-trivial overshoot as it may occur but not worth warning
        pass

    def sine(self, N=None, f=1, phi0=0, **tkw):
        """sin(2pi*f*t + phi)"""
        pass

    def cosine(self, N=None, f=1, phi0=0, **tkw):
        """cos(2pi*f*t + phi)"""
        pass

    def _generate(self, fn, N, fmin, fmax, **tkw):
        """Used by chirps."""
        pass

    def lchirp(self, N=None, fmin=0, fmax=None, **tkw):
        """
        >>>   f(t) = a*t + b
        >>> phi(t) = (a/2)*(t^2 - tmin^2) + b*(t - tmin)
        >>> a = (fmin - fmax) / (tmin - tmax)
            b = (fmin*tmax - fmax*tmin) / (tmax - tmin)
        """
        pass

    def _lchirp_fn(self, t, tmin, tmax, fmin, fmax, get_w=False):
        pass

    def echirp(self, N=None, fmin=1, fmax=None, **tkw):
        """
        >>> f(t)   = a*b^t
        >>> phi(t) = (a/ln(b)) * (b^t - b^tmin)
        >>> a = (fmin^tmax / fmax^tmin) ^ 1/(tmax - tmin)
            b = fmax^(1/tmax) * (1/a)^(1/tmax)
        """
        pass

    def _echirp_fn(self, t, tmin, tmax, fmin, fmax, get_w=False):
        pass

    def echirp_pc(self, N=None, fmin=0, fmax=None, **tkw):
        """Alternate design that keeps f'(t) fixed at `e`, but is no longer
        geometric in the sense `f(t2) / f(t1) = const.`. "echirp plus constant".

        >>> f(t)   = a*exp(t) + b
        >>> phi(t) = a*(exp(t) - exp(tmin)) + b*(t - tmin)
        >>> a = (fmax - fmin)/(exp(tmax) - exp(tmin))
            b = (fmin*exp(tmax) - fmax*exp(tmin)) / (exp(tmax) - exp(tmin))
        """
        pass

    def _echirp_pc_fn(self, t, tmin, tmax, fmin, fmax, get_w=False):
        pass

    def hchirp(self, N=None, fmin=.1, fmax=None, **tkw):
        """
        >>> f(t)   = A / (B - t)^2
        >>> phi(t) = A * (1/(B - t) + 1/(tmin - B))
        >>> a, b, c, d = fmin, fmax, tmin, tmax
            A = AN / AD, B = BN / BD,
            AN = 2*sqrt(a^3*b^3*(c - d)^4) + a^2*b*(c - d)^2 + a*b^2*(c - d)^2
            AD = (a - b)^2
            BN = sqrt(a^3*b^3*(c-d)^4) + a^2*b*c*(c-d) + a*b^2*d*(d - c)
            BD = a*b*(a - b)*(c - d)
        """
        pass

    def _hchirp_fn(self, t, tmin, tmax, fmin, fmax, get_w=False):
        pass

    def par_lchirp(self, N=None, fmin1=None, fmax1=None, fmin2=None, fmax2=None,
                   **tkw):
        """Linear frequency modulation in parallel. Should have
        `fmax2 > fmax1`, `fmin2 > fmin1`, and shared `tmin`, `tmax`.
        """
        pass

    def par_echirp(self, N=None, fmin1=None, fmax1=None, fmin2=None, fmax2=None,
                   **tkw):
        """Exponential frequency modulation in parallel. Should have
        `fmax2 > fmax1`, `fmin2 > fmin1`, and shared `tmin`, `tmax`.
        """
        pass

    def par_hchirp(self, N=None, fmin1=None, fmax1=None, fmin2=None, fmax2=None,
                   **tkw):
        """Hyperbolic frequency modulation in parallel. Should have
        `fmax2 > fmax1`, `fmin2 > fmin1`, and shared `tmin`, `tmax`.
        """
        pass

    def am_sine(self, N=None, f=1, amin=0, amax=1, phi=0, **tkw):
        """Sine amplitude modulation, `|sin(w) + 1| / 2`."""
        pass

    def am_cosine(self, N=None, f=1, amin=0, amax=1, phi=0, **tkw):
        """Cosine amplitude modulation, `|cos(w) + 1| / 2`."""
        pass

    def am_exp(self, N=None, amin=.1, amax=1, **tkw):
        """Uses `echirp`'s expression for `f(t)`."""
        pass

    def am_gauss(self, N=None, amin=.1, amax=1, **tkw):
        """Gaussian centered at center sample (`N/2`)."""
        pass

    def jumps(self, N=None, freqs=None, **tkw):
        """Large instant freq transitions, e.g. `cos(2pi f*t), f=2 -> f=100`."""
        pass

    def packed(self, N=None, freqs=None, overlap=.8, **tkw):
        """Closely-spaced bands of sinusoids with majority overlap, e.g.
            `cos(w*t[No:]) + cos((w+1)*t[-No:]) + cos((w+3)*t[No:]) + ...`,
            `No = .8*len(t)`.
        """
        pass

    def packed_poly(self, N=None, **tkw):
        """Closely-packed polynomial frequency modulations (non-configurable;
        adjusts with N to keep bands approx unmoved in time-frequency plane).

        Generates https://www.desmos.com/calculator/swbhgezpjk with A.M.
        """
        pass

    def poly_cubic(self, N=None, **tkw):
        """Cubic polynomial frequency variation + pure tone (non-configurable;
        adjusts with N to keep bands approx unmoved in time-frequency plane).
        """
        pass

    #### Test functions ######################################################
    def demo(self, signals='all', N=None, dft=None):
        """Plots signal waveforms, and optionally their DFTs.

        # Arguments:
            signals: str / [str] / [(str, dict)]
                'all' will set `signals = TestSignals.DEMO`, and plot in
                that order. Else, strings must be in `TestSignals.SUPPORTED`.
                Can also be `(str, dict)` pairs in a list, dict passed as
                keyword arguments to the generating function.
                Also see `help(ssqueezepy._test_signals)`, and
                `help(TestSignals.make_signals)`.

            N: int
                Length (# of samples) of generated signals.

            dft: None / str['rows', 'cols']
                If not None, will also plot DFT of each signal along the signal.
                If `'cols'`, will stack horizontally - if `'rows'`, vertically.
        """
        pass

    def test_transforms(self, fn, signals='all', N=None):
        """Make `fn` return `None` to skip visuals (e.g. if already done by `fn`).

        Input signature is `fn(x, t, params, ...)`, where
        `params = (name, fparams, aparams)`. Output, if not None, must be
        `(Tf, pkw)`, where `Tf` is a 2D np.ndarray time-frequency transform,
        and `pkw` is keyword arguments to `ssqueezepy.visuals.imshow`
        (can be empty dict).

        Also see `help(ssqueezepy._test_signals)`, and
        `help(TestSignals.make_signals)`.
        """
        pass

    #### utils ###############################################################
    def make_signals(self, signals='all', N=None, get_params=False):
        """Generates `signals` signals of length `N`.

        Returns list of signals `[x0, x1, ...]` (or if `get_params`, dictionary
        of `{name: x, t, (fparams, aparams)}`), where `x` is the signal,
        `t` is its time vector, `fparams` is a dict of keyword argsto the carrier,
        and `aparams` to the amplitude modulator (if applicable, e.g.
        `lchirp:am-sine').
        `fparams` may additionally contain a special kwarg: `snr`, not passed to
        carrier `fn`, that adds random normal noise of SNR `snr` to signal.

        Also see `help(ssqueezepy._test_signals)`.
        """
        pass

    @classmethod
    def _title(self, signal, N, fparams, aparams, x=None, wrap_len=70):
        pass

    @staticmethod
    def _process_varname_alias(signal, N, fparams):
        pass

    def _process_params(self, N, tkw, fn=None, fmin=None, fmax=None):
        pass

    def _est_N_nonalias(self, f_fn, tmin, tmax, fmin, fmax):
        """Find smallest `N` (number of samples) such that signal generated
        from `tmin` to `tmax` will not alias.

        https://dsp.stackexchange.com/a/72942/50076

        max_phi_increment = fmax_fn * (t[1] - t[0])
        t[1] - t[0] = (tmax - tmin) / (N - 1)  [[endpoint=True]]
        max_phi_increment = pi
        fmax_fn * (tmax - tmin) / (N - 1) = pi
        1 + fmax_fn * (tmax - tmin) / pi = N
        """
        pass

    def _process_input(self, signals):
        """
        `signals`:
            - Ensure is string, or list/tuple of strings or of lists/tuples,
            each list/tuple being a (str, dict) or (str, (dict, dict)) pair.
            - Ensure each string is in `SUPPORTED`, and has an accompanying
            `params` pair (if not, set from `defalt_args`).
            - Loads parameters into two separate dictionaries, one for
            'carrier' / base function, other for (amplitude) 'modulator'.
            Defaults loaded according to precedence: `name:am-name` overrides
            `name` and `am-name`, but latter two are used if former isn't set.
        """
        def raise_type_error(signal):
            raise TypeError("all tuple or list elements of `signals` "
                            "must be (str, dict) or (str, (dict, dict)) pairs "
                            "(got (%s))" % ', '.join(
                                map(lambda s: type(s).__name__, signal)))

        if isinstance(signals, (str, tuple)):
            if signals != 'all':
                signals = [signals]
        elif not isinstance(signals, list):
            raise TypeError("`signals` must be string, list, or tuple "
                            "(got %s)" % type(signals))

        if isinstance(signals, list):
            for signal in signals:
                if isinstance(signal, str):
                    if ':' in signal:
                        fname, aname = signal.split(':')
                    else:
                        fname, aname = signal, ''
                    fname = fname.lstrip('#')

                    for name in (fname, aname):
                        if name != '' and name not in self.SUPPORTED:
                            raise ValueError(f"'{name}' is not supported; "
                                             "must be one of: "
                                             + ", ".join(self.SUPPORTED))
                elif isinstance(signal, (list, tuple)):
                    if not (isinstance(signal[0], str) and
                            isinstance(signal[1], (dict, list, tuple))):
                        raise_type_error(signal)
                    elif (isinstance(signal[1], (list, tuple)) and
                          not (isinstance(signal[1][0], dict) and
                               isinstance(signal[1][1], dict))):
                        raise_type_error(signal)
                else:
                    raise TypeError("all elements of `signals` must be string, "
                                    "or tuple or list of (string, dict) or "
                                    "(string, (dict, dict)) pairs "
                                    "(found %s)" % type(signal))

        if signals == 'all':
            signals = self.DEMO.copy()
        elif not isinstance(signals, (list, tuple)):
            signals = [signals]

        names, params_all = [], []
        for signal in signals:
            if isinstance(signal, (tuple, list)):
                name, params = signal
                if isinstance(params, (list, tuple)):
                    fparams, aparams = params
                else:
                    fparams, aparams = params, {}
            else:
                name, fparams, aparams = signal, {}, {}

            if name[0] == '#':
                add_reversed = True
                name = name[1:]
            else:
                add_reversed = False

            if 'am-' in name:
                if name.startswith('am-'):
                    if name.endswith(':'):
                        name = name.rstrip(':')
                    fname, aname = 'cosine', name
                    defaults = (self.default_args.get(fname, {}),
                                self.default_args.get(aname, {}))
                    name = fname + ':' + aname
                else:
                    defaults = self.default_args.get(name, {})
                    fname, aname = name.split(':')

                if isinstance(defaults, (list, tuple)):
                    fdefaults, adefaults = defaults
                elif isinstance(defaults, dict) and defaults != {}:
                    fdefaults, adefaults = defaults, {}
                else:
                    fdefaults, adefaults = self.default_args.get(fname, {}), {}

                if adefaults == {}:
                    adefaults = self.default_args.get(aname, {})

                for k, v in fdefaults.items():
                    fparams[k] = fparams.get(k, v)
                for k, v in adefaults.items():
                    aparams[k] = aparams.get(k, v)

                if name.startswith('am-'):
                    fdefaults, adefaults = adefaults, fdefaults
            else:
                for k, v in self.default_args.get(name, {}).items():
                    fparams[k] = fparams.get(k, v)

            if add_reversed:
                name = '#' + name
            names.append(name)
            params_all.append([fparams, aparams])

        # store latest result for debug purposes
        self._names = names
        self._params_all = params_all
        return names, params_all

    #### prebuilt test methods ##############################################
    def wavcomp(self, wavelets, signals='all', N=None, w=1.2, h=None,
                tight_kw=None):
        """Plots CWT & SSQ_CWT taken with `wavelets` wavelets side by side,
        vertically.
        """
        pass

    def _wavcomp_fn(self, x, t, params, wavelets, w=1.2, h=None, tight_kw=None):
        pass

    def cwt_vs_stft(self, wavelet, window, signals='all', N=None,
                    win_len=None, n_fft=None, window_name=None, config_str='',
                    w=1.2, h=.9, tight_kw=None):
        """Plots CWT & SSQ_CWT, and STFT & SSQ_STFT of `signals` taken with
        `wavelet` and `window` along the rest of parameters.

        `window_name` & `config_str` are used to title STFT plots. `w` & `h`
        control plots' width & height. `tight_kw` is passed to
        `plt.subplots_adjust()`.
        """
        pass

    def _cwt_vs_stft_fn(self, x, t, params, wavelet, window, win_len=None,
                        n_fft=None, window_name=None, config_str='', w=1.2, h=.9,
                        tight_kw=None):
        pass

    @staticmethod
    def _title_cwt(wavelet, name, x, fparams, aparams, wrap_len=53):
        pass

    @staticmethod
    def _title_stft(window, name, x, fparams, aparams, win_len=None, n_fft=None,
                    window_name='', config_str='', wrap_len=53):
        pass

    def ridgecomp(self, signals='all', N=None, penalty=20, n_ridges=2, bw=None,
                  transform='cwt', w=1.2, h=.4, **transform_kw):
        """Plots extracted ridges from a CWT or STFT and them SSQ'd of `signals`,
        superimposed on the transform itself, passing in `transform_kw` to
        `ssq_cwt` or `ssq_stft`. `w` & `h` control plots' width & height.

        See `help(ridge_extraction.extract_ridges)`.
        """
        pass

    def _ridgecomp_fn(self, x, t, params, penalty=20, n_ridges=2, bw=None,
                      transform='cwt', w=1.2, h=.4, **transform_kw):
        pass


def _t(tmin, tmax, N, endpoint=False):
    pass
