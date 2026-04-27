# -*- coding: utf-8 -*-
"""
Contains `GDEFAULTS`, global defaults dictionary, set in `ssqueezepy.configs.ini`.

The .ini is parsed into a dict, then values are retrieved internally by functions
via `gdefaults()`, which sets default values if keyword arguments weren't set
to original functions (or were set to `None`).

E.g. calling `wavelets.morlet()`, the function has `mu=None` signature, so `mu`
will be drawn from `configs.ini`, unless calling like `wavelets.morlet(mu=1)`.
"""
import os
import inspect
import logging

logging.basicConfig(format='')
WARN = lambda msg: logging.warning("WARNING: %s" % msg)
path = os.path.join(os.path.dirname(__file__), 'configs.ini')

try:
    import torch
    import cupy
except:
    torch, cupy = None, None


def gdefaults(module_and_obj=None, get_all=False, as_dict=None,
              default_order=False, **kw):
    """Fetches default arguments from `ssqueezepy/configs.ini` and fills them
    in `kw` where they're None (or always if `get_all=True`). See code comments.
    """
    pass


def _get_gdefaults():
    """Global defaults fetched from configs.ini."""
    def float_if_number(s):
        """If float works, so should int."""
        pass
    def process_special(s):
        pass
    def process_value(value):
        pass
    pass


def IS_PARALLEL():
    """Returns False if 'SSQ_PARALLEL' environment flag was set to '0', or
    if `parallel` in `configs.ini` is set to `0`; former overrides latter.
    """
    pass


def USE_GPU():
    pass


GDEFAULTS = _get_gdefaults()
