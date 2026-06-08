# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import importlib

from pytest import warns

import pyhgf.distribution


def test_distribution_module_is_deprecated():
    """Loading the deprecated distribution module should display the warning."""
    with warns(DeprecationWarning, match="deprecated"):
        importlib.reload(pyhgf.distribution)
