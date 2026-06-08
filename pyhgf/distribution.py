# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Deprecated module.

The :mod:`pyhgf.distribution` module is deprecated, and its content has been removed. It
is recommended to create the log probability function and use pytensor.warap_jax
instead. Please refer to the main documentation for examples. Importing from this module
will raise a :class:`DeprecationWarning`, and using any of the objects it used to expose
will raise a :class:`NotImplementedError`.
"""

import warnings

_DEPRECATION_MESSAGE = (
    "The `pyhgf.distribution` module is deprecated and will be removed in a "
    "future release. Its content (`logp`, `hgf_logp`, `HGFLogpGradOp`, "
    "`HGFDistribution` and `HGFPointwise`) is no longer available."
)

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)


def logp(*args, **kwargs):
    """Raise an error; this function is deprecated.

    See :data:`_DEPRECATION_MESSAGE`.
    """
    raise NotImplementedError(_DEPRECATION_MESSAGE)


def hgf_logp(*args, **kwargs):
    """Raise an error; this function is deprecated.

    See :data:`_DEPRECATION_MESSAGE`.
    """
    raise NotImplementedError(_DEPRECATION_MESSAGE)


class HGFLogpGradOp:
    """Deprecated.

    See :data:`_DEPRECATION_MESSAGE`.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_DEPRECATION_MESSAGE)


class HGFDistribution:
    """Deprecated.

    See :data:`_DEPRECATION_MESSAGE`.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_DEPRECATION_MESSAGE)


class HGFPointwise:
    """Deprecated.

    See :data:`_DEPRECATION_MESSAGE`.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_DEPRECATION_MESSAGE)
