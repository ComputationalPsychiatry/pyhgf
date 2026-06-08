# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Deprecated module.

The :class:`pyhgf.model.HGF` class is deprecated, and its content has been removed. It
is recommended to build the network directly using the :class:`pyhgf.model.Network`
class together with ``add_nodes()``. Please refer to the main documentation for
examples. Importing this module will raise a :class:`DeprecationWarning`, and using the
:class:`HGF` class will raise a :class:`NotImplementedError`.
"""

import warnings

_DEPRECATION_MESSAGE = (
    "The `pyhgf.model.HGF` class is deprecated and will be removed in a future "
    "release. Build the network directly using the `pyhgf.model.Network` class "
    "together with `add_nodes()` instead."
)

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)


class HGF:
    """Deprecated.

    See :data:`_DEPRECATION_MESSAGE`.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_DEPRECATION_MESSAGE)
