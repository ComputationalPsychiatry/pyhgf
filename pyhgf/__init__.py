# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import os
import pkgutil
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from typing import Optional, Union

import numpy as np
import pandas as pd

__version__ = version("pyhgf")

# The jaxlib releases where XLA:CPU runs on the thunk runtime by default and can
# still be switched off it.
THUNK_WORKAROUND_FIRST = (0, 4, 32)
THUNK_WORKAROUND_LAST = (0, 6, 2)
THUNK_RUNTIME_FLAG = "--xla_cpu_use_thunk_runtime=false"

# The jaxlib releases after that window, up to the first one that uses the threshold
# below as its default. In between, XLA:CPU splits a while loop into thunks once its
# body is estimated to move more than 1024 bytes per iteration, which the belief
# propagation scan of the nodalised `Network` does. Not every release in the thunk
# window reads this option (jaxlib 0.4.33 ignores it), so that window keeps the
# runtime flag.
THRESHOLD_WORKAROUND_FIRST = (0, 7, 0)
THRESHOLD_WORKAROUND_END = (0, 12, 0)
THRESHOLD_FLAG = (
    "--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=1000000000"
)


def _parse_version(version_string: str) -> tuple[int, ...]:
    """Read the leading numeric components of a version string."""
    components = []
    for part in version_string.split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        components.append(int(digits))
    return tuple(components)


def thunk_workaround_applies(version_string: str) -> bool:
    """Whether this jaxlib runs XLA:CPU on thunks and can still switch them off."""
    parsed = _parse_version(version_string)[:3]
    if len(parsed) < 3:
        return False
    return THUNK_WORKAROUND_FIRST <= parsed <= THUNK_WORKAROUND_LAST


def threshold_workaround_applies(version_string: str) -> bool:
    """Whether this jaxlib splits small while loops and reads the threshold option."""
    parsed = _parse_version(version_string)[:3]
    if len(parsed) < 3:
        return False
    return THRESHOLD_WORKAROUND_FIRST <= parsed < THRESHOLD_WORKAROUND_END


def _backend_has_started() -> bool:
    """Whether a jax backend is already running, in which case XLA has read its flags.

    Importing jax, updating its configuration and wrapping a function in `jax.jit` do
    not start a backend; the first operation that creates an array does.
    """
    if "jax" not in sys.modules:
        return False
    try:
        from jax._src.xla_bridge import backends_are_initialized
    except ImportError:
        return True
    return backends_are_initialized()


def _work_around_the_thunk_runtime() -> Optional[str]:
    """Keep XLA:CPU from splitting the belief propagation scan into thunks.

    On jaxlib 0.4.32 to 0.6.2 this switches XLA:CPU off the thunk runtime, so a
    while loop compiles into a single kernel. On jaxlib 0.7.0 to 0.11.2, which
    reject or ignore that flag, it raises the byte threshold under which a while
    loop is kept whole to 1000000000, the default from jaxlib 0.12.0. In both
    cases a gradient through the scan no longer pays a dispatch per thunk at
    every time step.

    The flags are read when the backend starts, so this runs at import time,
    before anything in this package imports jax. It is skipped when the caller
    has already set the option it would change through ``XLA_FLAGS``, and can be
    disabled altogether by setting ``PYHGF_KEEP_THUNK_RUNTIME``.

    Returns
    -------
    The flag that was added, or `None` when nothing was changed.
    """
    if os.environ.get("PYHGF_KEEP_THUNK_RUNTIME"):
        return None

    try:
        jaxlib_version = version("jaxlib")
    except PackageNotFoundError:
        return None

    if thunk_workaround_applies(jaxlib_version):
        option, flag = "xla_cpu_use_thunk_runtime", THUNK_RUNTIME_FLAG
    elif threshold_workaround_applies(jaxlib_version):
        option, flag = "xla_cpu_small_while_loop_byte_threshold", THRESHOLD_FLAG
    else:
        return None

    flags = os.environ.get("XLA_FLAGS", "")
    if option in flags:
        return None

    os.environ["XLA_FLAGS"] = f"{flags} {flag}".strip()

    if _backend_has_started():
        warnings.warn(
            f"pyhgf sets {flag} to avoid a large slowdown in gradients through "
            f"the belief propagation scan on jaxlib {jaxlib_version}, but a jax "
            "computation has already run and XLA reads its flags only once, so "
            "this process will not use it. Import pyhgf before the first jax "
            "computation, or set XLA_FLAGS before Python starts.",
            RuntimeWarning,
            stacklevel=2,
        )

    return flag


THUNK_WORKAROUND_FLAG = _work_around_the_thunk_runtime()


def load_data(dataset: str) -> Union[tuple[np.ndarray, ...], np.ndarray]:
    """Load dataset for continuous or binary HGF.

    Parameters
    ----------
    dataset : str
        The type of data to load. Can be `"continous"` or `"binary"`.

    Returns
    -------
    data : np.ndarray
        The data (a 1d timeseries).

    Notes
    -----
    The continuous time series is the standard USD-CHF conversion rates over time used
    in the Matlab examples.

    The binary dataset is from Iglesias et al. (2013) [#]_ (see the full dataset
    `here <https://www.research-collection.ethz.ch/handle/20.500.11850/454711)>`_. The
    binary set consist of one vector *u*, the observations, and one vector *y*, the
    decisions.

    References
    ----------
    .. [#] Iglesias, S., Kasper, L., Harrison, S. J., Manka, R., Mathys, C., & Stephan,
      K. E. (2021). Cholinergic and dopaminergic effects on prediction error and
      uncertainty responses during sensory associative learning. In NeuroImage (Vol.
      226, p. 117590). Elsevier BV. https://doi.org/10.1016/j.neuroimage.2020.117590
    """
    if dataset == "continuous":
        data = pd.read_csv(
            BytesIO(pkgutil.get_data(__name__, "data/usdchf.txt")),  # type: ignore
            names=["x"],
        ).x.to_numpy()
    elif dataset == "binary":
        u = pd.read_csv(
            BytesIO(
                pkgutil.get_data(__name__, "data/binary_input.txt")  # type: ignore
            ),
            names=["x"],
        ).x.to_numpy()
        y = pd.read_csv(
            BytesIO(
                pkgutil.get_data(__name__, "data/binary_response.txt")  # type: ignore
            ),
            names=["x"],
        ).x.to_numpy()
        data = (u, y)
    else:
        raise ValueError("Invalid dataset argument. Should be 'continous' or 'binary'.")

    return data
