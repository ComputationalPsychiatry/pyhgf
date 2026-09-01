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

# The window of jaxlib releases where the legacy XLA:CPU runtime can be selected.
THUNK_WORKAROUND_FIRST = (0, 4, 32)
THUNK_WORKAROUND_LAST = (0, 6, 2)


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
    """Whether this jaxlib both needs the legacy CPU runtime and can select it."""
    parsed = _parse_version(version_string)[:3]
    if len(parsed) < 3:
        return False
    return THUNK_WORKAROUND_FIRST <= parsed <= THUNK_WORKAROUND_LAST


def _has_cuda_plugin() -> bool:
    """Whether a CUDA plugin is installed, which means the work runs on GPU."""
    for package in ["jax-cuda12-plugin", "jax-cuda13-plugin"]:
        try:
            version(package)
            return True
        except PackageNotFoundError:
            continue
    return False


def _warn_if_stuck_with_the_thunk_runtime(jaxlib_version: str) -> None:
    """Say so when the CPU runtime is slow here and can no longer be changed.

    Only above the workaround window, and only without a CUDA plugin: the regression is
    specific to XLA:CPU, so a GPU install is unaffected and the vectorised `DeepNetwork`
    path is array-shaped and barely touched either way.
    """
    if _parse_version(jaxlib_version)[:3] <= THUNK_WORKAROUND_LAST:
        return
    if _has_cuda_plugin():
        return

    warnings.warn(
        f"jaxlib {jaxlib_version} runs XLA:CPU on the thunk runtime, which "
        "cannot be turned off from 0.7.0 onwards. Gradients through the belief "
        "propagation scan of the nodalised `Network` are roughly twenty times "
        "slower there, so MCMC over its parameters will be slow. Pin jaxlib "
        "below 0.7 for that work, or use the vectorised `DeepNetwork` path, "
        "which is unaffected. Set PYHGF_KEEP_THUNK_RUNTIME to silence this.",
        RuntimeWarning,
        stacklevel=2,
    )


def _use_legacy_cpu_runtime() -> Optional[str]:
    """Ask XLA for the legacy CPU runtime, where that is still possible.

    The flag is read when the backend starts, so this runs at import time,
    before anything in this package imports jax. It is skipped when the caller
    has already expressed a preference through ``XLA_FLAGS``, and can be
    disabled altogether by setting ``PYHGF_KEEP_THUNK_RUNTIME``.

    Returns
    -------
    The flag that was added, or `None` when nothing was changed.
    """
    if os.environ.get("PYHGF_KEEP_THUNK_RUNTIME"):
        return None

    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_cpu_use_thunk_runtime" in flags:
        return None

    try:
        jaxlib_version = version("jaxlib")
    except PackageNotFoundError:
        return None

    if not thunk_workaround_applies(jaxlib_version):
        _warn_if_stuck_with_the_thunk_runtime(jaxlib_version)
        return None

    flag = "--xla_cpu_use_thunk_runtime=false"
    os.environ["XLA_FLAGS"] = f"{flags} {flag}".strip()

    if "jax" in sys.modules:
        warnings.warn(
            "pyhgf sets --xla_cpu_use_thunk_runtime=false to avoid a large "
            "slowdown in gradients through the belief propagation scan on "
            f"jaxlib {jaxlib_version}. jax was already imported, so the flag "
            "may arrive too late to take effect. Import pyhgf before jax, or "
            "set XLA_FLAGS yourself, to be sure of it.",
            RuntimeWarning,
            stacklevel=2,
        )

    return flag


LEGACY_CPU_RUNTIME_FLAG = _use_legacy_cpu_runtime()


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
