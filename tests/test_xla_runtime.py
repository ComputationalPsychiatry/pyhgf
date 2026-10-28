# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import os
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from unittest.mock import patch

import pytest

from pyhgf import (
    THUNK_WORKAROUND_FLAG,
    _backend_has_started,
    _work_around_the_thunk_runtime,
    threshold_workaround_applies,
    thunk_workaround_applies,
)

FLAG = "--xla_cpu_use_thunk_runtime=false"
THRESHOLD = (
    "--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=1000000000"
)


@pytest.fixture
def clean_env():
    """Run with no XLA_FLAGS and no opt-out, whatever the caller's environment."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("XLA_FLAGS", None)
        os.environ.pop("PYHGF_KEEP_THUNK_RUNTIME", None)
        yield


@pytest.mark.parametrize(
    "jaxlib_version, expected",
    [
        ("0.4.26", False),  # the thunk runtime is not the default yet
        ("0.4.31", False),  # the last release before the change
        ("0.4.32", True),  # the thunk runtime becomes the default here
        ("0.5.3.dev20260725", True),  # a development build still counts
        ("0.6.2", True),  # the last release that can refuse it
        ("0.7.0", False),  # setting the flag raises from here
        ("0.7.2", False),  # and is silently ignored from here
        ("0.11.1", False),
        ("0.6", False),  # too short to compare
        ("not-a-version", False),
    ],
)
def test_workaround_window(jaxlib_version, expected):
    """The runtime flag is claimed only where it both helps and is honoured."""
    assert thunk_workaround_applies(jaxlib_version) is expected


@pytest.mark.parametrize(
    "jaxlib_version, expected",
    [
        ("0.4.33", False),  # ignores the option; the runtime flag covers it
        ("0.6.2", False),  # reads it, but the runtime flag already covers it
        ("0.7.0", True),  # the runtime flag raises here; the threshold does not
        ("0.9.2.dev20260301", True),  # a development build still counts
        ("0.11.2", True),  # the last release with a 1024-byte default
        ("0.12.0", False),  # uses the raised threshold as its default
        ("0.12.0.1", False),
        ("0.11", False),  # too short to compare
        ("not-a-version", False),
    ],
)
def test_threshold_window(jaxlib_version, expected):
    """The threshold is raised on jaxlib 0.7.0 to 0.11.x and on no other release."""
    assert threshold_workaround_applies(jaxlib_version) is expected


@pytest.mark.parametrize(
    "jaxlib_version, environment, expected_flag",
    [
        # Inside either window, with nothing in the way.
        ("0.5.3", {}, FLAG),
        ("0.9.2", {}, THRESHOLD),
        # Outside both, in either direction.
        ("0.4.31", {}, None),
        ("0.12.0.1", {}, None),
        # The caller's own flags are kept and appended to.
        ("0.5.3", {"XLA_FLAGS": "--xla_force_host_platform_device_count=2"}, FLAG),
        (
            "0.9.2",
            {"XLA_FLAGS": "--xla_force_host_platform_device_count=2"},
            THRESHOLD,
        ),
        # An explicit choice stands, whichever way it goes.
        ("0.5.3", {"XLA_FLAGS": "--xla_cpu_use_thunk_runtime=true"}, None),
        ("0.5.3", {"XLA_FLAGS": FLAG}, None),
        (
            "0.9.2",
            {
                "XLA_FLAGS": (
                    "--xla_backend_extra_options="
                    "xla_cpu_small_while_loop_byte_threshold=1024"
                )
            },
            None,
        ),
        # And the opt-out is respected.
        ("0.5.3", {"PYHGF_KEEP_THUNK_RUNTIME": "1"}, None),
        ("0.9.2", {"PYHGF_KEEP_THUNK_RUNTIME": "1"}, None),
    ],
)
def test_selecting_the_runtime(clean_env, jaxlib_version, environment, expected_flag):
    """Whether a flag is set, given the jaxlib version and what is already set."""
    os.environ.update(environment)
    before = os.environ.get("XLA_FLAGS", "")

    with patch("pyhgf.version", return_value=jaxlib_version):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert _work_around_the_thunk_runtime() == expected_flag

    if expected_flag is None:
        assert os.environ.get("XLA_FLAGS", "") == before
    else:
        assert expected_flag in os.environ["XLA_FLAGS"]
        assert before in os.environ["XLA_FLAGS"]


def test_a_missing_jaxlib_is_not_an_error(clean_env):
    """Reading the metadata of an absent jaxlib must not break the import."""
    with patch("pyhgf.version", side_effect=PackageNotFoundError):
        assert _work_around_the_thunk_runtime() is None
    assert "XLA_FLAGS" not in os.environ


@pytest.mark.parametrize(
    "jaxlib_version, flag", [("0.5.3", FLAG), ("0.9.2", THRESHOLD)]
)
def test_it_warns_when_the_backend_has_started(clean_env, jaxlib_version, flag):
    """A flag set after the backend started has no effect, which is worth saying."""
    with patch("pyhgf.version", return_value=jaxlib_version):
        with patch("pyhgf._backend_has_started", return_value=True):
            with pytest.warns(RuntimeWarning, match="already run"):
                assert _work_around_the_thunk_runtime() == flag


@pytest.mark.parametrize(
    "jaxlib_version, flag", [("0.5.3", FLAG), ("0.9.2", THRESHOLD)]
)
def test_importing_jax_first_is_silent(clean_env, jaxlib_version, flag):
    """An imported jax whose backend has not started still reads the flag."""
    with patch("pyhgf.version", return_value=jaxlib_version):
        with patch("pyhgf._backend_has_started", return_value=False):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert _work_around_the_thunk_runtime() == flag


@pytest.mark.parametrize("jaxlib_version", ["0.4.31", "0.12.0", "0.12.0.1"])
def test_no_warning_where_nothing_is_needed(clean_env, jaxlib_version):
    """Outside both windows the import is silent and leaves XLA_FLAGS alone."""
    with patch("pyhgf.version", return_value=jaxlib_version):
        with patch("pyhgf._backend_has_started", return_value=True):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert _work_around_the_thunk_runtime() is None
    assert "XLA_FLAGS" not in os.environ


def test_detecting_a_started_backend():
    """No jax means no backend; a jax without the expected internals counts as one."""
    with patch.dict("sys.modules"):
        sys.modules.pop("jax", None)
        assert _backend_has_started() is False
    with patch.dict("sys.modules", {"jax._src.xla_bridge": None}):
        assert "jax" in sys.modules
        assert _backend_has_started() is True


@pytest.mark.parametrize(
    "prelude, expect_warning",
    [
        ("import jax; jax.config.update('jax_enable_x64', True)", False),
        ("import jax.numpy as jnp; jnp.ones(1)", True),
    ],
)
def test_import_order_in_a_fresh_process(prelude, expect_warning):
    """Only a computation run before the import makes the flag arrive too late."""
    installed = version("jaxlib")
    if not (
        thunk_workaround_applies(installed) or threshold_workaround_applies(installed)
    ):
        pytest.skip(f"jaxlib {installed} needs no flag")

    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("XLA_FLAGS", "PYHGF_KEEP_THUNK_RUNTIME")
    }
    result = subprocess.run(
        [sys.executable, "-c", f"{prelude}; import pyhgf"],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    assert ("already run" in result.stderr) is expect_warning


def test_the_flag_matches_the_installed_jaxlib():
    """What happened at import must agree with the jaxlib actually installed."""
    installed = version("jaxlib")
    if thunk_workaround_applies(installed):
        assert THUNK_WORKAROUND_FLAG == FLAG
    elif threshold_workaround_applies(installed):
        assert THUNK_WORKAROUND_FLAG == THRESHOLD
    else:
        assert THUNK_WORKAROUND_FLAG is None

    if THUNK_WORKAROUND_FLAG is not None:
        assert THUNK_WORKAROUND_FLAG in os.environ.get("XLA_FLAGS", "")
