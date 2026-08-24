# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import os
import warnings
from importlib.metadata import PackageNotFoundError, version
from unittest.mock import patch

import pytest

from pyhgf import (
    LEGACY_CPU_RUNTIME_FLAG,
    _use_legacy_cpu_runtime,
    thunk_workaround_applies,
)

FLAG = "--xla_cpu_use_thunk_runtime=false"


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
        ("0.4.26", False),  # the legacy runtime is already the default
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
    """The flag is claimed only where it both helps and is honoured."""
    assert thunk_workaround_applies(jaxlib_version) is expected


@pytest.mark.parametrize(
    "jaxlib_version, environment, expected_flag",
    [
        # Inside the window, with nothing in the way.
        ("0.5.3", {}, FLAG),
        # Outside it, in either direction.
        ("0.4.31", {}, None),
        ("0.11.1", {}, None),
        # The caller's own flags are kept and appended to.
        ("0.5.3", {"XLA_FLAGS": "--xla_force_host_platform_device_count=2"}, FLAG),
        # An explicit choice stands, whichever way it goes.
        ("0.5.3", {"XLA_FLAGS": "--xla_cpu_use_thunk_runtime=true"}, None),
        ("0.5.3", {"XLA_FLAGS": FLAG}, None),
        # And the opt-out is respected.
        ("0.5.3", {"PYHGF_KEEP_THUNK_RUNTIME": "1"}, None),
    ],
)
def test_selecting_the_runtime(clean_env, jaxlib_version, environment, expected_flag):
    """Whether the flag is set, given the jaxlib version and what is already set."""
    os.environ.update(environment)
    before = os.environ.get("XLA_FLAGS", "")

    with patch("pyhgf.version", return_value=jaxlib_version):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert _use_legacy_cpu_runtime() == expected_flag

    if expected_flag is None:
        assert os.environ.get("XLA_FLAGS", "") == before
    else:
        assert FLAG in os.environ["XLA_FLAGS"]
        assert before in os.environ["XLA_FLAGS"]


def test_a_missing_jaxlib_is_not_an_error(clean_env):
    """Reading the metadata of an absent jaxlib must not break the import."""
    with patch("pyhgf.version", side_effect=PackageNotFoundError):
        assert _use_legacy_cpu_runtime() is None
    assert "XLA_FLAGS" not in os.environ


def test_it_warns_when_jax_is_already_imported(clean_env):
    """The flag is read when the backend starts, so arriving late is worth saying."""
    with patch("pyhgf.version", return_value="0.5.3"):
        with patch.dict("sys.modules", {"jax": object()}):
            with pytest.warns(RuntimeWarning, match="already imported"):
                assert _use_legacy_cpu_runtime() == FLAG


@pytest.mark.parametrize(
    "jaxlib_version, has_cuda_plugin, expect_warning",
    [
        ("0.11.1", False, True),  # stuck with the slow runtime, and can say so
        ("0.11.1", True, False),  # a GPU install: the regression is XLA:CPU only
        ("0.4.31", False, False),  # below the window there is nothing wrong
    ],
)
def test_warning_above_the_window(
    clean_env, jaxlib_version, has_cuda_plugin, expect_warning
):
    """Being stuck with the thunk runtime is announced, but only where it bites."""
    with patch("pyhgf.version", return_value=jaxlib_version):
        with patch("pyhgf._has_cuda_plugin", return_value=has_cuda_plugin):
            if expect_warning:
                with pytest.warns(RuntimeWarning, match="thunk runtime"):
                    assert _use_legacy_cpu_runtime() is None
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    assert _use_legacy_cpu_runtime() is None


def test_the_flag_matches_the_installed_jaxlib():
    """What happened at import must agree with the jaxlib actually installed."""
    if thunk_workaround_applies(version("jaxlib")):
        assert LEGACY_CPU_RUNTIME_FLAG == FLAG
        assert FLAG in os.environ.get("XLA_FLAGS", "")
    else:
        assert LEGACY_CPU_RUNTIME_FLAG is None
