"""Generative inputs need separate noise and deterministic key replay."""

import jax
import numpy as np
import pytest

from pyhgf.model import Network


@pytest.mark.parametrize("kind", ["continuous-state", "binary-state"])
@pytest.mark.parametrize("key_factory", [jax.random.key, jax.random.PRNGKey])
def test_generative_input_noise_is_independent_and_reproducible(kind, key_factory):
    """Disconnected observations must not share draws for typed or legacy keys."""
    net = Network().add_nodes(kind=kind, n_nodes=2)
    if kind == "binary-state":
        # A fixed logit gives two conditionally independent Bernoulli(0.5) inputs.
        net.add_nodes(
            kind="constant-state",
            value_children=[0, 1],
            node_parameters={"mean": 0.0, "expected_mean": 0.0},
        )
    net.create_belief_propagation_fn(sampling_fn=True)
    time_steps = np.ones(256)
    net.sample(n_predictions=2, time_steps=time_steps, rng_key=key_factory(7))
    first = [np.asarray(net.samples[i]["mean"]).copy() for i in (0, 1)]
    assert np.std(first[0]) > 0.0
    assert np.std(first[1]) > 0.0
    assert not np.array_equal(first[0], first[1])

    net.sample(n_predictions=2, time_steps=time_steps, rng_key=key_factory(7))
    for i in (0, 1):
        np.testing.assert_array_equal(net.samples[i]["mean"], first[i])

    net.sample(n_predictions=2, time_steps=time_steps, rng_key=key_factory(8))
    for i in (0, 1):
        assert not np.array_equal(net.samples[i]["mean"], first[i])


@pytest.mark.parametrize("key_factory", [jax.random.key, jax.random.PRNGKey])
def test_single_input_retains_its_existing_random_stream(key_factory):
    """The multi-input fix must retain established single-input stream replay."""
    net = Network().add_nodes().create_belief_propagation_fn(sampling_fn=True)
    key = key_factory(7)
    net.sample(n_predictions=1, time_steps=np.ones(32), rng_key=key)
    prediction_key = jax.random.split(key, 1)[0]
    time_keys = jax.random.split(prediction_key, 32)
    expected = jax.vmap(jax.random.normal)(time_keys)
    np.testing.assert_allclose(net.samples[0]["mean"][0], expected)
