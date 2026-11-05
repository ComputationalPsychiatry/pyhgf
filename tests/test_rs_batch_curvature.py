# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
import pytest
from pyhgf.rshgf import DeepNetwork


@pytest.mark.parametrize("learning_kind", ["precision_weighted", "standard"])
@pytest.mark.parametrize(
    ("coupling", "n_layers"), [("tanh", 5), ("linear", 5), ("tanh", 3)]
)
def test_single_sample_batch_matches_scalar_curved_updates(
    learning_kind, coupling, n_layers
):
    """An interior nonlinear child gives the same scalar and batch update."""

    def build():
        net = DeepNetwork(
            volatility_updates="eHGF",
            max_posterior_precision=1e10,
            precision_clipping_value=1e-6,
            coupling_fn=coupling,
            predict_precision=True,
            feedforward_uncertainty=False,
            tonic_volatility=False,
            carry_precision=True,
        )
        for _ in range(n_layers):
            net.add_layer(1, volatility_parent=False, add_constant_input=True)
        net.set_weights([
            np.full(np.asarray(w).shape, 0.3, dtype=np.asarray(w).dtype)
            for w in net.get_weights()
        ])
        return net

    scalar, batch = build(), build()
    x, y = np.array([[-1.0]]), np.array([[10.0]])
    batch_prediction = batch.batch_forward(x, time_step=1.0)
    scalar_prediction = scalar.fit(
        x,
        y,
        optimiser="sgd",
        learning_rate=0.1,
        learning_kind=learning_kind,
        weight_update=True,
        time_step=1.0,
    )
    batch.batch_update(
        x,
        y,
        optimiser="sgd",
        learning_rate=0.1,
        learning_kind=learning_kind,
        update_precisions=True,
        time_step=1.0,
    )
    np.testing.assert_allclose(
        scalar_prediction, batch_prediction, rtol=3e-5, atol=1e-6
    )
    for scalar_weights, batch_weights in zip(
        scalar.get_weights(), batch.get_weights(), strict=True
    ):
        np.testing.assert_allclose(scalar_weights, batch_weights, rtol=3e-5, atol=1e-6)
