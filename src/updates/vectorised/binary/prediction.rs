//! Top-down prediction for binary leaf layers, mirroring
//! `pyhgf/updates/vectorized/binary/prediction.py`.

use crate::math::{sigmoid, with_coupling, CouplingFn};
use crate::updates::vectorised::push;
use crate::vectorised::layer::LayerState;
use crate::vectorised::mat::{Float, Matrix};

/// Predict a binary (Bernoulli) leaf layer: `μ̂ = σ(W @ g(parent))`, clipped
/// away from 0/1, with the Bernoulli variance stored as `expected_precision`.
///
/// Mirrors `vectorized_binary_prediction`.
pub fn binary_prediction(
    child: &mut LayerState,
    parent: &LayerState,
    weights: &Matrix,
    coupling_fn: &CouplingFn,
    parent_has_constant: bool,
    precision_clipping_value: Float,
) {
    let mut coupled_parents = with_coupling!(coupling_fn, |f, _df, _d2f| parent
        .expected_mean
        .mapv(|v| f(v as f64) as Float));
    if parent_has_constant {
        coupled_parents = push(&coupled_parents, 1.0);
    }
    let logit = weights.dot(&coupled_parents);
    let expected_mean = logit.mapv(|x| {
        (sigmoid(x as f64) as Float).clamp(precision_clipping_value, 1.0 - precision_clipping_value)
    });
    let expected_precision = expected_mean.mapv(|m| m * (1.0 - m));

    child.expected_mean = expected_mean;
    // Binary leaves store the Bernoulli variance in `expected_precision`; the
    // conditional coincides with the marginal (Limit 3 leaf convention).
    child.conditional_expected_precision = expected_precision.clone();
    child.expected_precision = expected_precision;
}
