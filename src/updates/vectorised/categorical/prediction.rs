//! Top-down prediction for categorical leaf layers, mirroring
//! `pyhgf/updates/vectorized/categorical/prediction.py`.

use crate::math::{with_coupling, CouplingFn};
use crate::updates::vectorised::{push, softmax_inplace};
use crate::vectorised::layer::LayerState;
use crate::vectorised::mat::{Float, Matrix};
use ndarray::Array1;

/// Predict a categorical (softmax) leaf layer: `μ̂ = softmax(W @ g(parent))`,
/// with unit precisions and zero effective precision.
///
/// Mirrors `vectorized_categorical_prediction`.
pub fn categorical_prediction(
    child: &mut LayerState,
    parent: &LayerState,
    weights: &Matrix,
    coupling_fn: &CouplingFn,
    parent_has_constant: bool,
) {
    let mut coupled_parents = with_coupling!(coupling_fn, |f, _df, _d2f| parent
        .expected_mean
        .mapv(|v| f(v as f64) as Float));
    if parent_has_constant {
        coupled_parents = push(&coupled_parents, 1.0);
    }
    let mut logits = weights.dot(&coupled_parents);
    softmax_inplace(logits.view_mut());

    let n = logits.len();
    child.expected_mean = logits;
    child.expected_precision = Array1::ones(n);
    child.conditional_expected_precision = Array1::ones(n);
    child.effective_precision = Array1::zeros(n);
}
