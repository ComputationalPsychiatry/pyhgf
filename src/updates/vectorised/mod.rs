//! Vectorised (whole-layer, matrix-multiply) update kernels, mirroring
//! `pyhgf/updates/vectorized/` file for file: one submodule per node kind
//! (`volatile`, `binary`, `categorical`), each with its `prediction`,
//! `prediction_error`, and (for volatile) `posterior` kernels, plus the
//! rank-one weight-gradient in [`learning`].
//!
//! ## Orientation
//!
//! A parent layer predicts the child below it. `weights` has shape
//! `(n_child, n_parent[+1])`. Prediction is `W @ g(parent)` (length `n_child`);
//! the bottom-up posterior message is `Wᵀ @ (…)` (length `n_parent`). The
//! coupling function and its derivatives are those of the **parent** layer, and
//! derivatives are evaluated at the parent's *expected* (predicted) mean — the
//! reference point the volatile-layer updates use.

pub mod binary;
pub mod categorical;
pub mod learning;
pub mod volatile;

use crate::vectorised::mat::Vector;
use ndarray::{s, Array1, ArrayViewMut1};

/// Append a scalar to a vector, returning a new length `n + 1` vector (used to
/// wire in the constant bias activation).
pub(crate) fn push(v: &Vector, value: f64) -> Vector {
    let n = v.len();
    let mut out = Array1::<f64>::zeros(n + 1);
    out.slice_mut(s![..n]).assign(v);
    out[n] = value;
    out
}

/// Numerically stable in-place softmax over one sample's logits — the single
/// implementation shared by the categorical prediction kernel and the batched
/// forward pass, so the two paths cannot drift.
pub(crate) fn softmax_inplace(mut v: ArrayViewMut1<f64>) {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    v.mapv_inplace(|z| (z - max).exp());
    let sum = v.sum();
    v.mapv_inplace(|z| z / sum);
}
