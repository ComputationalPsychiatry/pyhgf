//! Weight learning for the vectorised deep-network backend, mirroring
//! `pyhgf/updates/vectorized/learning.py`.
//!
//! The per-layer weight gradient is a rank-one (outer) product of a child-side
//! factor (the value prediction error, optionally scaled by the child's
//! posterior precision) and a parent-side factor (the coupled parent
//! activation). Only the two factors are materialised — the optimiser
//! ([`crate::vectorised::optimiser`]) consumes them directly, forming each
//! element `u[i]·v[j]` inline. The factors are in **descent** form (sign folded
//! into the child side), so the optimiser step `w ← w + update` reproduces
//! gradient descent.

use crate::math::{with_coupling, CouplingFn};
use crate::vectorised::layer::LayerState;
use crate::vectorised::mat::Vector;

/// Weight-gradient mode, matching the JAX `learning_kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightKind {
    /// Raw squared-error gradient `δ ⊗ a` (no precision metric).
    Standard,
    /// Free-energy gradient `δ · π_child ⊗ a` — the backprop-parity mode.
    PrecisionWeighted,
}

/// Zero out any non-finite entries so the optimizer never propagates NaN/inf
/// through its moment accumulators (matches the JAX `jnp.where(isfinite, …)`).
fn sanitize(v: &Vector) -> Vector {
    v.mapv(|x| if x.is_finite() { x } else { 0.0 })
}

/// Child- and parent-side factors `(u, v)` of the descent weight gradient, such
/// that `u[i] · v[j]` is the full gradient (shape `(n_child, n_parent[+1])`).
///
/// Mirrors `vectorized_weight_gradient_factors`: `u = -δ` (scaled by the
/// child's posterior precision in `PrecisionWeighted` unless the child is
/// binary), `v = g(parent.mean)` with a trailing `1` for the bias column.
pub fn weight_gradient_factors(
    parent: &LayerState,
    child: &LayerState,
    coupling_fn: &CouplingFn,
    kind: WeightKind,
    parent_has_constant: bool,
    child_is_binary: bool,
) -> (Vector, Vector) {
    let pe = &child.mean - &child.expected_mean;
    let mut coupled_parent = with_coupling!(coupling_fn, |f, _df, _d2f| parent.mean.mapv(f));
    if parent_has_constant {
        coupled_parent = super::push(&coupled_parent, 1.0);
    }

    let mut u = pe;
    if kind == WeightKind::PrecisionWeighted && !child_is_binary {
        u = &u * &child.precision;
    }

    let u = sanitize(&u);
    let v = sanitize(&coupled_parent);
    // Descent sign folded into the child-side factor.
    (u.mapv(|x| -x), v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::resolve_coupling_fn;
    use ndarray::Array1;

    fn linear() -> &'static CouplingFn {
        resolve_coupling_fn("linear")
    }

    #[test]
    fn test_gradient_is_rank_one_outer() {
        let mut parent = LayerState::create(2, true);
        parent.mean = Array1::from_vec(vec![1.0, 3.0]);
        let mut child = LayerState::create(2, true);
        child.mean = Array1::from_vec(vec![1.0, 1.0]);
        child.expected_mean = Array1::from_vec(vec![0.0, 0.0]);
        child.precision = Array1::from_vec(vec![1.0, 1.0]);

        let (u, v) = weight_gradient_factors(
            &parent,
            &child,
            linear(),
            WeightKind::Standard,
            false,
            false,
        );
        // δ = [1,1], a = [1,3]; descent grad g[i][j] = u[i]·v[j] = -(δ ⊗ a).
        assert_eq!((u.len(), v.len()), (2, 2));
        assert!((u[0] * v[0] - (-1.0)).abs() < 1e-12);
        assert!((u[0] * v[1] - (-3.0)).abs() < 1e-12);
        assert!((u[1] * v[1] - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_precision_weighting_scales_child_factor() {
        let mut parent = LayerState::create(1, true);
        parent.mean = Array1::from_vec(vec![2.0]);
        let mut child = LayerState::create(1, true);
        child.mean = Array1::from_vec(vec![1.0]);
        child.expected_mean = Array1::from_vec(vec![0.0]);
        child.precision = Array1::from_vec(vec![4.0]);

        let (u, v) = weight_gradient_factors(
            &parent,
            &child,
            linear(),
            WeightKind::PrecisionWeighted,
            false,
            false,
        );
        // δ·π = 1*4 = 4, a = 2 → descent -(4*2) = -8.
        assert!((u[0] * v[0] - (-8.0)).abs() < 1e-12);
    }

    #[test]
    fn test_bias_column_appends_unit_activation() {
        let mut parent = LayerState::create(1, true);
        parent.mean = Array1::from_vec(vec![5.0]);
        let mut child = LayerState::create(1, true);
        child.mean = Array1::from_vec(vec![1.0]);
        child.expected_mean = Array1::from_vec(vec![0.0]);
        child.precision = Array1::from_vec(vec![1.0]);

        let (_, v) =
            weight_gradient_factors(&parent, &child, linear(), WeightKind::Standard, true, false);
        // v = [g(5), 1] = [5, 1].
        assert_eq!(v.len(), 2);
        assert!((v[0] - 5.0).abs() < 1e-12);
        assert!((v[1] - 1.0).abs() < 1e-12);
    }
}
