//! Bottom-up value-level posterior for volatile-node layers, mirroring
//! `pyhgf/updates/vectorized/volatile/posterior.py`.

use crate::math::{with_coupling, CouplingFn};
use crate::vectorised::layer::LayerState;
use crate::vectorised::mat::{Float, Matrix};
use ndarray::s;

/// Update the value-level posterior (precision then mean) of a parent layer
/// from its child's prediction errors.
///
/// Mirrors `vectorized_layer_posterior_update`. `weights` is the parent's
/// `weights_in`; the bias column is stripped when `parent_has_constant`.
/// `coupling_fn` is the parent's, evaluated at the parent's expected mean.
pub fn layer_posterior_update(
    parent: &mut LayerState,
    child: &LayerState,
    weights: &Matrix,
    coupling_fn: &CouplingFn,
    parent_has_constant: bool,
    max_posterior_precision: Float,
    child_is_input_layer: bool,
) {
    // Bias column stripped via a view (no matrix clone).
    let ncols = weights.ncols();
    let w = if parent_has_constant {
        weights.slice(s![.., ..ncols - 1])
    } else {
        weights.view()
    };

    let (coupling_prime, coupling_second) = with_coupling!(coupling_fn, |_f, df, d2f| {
        (
            parent.expected_mean.mapv(|v| df(v as f64) as Float),
            parent.expected_mean.mapv(|v| d2f(v as f64) as Float),
        )
    });

    // The three weight contractions — `(W²)ᵀ @ eff`, `Wᵀ @ (eff∘δ)`, and
    // `Wᵀ @ (gain∘δ)` — computed in a single row-major pass over `w`, with the
    // per-child effective precision (smoothing/Schur correction) and gain
    // formed as scalars on the fly: no `W²` matrix and no per-child vector
    // temporaries.
    let n_child = w.nrows();
    let n_parent = w.ncols();
    let mut pc1_base = ndarray::Array1::<Float>::zeros(n_parent);
    let mut sum_pi_vpe = ndarray::Array1::<Float>::zeros(n_parent);
    let mut msg = ndarray::Array1::<Float>::zeros(n_parent);
    for i in 0..n_child {
        let cep = child.conditional_expected_precision[i];
        let prec = child.precision[i];
        let pi_y = prec - child.expected_precision[i];
        // Leaf children short-circuit to the canonical predicted precision.
        let eff = if child_is_input_layer {
            cep
        } else {
            cep * pi_y / (cep + pi_y)
        };
        let gain = cep * prec / (cep + pi_y);
        let d = child.value_prediction_error[i];
        let ed = eff * d;
        let gd = gain * d;
        let wrow = w.row(i);
        for j in 0..n_parent {
            let wij = wrow[j];
            pc1_base[j] += wij * wij * eff;
            sum_pi_vpe[j] += wij * ed;
            msg[j] += wij * gd;
        }
    }

    // Precision: clip(π̃_b + pc1·g'² − g''·(Wᵀ@(eff∘δ))), written in place.
    ndarray::Zip::from(&mut parent.precision)
        .and(&parent.expected_precision)
        .and(&pc1_base)
        .and(&sum_pi_vpe)
        .and(&coupling_prime)
        .and(&coupling_second)
        .for_each(|p, &ep, &b, &spv, &gp, &gs| {
            let raw = ep + b * (gp * gp) - gs * spv;
            *p = raw.max(ep).min(max_posterior_precision);
        });

    // Mean: μ̂_b + (Wᵀ@(g_a∘δ))·g'/π_b, written in place (reads the new precision).
    ndarray::Zip::from(&mut parent.mean)
        .and(&parent.expected_mean)
        .and(&msg)
        .and(&coupling_prime)
        .and(&parent.precision)
        .for_each(|m, &em, &mg, &gp, &pp| {
            *m = em + mg * gp / pp;
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::resolve_coupling_fn;
    use ndarray::Array1;

    fn linear() -> &'static CouplingFn {
        resolve_coupling_fn("linear")
    }

    /// Posterior mean update with linear coupling, unit precisions and a leaf
    /// child reduces to `μ̂_b + Wᵀ @ δ` scaled by the gain / posterior precision.
    #[test]
    fn test_posterior_update_leaf_message() {
        // Parent (2 nodes), child leaf (2 nodes), identity weights.
        let mut parent = LayerState::create(2, true);
        parent.expected_mean = Array1::from_vec(vec![0.0, 0.0]);
        parent.expected_precision = Array1::from_vec(vec![1.0, 1.0]);
        let mut child = LayerState::create(2, true);
        child.value_prediction_error = Array1::from_vec(vec![0.5, -0.5]);
        // Leaf convention: conditional == expected == precision.
        child.conditional_expected_precision = Array1::from_vec(vec![1.0, 1.0]);
        child.expected_precision = Array1::from_vec(vec![1.0, 1.0]);
        child.precision = Array1::from_vec(vec![1.0, 1.0]);
        let weights = Matrix::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        layer_posterior_update(
            &mut parent,
            &child,
            &weights,
            linear(),
            false,
            1e10,
            true, // child is a leaf
        );
        // Posterior mean must move in the direction of the (transposed) error.
        assert!(parent.mean[0] > 0.0);
        assert!(parent.mean[1] < 0.0);
        assert!(parent.precision.iter().all(|&x| x.is_finite() && x >= 1.0));
    }
}
