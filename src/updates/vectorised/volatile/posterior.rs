//! Bottom-up value-level posterior for volatile-node layers, mirroring
//! `pyhgf/updates/vectorised/volatile/posterior.py`.

use crate::math::{with_coupling, CouplingFn};
use crate::vectorised::layer::LayerState;
use crate::vectorised::mat::{Float, Matrix};
use ndarray::s;

/// Update the value-level posterior (precision then mean) of a parent layer
/// from its child's prediction errors.
///
/// Mirrors `vectorised_layer_posterior_update`. `weights` is the parent's
/// `weights_in`; the bias column is stripped when `parent_has_constant`.
/// `coupling_fn` is the parent's, evaluated at the parent's expected mean.
///
/// The precision update deviates from the generalised HGF (Weber et al., 2026),
/// for deep networks only: it is first-order in the coupling function, π̂_b + g'²
/// · Σ_a α_a² · π_a^eff, without the second-order term −α g'' π̂_a δ_a of Eq. 50.
/// That term takes either sign and can cancel the first-order one, after which
/// the mean step, the message divided by the precision, grows as 1/π̂_b. The
/// nodalised backend keeps the term.
///
/// With `exact_moments`, the coupling function is replaced by its statistical
/// linearisation under the parent's predicted belief N(μ̂_b, 1/π̂_b), as in the
/// prediction step: g' is the mean slope E[g'(x_b)].
#[allow(clippy::too_many_arguments)]
pub fn layer_posterior_update(
    parent: &mut LayerState,
    child: &LayerState,
    weights: &Matrix,
    coupling_fn: &CouplingFn,
    parent_has_constant: bool,
    max_posterior_precision: Float,
    child_is_input_layer: bool,
    exact_moments: bool,
) {
    // Bias column stripped via a view (no matrix clone).
    let ncols = weights.ncols();
    let w = if parent_has_constant {
        weights.slice(s![.., ..ncols - 1])
    } else {
        weights.view()
    };

    let coupling_prime = if exact_moments {
        ndarray::Zip::from(&parent.expected_mean)
            .and(&parent.expected_precision)
            .map_collect(|&m, &p| (coupling_fn.moments)(m as f64, 1.0 / p as f64).2 as Float)
    } else {
        with_coupling!(coupling_fn, |_f, df, _d2f| parent
            .expected_mean
            .mapv(|v| df(v as f64) as Float))
    };

    // The two weight contractions, `(W²)ᵀ @ eff` and `Wᵀ @ (gain∘δ)`, are computed
    // in a single row-major pass over `w`, with the per-child effective precision
    // (posterior-step relaxation) and gain formed as scalars on the fly: no `W²`
    // matrix and no per-child vector temporaries. `eff` weights the precision
    // increment; the gain weights the mean shift.
    let n_child = w.nrows();
    let n_parent = w.ncols();
    let mut pc1_base = ndarray::Array1::<Float>::zeros(n_parent);
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
        let gd = gain * child.value_prediction_error[i];
        let wrow = w.row(i);
        for j in 0..n_parent {
            let wij = wrow[j];
            pc1_base[j] += wij * wij * eff;
            msg[j] += wij * gd;
        }
    }

    // Precision: clip(π̂_b + pc1·g'²), written in place.
    ndarray::Zip::from(&mut parent.precision)
        .and(&parent.expected_precision)
        .and(&pc1_base)
        .and(&coupling_prime)
        .for_each(|p, &ep, &b, &gp| {
            let raw = ep + b * (gp * gp);
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
            true,  // child is a leaf
            false, // exact_moments
        );
        // Posterior mean must move in the direction of the (transposed) error.
        assert!(parent.mean[0] > 0.0);
        assert!(parent.mean[1] < 0.0);
        assert!(parent.precision.iter().all(|&x| x.is_finite() && x >= 1.0));
    }

    /// With a tanh coupling and interior children, each child contributes
    /// π̄_a · π_y / (π̄_a + π_y) · α² g'² to the posterior precision, with
    /// π_y = π_a − π̂_a and the derivative at the expected mean. The prediction
    /// errors do not enter: the update has no second-order term.
    #[test]
    fn test_posterior_precision_curved_coupling() {
        let mut parent = LayerState::create(1, true);
        parent.expected_mean = Array1::from_vec(vec![0.3]);
        parent.expected_precision = Array1::from_vec(vec![1.2]);
        let mut child = LayerState::create(2, true);
        child.precision = Array1::from_vec(vec![3.0, 2.4]);
        child.expected_precision = Array1::from_vec(vec![2.0, 1.5]);
        child.conditional_expected_precision = Array1::from_vec(vec![2.5, 1.8]);
        child.value_prediction_error = Array1::from_vec(vec![0.4, -0.7]);
        let weights = Matrix::from_shape_vec((2, 1), vec![0.8, -0.5]).unwrap();

        layer_posterior_update(
            &mut parent,
            &child,
            &weights,
            resolve_coupling_fn("tanh"),
            false,
            1e10,
            false, // interior children
            false, // exact_moments
        );

        let t = 0.3_f64.tanh();
        let g_prime = 1.0 - t * t;
        let mut expected = 1.2;
        for (alpha, pi_a, pi_hat_a, pi_bar_a) in [(0.8, 3.0, 2.0, 2.5), (-0.5, 2.4, 1.5, 1.8)] {
            let pi_y = pi_a - pi_hat_a;
            expected += pi_bar_a / (pi_bar_a + pi_y) * alpha * alpha * g_prime * g_prime * pi_y;
        }
        let precision = parent.precision[0] as f64;
        assert!(
            (precision - expected).abs() < 1e-5 * expected,
            "precision {precision} vs closed form {expected}"
        );
    }
}
