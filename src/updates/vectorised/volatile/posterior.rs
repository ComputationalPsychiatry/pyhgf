//! Bottom-up value-level posterior for volatile-node layers, mirroring
//! `pyhgf/updates/vectorised/volatile/posterior.py`.

use crate::math::{with_coupling, CouplingFn};
use crate::vectorised::layer::LayerState;
use crate::vectorised::mat::{Float, Matrix};
use ndarray::{s, ArrayView1, ArrayViewMut1};

/// The predicted beliefs of a child node that its parents' posterior update reads.
#[derive(Clone, Copy)]
pub(crate) struct ChildNode {
    /// Conditional predicted precision π̄_a, which conditions on every parent.
    pub conditional_expected_precision: Float,
    /// Marginal predicted precision π̂_a.
    pub expected_precision: Float,
    /// Posterior precision π_a.
    pub precision: Float,
    /// Value prediction error δ_a.
    pub value_prediction_error: Float,
}

/// Add the relaxed value messages of one child node a to its parents, one factor per
/// edge.
///
/// For every parent node m, adds `W[a, m]² · π_eff(a|m)` to `precision_sum[m]` and
/// `W[a, m] · g(a|m) · δ_a` to `message[m]`, with
///
/// ```text
/// π_eff(a|m) = π̄_{a|m} · π_y / (π̄_{a|m} + π_y),    g(a|m) = π̄_{a|m} · π_a / (π̄_{a|m} + π_y),
/// π_y = π_a − π̂_a,    1/π̄_{a|m} = 1/π̄_a + Σ_{m' ≠ m} W[a, m']² · h[m'],
/// ```
///
/// where `h[m] = A_m² / π̂_m` (`parent_variance`) makes `W[a, m]² · h[m]` the variance
/// that the edge adds to the prediction of the child, with A_m the slope of the
/// linear function that replaces the coupling function in the prediction step:
/// g'(μ̂_m), or the mean slope E[g'(x_m)] with the exact coupling moments. π̄_{a|m}
/// is the predicted precision of x_a given x_m alone, with the other parents of a
/// integrated over their predicted beliefs; the joint Gaussian of (x_a, x_m) with
/// this conditional has the marginal π̂_a for x_a. A clamped child takes the fully
/// observed limit π_y → ∞, π_eff = g = π̄_{a|m}.
///
/// For linear coupling, the factors of one child give the exact marginal posterior
/// of x_m from the evidence of this child. The messages of several children add up
/// to the exact marginal posterior when no two children of m share another parent.
/// When they do, as between two fully connected layers, each edge leaves out what
/// the other children say about the shared parents, and the sum is an
/// approximation.
///
/// The sums over the other parents are formed from prefix and suffix sums, not by
/// subtracting each term from the row total, so they keep their precision when one
/// edge dominates. `suffix` is scratch space, resized to `n_parent + 1`.
pub(crate) fn add_edge_messages(
    weights_row: ArrayView1<Float>,
    parent_variance: ArrayView1<Float>,
    child: ChildNode,
    child_is_clamped: bool,
    suffix: &mut Vec<Float>,
    mut precision_sum: ArrayViewMut1<Float>,
    mut message: ArrayViewMut1<Float>,
) {
    let n_parent = weights_row.len();
    suffix.clear();
    suffix.resize(n_parent + 1, 0.0);
    for j in (0..n_parent).rev() {
        let w = weights_row[j];
        suffix[j] = suffix[j + 1] + w * w * parent_variance[j];
    }
    let pi_y = child.precision - child.expected_precision;
    let conditional_variance = 1.0 / child.conditional_expected_precision;
    let mut prefix: Float = 0.0;
    for j in 0..n_parent {
        let w = weights_row[j];
        let conditional = 1.0 / (conditional_variance + prefix + suffix[j + 1]);
        prefix += w * w * parent_variance[j];
        let (eff, gain) = if child_is_clamped {
            (conditional, conditional)
        } else {
            let denominator = conditional + pi_y;
            (
                conditional * pi_y / denominator,
                conditional * child.precision / denominator,
            )
        };
        precision_sum[j] += w * w * eff;
        message[j] += w * gain * child.value_prediction_error;
    }
}

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
///
/// With `integrate_other_parents`, each edge conditions the child on that parent
/// alone, with the child's other parents integrated over their predicted beliefs
/// (see [`add_edge_messages`]); without it, every edge uses the child's conditional
/// predicted precision π̄_a, which treats the other parents as known. The network
/// sets it when the prediction step propagates the parents' uncertainty
/// (`feedforward_uncertainty` and `predict_precision`) and the child is a volatile
/// (Gaussian) layer.
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
    integrate_other_parents: bool,
) {
    // Bias column stripped via a view (no matrix clone).
    let ncols = weights.ncols();
    let w = if parent_has_constant {
        weights.slice(s![.., ..ncols - 1])
    } else {
        weights.view()
    };

    let coupling_prime = if exact_moments {
        // Mean slope cached by the prediction sweep for this step.
        parent.linear_slope.clone()
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
    if integrate_other_parents {
        let parent_variance = ndarray::Zip::from(&coupling_prime)
            .and(&parent.expected_precision)
            .map_collect(|&gp, &ep| gp * gp / ep);
        let mut suffix = Vec::with_capacity(n_parent + 1);
        for i in 0..n_child {
            add_edge_messages(
                w.row(i),
                parent_variance.view(),
                ChildNode {
                    conditional_expected_precision: child.conditional_expected_precision[i],
                    expected_precision: child.expected_precision[i],
                    precision: child.precision[i],
                    value_prediction_error: child.value_prediction_error[i],
                },
                child_is_input_layer,
                &mut suffix,
                pc1_base.view_mut(),
                msg.view_mut(),
            );
        }
    } else {
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
            false, // other parents treated as known
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
            false, // other parents treated as known
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

    /// Exact marginal posterior of parent `m` in the joint Gaussian of a child node
    /// `x_a = W[0] x_0 + W[1] x_1 + W[2] x_2 + ε`, ε of variance 1/π̄_a, observed with
    /// noise of variance 1/π_y at a distance `w` from its prediction. Returns
    /// `(precision, mean shift)`.
    fn exact_parent_posterior(
        weights: [f64; 3],
        parent_precisions: [f64; 3],
        conditional: f64,
        pi_y: f64,
        w: f64,
        m: usize,
    ) -> (f64, f64) {
        let marginal_variance: f64 = 1.0 / conditional
            + (0..3)
                .map(|j| weights[j] * weights[j] / parent_precisions[j])
                .sum::<f64>();
        let s = marginal_variance + 1.0 / pi_y;
        let covariance = weights[m] / parent_precisions[m];
        let variance = 1.0 / parent_precisions[m] - covariance * covariance / s;
        (1.0 / variance, covariance / s * w)
    }

    /// With `integrate_other_parents`, a child with three linear parents gives each
    /// parent its exact marginal posterior, for an interior child (finite evidence)
    /// and for a clamped leaf (the fully observed limit). Conditioning the child on
    /// every parent instead treats the other two as known and overstates the
    /// precision each parent gains.
    #[test]
    fn test_integrating_other_parents_is_exact_for_linear_coupling() {
        let weights = [0.8, -1.3, 0.5];
        let parent_precisions = [1.2, 0.7, 2.0];
        let conditional = 1.5;
        for clamped in [false, true] {
            // A clamped leaf is observed exactly: its evidence precision is infinite
            // and its stored precision is the conditional predicted precision.
            let (pi_y, w) = if clamped {
                (f64::INFINITY, 0.9)
            } else {
                (2.5, 0.9)
            };
            let marginal = 1.0
                / (1.0 / conditional
                    + (0..3)
                        .map(|j| weights[j] * weights[j] / parent_precisions[j])
                        .sum::<f64>());
            let (precision, expected_precision, delta) = if clamped {
                (conditional, conditional, w)
            } else {
                let posterior = marginal + pi_y;
                (posterior, marginal, pi_y * w / posterior)
            };

            let mut parent = LayerState::create(3, true);
            parent.expected_mean = Array1::from_vec(vec![0.0; 3]);
            parent.expected_precision =
                Array1::from_vec(parent_precisions.iter().map(|&p| p as Float).collect());
            let mut child = LayerState::create(1, true);
            child.conditional_expected_precision = Array1::from_vec(vec![conditional as Float]);
            child.expected_precision = Array1::from_vec(vec![expected_precision as Float]);
            child.precision = Array1::from_vec(vec![precision as Float]);
            child.value_prediction_error = Array1::from_vec(vec![delta as Float]);
            let w_matrix =
                Matrix::from_shape_vec((1, 3), weights.iter().map(|&x| x as Float).collect())
                    .unwrap();

            layer_posterior_update(
                &mut parent,
                &child,
                &w_matrix,
                linear(),
                false,
                1e10,
                clamped,
                false, // exact_moments
                true,
            );

            for m in 0..3 {
                let (exact_precision, exact_shift) =
                    exact_parent_posterior(weights, parent_precisions, conditional, pi_y, w, m);
                let precision = parent.precision[m] as f64;
                let shift = parent.mean[m] as f64;
                assert!(
                    (precision - exact_precision).abs() < 1e-4 * exact_precision,
                    "clamped={clamped} parent {m}: precision {precision} vs exact {exact_precision}"
                );
                assert!(
                    (shift - exact_shift).abs() < 1e-4 * (1.0 + exact_shift.abs()),
                    "clamped={clamped} parent {m}: mean {shift} vs exact {exact_shift}"
                );
            }
        }
    }

    /// The prefix and suffix sums of `add_edge_messages` give the sum over the other
    /// parents of every edge, as the direct sum does.
    #[test]
    fn test_add_edge_messages_sums_the_other_parents() {
        let weights = Array1::from_vec(vec![0.4, -1.1, 2.0, 0.3]);
        let parent_variance = Array1::from_vec(vec![0.5, 1.5, 0.25, 3.0]);
        let child = ChildNode {
            conditional_expected_precision: 2.0,
            expected_precision: 2.0,
            precision: 2.0,
            value_prediction_error: 1.0,
        };
        let mut precision_sum = Array1::<Float>::zeros(4);
        let mut message = Array1::<Float>::zeros(4);
        let mut suffix = Vec::new();
        add_edge_messages(
            weights.view(),
            parent_variance.view(),
            child,
            true,
            &mut suffix,
            precision_sum.view_mut(),
            message.view_mut(),
        );
        for m in 0..4 {
            let others: Float = (0..4)
                .filter(|&j| j != m)
                .map(|j| weights[j] * weights[j] * parent_variance[j])
                .sum();
            let conditional = 1.0 / (0.5 + others);
            assert!((message[m] - weights[m] * conditional).abs() < 1e-6);
            assert!((precision_sum[m] - weights[m] * weights[m] * conditional).abs() < 1e-6);
        }
    }

    /// With a tanh coupling, the other parents enter through the slope that the
    /// prediction step gives them: the derivative of the coupling function at their
    /// predicted means, or its mean slope under their predicted beliefs with the
    /// exact coupling moments. This is also the slope of the edge being updated, and
    /// the precision is first-order in the coupling function. The expected values
    /// come from the JAX kernel on the same inputs
    /// (`test_integrating_other_parents_with_curved_coupling` in
    /// `tests/test_updates/posterior/test_relaxed_edge_factors.py`).
    #[test]
    fn test_integrating_other_parents_with_curved_coupling() {
        let coupling_fn = resolve_coupling_fn("tanh");
        let weights = Matrix::from_shape_vec((2, 3), vec![0.8, -0.5, 0.3, -0.4, 0.9, 0.6]).unwrap();
        for (exact_moments, expected_precision, expected_mean) in [
            (
                false,
                [1.6092200770131118, 1.1962781289380784, 2.2266348300871837],
                [0.8512406330237166, -1.3480584013452446, 1.048347784353152],
            ),
            (
                true,
                [1.3969053308793788, 1.073988550186877, 2.243046824140868],
                [0.756508702322462, -1.258488016643296, 1.0325545280990016],
            ),
        ] {
            let mut parent = LayerState::create(3, true);
            parent.expected_mean = Array1::from_vec(vec![0.3, -0.6, 1.1]);
            parent.expected_precision = Array1::from_vec(vec![1.2, 0.9, 2.2]);
            parent.cache_linearisation(coupling_fn);
            let mut child = LayerState::create(2, true);
            child.precision = Array1::from_vec(vec![3.0, 2.4]);
            child.expected_precision = Array1::from_vec(vec![2.0, 1.5]);
            child.conditional_expected_precision = Array1::from_vec(vec![2.5, 1.8]);
            child.value_prediction_error = Array1::from_vec(vec![0.4, -0.7]);

            layer_posterior_update(
                &mut parent,
                &child,
                &weights,
                coupling_fn,
                false,
                1e10,
                false,
                exact_moments,
                true,
            );

            for m in 0..3 {
                let (p, e) = (parent.precision[m] as f64, expected_precision[m]);
                assert!(
                    (p - e).abs() < 1e-5 * e,
                    "exact_moments={exact_moments} parent {m}: precision {p} vs JAX {e}"
                );
                let (mu, e) = (parent.mean[m] as f64, expected_mean[m]);
                assert!(
                    (mu - e).abs() < 1e-5 * (1.0 + e.abs()),
                    "exact_moments={exact_moments} parent {m}: mean {mu} vs JAX {e}"
                );
            }
        }
    }
}
