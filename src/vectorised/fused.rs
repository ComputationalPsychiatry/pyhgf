//! Fused batch step for pinned-confidence networks.
//!
//! In the configuration the deep learning notebooks run, volatile layers
//! without a volatility level and with linear coupling, every precision the
//! batch step computes is independent of the data: the predicted, posterior,
//! and effective precisions and the smoothing gains depend only on the
//! weights, the prior precisions, and the time step, so across the batch
//! they are per-node *vectors*, not `(n_nodes, n_samples)` matrices. The
//! general kernels of [`crate::vectorised::batched`] nevertheless carry them
//! as matrices, one elementwise pass each over the whole batch.
//!
//! This module computes those vectors once per call ([`pinned_profile`]) and
//! runs the per-sample work of the sweep with them folded in as row
//! scalings ([`fused_sweep`]): one gemm per layer forward, one gemm per
//! interior layer for the routed errors, and one gemm per layer for the
//! batch-mean weight gradients, which is the same shape of computation as a
//! backpropagation step. The update it computes is identical to the general
//! path (asserted in the tests to floating point tolerance); only the
//! execution differs. [`pinned_profile`] returns `None` for any network
//! outside the regime (a volatility level, a nonlinear coupling function, a
//! binary or categorical layer), and the caller falls back to the general
//! kernels.

use crate::math::CouplingKind;
use crate::updates::vectorised::learning::WeightKind;
use crate::vectorised::batched::{coupled_activations, BatchedLayerState, ConfidenceIncrements};
use crate::vectorised::layer::{DeepNet, LayerKind};
use crate::vectorised::mat::{Float, Matrix, Vector};
use ndarray::{s, ArrayView2, Zip};

/// The data-independent quantities of one pinned-confidence batch step:
/// per-layer, per-node vectors, valid for the current weights.
pub struct PinnedProfile {
    /// Posterior precision of each layer after the update sweep (clipped);
    /// equal to the prior on the clamped boundary layers.
    posterior_precision: Vec<Vector>,
    /// Smoothing gain of each layer in its role as the child of the
    /// interleaved posterior updates (unused entry for the top layer).
    gain: Vec<Vector>,
}

/// Compute the pinned-confidence profile of `net`, or `None` when the
/// network is outside the pinned regime and the general kernels must run.
pub fn pinned_profile(net: &DeepNet, time_step: Float) -> Option<PinnedProfile> {
    let n = net.layers.len();
    if n < 2 {
        return None;
    }
    for layer in &net.layers {
        if layer.kind != LayerKind::Volatile
            || layer.has_volatility_parent
            || layer.coupling_fn.kind != CouplingKind::Linear
        {
            return None;
        }
    }

    // Prediction sweep, top-down: the marginal predicted precision
    // π̃ = 1/(1/π_prior + Σ_j W²·t²/π̃_parent); the clamped top keeps its
    // template value and the observed bottom leaf keeps its prior.
    let mut expected: Vec<Option<Vector>> = vec![None; n];
    expected[n - 1] = Some(net.layers[n - 1].state.expected_precision.clone());
    for i in (0..n - 1).rev() {
        let child_layer = &net.layers[i];
        let parent_layer = &net.layers[i + 1];
        let weights = parent_layer.weights_in.as_ref()?;
        let prior = &child_layer.state.precision;
        let value = if child_layer.is_input_layer {
            prior.clone()
        } else {
            let cols = weights.ncols() - usize::from(parent_layer.add_constant_input);
            let w = weights.slice(s![.., ..cols]);
            let parent_pi = expected[i + 1].as_ref().expect("computed top-down");
            let ppv = parent_pi.mapv(|p| (time_step * time_step) / p);
            let vcv = w.mapv(|x| x * x).dot(&ppv);
            Zip::from(prior)
                .and(&vcv)
                .map_collect(|&pr, &v| 1.0 / (1.0 / pr + v))
        };
        expected[i] = Some(value);
    }
    let expected: Vec<Vector> = expected.into_iter().map(|v| v.expect("filled")).collect();

    // Update sweep, bottom-up: posterior precisions and the smoothing gains
    // of each child. The conditional predicted precision is the prior
    // throughout (no volatility source), and the leaf's posterior equals its
    // prior (the leaf's value-level precision is not updated).
    let max_pp = net.max_posterior_precision;
    let mut posterior: Vec<Vector> = Vec::with_capacity(n);
    let mut gain: Vec<Vector> = Vec::with_capacity(n);
    posterior.push(net.layers[0].state.precision.clone());
    for i in 1..n {
        let child_layer = &net.layers[i - 1];
        let cond = &child_layer.state.precision;
        let child_pi = &posterior[i - 1];
        let child_expected = &expected[i - 1];
        // The child's smoothing gain and effective precision, as scalars per
        // node: gain = π̂·π/(π̂ + π − π̃), eff = π̂·(π − π̃)/(π̂ + π − π̃),
        // with the leaf short-circuiting eff to π̂.
        let mut eff = Vector::zeros(cond.len());
        let mut g = Vector::zeros(cond.len());
        Zip::from(&mut g)
            .and(&mut eff)
            .and(cond)
            .and(child_pi)
            .and(child_expected)
            .for_each(|g, e, &cep, &prec, &ep| {
                let pi_y = prec - ep;
                *e = if child_layer.is_input_layer {
                    cep
                } else {
                    cep * pi_y / (cep + pi_y)
                };
                *g = cep * prec / (cep + pi_y);
            });
        gain.push(g);

        if i == n - 1 {
            // The clamped top layer is never posterior updated.
            posterior.push(net.layers[i].state.precision.clone());
        } else {
            let parent_layer = &net.layers[i];
            let weights = parent_layer.weights_in.as_ref()?;
            let cols = weights.ncols() - usize::from(parent_layer.add_constant_input);
            let w = weights.slice(s![.., ..cols]);
            let pc1 = w.mapv(|x| x * x).t().dot(&eff);
            let value = Zip::from(&expected[i])
                .and(&pc1)
                .map_collect(|&ep, &b| (ep + b).max(ep).min(max_pp));
            posterior.push(value);
        }
    }
    gain.push(Vector::zeros(0)); // unused: the top layer is never a child

    Some(PinnedProfile {
        posterior_precision: posterior,
        gain,
    })
}

/// Multiply each row of `m` by the matching entry of `v`, in place.
fn scale_rows(m: &mut Matrix, v: &Vector) {
    Zip::from(m.rows_mut()).and(v).for_each(|mut row, &s| {
        row.mapv_inplace(|x| x * s);
    });
}

/// One pinned-confidence batch step over `states` (the per-chunk buffers of
/// the general path; only the mean fields are touched): returns the routed
/// input errors and, when requested, the batch-mean weight gradients and
/// the confidence increments. Mirrors the general sweep exactly, with every
/// precision read from `profile`.
#[allow(clippy::too_many_arguments)]
pub fn fused_sweep(
    net: &DeepNet,
    profile: &PinnedProfile,
    states: &mut [BatchedLayerState],
    x: ArrayView2<Float>,
    y: ArrayView2<Float>,
    learning: bool,
    kind: WeightKind,
    update_confidences: bool,
) -> (
    Matrix,
    Option<Vec<Option<Matrix>>>,
    Option<ConfidenceIncrements>,
) {
    let n = net.layers.len();
    let n_samples = x.nrows();

    // Forward: clamp the predictors on top and predict every expected mean
    // with one gemm per layer.
    {
        let top = states.last_mut().expect("network has no layers");
        top.expected_mean.assign(&x.t());
        top.mean.assign(&top.expected_mean);
    }
    for i in (1..n).rev() {
        let (lower, upper) = states.split_at_mut(i);
        let child = &mut lower[i - 1];
        let parent = &upper[0];
        let parent_layer = &net.layers[i];
        let weights = parent_layer.weights_in.as_ref().expect("has weights");
        let coupled = coupled_activations(
            &parent.expected_mean,
            parent_layer.coupling_fn,
            parent_layer.add_constant_input,
        );
        child.expected_mean = weights.dot(&coupled);
    }

    // Update: the leaf prediction error, then the interleaved posterior
    // means bottom-up. δ_parent = Wᵀ(gain∘δ_child)/π_parent, and the
    // posterior mean is the expected mean plus that shift.
    {
        let leaf = &mut states[0];
        leaf.mean.assign(&y.t());
        Zip::from(&mut leaf.value_prediction_error)
            .and(&leaf.mean)
            .and(&leaf.expected_mean)
            .for_each(|d, &m, &em| *d = m - em);
    }
    for i in 1..n.saturating_sub(1) {
        let (lower, upper) = states.split_at_mut(i);
        let child = &lower[i - 1];
        let parent = &mut upper[0];
        let parent_layer = &net.layers[i];
        let weights = parent_layer.weights_in.as_ref().expect("has weights");
        let cols = weights.ncols() - usize::from(parent_layer.add_constant_input);
        let w = weights.slice(s![.., ..cols]);

        let mut gd = child.value_prediction_error.clone();
        scale_rows(&mut gd, &profile.gain[i - 1]);
        let msg = w.t().dot(&gd);
        let pi = &profile.posterior_precision[i];
        Zip::from(parent.value_prediction_error.rows_mut())
            .and(msg.rows())
            .and(pi)
            .for_each(|mut out, m_row, &p| {
                Zip::from(&mut out).and(&m_row).for_each(|o, &m| *o = m / p);
            });
        Zip::from(&mut parent.mean)
            .and(&parent.expected_mean)
            .and(&parent.value_prediction_error)
            .for_each(|m, &em, &d| *m = em + d);
    }

    // Routed input errors: the top weights against the gain-scaled errors of
    // the layer below (linear coupling: no derivative factor).
    let errors = {
        let top_layer = &net.layers[n - 1];
        let child = &states[n - 2];
        let w = top_layer.weights_in.as_ref().expect("has weights");
        let cols = w.ncols() - usize::from(top_layer.add_constant_input);
        let w = w.slice(s![.., ..cols]);
        let mut gd = child.value_prediction_error.clone();
        scale_rows(&mut gd, &profile.gain[n - 2]);
        w.t().dot(&gd)
    };

    // Batch-mean weight gradients: u = -(δ [∘ π]), v = the parent's
    // posterior mean with a bias row of ones, one gemm per layer.
    let grads = learning.then(|| {
        let sanitize = |x: Float| if x.is_finite() { x } else { 0.0 };
        let mut out: Vec<Option<Matrix>> = Vec::with_capacity(n);
        out.push(None);
        for i in 1..n {
            let parent_layer = &net.layers[i];
            let parent = &states[i];
            let child = &states[i - 1];
            let mut u = Matrix::zeros(child.value_prediction_error.raw_dim());
            if kind == WeightKind::PrecisionWeighted {
                let pi = &profile.posterior_precision[i - 1];
                Zip::from(u.rows_mut())
                    .and(child.value_prediction_error.rows())
                    .and(pi)
                    .for_each(|mut urow, drow, &p| {
                        Zip::from(&mut urow)
                            .and(&drow)
                            .for_each(|u, &d| *u = sanitize(-d * p));
                    });
            } else {
                Zip::from(&mut u)
                    .and(&child.value_prediction_error)
                    .for_each(|u, &d| *u = sanitize(-d));
            }
            let n_parent = parent.mean.nrows();
            let rows = n_parent + usize::from(parent_layer.add_constant_input);
            let mut v = Matrix::zeros((rows, n_samples));
            Zip::from(v.slice_mut(s![..n_parent, ..]))
                .and(&parent.mean)
                .for_each(|c, &m| *c = sanitize(m));
            if parent_layer.add_constant_input {
                v.row_mut(n_parent).fill(1.0);
            }
            let mut grad = u.dot(&v.t());
            grad.mapv_inplace(|g| g / n_samples as Float);
            out.push(Some(grad));
        }
        out
    });

    // Confidence increments: identical for every sample, so the batch mean
    // is the posterior minus the prior directly (zero on the boundary
    // layers, matching the general path).
    let increments = update_confidences.then(|| {
        net.layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                let inc = &profile.posterior_precision[i] - &layer.state.precision;
                (inc, None)
            })
            .collect::<ConfidenceIncrements>()
    });

    (errors, grads, increments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorised::batched::{
        batched_confidence_increments, batched_input_prediction_error,
        batched_mean_weight_gradients, batched_prediction_sweep, batched_update_sweep,
    };
    use crate::vectorised::layer::{LayerConfig, LayerState};
    use ndarray::Array1;

    /// An eligible network: volatile layers without volatility levels,
    /// linear coupling, varied weights and per-node priors.
    fn make_pinned_net(sizes: &[usize], bias: bool) -> DeepNet {
        let configs: Vec<LayerConfig> = sizes
            .iter()
            .map(|&size| {
                let mut cfg = LayerConfig::new(size);
                cfg.volatility_parent = false;
                cfg.add_constant_input = bias;
                cfg
            })
            .collect();
        let mut net = DeepNet::from_configs(&configs).unwrap();
        for (l, layer) in net.layers.iter_mut().enumerate() {
            if let Some(w) = layer.weights_in.as_mut() {
                for ((i, j), v) in w.indexed_iter_mut() {
                    *v = 0.4 * ((i + 2 * j + l) as Float * 0.7).sin();
                }
            }
            let n = layer.state.precision.len();
            layer.state.precision =
                Array1::from_shape_fn(n, |k| 0.5 + 0.3 * ((k + l) as Float * 1.3).cos().abs());
            layer.state.expected_precision = layer.state.precision.clone();
        }
        net
    }

    fn data(net: &DeepNet, n_samples: usize) -> (Matrix, Matrix) {
        let top = net.layers.last().unwrap().state.precision.len();
        let bottom = net.layers[0].state.precision.len();
        let x = Matrix::from_shape_fn((n_samples, top), |(i, j)| {
            ((i * top + j) as Float * 0.9).cos()
        });
        let y = Matrix::from_shape_fn((n_samples, bottom), |(i, j)| {
            ((i * bottom + j) as Float * 0.4).sin()
        });
        (x, y)
    }

    fn close(a: Float, b: Float) -> bool {
        (a - b).abs() < 2e-4 * (1.0 + a.abs().max(b.abs()))
    }

    /// The fused sweep reproduces the general kernels on an eligible network:
    /// routed input errors, batch-mean weight gradients under both learning
    /// kinds, and confidence increments.
    #[test]
    fn test_fused_matches_general_kernels() {
        for bias in [true, false] {
            for kind in [WeightKind::PrecisionWeighted, WeightKind::Standard] {
                let net = make_pinned_net(&[3, 5, 4, 6], bias);
                let (x, y) = data(&net, 33);
                let templates: Vec<&LayerState> =
                    net.layers.iter().map(|layer| &layer.state).collect();

                let mut general: Vec<BatchedLayerState> = templates
                    .iter()
                    .map(|t| BatchedLayerState::from_template(t, 33))
                    .collect();
                batched_prediction_sweep(&net, &mut general, x.view(), 1.0);
                batched_update_sweep(&net, &mut general, y.view(), 1.0);
                let errors_g = batched_input_prediction_error(&net, &general);
                let grads_g = batched_mean_weight_gradients(&net, &general, kind);
                let incs_g = batched_confidence_increments(&general, &templates);

                let profile = pinned_profile(&net, 1.0).expect("eligible network");
                let mut fused: Vec<BatchedLayerState> = templates
                    .iter()
                    .map(|t| BatchedLayerState::from_template(t, 33))
                    .collect();
                let (errors_f, grads_f, incs_f) = fused_sweep(
                    &net,
                    &profile,
                    &mut fused,
                    x.view(),
                    y.view(),
                    true,
                    kind,
                    true,
                );

                for (a, b) in errors_f.iter().zip(errors_g.iter()) {
                    assert!(close(*a, *b), "input errors differ: {a} vs {b}");
                }
                for (gf, gg) in grads_f.unwrap().iter().zip(grads_g.iter()) {
                    match (gf, gg) {
                        (None, None) => {}
                        (Some(gf), Some(gg)) => {
                            for (a, b) in gf.iter().zip(gg.iter()) {
                                assert!(close(*a, *b), "gradients differ: {a} vs {b}");
                            }
                        }
                        _ => panic!("gradient layout differs"),
                    }
                }
                for ((pf, vf), (pg, vg)) in incs_f.unwrap().iter().zip(incs_g.iter()) {
                    assert!(vf.is_none() && vg.is_none());
                    for (a, b) in pf.iter().zip(pg.iter()) {
                        assert!(close(*a, *b), "increments differ: {a} vs {b}");
                    }
                }
            }
        }
    }

    /// Networks outside the pinned regime yield no profile.
    #[test]
    fn test_profile_eligibility() {
        assert!(pinned_profile(&make_pinned_net(&[3, 4, 5], true), 1.0).is_some());

        let mut with_vol = vec![LayerConfig::new(3), LayerConfig::new(4)];
        with_vol[1].volatility_parent = true;
        with_vol[0].volatility_parent = false;
        let net = DeepNet::from_configs(&with_vol).unwrap();
        assert!(pinned_profile(&net, 1.0).is_none());

        let mut tanh = LayerConfig::new(4);
        tanh.volatility_parent = false;
        tanh.coupling_fn = crate::math::parse_coupling_fn("tanh").unwrap();
        let mut bottom = LayerConfig::new(3);
        bottom.volatility_parent = false;
        let net = DeepNet::from_configs(&[bottom, tanh]).unwrap();
        assert!(pinned_profile(&net, 1.0).is_none());

        let mut binary = LayerConfig::new(3);
        binary.kind = LayerKind::Binary;
        let mut top = LayerConfig::new(4);
        top.volatility_parent = false;
        let net = DeepNet::from_configs(&[binary, top]).unwrap();
        assert!(pinned_profile(&net, 1.0).is_none());
    }

    /// A general call after a fused call on the same workspace is unaffected
    /// by the stale precision buffers the fused call leaves behind.
    #[test]
    fn test_fused_then_general_workspace_is_clean() {
        use crate::updates::vectorised::learning::WeightKind;
        use crate::vectorised::batched::BatchWorkspace;
        use crate::vectorised::optimiser::{OptState, Optimizer};

        // An ineligible net (volatility levels on) with the same layer sizes
        // as the eligible one, so the workspace buffers are shape-compatible.
        let make_general_net = || {
            let configs: Vec<LayerConfig> = [3usize, 5, 4]
                .iter()
                .map(|&s| LayerConfig::new(s))
                .collect();
            DeepNet::from_configs(&configs).unwrap()
        };
        let pinned = || make_pinned_net(&[3, 5, 4], true);
        let (x, y) = data(&pinned(), 17);
        let opt = Optimizer::adam(0.01);

        // Reference: the general net alone on a fresh workspace.
        let mut net_ref = make_general_net();
        let mut opt_ref = OptState::init(&net_ref);
        let mut ws_ref = BatchWorkspace::default();
        let errors_ref = net_ref.batch_update_with_workspace(
            &mut ws_ref,
            x.view(),
            y.view(),
            Some(&opt),
            Some(&mut opt_ref),
            1.0,
            WeightKind::PrecisionWeighted,
            true,
        );

        // The same general call on a workspace a fused call has dirtied.
        let mut ws = BatchWorkspace::default();
        let mut fused_net = pinned();
        let mut fused_opt = OptState::init(&fused_net);
        fused_net.batch_update_with_workspace(
            &mut ws,
            x.view(),
            y.view(),
            Some(&opt),
            Some(&mut fused_opt),
            1.0,
            WeightKind::PrecisionWeighted,
            true,
        );
        let mut net_b = make_general_net();
        let mut opt_b = OptState::init(&net_b);
        let errors_b = net_b.batch_update_with_workspace(
            &mut ws,
            x.view(),
            y.view(),
            Some(&opt),
            Some(&mut opt_b),
            1.0,
            WeightKind::PrecisionWeighted,
            true,
        );
        assert_eq!(errors_ref, errors_b);
        for (a, b) in net_ref.layers.iter().zip(net_b.layers.iter()) {
            assert_eq!(a.weights_in, b.weights_in);
            assert_eq!(a.state.precision, b.state.precision);
        }
    }
}
