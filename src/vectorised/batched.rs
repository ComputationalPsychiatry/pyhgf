//! Batch-synchronous sweeps over all samples at once, in a features × samples
//! layout: every phase of [`crate::vectorised::network::DeepNet::batch_update`]
//! is one `gemm` per layer instead of one `gemv` per layer per sample.
//!
//! Each sample of a batch starts from the same state template, so the whole
//! batch can be swept together: a [`BatchedLayerState`] holds every
//! [`LayerState`] field with a trailing samples axis (shape
//! `(n_nodes, n_samples)`), and the kernels below mirror the per-sample
//! kernels of [`crate::updates::vectorised`] term for term. Elementwise
//! formulas are shared with the per-sample path through the per-node scalar
//! functions of [`crate::updates::vectorised::volatile::prediction_error`],
//! so the two paths cannot drift; only the weight contractions change, from
//! matrix-vector to matrix-matrix products.

use crate::math::sigmoid;
use crate::updates::vectorised::learning::WeightKind;
use crate::updates::vectorised::softmax_inplace;
use crate::updates::vectorised::volatile::guarded_volatility;
use crate::updates::vectorised::volatile::prediction_error::{
    ehgf_vol_node, standard_vol_node, unbounded_vol_node, volatility_pe_node,
};
use crate::vectorised::layer::{DeepNet, Layer, LayerKind, LayerState, VolatilityUpdate};
use crate::vectorised::mat::{Matrix, Vector};
use ndarray::parallel::prelude::*;
use ndarray::{s, ArrayView2, Axis, Zip};

/// A [`LayerState`] with a trailing samples axis: every field is a
/// `(n_nodes, n_samples)` matrix, one column per sample. The
/// `effective_precision_vol` and `volatility_prediction_error` fields are
/// omitted: nothing in the batch step reads them (the volatility prediction
/// error is consumed inline by the volatility posterior).
pub struct BatchedLayerState {
    pub mean: Matrix,
    pub precision: Matrix,
    pub expected_mean: Matrix,
    pub expected_precision: Matrix,
    pub conditional_expected_precision: Matrix,
    pub effective_precision: Matrix,
    pub value_prediction_error: Matrix,
    pub mean_vol: Option<Matrix>,
    pub precision_vol: Option<Matrix>,
    pub expected_mean_vol: Option<Matrix>,
    pub expected_precision_vol: Option<Matrix>,
}

/// Repeat a per-node vector across `n_samples` columns.
fn tile(v: &Vector, n_samples: usize) -> Matrix {
    v.view()
        .insert_axis(Axis(1))
        .broadcast((v.len(), n_samples))
        .expect("broadcast to (n, samples)")
        .to_owned()
}

impl BatchedLayerState {
    /// Broadcast a state template to `n_samples` identical columns; the
    /// starting point of every batch step.
    ///
    /// Only the fields the sweeps read before writing carry the template
    /// values: the value-level posterior precision (every layer), the
    /// marginal predicted precision (read on the clamped top layer, which is
    /// never predicted), and the volatility level's belief. Every other field
    /// is written by the sweeps before it is read, so it starts as a zeroed
    /// allocation the copy never has to touch.
    pub fn from_template(state: &LayerState, n_samples: usize) -> Self {
        let n = state.n_nodes();
        let z = || Matrix::zeros((n, n_samples));
        Self {
            mean: z(),
            precision: tile(&state.precision, n_samples),
            expected_mean: z(),
            expected_precision: tile(&state.expected_precision, n_samples),
            conditional_expected_precision: z(),
            effective_precision: z(),
            value_prediction_error: z(),
            mean_vol: state.mean_vol.as_ref().map(|v| tile(v, n_samples)),
            precision_vol: state.precision_vol.as_ref().map(|v| tile(v, n_samples)),
            expected_mean_vol: state.expected_mean_vol.as_ref().map(|_| z()),
            expected_precision_vol: state.expected_precision_vol.as_ref().map(|_| z()),
        }
    }

    fn n_samples(&self) -> usize {
        self.mean.ncols()
    }
}

/// Coupled parent activations for the whole batch: `g(μ̂_parent)` with a
/// trailing bias row of ones, shape `(n_parent[+1], n_samples)`. Every entry
/// is written exactly once (activation rows by the coupling pass, the bias
/// row by a fill), so the zeroed allocation is never touched twice.
fn coupled_activations(
    parent_field: &Matrix,
    coupling_fn: &'static crate::math::CouplingFn,
    parent_has_constant: bool,
) -> Matrix {
    let (n_parent, n_samples) = parent_field.dim();
    let rows = n_parent + usize::from(parent_has_constant);
    let mut coupled = Matrix::zeros((rows, n_samples));
    crate::math::with_coupling!(coupling_fn, |f, _df, _d2f| {
        Zip::from(coupled.slice_mut(s![..n_parent, ..]))
            .and(parent_field)
            .for_each(|c, &m| *c = f(m));
    });
    if parent_has_constant {
        coupled.row_mut(n_parent).fill(1.0);
    }
    coupled
}

/// Predict a volatile child layer from its parent for the whole batch,
/// mirroring the per-sample `layer_prediction`.
#[allow(clippy::too_many_arguments)]
fn batched_volatile_prediction(
    child: &mut BatchedLayerState,
    child_layer: &Layer,
    parent: &BatchedLayerState,
    weights: &Matrix,
    parent_layer: &Layer,
    time_step: f64,
) {
    let params = &child_layer.params;
    let (n, n_samples) = child.mean.dim();
    let n_parent = parent.expected_mean.nrows();
    let p = weights.ncols();

    // 1. Volatility level (internal): the same elementwise formulas as the
    // per-sample kernel, over every column at once.
    if child_layer.has_volatility_parent {
        let mean_vol = child.mean_vol.as_ref().expect("volatility mean");
        let precision_vol = child.precision_vol.as_ref().expect("volatility precision");

        let mut emv = Matrix::zeros((n, n_samples));
        let mut epv = Matrix::zeros((n, n_samples));
        Zip::from(emv.rows_mut())
            .and(epv.rows_mut())
            .and(mean_vol.rows())
            .and(precision_vol.rows())
            .and(&params.autoconnection_strength_vol)
            .and(&params.tonic_volatility_vol)
            .for_each(|mut em_row, mut ep_row, mv, pv, &a, &tvv| {
                let pvv = guarded_volatility(tvv, time_step);
                Zip::from(&mut em_row)
                    .and(&mut ep_row)
                    .and(&mv)
                    .and(&pv)
                    .for_each(|em, ep, &m, &pv_prec| {
                        *em = a * m;
                        *ep = 1.0 / (1.0 / pv_prec + pvv);
                    });
            });
        child.expected_mean_vol = Some(emv);
        child.expected_precision_vol = Some(epv);
    }

    // 2. Value level: coupled activations and the per-parent Laplace variance
    // factor (t·g'(μ̂))² / π̃_parent (the bias row stays 0 there).
    let coupled = coupled_activations(
        &parent.expected_mean,
        parent_layer.coupling_fn,
        parent_layer.add_constant_input,
    );
    let mut ppv = Matrix::zeros((p, n_samples));
    crate::math::with_coupling!(parent_layer.coupling_fn, |_f, df, _d2f| {
        Zip::from(ppv.slice_mut(s![..n_parent, ..]))
            .and(&parent.expected_mean)
            .and(&parent.expected_precision)
            .for_each(|o, &mu, &ep| {
                let num = time_step * df(mu);
                *o = (num * num) / ep;
            });
    });

    // Mean prediction: one gemm over the whole batch.
    child.expected_mean = weights.dot(&coupled);

    // Laplace value-coupling variance: W² @ ppv, one gemm.
    let w2 = weights.mapv(|w| w * w);
    let value_coupling_variance = w2.dot(&ppv);

    // Conditional (π̂), marginal (π̃), and effective (γ) predicted precisions,
    // fused into a single pass over the state: π̂ = 1/(1/π + Ω),
    // π̃ = 1/(1/π̂ + value_coupling_variance), γ = Ω·π̃, with
    // Ω = t·exp(ω + κ·μ_vol + κ²/(2·π̂_vol)) the predicted volatility.
    let precision_triple = |c: &mut f64, e: &mut f64, ef: &mut f64, pr: f64, pv: f64, v: f64| {
        *c = 1.0 / (1.0 / pr + pv);
        *e = 1.0 / (1.0 / *c + v);
        *ef = pv * *e;
    };
    if child_layer.is_input_layer {
        // Input/leaf override: no random walk between observations.
        child
            .conditional_expected_precision
            .assign(&child.precision);
        child.expected_precision.assign(&child.precision);
        child.effective_precision.fill(0.0);
    } else if child_layer.has_volatility_parent {
        let emv = child.expected_mean_vol.as_ref().unwrap();
        let epv = child.expected_precision_vol.as_ref().unwrap();
        let mut predicted_vol = Matrix::zeros((n, n_samples));
        Zip::from(predicted_vol.rows_mut())
            .and(emv.rows())
            .and(epv.rows())
            .and(&params.tonic_volatility)
            .and(&params.volatility_coupling)
            .for_each(|mut out, em, ep, &t, &k| {
                Zip::from(&mut out).and(&em).and(&ep).for_each(|o, &m, &pvv| {
                    *o = guarded_volatility(t + k * m + (k * k) / (pvv * 2.0), time_step);
                });
            });
        Zip::from(&mut child.conditional_expected_precision)
            .and(&mut child.expected_precision)
            .and(&mut child.effective_precision)
            .and(&child.precision)
            .and(&predicted_vol)
            .and(&value_coupling_variance)
            .for_each(|c, e, ef, &pr, &pv, &v| precision_triple(c, e, ef, pr, pv, v));
    } else {
        // Without a volatility level Ω is per node, not per sample: no tiled
        // matrix, one row scalar each.
        let per_node = params
            .tonic_volatility
            .mapv(|t| guarded_volatility(t, time_step));
        Zip::from(child.conditional_expected_precision.rows_mut())
            .and(child.expected_precision.rows_mut())
            .and(child.effective_precision.rows_mut())
            .and(child.precision.rows())
            .and(value_coupling_variance.rows())
            .and(&per_node)
            .for_each(|mut crow, mut erow, mut efrow, prow, vrow, &pv| {
                Zip::from(&mut crow)
                    .and(&mut erow)
                    .and(&mut efrow)
                    .and(&prow)
                    .and(&vrow)
                    .for_each(|c, e, ef, &pr, &v| precision_triple(c, e, ef, pr, pv, v));
            });
    }
}

/// Predict a binary leaf layer for the whole batch, mirroring the per-sample
/// `binary_prediction`.
fn batched_binary_prediction(
    child: &mut BatchedLayerState,
    parent: &BatchedLayerState,
    weights: &Matrix,
    parent_layer: &Layer,
    precision_clipping_value: f64,
) {
    let coupled = coupled_activations(
        &parent.expected_mean,
        parent_layer.coupling_fn,
        parent_layer.add_constant_input,
    );
    let mut logit = weights.dot(&coupled);
    logit.mapv_inplace(|x| {
        sigmoid(x).clamp(precision_clipping_value, 1.0 - precision_clipping_value)
    });
    let expected_precision = logit.mapv(|m| m * (1.0 - m));
    child.expected_mean = logit;
    child.conditional_expected_precision = expected_precision.clone();
    child.expected_precision = expected_precision;
}

/// Predict a categorical leaf layer for the whole batch, mirroring the
/// per-sample `categorical_prediction` (one softmax per sample column).
fn batched_categorical_prediction(
    child: &mut BatchedLayerState,
    parent: &BatchedLayerState,
    weights: &Matrix,
    parent_layer: &Layer,
) {
    let coupled = coupled_activations(
        &parent.expected_mean,
        parent_layer.coupling_fn,
        parent_layer.add_constant_input,
    );
    let mut logits = weights.dot(&coupled);
    for col in logits.columns_mut() {
        softmax_inplace(col);
    }
    child.expected_mean = logits;
    child.expected_precision.fill(1.0);
    child.conditional_expected_precision.fill(1.0);
    child.effective_precision.fill(0.0);
}

/// Prediction errors and the volatility-level posterior for a layer, over the
/// whole batch; mirrors the per-sample `layer_prediction_error` (with the
/// binary and categorical leaf variants folded in via the layer kind).
fn batched_prediction_error(
    layer_state: &mut BatchedLayerState,
    layer: &Layer,
    volatility_updates: VolatilityUpdate,
    time_step: f64,
    max_posterior_precision: f64,
) {
    match layer.kind {
        LayerKind::Binary => {
            // δ = (mean − μ̂) / π̂, posterior precision = expected precision.
            Zip::from(&mut layer_state.value_prediction_error)
                .and(&layer_state.mean)
                .and(&layer_state.expected_mean)
                .and(&layer_state.expected_precision)
                .for_each(|d, &m, &em, &ep| *d = (m - em) / ep);
            layer_state.precision.assign(&layer_state.expected_precision);
            return;
        }
        LayerKind::Categorical => {
            Zip::from(&mut layer_state.value_prediction_error)
                .and(&layer_state.mean)
                .and(&layer_state.expected_mean)
                .for_each(|d, &m, &em| *d = m - em);
            layer_state.precision.assign(&layer_state.expected_precision);
            return;
        }
        LayerKind::Volatile => {}
    }

    // Value prediction error (always).
    Zip::from(&mut layer_state.value_prediction_error)
        .and(&layer_state.mean)
        .and(&layer_state.expected_mean)
        .for_each(|d, &m, &em| *d = m - em);

    if !layer.has_volatility_parent {
        return;
    }

    // Volatility PE and the volatility-level posterior, node by node, sample
    // by sample, through the same scalar functions as the per-sample path.
    // This is the transcendental-heavy part of the batch step (several
    // exponentials, and a Lambert-W solve in the unbounded scheme, per node
    // per sample), so the node rows run in parallel; every row's samples are
    // contiguous in the row-major layout.
    let (n, n_samples) = layer_state.mean.dim();
    let params = &layer.params;
    let emv = layer_state.expected_mean_vol.as_ref().expect("volatility level");
    let epv = layer_state
        .expected_precision_vol
        .as_ref()
        .expect("volatility level");
    let mut new_precision_vol = Matrix::zeros((n, n_samples));
    let mut new_mean_vol = Matrix::zeros((n, n_samples));
    let mean = &layer_state.mean;
    let expected_mean = &layer_state.expected_mean;
    let precision = &layer_state.precision;
    let expected_precision = &layer_state.expected_precision;
    let conditional = &layer_state.conditional_expected_precision;
    let effective = &layer_state.effective_precision;
    new_precision_vol
        .outer_iter_mut()
        .into_par_iter()
        .zip(new_mean_vol.outer_iter_mut().into_par_iter())
        .enumerate()
        .for_each(|(i, (mut pv_row, mut mv_row))| {
            let k = params.volatility_coupling[i];
            let t = params.tonic_volatility[i];
            for j in 0..n_samples {
                let dpe = volatility_pe_node(
                    expected_precision[[i, j]],
                    precision[[i, j]],
                    mean[[i, j]],
                    expected_mean[[i, j]],
                );
                let (pp, m) = match volatility_updates {
                    VolatilityUpdate::Standard => standard_vol_node(
                        k,
                        effective[[i, j]],
                        dpe,
                        epv[[i, j]],
                        emv[[i, j]],
                        max_posterior_precision,
                    ),
                    VolatilityUpdate::EHgf => ehgf_vol_node(
                        k,
                        t,
                        emv[[i, j]],
                        epv[[i, j]],
                        effective[[i, j]],
                        dpe,
                        conditional[[i, j]],
                        precision[[i, j]],
                        mean[[i, j]],
                        expected_mean[[i, j]],
                        time_step,
                        max_posterior_precision,
                    ),
                    VolatilityUpdate::Unbounded => unbounded_vol_node(
                        k,
                        t,
                        emv[[i, j]],
                        epv[[i, j]],
                        conditional[[i, j]],
                        expected_precision[[i, j]],
                        precision[[i, j]],
                        mean[[i, j]],
                        expected_mean[[i, j]],
                        time_step,
                        max_posterior_precision,
                    ),
                };
                pv_row[j] = pp;
                mv_row[j] = m;
            }
        });
    layer_state.precision_vol = Some(new_precision_vol);
    layer_state.mean_vol = Some(new_mean_vol);
}

/// The per-sample smoothing gain and effective (Schur-corrected) precision of
/// a child layer, `(gain, eff)`, each `(n_child, n_samples)`: the same
/// scalars the per-sample posterior forms on the fly.
fn gain_and_effective(child: &BatchedLayerState, child_is_input_layer: bool) -> (Matrix, Matrix) {
    let dim = child.mean.raw_dim();
    let mut gain = Matrix::zeros(dim);
    let mut eff = Matrix::zeros(dim);
    Zip::from(&mut gain)
        .and(&mut eff)
        .and(&child.conditional_expected_precision)
        .and(&child.precision)
        .and(&child.expected_precision)
        .for_each(|g, e, &cep, &prec, &ep| {
            let pi_y = prec - ep;
            // Leaf children short-circuit to the canonical predicted precision.
            *e = if child_is_input_layer {
                cep
            } else {
                cep * pi_y / (cep + pi_y)
            };
            *g = cep * prec / (cep + pi_y);
        });
    (gain, eff)
}

/// Value-level posterior (precision then mean) of a parent layer from its
/// child's prediction errors, over the whole batch; mirrors the per-sample
/// `layer_posterior_update` with the three weight contractions as gemms.
fn batched_posterior_update(
    parent: &mut BatchedLayerState,
    child: &BatchedLayerState,
    weights: &Matrix,
    parent_layer: &Layer,
    max_posterior_precision: f64,
    child_is_input_layer: bool,
) {
    let ncols = weights.ncols();
    let w = if parent_layer.add_constant_input {
        weights.slice(s![.., ..ncols - 1])
    } else {
        weights.view()
    };
    let w2 = w.mapv(|x| x * x);

    let (gain, eff) = gain_and_effective(child, child_is_input_layer);
    let gd = &gain * &child.value_prediction_error;

    // The weight contractions, one gemm each over the whole batch.
    let pc1_base = w2.t().dot(&eff);
    let msg = w.t().dot(&gd);

    if parent_layer.coupling_fn.kind == crate::math::CouplingKind::Linear {
        // Linear coupling: g' ≡ 1 and g'' ≡ 0, so the curvature term of the
        // precision update vanishes with its gemm, and no derivative matrices
        // are needed.
        Zip::from(&mut parent.precision)
            .and(&parent.expected_precision)
            .and(&pc1_base)
            .for_each(|p, &ep, &b| *p = (ep + b).max(ep).min(max_posterior_precision));
        Zip::from(&mut parent.mean)
            .and(&parent.expected_mean)
            .and(&msg)
            .and(&parent.precision)
            .for_each(|m, &em, &mg, &pp| *m = em + mg / pp);
        return;
    }

    let (mut coupling_prime, mut coupling_second) = {
        let dim = parent.expected_mean.raw_dim();
        (Matrix::zeros(dim), Matrix::zeros(dim))
    };
    crate::math::with_coupling!(parent_layer.coupling_fn, |_f, df, d2f| {
        Zip::from(&mut coupling_prime)
            .and(&mut coupling_second)
            .and(&parent.expected_mean)
            .for_each(|p, sec, &m| {
                *p = df(m);
                *sec = d2f(m);
            });
    });

    let ed = &eff * &child.value_prediction_error;
    let sum_pi_vpe = w.t().dot(&ed);

    // Precision: clip(π̃_b + pc1·g'² − g''·(Wᵀ@(eff∘δ))).
    Zip::from(&mut parent.precision)
        .and(&parent.expected_precision)
        .and(&pc1_base)
        .and(&sum_pi_vpe)
        .and(&coupling_prime)
        .and(&coupling_second)
        .for_each(|p, &ep, &b, &spv, &gp, &gs| {
            let raw = ep + b * (gp * gp) - gs * spv;
            *p = raw.max(ep).min(max_posterior_precision);
        });

    // Mean: μ̂_b + (Wᵀ@(g_a∘δ))·g'/π_b (reads the new precision).
    Zip::from(&mut parent.mean)
        .and(&parent.expected_mean)
        .and(&msg)
        .and(&coupling_prime)
        .and(&parent.precision)
        .for_each(|m, &em, &mg, &gp, &pp| {
            *m = em + mg * gp / pp;
        });
}

/// Top-down prediction sweep for the whole batch: clamp the predictors
/// (`x` is `(n_samples, top_size)`) on the top layer and predict every layer
/// from the one above it. The batched twin of `DeepNet::prediction_sweep`.
pub fn batched_prediction_sweep(
    net: &DeepNet,
    states: &mut [BatchedLayerState],
    x: ArrayView2<f64>,
    time_step: f64,
) {
    let n = net.layers.len();
    // Clamp x (transposed to features × samples) on the top layer: one
    // strided copy from the view, one contiguous copy between the fields.
    let top = states.last_mut().expect("network has no layers");
    top.expected_mean.assign(&x.t());
    top.mean.assign(&top.expected_mean);

    for i in (1..n).rev() {
        let (lower, upper) = states.split_at_mut(i);
        let child = &mut lower[i - 1];
        let parent = &upper[0];
        let child_layer = &net.layers[i - 1];
        let parent_layer = &net.layers[i];
        let weights = parent_layer
            .weights_in
            .as_ref()
            .expect("parent has no weights");
        match child_layer.kind {
            LayerKind::Volatile => batched_volatile_prediction(
                child,
                child_layer,
                parent,
                weights,
                parent_layer,
                time_step,
            ),
            LayerKind::Binary => batched_binary_prediction(
                child,
                parent,
                weights,
                parent_layer,
                net.precision_clipping_value,
            ),
            LayerKind::Categorical => {
                batched_categorical_prediction(child, parent, weights, parent_layer)
            }
        }
    }
}

/// Bottom-up update sweep for the whole batch: clamp the observations
/// (`y` is `(n_samples, bottom_size)`), compute the leaf prediction error,
/// then interleave posterior + prediction error over interior layers (the
/// clamped top is not updated). The batched twin of `DeepNet::update_sweep`.
pub fn batched_update_sweep(
    net: &DeepNet,
    states: &mut [BatchedLayerState],
    y: ArrayView2<f64>,
    time_step: f64,
) {
    let n = net.layers.len();
    let vol = net.volatility_updates;
    let max_pp = net.max_posterior_precision;

    // Clamp observations and compute the leaf prediction error.
    states[0].mean.assign(&y.t());
    batched_prediction_error(&mut states[0], &net.layers[0], vol, time_step, max_pp);

    // Interior posterior + PE, bottom-up (exclude the clamped top layer).
    for i in 1..n.saturating_sub(1) {
        let (lower, upper) = states.split_at_mut(i);
        let child = &lower[i - 1];
        let parent = &mut upper[0];
        let parent_layer = &net.layers[i];
        let weights = parent_layer
            .weights_in
            .as_ref()
            .expect("parent has no weights");
        batched_posterior_update(
            parent,
            child,
            weights,
            parent_layer,
            max_pp,
            net.layers[i - 1].is_input_layer,
        );
        batched_prediction_error(parent, parent_layer, vol, time_step, max_pp);
    }
}

/// Per-sample prediction errors routed to the input (top) layer, shape
/// `(n_top, n_samples)`: the batched twin of
/// `DeepNet::input_prediction_error`, one gemm for the whole batch.
pub fn batched_input_prediction_error(net: &DeepNet, states: &[BatchedLayerState]) -> Matrix {
    let n = net.layers.len();
    assert!(n >= 2, "input_prediction_error needs at least two layers");
    let top_layer = &net.layers[n - 1];
    let child = &states[n - 2];
    let w = top_layer
        .weights_in
        .as_ref()
        .expect("the top layer of a multi-layer network has weights");
    // The bias column connects the constant node, not a real input.
    let cols = w.ncols() - usize::from(top_layer.add_constant_input);
    let w = w.slice(s![.., ..cols]);

    // Smoothing gain of the child layer, identical to the gain of the
    // interior posterior mean update, fused directly into the message:
    // gain∘δ in one pass, no gain matrix materialised.
    let mut message = Matrix::zeros(child.mean.raw_dim());
    Zip::from(&mut message)
        .and(&child.value_prediction_error)
        .and(&child.conditional_expected_precision)
        .and(&child.precision)
        .and(&child.expected_precision)
        .for_each(|out, &d, &cep, &prec, &ep| {
            let pi_y = prec - ep;
            *out = cep * prec / (cep + pi_y) * d;
        });
    let mut routed = w.t().dot(&message);
    if top_layer.coupling_fn.kind != crate::math::CouplingKind::Linear {
        crate::math::with_coupling!(top_layer.coupling_fn, |_f, df, _d2f| {
            Zip::from(&mut routed)
                .and(&states[n - 1].expected_mean)
                .for_each(|r, &m| *r *= df(m));
        });
    }
    routed
}

/// Batch-mean descent weight gradients, matched 1:1 to `net.layers` (`None`
/// for the bottom layer). The per-sample rank-one factors of
/// `weight_gradient_factors` become two factor matrices, and the batch mean
/// of their outer products is a single contraction per layer:
/// `mean_grad = U @ Vᵀ / n_samples`.
pub fn batched_mean_weight_gradients(
    net: &DeepNet,
    states: &[BatchedLayerState],
    kind: WeightKind,
) -> Vec<Option<Matrix>> {
    let n = net.layers.len();
    let mut grads: Vec<Option<Matrix>> = Vec::with_capacity(n);
    grads.push(None); // bottom layer has no incoming weights
    for i in 1..n {
        let parent_layer = &net.layers[i];
        let child_layer = &net.layers[i - 1];
        let parent = &states[i];
        let child = &states[i - 1];
        let n_samples = child.n_samples();

        // Child-side factor u = -(δ [∘ π]) and parent-side factor
        // v = g(parent posterior mean) with a bias row of ones, each built
        // sanitized (non-finite entries zeroed) in a single pass.
        let sanitize = |x: f64| if x.is_finite() { x } else { 0.0 };
        let mut u = Matrix::zeros(child.value_prediction_error.raw_dim());
        if kind == WeightKind::PrecisionWeighted && child_layer.kind != LayerKind::Binary {
            Zip::from(&mut u)
                .and(&child.value_prediction_error)
                .and(&child.precision)
                .for_each(|u, &d, &p| *u = sanitize(-d * p));
        } else {
            Zip::from(&mut u)
                .and(&child.value_prediction_error)
                .for_each(|u, &d| *u = sanitize(-d));
        }
        let n_parent = parent.mean.nrows();
        let rows = n_parent + usize::from(parent_layer.add_constant_input);
        let mut v = Matrix::zeros((rows, n_samples));
        crate::math::with_coupling!(parent_layer.coupling_fn, |f, _df, _d2f| {
            Zip::from(v.slice_mut(s![..n_parent, ..]))
                .and(&parent.mean)
                .for_each(|c, &m| *c = sanitize(f(m)));
        });
        if parent_layer.add_constant_input {
            v.row_mut(n_parent).fill(1.0);
        }

        let mut grad = u.dot(&v.t());
        grad.mapv_inplace(|g| g / n_samples as f64);
        grads.push(Some(grad));
    }
    grads
}

/// Per-layer batch-mean increments of the carried confidence fields: the
/// value-level posterior precision and, where present, the volatility
/// level's belief `(mean_vol, precision_vol)`.
pub type ConfidenceIncrements = Vec<(Vector, Option<(Vector, Vector)>)>;

/// Batch-mean increments of the carried confidence fields relative to the
/// template: `row_mean(field) − template`, per layer.
pub fn batched_confidence_increments(
    states: &[BatchedLayerState],
    templates: &[&LayerState],
) -> ConfidenceIncrements {
    states
        .iter()
        .zip(templates.iter())
        .map(|(state, template)| {
            let precision = state.precision.mean_axis(Axis(1)).expect("non-empty batch")
                - &template.precision;
            let vol = template.mean_vol.as_ref().map(|template_mv| {
                let mv = state.mean_vol.as_ref().expect("volatility level");
                let pv = state.precision_vol.as_ref().expect("volatility level");
                let template_pv = template.precision_vol.as_ref().expect("volatility level");
                (
                    mv.mean_axis(Axis(1)).expect("non-empty batch") - template_mv,
                    pv.mean_axis(Axis(1)).expect("non-empty batch") - template_pv,
                )
            });
            (precision, vol)
        })
        .collect()
}
