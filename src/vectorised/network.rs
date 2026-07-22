//! Whole-network sweeps for the vectorised deep-network backend.
//!
//! These compose the per-layer kernels of [`crate::updates::vectorised`] into the
//! belief-propagation passes of the JAX driver
//! (`pyhgf.utils.vectorized_belief_propagation`):
//!
//! * [`DeepNet::prediction_sweep`] — clamp the predictors on top, predict every
//!   layer top→down.
//! * [`DeepNet::update_sweep`] — clamp the observations at the bottom, compute
//!   the leaf PE, then interleave posterior + PE bottom→up over interior layers.
//! * [`DeepNet::propagation_step`] — one prediction + update pass with an
//!   optional weight-learning step.
//! * [`DeepNet::predict_batch`] — a batched, read-only forward pass returning
//!   the bottom layer's expected mean for every sample.

use crate::math::sigmoid;
use crate::updates::vectorised::learning::{weight_gradient_factors, WeightKind};
use crate::updates::vectorised::{binary, categorical, softmax_inplace, volatile};
use crate::vectorised::layer::{DeepNet, Layer, LayerKind};
use crate::vectorised::mat::{Matrix, Vector};
use crate::vectorised::optimiser::{OptState, Optimizer};
use ndarray::{s, ArrayView1, ArrayView2};

/// Predict a single child layer from its parent, in place, dispatching on the
/// child kind. `child` is borrowed mutably; `parent` provides the read-only
/// weights/coupling/bias.
fn predict_child(child: &mut Layer, parent: &Layer, weights: &Matrix, time_step: f64, pcv: f64) {
    match child.kind {
        LayerKind::Volatile => volatile::prediction::layer_prediction(
            &mut child.state,
            &child.params,
            &parent.state,
            weights,
            parent.coupling_fn,
            time_step,
            parent.add_constant_input,
            child.has_volatility_parent,
            child.is_input_layer,
        ),
        LayerKind::Binary => binary::prediction::binary_prediction(
            &mut child.state,
            &parent.state,
            weights,
            parent.coupling_fn,
            parent.add_constant_input,
            pcv,
        ),
        LayerKind::Categorical => categorical::prediction::categorical_prediction(
            &mut child.state,
            &parent.state,
            weights,
            parent.coupling_fn,
            parent.add_constant_input,
        ),
    }
}

/// Compute a layer's prediction error in place, dispatching on its kind.
fn layer_pe(
    layer: &mut Layer,
    volatility_updates: crate::vectorised::layer::VolatilityUpdate,
    time_step: f64,
    max_pp: f64,
) {
    match layer.kind {
        LayerKind::Volatile => volatile::prediction_error::layer_prediction_error(
            &mut layer.state,
            &layer.params,
            volatility_updates,
            time_step,
            layer.has_volatility_parent,
            max_pp,
        ),
        LayerKind::Binary => binary::prediction_error::binary_prediction_error(&mut layer.state),
        LayerKind::Categorical => {
            categorical::prediction_error::categorical_prediction_error(&mut layer.state)
        }
    }
}

impl DeepNet {
    /// Clamp the predictors `x` on the top layer (where the predictors enter),
    /// writing into the existing state buffers (no allocation).
    fn set_top_predictors(&mut self, x: ArrayView1<f64>) {
        let top = self.layers.last_mut().expect("network has no layers");
        top.state.expected_mean.assign(&x);
        top.state.mean.assign(&x);
    }

    /// Clamp the observations `y` on the bottom (output) layer, in place.
    fn set_bottom_observations(&mut self, y: ArrayView1<f64>) {
        self.layers[0].state.mean.assign(&y);
    }

    /// Top-down prediction sweep: clamp `x` on top and predict every layer from
    /// the one above it. No prediction errors or posterior updates.
    pub fn prediction_sweep(&mut self, x: ArrayView1<f64>, time_step: f64) {
        let n = self.layers.len();
        self.set_top_predictors(x);
        let pcv = self.precision_clipping_value;
        for i in (1..n).rev() {
            let (lower, upper) = self.layers.split_at_mut(i);
            let child = &mut lower[i - 1];
            let parent = &upper[0];
            let weights = parent.weights_in.as_ref().expect("parent has no weights");
            predict_child(child, parent, weights, time_step, pcv);
        }
    }

    /// Bottom-up update sweep: clamp `y`, compute the leaf PE, then interleave
    /// posterior + PE over interior layers (the clamped top is not updated).
    pub fn update_sweep(&mut self, y: ArrayView1<f64>, time_step: f64) {
        let n = self.layers.len();
        let vol = self.volatility_updates;
        let max_pp = self.max_posterior_precision;

        // Clamp observations and compute the leaf prediction error.
        self.set_bottom_observations(y);
        layer_pe(&mut self.layers[0], vol, time_step, max_pp);

        // Interior posterior + PE, bottom-up (exclude the clamped top layer).
        for i in 1..n.saturating_sub(1) {
            let (lower, upper) = self.layers.split_at_mut(i);
            let child = &lower[i - 1];
            let parent = &mut upper[0];
            let weights = parent.weights_in.as_ref().expect("parent has no weights");

            // Posterior update of the parent from the child, in place.
            volatile::posterior::layer_posterior_update(
                &mut parent.state,
                &child.state,
                weights,
                parent.coupling_fn,
                parent.add_constant_input,
                max_pp,
                child.is_input_layer,
            );
            // Then the parent's own prediction error, in place.
            layer_pe(parent, vol, time_step, max_pp);
        }
    }

    /// Per-layer descent weight-gradient factors `(u, v)`, matched 1:1 to
    /// `layers` (`None` for the bottom layer, which has no incoming weights).
    /// The gradient element is `u[i]·v[j]`; the optimiser consumes the factors
    /// directly, so no gradient matrix is materialised. Must run after
    /// [`Self::update_sweep`] so the layer states carry their PEs/posteriors.
    pub fn weight_gradient_factors(&self, kind: WeightKind) -> Vec<Option<(Vector, Vector)>> {
        let n = self.layers.len();
        let mut grads: Vec<Option<(Vector, Vector)>> = Vec::with_capacity(n);
        grads.push(None); // bottom layer
        for i in 1..n {
            let parent = &self.layers[i];
            let child = &self.layers[i - 1];
            let g = weight_gradient_factors(
                &parent.state,
                &child.state,
                parent.coupling_fn,
                kind,
                parent.add_constant_input,
                child.kind == LayerKind::Binary,
            );
            grads.push(Some(g));
        }
        grads
    }

    /// One propagation step: prediction sweep (clamp `x`) → update sweep (clamp
    /// `y`) → optional weight-learning step. Returns the bottom layer's
    /// predicted mean (the network's output for this step).
    #[allow(clippy::too_many_arguments)]
    pub fn propagation_step(
        &mut self,
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
        optimizer: &Optimizer,
        opt_state: &mut OptState,
        time_step: f64,
        learning_kind: WeightKind,
        weight_update: bool,
    ) -> Vector {
        self.prediction_sweep(x, time_step);
        let output_pred = self.layers[0].state.expected_mean.clone();
        self.update_sweep(y, time_step);
        if weight_update {
            let grads = self.weight_gradient_factors(learning_kind);
            optimizer.apply(opt_state, &mut self.layers, &grads);
        }
        output_pred
    }

    /// Prediction error routed to the network's input (top) layer, mirroring
    /// the JAX `input_prediction_error`.
    ///
    /// The bottom-up sweep never touches the top layer (its values are clamped
    /// to the predictors), so this computes the error message the top layer
    /// would receive from the layer below it: the child layer's value
    /// prediction error, weighted by its smoothing gain (the same gain used by
    /// the interior posterior mean update), routed back through the connecting
    /// weights (bias column excluded) and scaled by the derivative of the top
    /// layer's coupling function at the clamped predictors. Must run after
    /// [`Self::update_sweep`], so the child layer carries its prediction
    /// error. Panics if the network has a single layer (there is no layer
    /// below the input layer to route an error from); the Python boundary
    /// validates this.
    pub fn input_prediction_error(&self) -> Vector {
        let n = self.layers.len();
        assert!(n >= 2, "input_prediction_error needs at least two layers");
        let top = &self.layers[n - 1];
        let child = &self.layers[n - 2].state;
        let w = top
            .weights_in
            .as_ref()
            .expect("the top layer of a multi-layer network has weights");
        // The bias column connects the constant node, not a real input.
        let cols = w.ncols() - usize::from(top.add_constant_input);
        let w = w.slice(s![.., ..cols]);

        // Smoothing gain of the child layer, identical to the gain of the
        // interior posterior mean update, fused directly into the message so
        // the top layer sees exactly the message any interior layer would see.
        let mut message = Vector::zeros(child.mean.len());
        ndarray::Zip::from(&mut message)
            .and(&child.value_prediction_error)
            .and(&child.conditional_expected_precision)
            .and(&child.precision)
            .and(&child.expected_precision)
            .for_each(|out, &d, &cep, &prec, &ep| {
                let pi_y = prec - ep;
                *out = cep * prec / (cep + pi_y) * d;
            });
        let mut routed = w.t().dot(&message);
        if top.coupling_fn.kind != crate::math::CouplingKind::Linear {
            crate::math::with_coupling!(top.coupling_fn, |_f, df, _d2f| {
                ndarray::Zip::from(&mut routed)
                    .and(&top.state.expected_mean)
                    .for_each(|r, &m| *r *= df(m));
            });
        }
        routed
    }

    /// One batch-synchronous learning step over many samples at once,
    /// mirroring the JAX `batch_step`.
    ///
    /// Every sample is processed from the same state template (same weights,
    /// same confidences); the whole batch is swept together through the
    /// batched kernels of [`crate::vectorised::batched`], so every phase is
    /// one `gemm` per layer in a features × samples layout. The per-sample
    /// weight gradients and confidence changes are averaged and applied once,
    /// so the whole batch counts as a single observation. This differs from
    /// [`Self::propagation_step`] driven sequentially, where the confidences
    /// adapt from one sample to the next.
    ///
    /// With `optimizer` set, the batch-mean descent gradient drives one
    /// optimiser step; with `None` the weights are frozen. With
    /// `update_confidences`, the batch-mean change of the carried confidence
    /// fields (the value-level posterior precision and the volatility level's
    /// belief) is added to the template state; everything else is left on the
    /// template, as the next batch's sweeps rewrite it anyway.
    ///
    /// Returns the per-sample prediction errors at the input layer, shape
    /// `(n_samples, top_size)`; see [`Self::input_prediction_error`] for their
    /// meaning and sign convention.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_update(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
        optimizer: Option<&Optimizer>,
        opt_state: Option<&mut OptState>,
        time_step: f64,
        learning_kind: WeightKind,
        update_confidences: bool,
    ) -> Matrix {
        // Samples are exchangeable by construction, so the batch splits into
        // per-thread slabs swept independently; small batches stay in one
        // chunk and take exactly the single-threaded path.
        const MIN_CHUNK: usize = 256;
        let n_chunks = rayon::current_num_threads()
            .min(x.nrows().div_ceil(MIN_CHUNK))
            .max(1);
        self.batch_update_chunked(
            x,
            y,
            optimizer,
            opt_state,
            time_step,
            learning_kind,
            update_confidences,
            n_chunks,
        )
    }

    /// The chunked executor behind [`Self::batch_update`]: sweep each slab of
    /// samples through the batched kernels in parallel, then combine the
    /// per-chunk results (input-error slices concatenated; gradient and
    /// confidence means weighted by chunk size) and apply them once.
    #[allow(clippy::too_many_arguments)]
    fn batch_update_chunked(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
        optimizer: Option<&Optimizer>,
        opt_state: Option<&mut OptState>,
        time_step: f64,
        learning_kind: WeightKind,
        update_confidences: bool,
        n_chunks: usize,
    ) -> Matrix {
        use crate::vectorised::batched::{
            batched_confidence_increments, batched_input_prediction_error,
            batched_mean_weight_gradients, batched_prediction_sweep, batched_update_sweep,
            BatchedLayerState, ConfidenceIncrements,
        };
        use crate::vectorised::layer::LayerState;
        use rayon::prelude::*;

        let n_samples = x.nrows();
        let n_layers = self.layers.len();
        let top_size = self.layers[n_layers - 1].n_nodes();
        let learning = optimizer.is_some();

        // Even sample ranges, the remainder spread over the first chunks.
        let base = n_samples / n_chunks;
        let remainder = n_samples % n_chunks;
        let mut ranges = Vec::with_capacity(n_chunks);
        let mut start = 0;
        for c in 0..n_chunks {
            let len = base + usize::from(c < remainder);
            ranges.push(start..start + len);
            start += len;
        }

        struct ChunkOut {
            range: std::ops::Range<usize>,
            errors: Matrix,
            grads: Option<Vec<Option<Matrix>>>,
            increments: Option<ConfidenceIncrements>,
        }

        let net: &DeepNet = self;
        let templates: Vec<&LayerState> = net.layers.iter().map(|layer| &layer.state).collect();
        let chunk_outs: Vec<ChunkOut> = ranges
            .into_par_iter()
            .map(|range| {
                let mut states: Vec<BatchedLayerState> = templates
                    .iter()
                    .map(|state| BatchedLayerState::from_template(state, range.len()))
                    .collect();
                batched_prediction_sweep(
                    net,
                    &mut states,
                    x.slice(s![range.clone(), ..]),
                    time_step,
                );
                batched_update_sweep(net, &mut states, y.slice(s![range.clone(), ..]), time_step);
                let errors = batched_input_prediction_error(net, &states);
                let grads =
                    learning.then(|| batched_mean_weight_gradients(net, &states, learning_kind));
                let increments = update_confidences
                    .then(|| batched_confidence_increments(&states, &templates));
                ChunkOut {
                    range,
                    errors,
                    grads,
                    increments,
                }
            })
            .collect();

        let mut input_errors = Matrix::zeros((top_size, n_samples));
        for chunk in &chunk_outs {
            input_errors
                .slice_mut(s![.., chunk.range.clone()])
                .assign(&chunk.errors);
        }

        if let (Some(opt), Some(state)) = (optimizer, opt_state) {
            // Batch mean = chunk means weighted by chunk size.
            let mut grads: Vec<Option<Matrix>> = self
                .layers
                .iter()
                .map(|layer| layer.weights_in.as_ref().map(|w| Matrix::zeros(w.raw_dim())))
                .collect();
            for chunk in &chunk_outs {
                let weight = chunk.range.len() as f64 / n_samples as f64;
                for (total, grad) in grads.iter_mut().zip(chunk.grads.as_ref().unwrap()) {
                    if let (Some(total), Some(grad)) = (total.as_mut(), grad.as_ref()) {
                        total.scaled_add(weight, grad);
                    }
                }
            }
            opt.apply_dense(state, &mut self.layers, &grads);
        }

        if update_confidences {
            for (l, layer) in self.layers.iter_mut().enumerate() {
                for chunk in &chunk_outs {
                    let weight = chunk.range.len() as f64 / n_samples as f64;
                    let (precision_inc, vol_inc) = &chunk.increments.as_ref().unwrap()[l];
                    layer.state.precision.scaled_add(weight, precision_inc);
                    if let Some((mean_vol_inc, precision_vol_inc)) = vol_inc {
                        layer
                            .state
                            .mean_vol
                            .as_mut()
                            .expect("volatility level")
                            .scaled_add(weight, mean_vol_inc);
                        layer
                            .state
                            .precision_vol
                            .as_mut()
                            .expect("volatility level")
                            .scaled_add(weight, precision_vol_inc);
                    }
                }
            }
        }

        // Rows become samples again at the boundary (stride flip, no copy).
        input_errors.reversed_axes()
    }

    /// Batched, read-only forward pass over `n` samples at once.
    ///
    /// `x` is a `(n_samples, top_size)` view (borrowed directly from the numpy
    /// buffer at the Python boundary — no input copy); returns the bottom
    /// layer's predicted mean `(n_samples, bottom_size)`. Mirrors JAX's
    /// `batched_prediction_pass`: every sample is predicted from the same
    /// (current) network — the pass does not mutate state. The returned
    /// expected mean is a pure matmul chain through the coupling functions and
    /// never depends on the precision or volatility levels, so this is exact,
    /// not an approximation, and it collapses the per-sample `gemv`s into one
    /// `gemm` per layer.
    pub fn predict_batch(&self, x: ArrayView2<f64>) -> Matrix {
        let n = self.layers.len();
        if n == 1 {
            return x.to_owned(); // degenerate single-layer network (no weights)
        }
        // Work in a **features × samples** layout so every layer is a contiguous
        // `W @ A` gemm. In (samples × features) the forward would be `A @ Wᵀ`,
        // whose transposed right operand is non-contiguous and much slower for
        // the thin inner dimensions typical of HGF layers. This transposed
        // materialisation is the single input copy of the whole pass.
        let mut a = x.t().to_owned(); // (top_size, n_samples)
        for i in (1..n).rev() {
            let parent = &self.layers[i];
            let child = &self.layers[i - 1];
            let w = parent.weights_in.as_ref().expect("parent has no weights"); // (child, parent[+1])

            // Coupled parent activations, bias node as a trailing row of ones.
            // The ones row is written once; the parent rows are overwritten with
            // g(a) in a single fused pass.
            let (psize, n_samples) = a.dim();
            let rows = psize + usize::from(parent.add_constant_input);
            let mut coupled = Matrix::from_elem((rows, n_samples), 1.0);
            crate::math::with_coupling!(parent.coupling_fn, |f, _df, _d2f| {
                ndarray::Zip::from(coupled.slice_mut(s![..psize, ..]))
                    .and(&a)
                    .for_each(|c, &e| *c = f(e));
            });

            // (child, parent[+1]) @ (parent[+1], n_samples) = (child, n_samples).
            let mut child_a = w.dot(&coupled);

            match child.kind {
                LayerKind::Binary => {
                    let pcv = self.precision_clipping_value;
                    child_a.mapv_inplace(|z| sigmoid(z).clamp(pcv, 1.0 - pcv));
                }
                LayerKind::Categorical => {
                    // One sample per column in the features×samples layout.
                    for col in child_a.columns_mut() {
                        softmax_inplace(col);
                    }
                }
                LayerKind::Volatile => {}
            }
            a = child_a;
        }
        // Back to (n_samples, bottom_size) by flipping strides — no copy; the
        // Python boundary's `to_pyarray` materialises C order in its own single
        // copy.
        a.reversed_axes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorised::layer::{DeepNet, LayerConfig};
    use ndarray::Array1;

    /// A two-layer linear network with identity coupling and no bias predicts
    /// `W @ x` at the output (all weights initialised to 1).
    #[test]
    fn test_predict_two_layer_linear() {
        // Bottom (output) size 2, top (input) size 3, no bias for a clean matmul.
        let configs = vec![
            LayerConfig::new(2),
            LayerConfig {
                add_constant_input: false,
                ..LayerConfig::new(3)
            },
        ];
        let net = DeepNet::from_configs(&configs).unwrap();
        let x = Matrix::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let out = net.predict_batch(x.view());
        // weights_in on layer 1 is ones (2,3); each output = sum(x) = 6.
        assert_eq!(out.shape(), &[1, 2]);
        assert!((out[[0, 0]] - 6.0).abs() < 1e-10);
        assert!((out[[0, 1]] - 6.0).abs() < 1e-10);
    }

    /// The bias column adds a constant of `sum(bias_weights)`; here a 1-node
    /// output with a bias predicts `sum(x) + bias`.
    #[test]
    fn test_predict_with_bias() {
        let configs = vec![LayerConfig::new(1), LayerConfig::new(2)];
        let net = DeepNet::from_configs(&configs).unwrap();
        // weights_in on layer 1: shape (1, 2 + bias) = (1, 3), all ones.
        let x = Matrix::from_shape_vec((1, 2), vec![1.0, 4.0]).unwrap();
        let out = net.predict_batch(x.view());
        // sum(x)=5 plus bias weight 1 = 6.
        assert!((out[[0, 0]] - 6.0).abs() < 1e-10);
    }

    /// A full propagation step with SGD reduces the output error on a linear
    /// regression target after several steps.
    #[test]
    fn test_training_reduces_error() {
        let configs = vec![
            LayerConfig {
                add_constant_input: false,
                ..LayerConfig::new(1)
            },
            LayerConfig {
                add_constant_input: false,
                ..LayerConfig::new(1)
            },
        ];
        let mut net = DeepNet::from_configs(&configs).unwrap();
        let opt = Optimizer::sgd(0.05);
        let mut state = OptState::init(&net);

        let x = Array1::from_vec(vec![1.0]);
        let y = Array1::from_vec(vec![0.5]); // target below the initial prediction (=x=1)

        let first = net.predict_batch(x.view().insert_axis(ndarray::Axis(0)))[[0, 0]];
        let mut last = first;
        for _ in 0..20 {
            last = net.propagation_step(
                x.view(),
                y.view(),
                &opt,
                &mut state,
                1.0,
                WeightKind::PrecisionWeighted,
                true,
            )[0];
        }
        // Prediction should move from ~1.0 toward the target 0.5.
        assert!((last - 0.5).abs() < (first - 0.5).abs());
        assert!(last < first);
    }

    /// The batched (gemm) step reproduces the per-sample reference: sweeping
    /// each sample from the template with the per-sample kernels, averaging
    /// the rank-one gradients and the carried-field increments, and applying
    /// both once. Covers a three-layer network with a nonlinear (tanh)
    /// coupling, Adam, and confidence carrying.
    #[test]
    fn test_batch_update_matches_per_sample_reference() {
        use crate::math::resolve_coupling_fn;

        let make_net = || {
            let configs = vec![
                LayerConfig::new(2),
                LayerConfig {
                    coupling_fn: resolve_coupling_fn("tanh"),
                    ..LayerConfig::new(4)
                },
                LayerConfig::new(3),
            ];
            let mut net = DeepNet::from_configs(&configs).unwrap();
            // Deterministic, non-uniform weights.
            for (l, layer) in net.layers.iter_mut().enumerate().skip(1) {
                let w = layer.weights_in.as_mut().unwrap();
                for ((i, j), v) in w.indexed_iter_mut() {
                    *v = 0.3 * ((i + 2 * j + l) as f64 * 0.7).sin();
                }
            }
            net
        };

        let n_samples = 5;
        let x = Matrix::from_shape_fn((n_samples, 3), |(i, j)| ((i * 3 + j) as f64 * 0.9).cos());
        let y = Matrix::from_shape_fn((n_samples, 2), |(i, j)| ((i * 2 + j) as f64 * 0.4).sin());
        let opt = Optimizer::adam(0.01);

        // Reference: the per-sample loop over the same template.
        let mut reference = make_net();
        let mut ref_state = OptState::init(&reference);
        let template: Vec<_> = reference.layers.iter().map(|l| l.state.clone()).collect();
        let n_layers = reference.layers.len();
        let mut grad_sums: Vec<Option<Matrix>> = reference
            .layers
            .iter()
            .map(|l| l.weights_in.as_ref().map(|w| Matrix::zeros(w.raw_dim())))
            .collect();
        let mut precision_sums: Vec<Vector> = template
            .iter()
            .map(|s| Vector::zeros(s.precision.len()))
            .collect();
        let mut vol_sums: Vec<(Vector, Vector)> = template
            .iter()
            .map(|s| {
                let v = s.mean_vol.as_ref().unwrap();
                (Vector::zeros(v.len()), Vector::zeros(v.len()))
            })
            .collect();
        let mut ref_errors = Matrix::zeros((n_samples, 3));
        for i in 0..n_samples {
            for (layer, state) in reference.layers.iter_mut().zip(template.iter()) {
                layer.state = state.clone();
            }
            reference.prediction_sweep(x.row(i), 1.0);
            reference.update_sweep(y.row(i), 1.0);
            ref_errors
                .row_mut(i)
                .assign(&reference.input_prediction_error());
            let factors = reference.weight_gradient_factors(WeightKind::PrecisionWeighted);
            for (sum, factor) in grad_sums.iter_mut().zip(factors.iter()) {
                if let (Some(g), Some((u, v))) = (sum.as_mut(), factor.as_ref()) {
                    for (r, &ui) in u.iter().enumerate() {
                        for (c, &vj) in v.iter().enumerate() {
                            g[[r, c]] += ui * vj;
                        }
                    }
                }
            }
            for l in 0..n_layers {
                let state = &reference.layers[l].state;
                precision_sums[l] += &(&state.precision - &template[l].precision);
                vol_sums[l].0 +=
                    &(state.mean_vol.as_ref().unwrap() - template[l].mean_vol.as_ref().unwrap());
                vol_sums[l].1 += &(state.precision_vol.as_ref().unwrap()
                    - template[l].precision_vol.as_ref().unwrap());
            }
        }
        for (layer, state) in reference.layers.iter_mut().zip(template) {
            layer.state = state;
        }
        let inv_n = 1.0 / n_samples as f64;
        for g in grad_sums.iter_mut().flatten() {
            g.mapv_inplace(|v| v * inv_n);
        }
        opt.apply_dense(&mut ref_state, &mut reference.layers, &grad_sums);
        for l in 0..n_layers {
            let state = &mut reference.layers[l].state;
            state.precision.zip_mut_with(&precision_sums[l], |p, &s| *p += s * inv_n);
            state
                .mean_vol
                .as_mut()
                .unwrap()
                .zip_mut_with(&vol_sums[l].0, |p, &s| *p += s * inv_n);
            state
                .precision_vol
                .as_mut()
                .unwrap()
                .zip_mut_with(&vol_sums[l].1, |p, &s| *p += s * inv_n);
        }

        // The batched step on an identically built network.
        let mut batched = make_net();
        let mut bat_state = OptState::init(&batched);
        let errors = batched.batch_update(
            x.view(),
            y.view(),
            Some(&opt),
            Some(&mut bat_state),
            1.0,
            WeightKind::PrecisionWeighted,
            true,
        );

        let close = |a: f64, b: f64| (a - b).abs() < 1e-12 * (1.0 + a.abs().max(b.abs()));
        for (a, b) in errors.iter().zip(ref_errors.iter()) {
            assert!(close(*a, *b), "input errors differ: {a} vs {b}");
        }
        for (bl, rl) in batched.layers.iter().zip(reference.layers.iter()) {
            if let (Some(bw), Some(rw)) = (bl.weights_in.as_ref(), rl.weights_in.as_ref()) {
                for (a, b) in bw.iter().zip(rw.iter()) {
                    assert!(close(*a, *b), "weights differ: {a} vs {b}");
                }
            }
            for (a, b) in bl.state.precision.iter().zip(rl.state.precision.iter()) {
                assert!(close(*a, *b), "carried precision differs: {a} vs {b}");
            }
            for (a, b) in bl
                .state
                .mean_vol
                .as_ref()
                .unwrap()
                .iter()
                .zip(rl.state.mean_vol.as_ref().unwrap().iter())
            {
                assert!(close(*a, *b), "carried mean_vol differs: {a} vs {b}");
            }
        }
    }

    /// Splitting the batch across chunks leaves the result unchanged: the
    /// chunk means recombine to the batch mean, so a four-chunk run matches a
    /// single-chunk run to floating point reordering noise.
    #[test]
    fn test_batch_update_chunked_matches_single_chunk() {
        let make_net = || {
            let configs = vec![LayerConfig::new(2), LayerConfig::new(4), LayerConfig::new(3)];
            let mut net = DeepNet::from_configs(&configs).unwrap();
            for (l, layer) in net.layers.iter_mut().enumerate().skip(1) {
                let w = layer.weights_in.as_mut().unwrap();
                for ((i, j), v) in w.indexed_iter_mut() {
                    *v = 0.3 * ((i + 2 * j + l) as f64 * 0.7).sin();
                }
            }
            net
        };
        let n_samples = 601; // odd, so the chunks are uneven
        let x = Matrix::from_shape_fn((n_samples, 3), |(i, j)| ((i * 3 + j) as f64 * 0.9).cos());
        let y = Matrix::from_shape_fn((n_samples, 2), |(i, j)| ((i * 2 + j) as f64 * 0.4).sin());
        let opt = Optimizer::adam(0.01);

        let run = |n_chunks: usize| {
            let mut net = make_net();
            let mut state = OptState::init(&net);
            let errors = net.batch_update_chunked(
                x.view(),
                y.view(),
                Some(&opt),
                Some(&mut state),
                1.0,
                WeightKind::PrecisionWeighted,
                true,
                n_chunks,
            );
            (net, errors)
        };
        let (net_one, errors_one) = run(1);
        let (net_four, errors_four) = run(4);

        let close = |a: f64, b: f64| (a - b).abs() < 1e-9 * (1.0 + a.abs().max(b.abs()));
        for (a, b) in errors_one.iter().zip(errors_four.iter()) {
            assert!(close(*a, *b), "input errors differ: {a} vs {b}");
        }
        for (l1, l4) in net_one.layers.iter().zip(net_four.layers.iter()) {
            if let (Some(w1), Some(w4)) = (l1.weights_in.as_ref(), l4.weights_in.as_ref()) {
                for (a, b) in w1.iter().zip(w4.iter()) {
                    assert!(close(*a, *b), "weights differ: {a} vs {b}");
                }
            }
            for (a, b) in l1.state.precision.iter().zip(l4.state.precision.iter()) {
                assert!(close(*a, *b), "carried precision differs: {a} vs {b}");
            }
        }
    }

    /// A single-sample batch step applies exactly the weight update of one
    /// sequential propagation step (the batch mean over one sample is that
    /// sample's gradient).
    #[test]
    fn test_batch_update_single_sample_matches_propagation_step() {
        let configs = vec![LayerConfig::new(2), LayerConfig::new(3)];
        let mut seq = DeepNet::from_configs(&configs).unwrap();
        let mut bat = DeepNet::from_configs(&configs).unwrap();
        let opt = Optimizer::sgd(0.05);
        let mut seq_state = OptState::init(&seq);
        let mut bat_state = OptState::init(&bat);

        let x = Array1::from_vec(vec![0.3, -0.2, 0.8]);
        let y = Array1::from_vec(vec![0.5, -0.1]);

        seq.propagation_step(
            x.view(),
            y.view(),
            &opt,
            &mut seq_state,
            1.0,
            WeightKind::PrecisionWeighted,
            true,
        );
        let errors = bat.batch_update(
            x.view().insert_axis(ndarray::Axis(0)),
            y.view().insert_axis(ndarray::Axis(0)),
            Some(&opt),
            Some(&mut bat_state),
            1.0,
            WeightKind::PrecisionWeighted,
            false,
        );
        assert_eq!(errors.shape(), &[1, 3]);
        let ws = seq.layers[1].weights_in.as_ref().unwrap();
        let wb = bat.layers[1].weights_in.as_ref().unwrap();
        for (a, b) in ws.iter().zip(wb.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// A binary leaf produces sigmoid outputs in (0, 1).
    #[test]
    fn test_binary_leaf_predict_in_unit_interval() {
        let configs = vec![
            LayerConfig {
                kind: LayerKind::Binary,
                ..LayerConfig::new(1)
            },
            LayerConfig::new(2),
        ];
        let net = DeepNet::from_configs(&configs).unwrap();
        let x = Matrix::from_shape_vec((1, 2), vec![0.5, -0.5]).unwrap();
        let out = net.predict_batch(x.view());
        assert!(out[[0, 0]] > 0.0 && out[[0, 0]] < 1.0);
    }

    /// A categorical leaf produces a normalised probability vector.
    #[test]
    fn test_categorical_leaf_predict_sums_to_one() {
        let configs = vec![
            LayerConfig {
                kind: LayerKind::Categorical,
                ..LayerConfig::new(3)
            },
            LayerConfig::new(2),
        ];
        let net = DeepNet::from_configs(&configs).unwrap();
        let x = Matrix::from_shape_vec((1, 2), vec![1.0, -1.0]).unwrap();
        let out = net.predict_batch(x.view());
        assert_eq!(out.shape(), &[1, 3]);
        assert!((out.sum() - 1.0).abs() < 1e-10);
        assert!(out.iter().all(|&p| p > 0.0));
    }
}
