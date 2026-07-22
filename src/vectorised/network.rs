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
        let mut net = DeepNet::from_configs(&configs).unwrap();
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
        let mut net = DeepNet::from_configs(&configs).unwrap();
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
        let mut net = DeepNet::from_configs(&configs).unwrap();
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
        let mut net = DeepNet::from_configs(&configs).unwrap();
        let x = Matrix::from_shape_vec((1, 2), vec![1.0, -1.0]).unwrap();
        let out = net.predict_batch(x.view());
        assert_eq!(out.shape(), &[1, 3]);
        assert!((out.sum() - 1.0).abs() < 1e-10);
        assert!(out.iter().all(|&p| p > 0.0));
    }
}
