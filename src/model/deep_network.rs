//! Python-facing `DeepNetwork` class: the fluent builder + `predict`/`fit`
//! surface over the vectorised backend ([`crate::vectorised`]).
//!
//! Mirrors the JAX `pyhgf.model.DeepNetwork` API. Layers are declared bottom
//! (output) first with `DeepNetwork::add_layer`; `DeepNetwork::predict` runs
//! a batched, read-only forward sweep and `DeepNetwork::fit` runs prediction +
//! update + weight learning sequentially over the samples.

use crate::math::parse_coupling_fn;
use crate::updates::vectorised::learning::WeightKind;
use crate::vectorised::layer::{
    DeepNet, LayerConfig, LayerKind, VolatilityUpdate, LAYER_STATE_FIELDS,
};
use crate::vectorised::mat::{Float, Matrix};
use crate::vectorised::optimiser::{OptState, Optimizer};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// A vectorised deep predictive-coding network with a matmul engine.
#[pyclass]
pub struct DeepNetwork {
    /// Layer specs in add order (bottom/output first).
    configs: Vec<LayerConfig>,
    /// The built engine. Rebuilt eagerly on every `add_layer` (like the JAX
    /// builder), so it is `Some` whenever `configs` is non-empty and
    /// construction errors surface at the offending `add_layer` call.
    net: Option<DeepNet>,
    /// Optimizer state, allocated on the first `fit` after a (re)build and
    /// reset whenever `fit` is called with a different optimizer.
    opt_state: Option<OptState>,
    /// The optimizer used by the previous `fit` call (for the reset-on-change
    /// semantics, mirroring the JAX class re-initialising on a new optax
    /// object).
    last_optimizer: Option<Optimizer>,
    volatility_updates: VolatilityUpdate,
    max_posterior_precision: Float,
    precision_clipping_value: Float,
    /// Network-level default coupling function name (validated at
    /// construction).
    default_coupling: String,
}

const NO_LAYER_ERR: &str = "add at least one layer (via add_layer) before using the network.";

impl DeepNetwork {
    /// Borrow the built engine, erroring if no layer has been added yet.
    fn require_net(&self) -> PyResult<&DeepNet> {
        self.net
            .as_ref()
            .ok_or_else(|| PyValueError::new_err(NO_LAYER_ERR))
    }

    /// Mutably borrow the built engine, erroring if no layer has been added.
    fn require_net_mut(&mut self) -> PyResult<&mut DeepNet> {
        self.net
            .as_mut()
            .ok_or_else(|| PyValueError::new_err(NO_LAYER_ERR))
    }

    /// Rebuild the engine from `configs` (eager, like the JAX builder).
    fn rebuild(&mut self) -> Result<(), String> {
        let net = DeepNet::from_configs(&self.configs)?.with_settings(
            self.volatility_updates,
            self.max_posterior_precision,
            self.precision_clipping_value,
        );
        self.net = Some(net);
        self.opt_state = None;
        self.last_optimizer = None;
        Ok(())
    }

    /// The top (predictor) layer's node count.
    fn top_size(&self) -> usize {
        self.configs.last().map_or(0, |c| c.size)
    }

    /// The bottom (output) layer's node count.
    fn bottom_size(&self) -> usize {
        self.configs.first().map_or(0, |c| c.size)
    }
}

/// Extract a numpy 2D array (or 1D single-sample) into an owned `(n, features)`
/// matrix, reading the numpy buffer directly (one copy). Returns the matrix and
/// whether the input was 1-D (a single sample), so callers can reproduce the
/// JAX convention of returning a 1-D output for a 1-D input.
fn extract_matrix(x: &Bound<'_, PyAny>) -> PyResult<(Matrix, bool)> {
    if let Ok(arr) = x.extract::<PyReadonlyArray2<Float>>() {
        Ok((arr.as_array().to_owned(), false))
    } else if let Ok(arr) = x.extract::<PyReadonlyArray1<Float>>() {
        Ok((
            arr.as_array().to_owned().insert_axis(ndarray::Axis(0)),
            true,
        ))
    } else if let Ok((mat, was_1d)) = extract_matrix_f64(x) {
        // f32 engine: numpy's default float64 buffers still land on a
        // vectorised cast, not the per-element fallback below.
        Ok((mat, was_1d))
    } else {
        // Fallback: a Python nested list / non-Float array.
        if let Ok(rows) = x.extract::<Vec<Vec<Float>>>() {
            let n = rows.len();
            let m = rows.first().map_or(0, Vec::len);
            let flat: Vec<Float> = rows.into_iter().flatten().collect();
            let mat = Matrix::from_shape_vec((n, m), flat)
                .map_err(|e| PyValueError::new_err(format!("ragged input array: {e}")))?;
            Ok((mat, false))
        } else {
            let flat: Vec<Float> = x.extract()?;
            let m = flat.len();
            Ok((Matrix::from_shape_vec((1, m), flat).unwrap(), true))
        }
    }
}

/// Vectorised-cast extraction of a float64 numpy buffer (numpy's default
/// dtype) for the f32 engine: one `mapv` cast instead of the per-element
/// sequence fallback.
#[cfg(not(feature = "f64"))]
fn extract_matrix_f64(x: &Bound<'_, PyAny>) -> PyResult<(Matrix, bool)> {
    if let Ok(arr) = x.extract::<PyReadonlyArray2<f64>>() {
        return Ok((arr.as_array().mapv(|v| v as Float), false));
    }
    let arr = x.extract::<PyReadonlyArray1<f64>>()?;
    Ok((
        arr.as_array()
            .mapv(|v| v as Float)
            .insert_axis(ndarray::Axis(0)),
        true,
    ))
}

/// Under the `f64` feature `Float` *is* f64, so the `Float` paths above have
/// already matched any float64 buffer; always defer to the caller's fallback.
#[cfg(feature = "f64")]
fn extract_matrix_f64(x: &Bound<'_, PyAny>) -> PyResult<(Matrix, bool)> {
    let _ = x;
    Err(PyValueError::new_err(
        "Float is f64; handled by the Float paths",
    ))
}

/// Error unless the input's column count matches the top layer's node count.
fn check_predict_cols(actual: usize, top: usize) -> PyResult<()> {
    if actual != top {
        return Err(PyValueError::new_err(format!(
            "x has {actual} feature column(s) but the input (top) layer has {top} node(s)."
        )));
    }
    Ok(())
}

/// Extract training data for `fit`, resolving the 1-D ambiguity against the
/// expected feature count: a 1-D array is `n` samples of one feature when the
/// layer has a single node, otherwise a single sample. The column count is
/// validated either way.
fn extract_for_fit(x: &Bound<'_, PyAny>, expected_cols: usize, name: &str) -> PyResult<Matrix> {
    let (mat, was_1d) = extract_matrix(x)?;
    let mat = if was_1d && expected_cols == 1 && mat.ncols() != 1 {
        // (1, n) → (n, 1): n samples of the single feature.
        let n = mat.ncols();
        mat.into_shape_with_order((n, 1))
            .expect("contiguous reshape")
    } else {
        mat
    };
    if mat.ncols() != expected_cols {
        return Err(PyValueError::new_err(format!(
            "{name} has {} feature column(s) but the corresponding layer has \
             {expected_cols} node(s).",
            mat.ncols()
        )));
    }
    Ok(mat)
}

/// Parse an optimizer name (`"adam"` or `"sgd"`) into an [`Optimizer`].
fn parse_optimizer(name: &str, learning_rate: Float) -> PyResult<Optimizer> {
    match name.to_lowercase().as_str() {
        "adam" => Ok(Optimizer::adam(learning_rate)),
        "sgd" => Ok(Optimizer::sgd(learning_rate)),
        other => Err(PyValueError::new_err(format!(
            "Unknown optimizer '{other}'. Use 'adam' or 'sgd'."
        ))),
    }
}

/// Parse a `learning_kind` name into a [`WeightKind`].
fn parse_weight_kind(kind: &str) -> PyResult<WeightKind> {
    match kind {
        "precision_weighted" => Ok(WeightKind::PrecisionWeighted),
        "standard" => Ok(WeightKind::Standard),
        other => Err(PyValueError::new_err(format!(
            "Unknown learning_kind '{other}'. Use 'precision_weighted' or 'standard'."
        ))),
    }
}

#[pymethods]
impl DeepNetwork {
    #[new]
    #[pyo3(signature = (
        volatility_updates = "unbounded",
        max_posterior_precision = 1e10,
        precision_clipping_value = 1e-6,
        coupling_fn = "linear",
    ))]
    fn py_new(
        volatility_updates: &str,
        max_posterior_precision: Float,
        precision_clipping_value: Float,
        coupling_fn: &str,
    ) -> PyResult<Self> {
        let volatility_updates =
            VolatilityUpdate::parse(volatility_updates).map_err(PyValueError::new_err)?;
        // Validate the default coupling name now so a typo fails at
        // construction, not silently later.
        parse_coupling_fn(coupling_fn).map_err(PyValueError::new_err)?;
        if !(precision_clipping_value > 0.0 && precision_clipping_value < 0.5) {
            return Err(PyValueError::new_err(format!(
                "precision_clipping_value must be in (0, 0.5), got \
                 {precision_clipping_value}."
            )));
        }
        Ok(DeepNetwork {
            configs: Vec::new(),
            net: None,
            opt_state: None,
            last_optimizer: None,
            volatility_updates,
            max_posterior_precision,
            precision_clipping_value,
            default_coupling: coupling_fn.to_string(),
        })
    }

    /// Add a layer of nodes (bottom/output layer first). Chainable. The
    /// network is rebuilt eagerly, so configuration errors (e.g. a categorical
    /// layer that is not the bottom layer) surface here — and, like the JAX
    /// builder, the rebuild resets all weights to their default value (1.0),
    /// so call `weight_initialisation`/`set_weights` after the last layer is
    /// added.
    #[pyo3(signature = (
        size,
        kind = "volatile",
        add_constant_input = true,
        fully_connected = true,
        coupling_fn = None,
        volatility_parent = true,
        **kwargs,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_layer<'py>(
        mut slf: PyRefMut<'py, Self>,
        size: usize,
        kind: &str,
        add_constant_input: bool,
        fully_connected: bool,
        coupling_fn: Option<String>,
        volatility_parent: bool,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let layer_kind = LayerKind::parse(kind).map_err(PyValueError::new_err)?;
        let cf = parse_coupling_fn(coupling_fn.as_deref().unwrap_or(&slf.default_coupling))
            .map_err(PyValueError::new_err)?;

        let mut cfg = LayerConfig::new(size);
        cfg.kind = layer_kind;
        cfg.add_constant_input = add_constant_input;
        cfg.fully_connected = fully_connected;
        cfg.coupling_fn = cf;
        cfg.volatility_parent = volatility_parent;

        if let Some(dict) = kwargs {
            for (key, value) in dict.iter() {
                let key: String = key.extract()?;
                let val: Float = value.extract()?;
                match key.as_str() {
                    // LayerParams overrides (typed fields on the config).
                    "tonic_volatility_vol" => cfg.tonic_volatility_vol = val,
                    // LayerState (initial-belief) overrides, applied at build
                    // time — e.g. `precision=2.0` or `expected_precision=1e10`.
                    name if LAYER_STATE_FIELDS.contains(&name) => {
                        cfg.state_overrides.push((key.clone(), val));
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown layer override '{other}'. Valid: \
                             tonic_volatility_vol, or any LayerState field \
                             ({}).",
                            LAYER_STATE_FIELDS.join(", ")
                        )))
                    }
                }
            }
        }

        slf.configs.push(cfg);
        // Eager rebuild: on error, pop the offending config so the network
        // object stays consistent (the failed layer is not retained).
        if let Err(e) = slf.rebuild() {
            slf.configs.pop();
            return Err(PyValueError::new_err(e));
        }
        Ok(slf)
    }

    /// Batched, read-only forward pass. `x` is `(n_samples, n_input_features)`
    /// or a 1-D single sample; returns the bottom layer's expected mean —
    /// `(n_samples, n_output_features)` for 2-D input, `(n_output_features,)`
    /// for 1-D input, matching the JAX `predict`. Every sample is predicted
    /// from the current network as one `gemm` per layer; the network state is
    /// not modified.
    fn predict<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        x: Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let net = slf.require_net()?;
        let top = slf.top_size();
        // Fast path: borrow a 2-D Float numpy buffer directly — the forward pass
        // then makes exactly one input copy (the features×samples layout) and
        // one output copy (`to_pyarray`).
        if let Ok(arr) = x.extract::<PyReadonlyArray2<Float>>() {
            let xv = arr.as_array();
            check_predict_cols(xv.ncols(), top)?;
            let out = net.predict_batch(xv);
            return Ok(out.to_pyarray(py).into_any().unbind());
        }
        // f32 engine: a 2-D float64 buffer pays one vectorised cast instead.
        #[cfg(not(feature = "f64"))]
        if let Ok(arr) = x.extract::<PyReadonlyArray2<f64>>() {
            let xmat = arr.as_array().mapv(|v| v as Float);
            check_predict_cols(xmat.ncols(), top)?;
            let out = net.predict_batch(xmat.view());
            return Ok(out.to_pyarray(py).into_any().unbind());
        }
        // Fallback: 1-D single sample or a Python list.
        let (xmat, was_1d) = extract_matrix(&x)?;
        check_predict_cols(xmat.ncols(), top)?;
        let out = net.predict_batch(xmat.view());
        if was_1d {
            Ok(out.row(0).to_pyarray(py).into_any().unbind())
        } else {
            Ok(out.to_pyarray(py).into_any().unbind())
        }
    }

    /// Train on `(x, y)`: prediction + update + weight learning per sample.
    /// Returns the per-sample output predictions `(n_samples, n_output)`.
    /// Passing a different optimizer (or learning rate) than the previous call
    /// resets the optimizer state, mirroring the JAX class.
    #[pyo3(signature = (
        x, y,
        optimizer = "adam",
        learning_rate = 1e-3,
        learning_kind = "precision_weighted",
        weight_update = true,
        time_step = 1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: Bound<'py, PyAny>,
        y: Bound<'py, PyAny>,
        optimizer: &str,
        learning_rate: Float,
        learning_kind: &str,
        weight_update: bool,
        time_step: Float,
    ) -> PyResult<Py<PyAny>> {
        let opt = parse_optimizer(optimizer, learning_rate)?;
        let kind = parse_weight_kind(learning_kind)?;

        slf.require_net()?;
        let xmat = extract_for_fit(&x, slf.top_size(), "x")?;
        let ymat = extract_for_fit(&y, slf.bottom_size(), "y")?;
        if xmat.nrows() != ymat.nrows() {
            return Err(PyValueError::new_err(format!(
                "x and y must have the same number of samples, got {} and {}.",
                xmat.nrows(),
                ymat.nrows()
            )));
        }

        let this = &mut *slf;
        // Reset the optimizer state when the optimizer changed since the last
        // fit (JAX re-initialises opt_state on a new optax object).
        if this.last_optimizer != Some(opt) {
            this.opt_state = None;
            this.last_optimizer = Some(opt);
        }
        if this.opt_state.is_none() {
            this.opt_state = Some(OptState::init(this.net.as_ref().unwrap()));
        }
        let net = this.net.as_mut().unwrap();
        let opt_state = this.opt_state.as_mut().unwrap();

        let n = xmat.nrows();
        let mut out = Matrix::zeros((n, ymat.ncols()));
        // Release the GIL for the sample loop: it touches only owned Rust
        // data (`xmat`/`ymat` were copied out of the numpy buffers above), so
        // other Python threads can run while the network trains.
        py.detach(|| {
            for i in 0..n {
                let pred = net.propagation_step(
                    xmat.row(i),
                    ymat.row(i),
                    &opt,
                    opt_state,
                    time_step,
                    kind,
                    weight_update,
                );
                out.row_mut(i).assign(&pred);
            }
        });
        Ok(out.to_pyarray(py).into_any().unbind())
    }

    /// One batch-synchronous learning step over many samples at once,
    /// mirroring the JAX `DeepNetwork.batch_update`. Every sample in the
    /// batch is processed from the same state (same weights, same
    /// confidences); the per-sample weight gradients and confidence changes
    /// are then averaged and applied once, so the whole batch counts as a
    /// single observation. This differs from `fit`, which scans samples
    /// sequentially and lets the confidences adapt from one sample to the
    /// next. `optimizer=None` (default) freezes the weights; pass `"adam"` or
    /// `"sgd"` with `learning_rate` to learn. `update_confidences=False`
    /// keeps the carried precisions pinned, the setting used for exact
    /// comparisons against backpropagation. Returns the per-sample prediction
    /// errors at the input (top) layer, shape `(n_samples, n_input_features)`;
    /// they follow the observed minus predicted convention, the negative of a
    /// squared-error loss gradient with respect to the predictors.
    #[pyo3(signature = (
        x, y,
        optimizer = None,
        learning_rate = 1e-3,
        learning_kind = "precision_weighted",
        update_confidences = true,
        time_step = 1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn batch_update<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: Bound<'py, PyAny>,
        y: Bound<'py, PyAny>,
        optimizer: Option<&str>,
        learning_rate: Float,
        learning_kind: &str,
        update_confidences: bool,
        time_step: Float,
    ) -> PyResult<Py<PyAny>> {
        let opt = optimizer
            .map(|name| parse_optimizer(name, learning_rate))
            .transpose()?;
        let kind = parse_weight_kind(learning_kind)?;

        slf.require_net()?;
        if slf.configs.len() < 2 {
            return Err(PyValueError::new_err(
                "The network has a single layer: there is no layer below the \
                 input layer to route an error from.",
            ));
        }
        let (xmat, x_was_1d) = extract_matrix(&x)?;
        let (ymat, y_was_1d) = extract_matrix(&y)?;
        if x_was_1d || y_was_1d {
            return Err(PyValueError::new_err(
                "batch_update() operates on batches: x and y must be 2D \
                 (batch, n_features).",
            ));
        }
        check_predict_cols(xmat.ncols(), slf.top_size())?;
        if ymat.ncols() != slf.bottom_size() {
            return Err(PyValueError::new_err(format!(
                "y has {} feature column(s) but the output (bottom) layer has {} node(s).",
                ymat.ncols(),
                slf.bottom_size()
            )));
        }
        if xmat.nrows() != ymat.nrows() {
            return Err(PyValueError::new_err(format!(
                "x and y must have the same number of samples, got {} and {}.",
                xmat.nrows(),
                ymat.nrows()
            )));
        }

        let this = &mut *slf;
        if let Some(opt) = opt {
            // Same reset-on-change semantics as `fit`.
            if this.last_optimizer != Some(opt) {
                this.opt_state = None;
                this.last_optimizer = Some(opt);
            }
            if this.opt_state.is_none() {
                this.opt_state = Some(OptState::init(this.net.as_ref().unwrap()));
            }
        }
        let net = this.net.as_mut().unwrap();
        // A stale opt_state from an earlier call is inert without an
        // optimizer: the engine only steps the weights when both are given.
        let opt_state = this.opt_state.as_mut();

        // Release the GIL for the batched step, as in `fit`: it touches only
        // owned Rust data, and the chunk workers never call into Python.
        let errors = py.detach(|| {
            net.batch_update(
                xmat.view(),
                ymat.view(),
                opt.as_ref(),
                opt_state,
                time_step,
                kind,
                update_confidences,
            )
        });
        Ok(errors.to_pyarray(py).into_any().unbind())
    }

    /// Initialise all inter-layer weights with a named strategy — `"xavier"`,
    /// `"he"`, `"orthogonal"`, or `"sparse"`. Chainable. The full matrices are
    /// re-drawn (bias column included) with the same seed for every layer,
    /// matching the JAX `DeepNetwork.weight_initialisation` semantics. Note:
    /// like the JAX builder, a later `add_layer` rebuilds the network and
    /// resets weights to 1.0 — initialise after the last layer is added.
    #[pyo3(signature = (strategy, seed=None))]
    fn weight_initialisation<'py>(
        mut slf: PyRefMut<'py, Self>,
        strategy: &str,
        seed: Option<u64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.require_net_mut()?
            .weight_initialisation(strategy, seed)
            .map_err(PyValueError::new_err)?;
        Ok(slf)
    }

    /// The inter-layer weight matrices for layers `1..n` (the bottom layer has
    /// none), as a list of `(child_size, self_size[+1])` numpy arrays.
    fn get_weights<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Py<PyList>> {
        let net = slf.require_net()?;
        let list = PyList::empty(py);
        for layer in net.layers.iter().skip(1) {
            let w = layer
                .weights_in
                .as_ref()
                .expect("interior layer has weights");
            list.append(w.to_pyarray(py))?;
        }
        Ok(list.into())
    }

    /// Set the inter-layer weight matrices for layers `1..n` from a list of 2D
    /// arrays (same order/shapes as [`Self::get_weights`]).
    fn set_weights(mut slf: PyRefMut<'_, Self>, weights: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        // Accept `Float` buffers directly and float64 buffers (numpy's
        // default dtype) through a vectorised cast, mirroring the data paths.
        let weights: Vec<Matrix> = weights
            .iter()
            .map(|w| {
                if let Ok(arr) = w.extract::<PyReadonlyArray2<Float>>() {
                    return Ok(arr.as_array().to_owned());
                }
                #[cfg(not(feature = "f64"))]
                if let Ok(arr) = w.extract::<PyReadonlyArray2<f64>>() {
                    return Ok(arr.as_array().mapv(|v| v as Float));
                }
                Err(PyValueError::new_err(
                    "each weight must be a 2-D float array".to_string(),
                ))
            })
            .collect::<PyResult<_>>()?;
        let net = slf.require_net_mut()?;
        let n_weighted = net.layers.len() - 1;
        if weights.len() != n_weighted {
            return Err(PyValueError::new_err(format!(
                "expected {n_weighted} weight matrices (layers 1..n), got {}.",
                weights.len()
            )));
        }
        // Validate every shape before assigning any, so an error leaves the
        // network's weights untouched rather than partially replaced.
        for (layer, w) in net.layers.iter().skip(1).zip(weights.iter()) {
            let expected = layer
                .weights_in
                .as_ref()
                .expect("interior layer has weights")
                .dim();
            if w.dim() != expected {
                return Err(PyValueError::new_err(format!(
                    "weight shape mismatch: expected {expected:?}, got {:?}.",
                    w.dim()
                )));
            }
        }
        for (layer, w) in net.layers.iter_mut().skip(1).zip(weights) {
            layer.weights_in = Some(w);
        }
        // Weight values changed but shapes did not; existing optimiser moments
        // stay valid, so opt_state is left as-is.
        Ok(())
    }

    /// Number of layers.
    #[getter]
    fn n_layers(&self) -> usize {
        self.configs.len()
    }

    /// Node count of each layer, bottom to top.
    #[getter]
    fn layer_sizes(&self) -> Vec<usize> {
        self.configs.iter().map(|c| c.size).collect()
    }

    /// Total number of nodes across all layers.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.configs.iter().map(|c| c.size).sum()
    }
}
