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
use crate::vectorised::mat::Matrix;
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
    max_posterior_precision: f64,
    precision_clipping_value: f64,
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
    if let Ok(arr) = x.extract::<PyReadonlyArray2<f64>>() {
        Ok((arr.as_array().to_owned(), false))
    } else if let Ok(arr) = x.extract::<PyReadonlyArray1<f64>>() {
        Ok((
            arr.as_array().to_owned().insert_axis(ndarray::Axis(0)),
            true,
        ))
    } else {
        // Fallback: a Python nested list / non-f64 array.
        if let Ok(rows) = x.extract::<Vec<Vec<f64>>>() {
            let n = rows.len();
            let m = rows.first().map_or(0, Vec::len);
            let flat: Vec<f64> = rows.into_iter().flatten().collect();
            let mat = Matrix::from_shape_vec((n, m), flat)
                .map_err(|e| PyValueError::new_err(format!("ragged input array: {e}")))?;
            Ok((mat, false))
        } else {
            let flat: Vec<f64> = x.extract()?;
            let m = flat.len();
            Ok((Matrix::from_shape_vec((1, m), flat).unwrap(), true))
        }
    }
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
        max_posterior_precision: f64,
        precision_clipping_value: f64,
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
                let val: f64 = value.extract()?;
                match key.as_str() {
                    // LayerParams overrides (typed fields on the config).
                    "tonic_volatility" => cfg.tonic_volatility = val,
                    "tonic_volatility_vol" => cfg.tonic_volatility_vol = val,
                    "volatility_coupling" => cfg.volatility_coupling = val,
                    "autoconnection_strength_vol" => cfg.autoconnection_strength_vol = val,
                    // LayerState (initial-belief) overrides, applied at build
                    // time — e.g. `precision=2.0` or `expected_precision=1e10`.
                    name if LAYER_STATE_FIELDS.contains(&name) => {
                        cfg.state_overrides.push((key.clone(), val));
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown layer override '{other}'. Valid: tonic_volatility, \
                             tonic_volatility_vol, volatility_coupling, \
                             autoconnection_strength_vol, or any LayerState field \
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
        // Fast path: borrow a 2-D f64 numpy buffer directly — the forward pass
        // then makes exactly one input copy (the features×samples layout) and
        // one output copy (`to_pyarray`).
        if let Ok(arr) = x.extract::<PyReadonlyArray2<f64>>() {
            let xv = arr.as_array();
            check_predict_cols(xv.ncols(), top)?;
            let out = net.predict_batch(xv);
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
        learning_rate: f64,
        learning_kind: &str,
        weight_update: bool,
        time_step: f64,
    ) -> PyResult<Py<PyAny>> {
        let opt = match optimizer.to_lowercase().as_str() {
            "adam" => Optimizer::adam(learning_rate),
            "sgd" => Optimizer::sgd(learning_rate),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown optimizer '{other}'. Use 'adam' or 'sgd'."
                )))
            }
        };
        let kind = match learning_kind {
            "precision_weighted" => WeightKind::PrecisionWeighted,
            "standard" => WeightKind::Standard,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown learning_kind '{other}'. Use 'precision_weighted' or 'standard'."
                )))
            }
        };

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
    fn set_weights(
        mut slf: PyRefMut<'_, Self>,
        weights: Vec<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
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
            if w.as_array().dim() != expected {
                return Err(PyValueError::new_err(format!(
                    "weight shape mismatch: expected {expected:?}, got {:?}.",
                    w.as_array().dim()
                )));
            }
        }
        for (layer, w) in net.layers.iter_mut().skip(1).zip(weights.iter()) {
            layer.weights_in = Some(w.as_array().to_owned());
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
