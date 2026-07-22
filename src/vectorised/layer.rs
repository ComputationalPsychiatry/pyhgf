//! Columnar layer representation for the vectorised deep-network backend.
//!
//! This mirrors the Equinox PyTree types of the JAX vectorised backend
//! (`pyhgf.typing.vectorised`) so the two backends can be checked for
//! numerical parity. A whole layer of HGF nodes is stored **columnar** — one
//! [`Vector`] per belief field with one entry per node — instead of a `Vec` of
//! per-node structs, so the update kernels are contiguous vector / matrix ops.
//!
//! ## Layer ordering (matches the Python builder)
//!
//! Layers are stored in *add order*: index `0` is the **bottom** layer — the
//! observation / output leaf where `y` is clamped during `fit` — and the last
//! layer is the **top**, where the predictors `x` enter. Prediction flows
//! top→down (a parent predicts the child below it); errors flow bottom→up.
//!
//! ## Weight orientation
//!
//! [`Layer::weights_in`] connects the layer *below* (the child) up into this
//! layer (the parent). Its shape is `(child_size, self_size[+1])` — rows index
//! the child nodes, columns index this layer's nodes, plus an optional trailing
//! **bias** column when `add_constant_input` is set. The bottom layer (index 0)
//! has `weights_in = None`, since no layer sits below it.

use crate::math::{CouplingFn, LINEAR};
use crate::utils::weight_initialisation::weight_init_by_name;
use crate::vectorised::mat::{eye, Matrix, Vector};
use ndarray::Array1;

/// Which volatility-level posterior update the network applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityUpdate {
    /// Precision first, then mean.
    Standard,
    /// Mean first (expected precision), then the safe precision increment.
    EHgf,
    /// Lambert-W dual-quadratic expansion with a variational softmax blend and
    /// Gaussian moment matching (the uHGF update; the JAX default).
    Unbounded,
}

impl VolatilityUpdate {
    /// Parse a volatility-update name (`"unbounded"` / `"eHGF"` / `"standard"`).
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "eHGF" | "ehgf" => Ok(VolatilityUpdate::EHgf),
            "standard" => Ok(VolatilityUpdate::Standard),
            "unbounded" => Ok(VolatilityUpdate::Unbounded),
            other => Err(format!(
                "Invalid volatility update '{other}'. Choose from \
                 [\"unbounded\", \"eHGF\", \"standard\"]."
            )),
        }
    }
}

/// The kind of nodes a layer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Volatile state nodes with a value level and an internal volatility level.
    Volatile,
    /// Binary (Bernoulli) state nodes; one independent belief per node.
    Binary,
    /// A single joint N-way choice over the layer's nodes (softmax). Only valid
    /// as the bottom (output) layer.
    Categorical,
}

impl LayerKind {
    /// Parse a layer kind, accepting both the short (`"volatile"`) and the
    /// Rust-backend (`"volatile-state"`) spellings.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "volatile" | "volatile-state" => Ok(LayerKind::Volatile),
            "binary" | "binary-state" => Ok(LayerKind::Binary),
            "categorical" | "categorical-state" => Ok(LayerKind::Categorical),
            other => Err(format!(
                "Invalid layer kind '{other}'. Choose from \
                 [\"volatile\", \"binary\", \"categorical\"]."
            )),
        }
    }
}

/// Per-node value- and (optional) volatility-level beliefs for one layer.
///
/// Every field is a [`Vector`] of length `n_nodes`. The six volatility-level
/// fields are `None` when the layer has no volatility parent — a frozen
/// volatility level is never predicted or updated, so allocating it would only
/// carry dead arrays through the state (mirrors the `None` PyTree leaves of the
/// JAX `LayerState`).
#[derive(Debug, Clone)]
pub struct LayerState {
    // Value level (external).
    /// Posterior mean of the value level.
    pub mean: Vector,
    /// Posterior precision of the value level.
    pub precision: Vector,
    /// Predicted (expected) mean of the value level.
    pub expected_mean: Vector,
    /// Marginal predicted precision of the value level (π̃).
    pub expected_precision: Vector,
    /// Conditional predicted precision of the value level (π̂), used by the
    /// structured-Gaussian (smoothing) update.
    pub conditional_expected_precision: Vector,
    /// Effective precision of the value-level prediction (γ).
    pub effective_precision: Vector,
    /// Value prediction error (δ).
    pub value_prediction_error: Vector,
    // Volatility level (internal); `None` without a volatility parent.
    /// Posterior mean of the volatility level.
    pub mean_vol: Option<Vector>,
    /// Posterior precision of the volatility level.
    pub precision_vol: Option<Vector>,
    /// Predicted (expected) mean of the volatility level.
    pub expected_mean_vol: Option<Vector>,
    /// Marginal predicted precision of the volatility level.
    pub expected_precision_vol: Option<Vector>,
    /// Effective precision of the volatility-level prediction.
    pub effective_precision_vol: Option<Vector>,
    /// Volatility prediction error.
    pub volatility_prediction_error: Option<Vector>,
}

impl LayerState {
    /// Initialise a layer state with the JAX defaults: means `0`, precisions
    /// `1`, effective precisions and prediction errors `0`. With
    /// `has_volatility_parent = false` the six volatility-level fields are
    /// `None`.
    pub fn create(n_nodes: usize, has_volatility_parent: bool) -> Self {
        let zeros = || Array1::<f64>::zeros(n_nodes);
        let ones = || Array1::<f64>::from_elem(n_nodes, 1.0);
        let vol = |v: f64| {
            if has_volatility_parent {
                Some(Array1::<f64>::from_elem(n_nodes, v))
            } else {
                None
            }
        };
        Self {
            mean: zeros(),
            precision: ones(),
            expected_mean: zeros(),
            expected_precision: ones(),
            conditional_expected_precision: ones(),
            effective_precision: zeros(),
            value_prediction_error: zeros(),
            mean_vol: vol(0.0),
            precision_vol: vol(1.0),
            expected_mean_vol: vol(0.0),
            expected_precision_vol: vol(1.0),
            effective_precision_vol: vol(0.0),
            volatility_prediction_error: vol(0.0),
        }
    }

    /// Number of nodes in the layer.
    pub fn n_nodes(&self) -> usize {
        self.mean.len()
    }

    /// Override one belief field by name, broadcasting `value` to every node.
    ///
    /// Mirrors the `add_layer(**kwargs)` state overrides of the JAX
    /// `DeepNetwork`: any [`LayerState`] field can be set at build time. A
    /// volatility-level override on a layer without a volatility parent is
    /// silently dropped (the frozen level never reads it; same rule as the
    /// Python `_init_state`). Unknown names are an error.
    pub fn set_field(&mut self, name: &str, value: f64) -> Result<(), String> {
        let n = self.n_nodes();
        let full = || Array1::<f64>::from_elem(n, value);
        // A volatility slot is written only when present; a frozen level (no
        // volatility parent) silently ignores the override.
        fn set_vol(slot: &mut Option<Vector>, value: Vector) {
            if slot.is_some() {
                *slot = Some(value);
            }
        }
        match name {
            "mean" => self.mean = full(),
            "precision" => self.precision = full(),
            "expected_mean" => self.expected_mean = full(),
            "expected_precision" => self.expected_precision = full(),
            "conditional_expected_precision" => self.conditional_expected_precision = full(),
            "effective_precision" => self.effective_precision = full(),
            "value_prediction_error" => self.value_prediction_error = full(),
            "mean_vol" => set_vol(&mut self.mean_vol, full()),
            "precision_vol" => set_vol(&mut self.precision_vol, full()),
            "expected_mean_vol" => set_vol(&mut self.expected_mean_vol, full()),
            "expected_precision_vol" => set_vol(&mut self.expected_precision_vol, full()),
            "effective_precision_vol" => set_vol(&mut self.effective_precision_vol, full()),
            "volatility_prediction_error" => set_vol(&mut self.volatility_prediction_error, full()),
            other => {
                return Err(format!(
                    "Unknown LayerState field '{other}'. Valid fields: {}.",
                    LAYER_STATE_FIELDS.join(", ")
                ))
            }
        }
        Ok(())
    }
}

/// The [`LayerState`] field names accepted as per-layer overrides.
pub const LAYER_STATE_FIELDS: &[&str] = &[
    "mean",
    "precision",
    "expected_mean",
    "expected_precision",
    "conditional_expected_precision",
    "effective_precision",
    "value_prediction_error",
    "mean_vol",
    "precision_vol",
    "expected_mean_vol",
    "expected_precision_vol",
    "effective_precision_vol",
    "volatility_prediction_error",
];

/// Per-node static parameters for one layer. Each field is length `n_nodes`.
#[derive(Debug, Clone)]
pub struct LayerParams {
    /// Tonic (baseline) volatility of the value level.
    pub tonic_volatility: Vector,
    /// Tonic (baseline) volatility of the volatility level.
    pub tonic_volatility_vol: Vector,
    /// Volatility-coupling strength (κ) between the value and volatility levels.
    pub volatility_coupling: Vector,
    /// Autoconnection (self-coupling) strength of the volatility level.
    pub autoconnection_strength_vol: Vector,
}

impl LayerParams {
    /// Default per-node parameters, matching `LayerParams.create` in the JAX
    /// backend: `tonic_volatility = tonic_volatility_vol = -4`,
    /// `volatility_coupling = autoconnection_strength_vol = 1`.
    pub fn create(n_nodes: usize) -> Self {
        Self::create_with(n_nodes, -4.0, -4.0, 1.0, 1.0)
    }

    /// Per-node parameters with explicit scalar values broadcast to `n_nodes`.
    pub fn create_with(
        n_nodes: usize,
        tonic_volatility: f64,
        tonic_volatility_vol: f64,
        volatility_coupling: f64,
        autoconnection_strength_vol: f64,
    ) -> Self {
        Self {
            tonic_volatility: Array1::from_elem(n_nodes, tonic_volatility),
            tonic_volatility_vol: Array1::from_elem(n_nodes, tonic_volatility_vol),
            volatility_coupling: Array1::from_elem(n_nodes, volatility_coupling),
            autoconnection_strength_vol: Array1::from_elem(n_nodes, autoconnection_strength_vol),
        }
    }
}

/// One layer of the vectorised deep network.
#[derive(Debug, Clone)]
pub struct Layer {
    /// Per-node beliefs.
    pub state: LayerState,
    /// Per-node static parameters.
    pub params: LayerParams,
    /// Incoming weights from the child layer below, shape
    /// `(child_size, self_size[+1])`, or `None` for the bottom layer.
    pub weights_in: Option<Matrix>,
    /// Coupling (activation) function applied to this layer's activations.
    pub coupling_fn: &'static CouplingFn,
    /// Whether a bias column is appended to `weights_in`.
    pub add_constant_input: bool,
    /// Whether the layer has an internal volatility parent.
    pub has_volatility_parent: bool,
    /// Whether this is the bottom observation layer (where `y` is clamped).
    pub is_input_layer: bool,
    /// Whether the incoming weights are fully connected (dense) vs one-to-one
    /// (diagonal).
    pub fully_connected: bool,
    /// The kind of nodes in the layer.
    pub kind: LayerKind,
}

impl Layer {
    /// Number of nodes in this layer.
    pub fn n_nodes(&self) -> usize {
        self.state.n_nodes()
    }
}

/// Declarative description of a single layer, used to build a [`DeepNet`].
///
/// Field defaults match the Python `add_layer` signature: `kind = Volatile`,
/// `add_constant_input = true`, `fully_connected = true`, coupling = identity,
/// `volatility_parent = true`.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Number of nodes in the layer.
    pub size: usize,
    /// The kind of nodes.
    pub kind: LayerKind,
    /// Append a bias input column to this layer's weights.
    pub add_constant_input: bool,
    /// Dense (`true`) vs one-to-one diagonal (`false`) incoming weights.
    pub fully_connected: bool,
    /// Coupling function for this layer.
    pub coupling_fn: &'static CouplingFn,
    /// Give the layer an internal volatility parent.
    pub volatility_parent: bool,
    /// Tonic volatility of the value level (per node, `LayerParams` default −4).
    pub tonic_volatility: f64,
    /// Tonic volatility of the volatility level (default −4).
    pub tonic_volatility_vol: f64,
    /// Volatility-coupling strength κ (default 1).
    pub volatility_coupling: f64,
    /// Volatility-level autoconnection strength (default 1).
    pub autoconnection_strength_vol: f64,
    /// Initial-belief overrides applied to the layer's [`LayerState`] at build
    /// time, as `(field_name, value)` pairs broadcast to every node (e.g.
    /// `("expected_precision", 1e10)` for a noiseless input layer). See
    /// [`LayerState::set_field`] for the accepted names and the
    /// volatility-drop rule.
    pub state_overrides: Vec<(String, f64)>,
}

impl LayerConfig {
    /// A layer of `size` volatile nodes with the Python `add_layer` defaults.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            kind: LayerKind::Volatile,
            add_constant_input: true,
            fully_connected: true,
            coupling_fn: &LINEAR,
            volatility_parent: true,
            tonic_volatility: -4.0,
            tonic_volatility_vol: -4.0,
            volatility_coupling: 1.0,
            autoconnection_strength_vol: 1.0,
            state_overrides: Vec::new(),
        }
    }
}

/// A vectorised deep network: an ordered stack of [`Layer`]s plus the
/// network-level settings shared by every update.
///
/// Index `0` is the bottom observation layer; the last layer is the top.
#[derive(Debug, Clone)]
pub struct DeepNet {
    /// The layers, bottom (index 0) to top.
    pub layers: Vec<Layer>,
    /// The volatility-level posterior update applied network-wide.
    pub volatility_updates: VolatilityUpdate,
    /// Upper bound on posterior precisions.
    pub max_posterior_precision: f64,
    /// Lower/upper clamp bounding binary predictions away from 0 and 1.
    pub precision_clipping_value: f64,
}

impl DeepNet {
    /// Build a network from an ordered list of layer configs (bottom first),
    /// mirroring `DeepNetwork._init_state`. Network settings default to the
    /// Python defaults (unbounded volatility updates,
    /// `max_posterior_precision = 1e10`, `precision_clipping_value = 1e-6`);
    /// override with [`Self::with_settings`].
    ///
    /// All inter-layer weights are initialised to `1.0` (dense) or the identity
    /// (one-to-one); apply a weight-initialisation strategy afterwards for
    /// anything else.
    ///
    /// # Errors
    /// Returns an error if a categorical layer is not the bottom layer, or if a
    /// one-to-one layer (`fully_connected = false`) uses a bias column or does
    /// not match the child layer's size.
    pub fn from_configs(configs: &[LayerConfig]) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(configs.len());

        for (i, cfg) in configs.iter().enumerate() {
            if cfg.kind == LayerKind::Categorical && i != 0 {
                return Err(
                    "Categorical layers represent a single N-way choice and are \
                     only supported as the output (bottom) layer."
                        .to_string(),
                );
            }
            if !cfg.fully_connected {
                if cfg.add_constant_input {
                    return Err("One-to-one layers (fully_connected=false) cannot use \
                         add_constant_input=true."
                        .to_string());
                }
                if i > 0 && configs[i - 1].size != cfg.size {
                    return Err(format!(
                        "One-to-one layers require the same size as the child \
                         layer ({}), got {}.",
                        configs[i - 1].size,
                        cfg.size
                    ));
                }
            }

            let mut state = LayerState::create(cfg.size, cfg.volatility_parent);
            for (name, value) in &cfg.state_overrides {
                state.set_field(name, *value)?;
            }
            let params = LayerParams::create_with(
                cfg.size,
                cfg.tonic_volatility,
                cfg.tonic_volatility_vol,
                cfg.volatility_coupling,
                cfg.autoconnection_strength_vol,
            );

            // `weights_in` lives on the parent (this layer); the bottom layer
            // has no child below it, hence `None`.
            let weights_in = if i > 0 {
                let child_size = configs[i - 1].size;
                let cols = cfg.size + usize::from(cfg.add_constant_input);
                Some(if cfg.fully_connected {
                    Matrix::from_elem((child_size, cols), 1.0)
                } else {
                    // Guaranteed square here (child_size == size, no bias).
                    eye(child_size)
                })
            } else {
                None
            };

            layers.push(Layer {
                state,
                params,
                weights_in,
                coupling_fn: cfg.coupling_fn,
                add_constant_input: cfg.add_constant_input,
                has_volatility_parent: cfg.volatility_parent,
                is_input_layer: i == 0,
                fully_connected: cfg.fully_connected,
                kind: cfg.kind,
            });
        }

        Ok(DeepNet {
            layers,
            // The JAX DeepNetwork default.
            volatility_updates: VolatilityUpdate::Unbounded,
            max_posterior_precision: 1e10,
            precision_clipping_value: 1e-6,
        })
    }

    /// Override the network-level settings (builder style).
    pub fn with_settings(
        mut self,
        volatility_updates: VolatilityUpdate,
        max_posterior_precision: f64,
        precision_clipping_value: f64,
    ) -> Self {
        self.volatility_updates = volatility_updates;
        self.max_posterior_precision = max_posterior_precision;
        self.precision_clipping_value = precision_clipping_value;
        self
    }

    /// Initialise every inter-layer weight matrix with a named strategy
    /// (`"xavier"`, `"he"`, `"orthogonal"`, `"sparse"`), reusing the same
    /// generators as the per-node backend
    /// ([`crate::utils::weight_initialisation`]).
    ///
    /// Matches the JAX `DeepNetwork.weight_initialisation` semantics: the full
    /// matrix is re-drawn **including the bias column**, and the same seed is
    /// used for every layer (identically-shaped layers start identical).
    pub fn weight_initialisation(
        &mut self,
        strategy: &str,
        seed: Option<u64>,
    ) -> Result<(), String> {
        for layer in self.layers.iter_mut() {
            let Some(w) = layer.weights_in.as_mut() else {
                continue;
            };
            let (n_children, cols) = w.dim();
            // Flat, row-major (n_children × cols), same layout as the JAX
            // helpers' `.reshape(n_children, n_parents)`.
            let flat = weight_init_by_name(strategy, cols, n_children, seed)?;
            *w = Matrix::from_shape_vec((n_children, cols), flat)
                .expect("init vector length matches the weight shape");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_state_defaults() {
        let s = LayerState::create(3, true);
        assert_eq!(s.n_nodes(), 3);
        assert!(s.mean.iter().all(|&x| x == 0.0));
        assert!(s.precision.iter().all(|&x| x == 1.0));
        assert!(s.conditional_expected_precision.iter().all(|&x| x == 1.0));
        assert!(s.effective_precision.iter().all(|&x| x == 0.0));
        assert!(s.mean_vol.is_some());
        assert_eq!(s.precision_vol.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_layer_state_no_volatility() {
        let s = LayerState::create(4, false);
        assert!(s.mean_vol.is_none());
        assert!(s.precision_vol.is_none());
        assert!(s.volatility_prediction_error.is_none());
    }

    #[test]
    fn test_layer_params_defaults() {
        let p = LayerParams::create(2);
        assert!(p.tonic_volatility.iter().all(|&x| x == -4.0));
        assert!(p.volatility_coupling.iter().all(|&x| x == 1.0));
        assert!(p.autoconnection_strength_vol.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_kind_parse_roundtrip() {
        assert_eq!(LayerKind::parse("volatile").unwrap(), LayerKind::Volatile);
        assert_eq!(LayerKind::parse("binary-state").unwrap(), LayerKind::Binary);
        assert!(LayerKind::parse("nope").is_err());
    }

    #[test]
    fn test_deepnet_shapes_and_orientation() {
        // Bottom (output) layer first: 2 output nodes, then a 3-node hidden
        // layer, then a 4-node top (input) layer.
        let configs = vec![
            LayerConfig::new(2),
            LayerConfig::new(3),
            LayerConfig::new(4),
        ];
        let net = DeepNet::from_configs(&configs).unwrap();

        assert_eq!(net.layers.len(), 3);
        let sizes: Vec<usize> = net.layers.iter().map(Layer::n_nodes).collect();
        assert_eq!(sizes, vec![2, 3, 4]);

        // Bottom layer is the observation layer, no incoming weights.
        assert!(net.layers[0].is_input_layer);
        assert!(net.layers[0].weights_in.is_none());
        assert!(!net.layers[1].is_input_layer);

        // weights_in on layer i has shape (child_size, self_size + bias).
        // Layer 1: child = layer 0 (size 2), self = 3, +1 bias => (2, 4).
        let w1 = net.layers[1].weights_in.as_ref().unwrap();
        assert_eq!(w1.shape(), &[2, 4]);
        assert!(w1.iter().all(|&x| x == 1.0));
        // Layer 2: child = layer 1 (size 3), self = 4, +1 bias => (3, 5).
        let w2 = net.layers[2].weights_in.as_ref().unwrap();
        assert_eq!(w2.shape(), &[3, 5]);
    }

    #[test]
    fn test_one_to_one_layer_is_diagonal() {
        let configs = vec![
            LayerConfig::new(3),
            LayerConfig {
                add_constant_input: false,
                fully_connected: false,
                ..LayerConfig::new(3)
            },
        ];
        let net = DeepNet::from_configs(&configs).unwrap();
        let w = net.layers[1].weights_in.as_ref().unwrap();
        assert_eq!(w.shape(), &[3, 3]);
        // Identity: ones on the diagonal, zeros off it.
        assert_eq!(w[[0, 0]], 1.0);
        assert_eq!(w[[1, 1]], 1.0);
        assert_eq!(w[[0, 1]], 0.0);
    }

    #[test]
    fn test_categorical_only_at_bottom() {
        // Categorical as the bottom layer is fine.
        let ok = DeepNet::from_configs(&[LayerConfig {
            kind: LayerKind::Categorical,
            ..LayerConfig::new(3)
        }]);
        assert!(ok.is_ok());

        // Categorical above the bottom is rejected.
        let bad = DeepNet::from_configs(&[
            LayerConfig::new(2),
            LayerConfig {
                kind: LayerKind::Categorical,
                ..LayerConfig::new(3)
            },
        ]);
        assert!(bad.is_err());
    }

    #[test]
    fn test_state_overrides_applied() {
        let mut cfg = LayerConfig::new(3);
        cfg.state_overrides = vec![
            ("precision".to_string(), 2.0),
            ("expected_precision".to_string(), 1e10),
        ];
        let net = DeepNet::from_configs(&[cfg]).unwrap();
        let s = &net.layers[0].state;
        assert!(s.precision.iter().all(|&x| x == 2.0));
        assert!(s.expected_precision.iter().all(|&x| x == 1e10));
        // Untouched fields keep their defaults.
        assert!(s.mean.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_vol_override_dropped_without_volatility_parent() {
        let mut cfg = LayerConfig::new(2);
        cfg.volatility_parent = false;
        cfg.state_overrides = vec![("precision_vol".to_string(), 5.0)];
        // Silently dropped (frozen level never reads it), like the JAX builder.
        let net = DeepNet::from_configs(&[cfg]).unwrap();
        assert!(net.layers[0].state.precision_vol.is_none());
    }

    #[test]
    fn test_unknown_state_override_errors() {
        let mut cfg = LayerConfig::new(2);
        cfg.state_overrides = vec![("not_a_field".to_string(), 1.0)];
        assert!(DeepNet::from_configs(&[cfg]).is_err());
    }

    #[test]
    fn test_weight_initialisation() {
        let configs = vec![
            LayerConfig::new(2),
            LayerConfig::new(3),
            LayerConfig::new(3),
        ];
        let mut a = DeepNet::from_configs(&configs).unwrap();
        let mut b = DeepNet::from_configs(&configs).unwrap();

        a.weight_initialisation("he", Some(42)).unwrap();
        b.weight_initialisation("he", Some(42)).unwrap();
        let w_a = a.layers[1].weights_in.as_ref().unwrap();
        // No longer the all-ones default, bias column included.
        assert!(w_a.iter().any(|&x| x != 1.0));
        let cols = w_a.ncols();
        assert!((0..w_a.nrows()).any(|r| w_a[[r, cols - 1]] != 1.0));
        // Deterministic for the same seed, on every layer.
        assert_eq!(w_a, b.layers[1].weights_in.as_ref().unwrap());
        assert_eq!(
            a.layers[2].weights_in.as_ref().unwrap(),
            b.layers[2].weights_in.as_ref().unwrap()
        );
        // Unknown strategy errors.
        assert!(a.weight_initialisation("nope", None).is_err());
    }

    #[test]
    fn test_one_to_one_rejects_bias() {
        let bad = DeepNet::from_configs(&[
            LayerConfig::new(3),
            LayerConfig {
                add_constant_input: true,
                fully_connected: false,
                ..LayerConfig::new(3)
            },
        ]);
        assert!(bad.is_err());
    }
}
