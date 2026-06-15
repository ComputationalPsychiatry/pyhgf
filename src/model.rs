use std::collections::HashMap;
use crate::utils::function_pointer::UpdateStep;
use crate::utils::set_sequence::set_update_sequence;
use crate::utils::beliefs_propagation::belief_propagation;
use crate::utils::set_learning_sequence::build_learning_sequence;
use crate::utils::weight_initialisation::weight_init_by_name;
use crate::updates::observations::{set_predictors, set_observation};
use crate::optimiser::AdamState;
use pyo3::types::PyTuple;
use pyo3::{prelude::*, types::{PyList, PyDict}};
use numpy::{PyArray1, PyArray, PyArrayMethods};

/// Accepts either a single int or a list of ints from Python.
/// Allows `value_children=0` or `value_children=[0, 1]`.
#[derive(Debug, Clone)]
pub enum IntOrList {
    Single(usize),
    List(Vec<usize>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for IntOrList {
    type Error = PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(val) = ob.extract::<usize>() {
            Ok(IntOrList::Single(val))
        } else {
            Ok(IntOrList::List(ob.extract::<Vec<usize>>()?))
        }
    }
}

impl From<Vec<usize>> for IntOrList {
    fn from(v: Vec<usize>) -> Self { IntOrList::List(v) }
}

impl From<usize> for IntOrList {
    fn from(v: usize) -> Self { IntOrList::Single(v) }
}

impl IntOrList {
    fn into_vec(self) -> Vec<usize> {
        match self {
            IntOrList::Single(v) => vec![v],
            IntOrList::List(v) => v,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(skip_from_py_object)]
pub struct AdjacencyLists{
    #[pyo3(get, set)]
    pub node_type: String,
    #[pyo3(get, set)]
    pub learning_kind: String,
    #[pyo3(get, set)]
    pub value_parents: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub value_children: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub volatility_parents: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub volatility_children: Option<Vec<usize>>,
}

#[derive(Debug)]
pub struct UpdateSequence {
    pub predictions: Vec<(usize, UpdateStep)>,
    pub updates: Vec<(usize, UpdateStep)>,
}

// =============================================================================
// Flat struct types — replacing HashMap<String, f64>
// =============================================================================

/// Flat struct for all node scalar attributes.
/// Unused fields default to 0.0 (or NaN for optional fields like `lr`).
#[derive(Debug, Clone, Copy)]
pub struct NodeState {
    pub mean: f64,
    pub expected_mean: f64,
    pub precision: f64,
    pub expected_precision: f64,
    /// Conditional predicted precision π̂_a (own variance + volatility,
    /// without the parent-uncertainty value-coupling term). Transient: recomputed
    /// each prediction step and consumed by the parent's posterior-step Schur
    /// correction; not recorded in `NodeTrajectory`.
    pub conditional_expected_precision: f64,
    pub observed: f64,
    pub tonic_volatility: f64,
    pub tonic_drift: f64,
    pub autoconnection_strength: f64,
    pub current_variance: f64,
    pub effective_precision: f64,
    pub value_prediction_error: f64,
    pub volatility_prediction_error: f64,
    // Volatile-state internal volatility level
    pub mean_vol: f64,
    pub expected_mean_vol: f64,
    pub precision_vol: f64,
    pub expected_precision_vol: f64,
    pub tonic_volatility_vol: f64,
    pub tonic_drift_vol: f64,
    pub autoconnection_strength_vol: f64,
    pub volatility_coupling_internal: f64,
    pub effective_precision_vol: f64,
    // EF-state
    pub nus: f64,
    // Learning
    pub lr: f64,
}

impl Default for NodeState {
    fn default() -> Self {
        NodeState {
            mean: 0.0,
            expected_mean: 0.0,
            precision: 1.0,
            expected_precision: 1.0,
            conditional_expected_precision: 1.0,
            observed: 1.0,
            tonic_volatility: 0.0,
            tonic_drift: 0.0,
            autoconnection_strength: 0.0,
            current_variance: 1.0,
            effective_precision: 0.0,
            value_prediction_error: 0.0,
            volatility_prediction_error: 0.0,
            mean_vol: 0.0,
            expected_mean_vol: 0.0,
            precision_vol: 1.0,
            expected_precision_vol: 1.0,
            tonic_volatility_vol: -4.0,
            tonic_drift_vol: 0.0,
            autoconnection_strength_vol: 1.0,
            volatility_coupling_internal: 1.0,
            effective_precision_vol: 0.0,
            nus: 0.0,
            lr: f64::NAN,
        }
    }
}

/// Per-node variable-length vector attributes.
#[derive(Debug, Clone, Default)]
pub struct NodeVectors {
    pub value_coupling_parents: Vec<f64>,
    pub value_coupling_children: Vec<f64>,
    pub volatility_coupling_parents: Vec<f64>,
    pub volatility_coupling_children: Vec<f64>,
    pub xis: Vec<f64>,
}

/// Per-node function pointer attributes.
///
/// The coupling function is defined on the **parent** node and applies to all
/// its value children.  `None` means linear coupling (the default) and avoids
/// any function-pointer call overhead at runtime.
#[derive(Debug, Clone, Copy)]
pub struct NodeFnPtrs {
    pub coupling_fn: Option<&'static crate::math::CouplingFn>,
}

impl Default for NodeFnPtrs {
    fn default() -> Self {
        NodeFnPtrs { coupling_fn: None }
    }
}

#[derive(Debug, Clone)]
pub struct Attributes {
    pub states: Vec<NodeState>,
    pub vectors: Vec<NodeVectors>,
    pub fn_ptrs: Vec<NodeFnPtrs>,
}

/// Trajectory recording for a single node.
#[derive(Debug)]
pub struct NodeTrajectory {
    pub mean: Vec<f64>,
    pub expected_mean: Vec<f64>,
    pub precision: Vec<f64>,
    pub expected_precision: Vec<f64>,
    pub observed: Vec<f64>,
    pub tonic_volatility: Vec<f64>,
    pub tonic_drift: Vec<f64>,
    pub autoconnection_strength: Vec<f64>,
    pub current_variance: Vec<f64>,
    pub effective_precision: Vec<f64>,
    pub value_prediction_error: Vec<f64>,
    pub volatility_prediction_error: Vec<f64>,
    pub mean_vol: Vec<f64>,
    pub expected_mean_vol: Vec<f64>,
    pub precision_vol: Vec<f64>,
    pub expected_precision_vol: Vec<f64>,
    pub tonic_volatility_vol: Vec<f64>,
    pub tonic_drift_vol: Vec<f64>,
    pub autoconnection_strength_vol: Vec<f64>,
    pub volatility_coupling_internal: Vec<f64>,
    pub effective_precision_vol: Vec<f64>,
    pub nus: Vec<f64>,
    pub lr: Vec<f64>,
    // Vector trajectory
    pub xis: Vec<Vec<f64>>,
    pub value_coupling_parents: Vec<Vec<f64>>,
    pub value_coupling_children: Vec<Vec<f64>>,
    pub volatility_coupling_parents: Vec<Vec<f64>>,
    pub volatility_coupling_children: Vec<Vec<f64>>,
}

impl NodeTrajectory {
    pub fn with_capacity(n: usize) -> Self {
        NodeTrajectory {
            mean: Vec::with_capacity(n),
            expected_mean: Vec::with_capacity(n),
            precision: Vec::with_capacity(n),
            expected_precision: Vec::with_capacity(n),
            observed: Vec::with_capacity(n),
            tonic_volatility: Vec::with_capacity(n),
            tonic_drift: Vec::with_capacity(n),
            autoconnection_strength: Vec::with_capacity(n),
            current_variance: Vec::with_capacity(n),
            effective_precision: Vec::with_capacity(n),
            value_prediction_error: Vec::with_capacity(n),
            volatility_prediction_error: Vec::with_capacity(n),
            mean_vol: Vec::with_capacity(n),
            expected_mean_vol: Vec::with_capacity(n),
            precision_vol: Vec::with_capacity(n),
            expected_precision_vol: Vec::with_capacity(n),
            tonic_volatility_vol: Vec::with_capacity(n),
            tonic_drift_vol: Vec::with_capacity(n),
            autoconnection_strength_vol: Vec::with_capacity(n),
            volatility_coupling_internal: Vec::with_capacity(n),
            effective_precision_vol: Vec::with_capacity(n),
            nus: Vec::with_capacity(n),
            lr: Vec::with_capacity(n),
            xis: Vec::with_capacity(n),
            value_coupling_parents: Vec::with_capacity(n),
            value_coupling_children: Vec::with_capacity(n),
            volatility_coupling_parents: Vec::with_capacity(n),
            volatility_coupling_children: Vec::with_capacity(n),
        }
    }

    pub fn push_state(&mut self, s: &NodeState) {
        self.mean.push(s.mean);
        self.expected_mean.push(s.expected_mean);
        self.precision.push(s.precision);
        self.expected_precision.push(s.expected_precision);
        self.observed.push(s.observed);
        self.tonic_volatility.push(s.tonic_volatility);
        self.tonic_drift.push(s.tonic_drift);
        self.autoconnection_strength.push(s.autoconnection_strength);
        self.current_variance.push(s.current_variance);
        self.effective_precision.push(s.effective_precision);
        self.value_prediction_error.push(s.value_prediction_error);
        self.volatility_prediction_error.push(s.volatility_prediction_error);
        self.mean_vol.push(s.mean_vol);
        self.expected_mean_vol.push(s.expected_mean_vol);
        self.precision_vol.push(s.precision_vol);
        self.expected_precision_vol.push(s.expected_precision_vol);
        self.tonic_volatility_vol.push(s.tonic_volatility_vol);
        self.tonic_drift_vol.push(s.tonic_drift_vol);
        self.autoconnection_strength_vol.push(s.autoconnection_strength_vol);
        self.volatility_coupling_internal.push(s.volatility_coupling_internal);
        self.effective_precision_vol.push(s.effective_precision_vol);
        self.nus.push(s.nus);
        self.lr.push(s.lr);
    }

    pub fn push_vectors(&mut self, v: &NodeVectors) {
        if !v.xis.is_empty() {
            self.xis.push(v.xis.clone());
        }
        if !v.value_coupling_parents.is_empty() {
            self.value_coupling_parents.push(v.value_coupling_parents.clone());
        }
        if !v.value_coupling_children.is_empty() {
            self.value_coupling_children.push(v.value_coupling_children.clone());
        }
        if !v.volatility_coupling_parents.is_empty() {
            self.volatility_coupling_parents.push(v.volatility_coupling_parents.clone());
        }
        if !v.volatility_coupling_children.is_empty() {
            self.volatility_coupling_children.push(v.volatility_coupling_children.clone());
        }
    }
}

#[derive(Debug)]
pub struct NodeTrajectories {
    pub nodes: Vec<NodeTrajectory>,
}

#[derive(Debug)]
#[pyclass]
pub struct Network{
    pub attributes: Attributes,
    pub edges: Vec<AdjacencyLists>,
    pub inputs: Vec<usize>,
    pub volatility_updates: String,
    pub mean_field_updates: bool,
    pub update_sequence: UpdateSequence,
    pub node_trajectories: NodeTrajectories,
    pub layers: Vec<Vec<usize>>,
    /// Optional Adam optimiser state, initialised when `fit()` is called with `optimizer="adam"`.
    pub adam_state: Option<AdamState>,
    /// Root nodes: nodes that have no children (neither value nor volatility).
    pub roots: Vec<usize>,
    /// Leaf nodes: nodes that have no parents (neither value nor volatility).
    pub leafs: Vec<usize>,
    /// Upper bound applied to every posterior precision write (value level for
    /// continuous/volatile nodes and the implicit volatility level for volatile
    /// nodes). Defaults to ``1e10`` and is shared with the JAX backends.
    pub max_posterior_precision: f64,
}

/// Helper: get the list of trajectory field names to export for a given node type.
fn trajectory_fields_for_type(node_type: &str) -> &'static [&'static str] {
    match node_type {
        "binary-state" => &[
            "observed", "mean", "expected_mean", "precision",
            "expected_precision", "value_prediction_error",
        ],
        "continuous-state" => &[
            "mean", "expected_mean", "precision", "expected_precision",
            "tonic_volatility", "tonic_drift", "autoconnection_strength",
            "current_variance", "effective_precision",
            "value_prediction_error", "volatility_prediction_error",
        ],
        "volatile-state" => &[
            "mean", "expected_mean", "precision", "expected_precision",
            "tonic_volatility", "tonic_drift", "autoconnection_strength",
            "current_variance", "effective_precision",
            "value_prediction_error", "volatility_prediction_error",
            "mean_vol", "expected_mean_vol", "precision_vol",
            "expected_precision_vol", "tonic_volatility_vol",
            "tonic_drift_vol", "autoconnection_strength_vol",
            "volatility_coupling_internal", "effective_precision_vol",
            "observed",
        ],
        "ef-state" => &["mean", "nus"],
        "constant-state" => &["mean", "expected_mean"],
        _ => &[],
    }
}

/// Helper: get a reference to the trajectory Vec<f64> for a given field name.
fn trajectory_field_ref<'a>(traj: &'a NodeTrajectory, field: &str) -> &'a Vec<f64> {
    match field {
        "mean" => &traj.mean,
        "expected_mean" => &traj.expected_mean,
        "precision" => &traj.precision,
        "expected_precision" => &traj.expected_precision,
        "observed" => &traj.observed,
        "tonic_volatility" => &traj.tonic_volatility,
        "tonic_drift" => &traj.tonic_drift,
        "autoconnection_strength" => &traj.autoconnection_strength,
        "current_variance" => &traj.current_variance,
        "effective_precision" => &traj.effective_precision,
        "value_prediction_error" => &traj.value_prediction_error,
        "volatility_prediction_error" => &traj.volatility_prediction_error,
        "mean_vol" => &traj.mean_vol,
        "expected_mean_vol" => &traj.expected_mean_vol,
        "precision_vol" => &traj.precision_vol,
        "expected_precision_vol" => &traj.expected_precision_vol,
        "tonic_volatility_vol" => &traj.tonic_volatility_vol,
        "tonic_drift_vol" => &traj.tonic_drift_vol,
        "autoconnection_strength_vol" => &traj.autoconnection_strength_vol,
        "volatility_coupling_internal" => &traj.volatility_coupling_internal,
        "effective_precision_vol" => &traj.effective_precision_vol,
        "nus" => &traj.nus,
        "lr" => &traj.lr,
        _ => &traj.mean, // fallback
    }
}

// Core Rust methods (also callable from Python via chaining wrappers below)
impl Network {

    pub fn new(volatility_updates: &str) -> Self {
        Network {
            attributes: Attributes { states: Vec::new(), vectors: Vec::new(), fn_ptrs: Vec::new() },
            edges: Vec::new(),
            inputs: Vec::new(),
            volatility_updates: String::from(volatility_updates),
            mean_field_updates: false,
            update_sequence: UpdateSequence { predictions: Vec::new(), updates: Vec::new() },
            node_trajectories: NodeTrajectories { nodes: Vec::new() },
            layers: Vec::new(),
            adam_state: None,
            roots: Vec::new(),
            leafs: Vec::new(),
            max_posterior_precision: 1e10,
        }
    }

    pub fn add_nodes(
        &mut self,
        kind: &str,
        n_nodes: usize,
        value_parents: Option<IntOrList>,
        value_children: Option<IntOrList>,
        volatility_parents: Option<IntOrList>,
        volatility_children: Option<IntOrList>,
        coupling_fn: Option<String>,
        additional_parameters: Option<HashMap<String, f64>>,
    ) {
        let coupling_fn_opt: Option<&'static crate::math::CouplingFn> = match coupling_fn.as_deref().unwrap_or("linear") {
            "linear" => None,
            name => Some(crate::math::resolve_coupling_fn(name)),
        };
        let value_parents = value_parents.map(|v| v.into_vec());
        let value_children = value_children.map(|v| v.into_vec());
        let volatility_parents = volatility_parents.map(|v| v.into_vec());
        let volatility_children = volatility_children.map(|v| v.into_vec());

      for _ in 0..n_nodes {
        let node_id = self.edges.len();

        let has_children = value_children.is_some() || volatility_children.is_some();
        let has_parents = value_parents.is_some() || volatility_parents.is_some();

        let is_input = !has_children;
        if is_input {
            self.inputs.push(node_id);
        }

        // Update roots/leafs tracking
        if !has_children {
            self.roots.push(node_id);
        }
        if !has_parents && kind != "constant-state" {
            self.leafs.push(node_id);
        }
        // Children that gain this node as a parent are no longer leafs
        if let Some(ref vc) = value_children {
            for &child_idx in vc {
                self.leafs.retain(|&x| x != child_idx);
            }
        }
        if let Some(ref volc) = volatility_children {
            for &child_idx in volc {
                self.leafs.retain(|&x| x != child_idx);
            }
        }
        // Parents that gain this node as a child are no longer roots
        if let Some(ref vp) = value_parents {
            for &parent_idx in vp {
                self.roots.retain(|&x| x != parent_idx);
            }
        }
        if let Some(ref volp) = volatility_parents {
            for &parent_idx in volp {
                self.roots.retain(|&x| x != parent_idx);
            }
        }

        let edges = AdjacencyLists {
            node_type: String::from(kind),
            learning_kind: String::from("precision_weighted"),
            value_parents: value_parents.clone(),
            value_children: value_children.clone(),
            volatility_parents: volatility_parents.clone(),
            volatility_children: volatility_children.clone(),
        };

        match kind {
            "continuous-state" => {
                let (autoconnection, tonic_vol) = if is_input { (0.0, 0.0) } else { (1.0, -4.0) };

                let mut state = NodeState {
                    mean: 0.0,
                    expected_mean: 0.0,
                    precision: 1.0,
                    expected_precision: 1.0,
                    tonic_volatility: tonic_vol,
                    tonic_drift: 0.0,
                    autoconnection_strength: autoconnection,
                    current_variance: 1.0,
                    ..Default::default()
                };

                // Apply additional_parameters overrides
                if let Some(ref overrides) = additional_parameters {
                    apply_overrides_continuous(&mut state, overrides);
                }

                self.attributes.states.push(state);
                self.edges.push(edges);

                let mut vecs = NodeVectors::default();
                let fns = NodeFnPtrs { coupling_fn: coupling_fn_opt };

                if let Some(ref vp) = value_parents {
                    vecs.value_coupling_parents = vec![1.0; vp.len()];
                }
                if let Some(ref vc) = value_children {
                    vecs.value_coupling_children = vec![1.0; vc.len()];
                    for &child_idx in vc {
                        if let Some(child_edges) = self.edges.get_mut(child_idx) {
                            match &mut child_edges.value_parents {
                                Some(parents) => parents.push(node_id),
                                None => child_edges.value_parents = Some(vec![node_id]),
                            }
                        }
                        if child_idx < self.attributes.vectors.len() {
                            self.attributes.vectors[child_idx].value_coupling_parents.push(1.0);
                        }
                    }
                }
                if let Some(ref volp) = volatility_parents {
                    vecs.volatility_coupling_parents = vec![1.0; volp.len()];
                }
                if let Some(ref volc) = volatility_children {
                    vecs.volatility_coupling_children = vec![1.0; volc.len()];
                    for &child_idx in volc {
                        if let Some(child_edges) = self.edges.get_mut(child_idx) {
                            match &mut child_edges.volatility_parents {
                                Some(parents) => parents.push(node_id),
                                None => child_edges.volatility_parents = Some(vec![node_id]),
                            }
                        }
                        if child_idx < self.attributes.vectors.len() {
                            self.attributes.vectors[child_idx].volatility_coupling_parents.push(1.0);
                        }
                    }
                }

                self.attributes.vectors.push(vecs);
                self.attributes.fn_ptrs.push(fns);
            }
            "ef-state" => {
                let state = NodeState {
                    mean: 0.0,
                    nus: 3.0,
                    ..Default::default()
                };
                self.attributes.states.push(state);
                self.edges.push(edges);
                let vecs = NodeVectors {
                    xis: vec![0.0, 1.0],
                    ..Default::default()
                };
                self.attributes.vectors.push(vecs);
                self.attributes.fn_ptrs.push(NodeFnPtrs::default());
            }
            "volatile-state" => {
                let volatile_edges = AdjacencyLists {
                    node_type: String::from(kind),
                    learning_kind: String::from("precision_weighted"),
                    value_parents: value_parents.clone(),
                    value_children: value_children.clone(),
                    volatility_parents: None,
                    volatility_children: None,
                };

                let mut state = NodeState {
                    mean: 0.0,
                    expected_mean: 0.0,
                    precision: 1.0,
                    expected_precision: 1.0,
                    tonic_volatility: -4.0,
                    tonic_drift: 0.0,
                    autoconnection_strength: 0.0,
                    current_variance: 1.0,
                    mean_vol: 0.0,
                    expected_mean_vol: 0.0,
                    precision_vol: 1.0,
                    expected_precision_vol: 1.0,
                    tonic_volatility_vol: -4.0,
                    tonic_drift_vol: 0.0,
                    autoconnection_strength_vol: 1.0,
                    volatility_coupling_internal: 1.0,
                    effective_precision: 0.0,
                    value_prediction_error: 0.0,
                    volatility_prediction_error: 0.0,
                    effective_precision_vol: 0.0,
                    ..Default::default()
                };

                if let Some(ref overrides) = additional_parameters {
                    apply_overrides_volatile(&mut state, overrides);
                }

                self.attributes.states.push(state);
                self.edges.push(volatile_edges);

                let mut vecs = NodeVectors::default();

                if let Some(ref vp) = value_parents {
                    vecs.value_coupling_parents = vec![1.0; vp.len()];
                }
                if let Some(ref vc) = value_children {
                    vecs.value_coupling_children = vec![1.0; vc.len()];
                    for &child_idx in vc {
                        if let Some(child_edges) = self.edges.get_mut(child_idx) {
                            match &mut child_edges.value_parents {
                                Some(parents) => parents.push(node_id),
                                None => child_edges.value_parents = Some(vec![node_id]),
                            }
                        }
                        if child_idx < self.attributes.vectors.len() {
                            self.attributes.vectors[child_idx].value_coupling_parents.push(1.0);
                        }
                    }
                }

                self.attributes.vectors.push(vecs);
                self.attributes.fn_ptrs.push(NodeFnPtrs { coupling_fn: coupling_fn_opt });
            }
            "binary-state" => {
                let state = NodeState {
                    observed: 1.0,
                    mean: 0.0,
                    expected_mean: 0.5,
                    precision: 1.0,
                    expected_precision: 1.0,
                    value_prediction_error: 0.0,
                    ..Default::default()
                };
                self.attributes.states.push(state);
                self.edges.push(edges);

                let mut vecs = NodeVectors::default();

                if let Some(ref vp) = value_parents {
                    vecs.value_coupling_parents = vec![1.0; vp.len()];
                }
                if let Some(ref vc) = value_children {
                    vecs.value_coupling_children = vec![1.0; vc.len()];
                    for &child_idx in vc {
                        if let Some(child_edges) = self.edges.get_mut(child_idx) {
                            match &mut child_edges.value_parents {
                                Some(parents) => parents.push(node_id),
                                None => child_edges.value_parents = Some(vec![node_id]),
                            }
                        }
                        if child_idx < self.attributes.vectors.len() {
                            self.attributes.vectors[child_idx].value_coupling_parents.push(1.0);
                        }
                    }
                }

                self.attributes.vectors.push(vecs);
                self.attributes.fn_ptrs.push(NodeFnPtrs { coupling_fn: coupling_fn_opt });
            }
            "constant-state" => {
                // Constant state nodes are assumed to have mean = 1.0 and
                // precision = 1.0 (fully known bias). They are always wired to
                // their children linearly (no coupling function), regardless
                // of the layer's coupling_fn.
                //
                // ``expected_precision`` is set to infinity so that the piHGF
                // Laplace value-coupling term `(t · α · g'(µ̂))² / π̂_parent`
                // contributes zero for the bias parent — matching the JAX
                // vectorised backend, which concatenates an `inf` into the
                // parent-precision vector for the constant column. (The
                // posterior-level ``precision`` is kept at 1.0 because the
                // ``precision_ratio`` learning gain reads it directly.)
                let state = NodeState {
                    mean: 1.0,
                    expected_mean: 1.0,
                    precision: 1.0,
                    expected_precision: f64::INFINITY,
                    ..Default::default()
                };
                self.attributes.states.push(state);
                self.edges.push(edges);

                let mut vecs = NodeVectors::default();

                if let Some(ref vc) = value_children {
                    vecs.value_coupling_children = vec![1.0; vc.len()];
                    for &child_idx in vc {
                        if let Some(child_edges) = self.edges.get_mut(child_idx) {
                            match &mut child_edges.value_parents {
                                Some(parents) => parents.push(node_id),
                                None => child_edges.value_parents = Some(vec![node_id]),
                            }
                        }
                        if child_idx < self.attributes.vectors.len() {
                            self.attributes.vectors[child_idx].value_coupling_parents.push(1.0);
                        }
                    }
                }
                if let Some(ref volc) = volatility_children {
                    vecs.volatility_coupling_children = vec![1.0; volc.len()];
                    for &child_idx in volc {
                        if let Some(child_edges) = self.edges.get_mut(child_idx) {
                            match &mut child_edges.volatility_parents {
                                Some(parents) => parents.push(node_id),
                                None => child_edges.volatility_parents = Some(vec![node_id]),
                            }
                        }
                        if child_idx < self.attributes.vectors.len() {
                            self.attributes.vectors[child_idx].volatility_coupling_parents.push(1.0);
                        }
                    }
                }

                self.attributes.vectors.push(vecs);
                // Force constant-state nodes to use no coupling (identity)
                // regardless of what the caller passed.
                self.attributes.fn_ptrs.push(NodeFnPtrs { coupling_fn: None });
            }
            _ => {}
        }

        // Reciprocal updates: when value_parents or volatility_parents are
        // specified, update each parent's children list so the parent knows
        // about this new child.  (The reverse direction — value_children
        // updating the child's parents — is already handled above.)
        let vp_clone = self.edges[node_id].value_parents.clone();
        let volp_clone = self.edges[node_id].volatility_parents.clone();

        if let Some(ref vp) = vp_clone {
            for &parent_idx in vp {
                // Skip if the parent node hasn't been created yet (it will
                // perform the reciprocal update via its own value_children).
                if parent_idx >= self.edges.len() { continue; }
                match &mut self.edges[parent_idx].value_children {
                    Some(children) => {
                        if !children.contains(&node_id) {
                            children.push(node_id);
                        }
                    }
                    None => self.edges[parent_idx].value_children = Some(vec![node_id]),
                }
                // Add coupling strength on the parent side only if not already
                // present (the value_children branch in each node-type arm
                // already handles couplings for the child→parent direction).
                let parent_n_children = self.edges[parent_idx]
                    .value_children.as_ref().map(|c| c.len()).unwrap_or(0);
                let parent_coupling_len = self.attributes.vectors[parent_idx]
                    .value_coupling_children.len();
                if parent_coupling_len < parent_n_children {
                    self.attributes.vectors[parent_idx]
                        .value_coupling_children.push(1.0);
                }
            }
        }
        if let Some(ref volp) = volp_clone {
            for &parent_idx in volp {
                if parent_idx >= self.edges.len() { continue; }
                match &mut self.edges[parent_idx].volatility_children {
                    Some(children) => {
                        if !children.contains(&node_id) {
                            children.push(node_id);
                        }
                    }
                    None => self.edges[parent_idx].volatility_children = Some(vec![node_id]),
                }
                let parent_n_children = self.edges[parent_idx]
                    .volatility_children.as_ref().map(|c| c.len()).unwrap_or(0);
                let parent_coupling_len = self.attributes.vectors[parent_idx]
                    .volatility_coupling_children.len();
                if parent_coupling_len < parent_n_children {
                    self.attributes.vectors[parent_idx]
                        .volatility_coupling_children.push(1.0);
                }
            }
        }
      } // end for n_nodes
    }

    pub fn set_update_sequence(&mut self) {
        self.update_sequence = set_update_sequence(self);
    }

    pub fn input_data(&mut self, input_data: Vec<Vec<f64>>, time_steps: Option<Vec<f64>>, record_trajectories: bool) {
        if self.update_sequence.predictions.is_empty() && self.update_sequence.updates.is_empty() {
            self.set_update_sequence();
        }

        let n_time = input_data.len();
        let time_steps = time_steps.unwrap_or_else(|| vec![1.0; n_time]);
        let predictions = self.update_sequence.predictions.clone();
        let updates = self.update_sequence.updates.clone();

        let mut node_trajectories = NodeTrajectories { nodes: Vec::new() };

        if record_trajectories {
            for _ in 0..self.attributes.states.len() {
                node_trajectories.nodes.push(NodeTrajectory::with_capacity(n_time));
            }
        }

        for (t, observations) in input_data.iter().enumerate() {
            belief_propagation(self, observations, &predictions, &updates, time_steps[t]);

            if record_trajectories {
                for (i, state) in self.attributes.states.iter().enumerate() {
                    node_trajectories.nodes[i].push_state(state);
                    node_trajectories.nodes[i].push_vectors(&self.attributes.vectors[i]);
                }
            }
        }

        if record_trajectories {
            self.node_trajectories = node_trajectories;
        }
    }

    pub fn add_layer(
        &mut self,
        size: usize,
        kind: &str,
        value_children: Option<Vec<usize>>,
        coupling_strengths: f64,
        coupling_fn: Option<String>,
        additional_parameters: Option<HashMap<String, f64>>,
        add_constant_input: bool,
    ) {
        let n_nodes_before = self.edges.len();

        let children: Vec<usize> = match value_children {
            Some(vc) => vc,
            None => {
                // If layers exist, connect to the last layer's nodes;
                // otherwise connect to all leaf nodes (nodes without parents).
                if let Some(last_layer) = self.layers.last() {
                    last_layer.clone()
                } else {
                    self.leafs.clone()
                }
            }
        }.into_iter()
            .filter(|&idx| self.edges[idx].node_type != "constant-state")
            .collect();

        let additional_parameters = {
            let mut params = additional_parameters.unwrap_or_default();
            params.entry("autoconnection_strength".into()).or_insert(0.0);
            Some(params)
        };

        for _ in 0..size {
            let vc = IntOrList::List(children.clone());
            self.add_nodes(kind, 1, None, Some(vc), None, None, coupling_fn.clone(), additional_parameters.clone());

            let node_id = self.edges.len() - 1;
            for v in self.attributes.vectors[node_id].value_coupling_children.iter_mut() {
                *v = coupling_strengths;
            }
            for &child_idx in &children {
                if let Some(last) = self.attributes.vectors[child_idx].value_coupling_parents.last_mut() {
                    *last = coupling_strengths;
                }
            }
        }

        if add_constant_input {
            let non_constant_children: Vec<usize> = children.iter()
                .filter(|&&idx| self.edges[idx].node_type != "constant-state")
                .copied()
                .collect();

            if !non_constant_children.is_empty() {
                let vc = IntOrList::List(non_constant_children);
                self.add_nodes("constant-state", 1, None, Some(vc), None, None, coupling_fn.clone(), None);
            }
        }

        let new_layer: Vec<usize> = (n_nodes_before..self.edges.len()).collect();
        self.layers.push(new_layer);
    }

    pub fn add_layer_stack(
        &mut self,
        layer_sizes: Vec<usize>,
        kind: &str,
        value_children: Option<Vec<usize>>,
        coupling_strengths: f64,
        coupling_fn: Option<String>,
        additional_parameters: Option<HashMap<String, f64>>,
        add_constant_input: bool,
    ) {
        for (i, &size) in layer_sizes.iter().enumerate() {
            if i == 0 {
                self.add_layer(size, kind, value_children.clone(), coupling_strengths, coupling_fn.clone(), additional_parameters.clone(), add_constant_input);
            } else {
                self.add_layer(size, kind, None, coupling_strengths, coupling_fn.clone(), additional_parameters.clone(), add_constant_input);
            }
        }
    }

    /// Train the network on input/output pairs.
    ///
    /// # Arguments
    /// * `x` - Input data, one row per time step.
    /// * `y` - Target data, one row per time step.
    /// * `inputs_x_idxs` - Node indices that receive input observations. Defaults
    ///   to the leaf nodes (nodes without parents) when not provided from Python.
    /// * `inputs_y_idxs` - Node indices that receive target observations. Defaults
    ///   to the root nodes (nodes without children) when not provided from Python.
    /// * `lr` - Gradient application. `Some(f)` sets a fixed learning rate on all
    ///   non-input nodes. `None` triggers the Adam optimiser (equivalent to
    ///   `lr="adam"` from Python); the Adam step size is taken from
    ///   `params["lr"]` (default 1e-3).
    /// * `record_trajectories` - When `true`, stores the full state history for
    ///   every node at each time step, accessible via `node_trajectories`.
    /// * `params` - Optional dictionary of Adam hyper-parameters (only used when
    ///   `lr == None`): `beta1` (default 0.9), `beta2` (default 0.999),
    ///   `epsilon` (default 1e-8), and `lr` (default 1e-3, the Adam step size).
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        y: &[Vec<f64>],
        inputs_x_idxs: &[usize],
        inputs_y_idxs: &[usize],
        lr: Option<f64>,
        record_trajectories: bool,
        params: Option<&HashMap<String, f64>>,
        learning_kind: &str,
    ) {
        if self.update_sequence.predictions.is_empty()
            && self.update_sequence.updates.is_empty()
        {
            self.set_update_sequence();
        }

        // Set learning_kind on all non-input nodes
        for (node_idx, edge) in self.edges.iter_mut().enumerate() {
            if !inputs_x_idxs.contains(&node_idx) {
                edge.learning_kind = String::from(learning_kind);
            }
        }

        // Always set a fixed lr on non-input nodes so learning_weights never skips
        // updates when lr is NaN.  When Adam is requested (lr == None), the Adam
        // step size from params overrides this.
        let fixed_lr = lr.unwrap_or(1e-3);
        for (node_idx, state) in self.attributes.states.iter_mut().enumerate() {
            if !inputs_x_idxs.contains(&node_idx) {
                state.lr = fixed_lr;
            }
        }

        // Initialise Adam optimiser state when lr == None ("adam" on the Python side)
        if lr.is_none() {
            let coupling_sizes: Vec<usize> = self
                .attributes
                .vectors
                .iter()
                .map(|v| v.value_coupling_parents.len())
                .collect();
            let beta1 = params.and_then(|p| p.get("beta1").copied()).unwrap_or(0.9);
            let beta2 = params.and_then(|p| p.get("beta2").copied()).unwrap_or(0.999);
            let epsilon = params.and_then(|p| p.get("epsilon").copied()).unwrap_or(1e-8);
            let adam_lr = params.and_then(|p| p.get("lr").copied()).unwrap_or(1e-3);
            let mut adam = AdamState::new(&coupling_sizes, beta1, beta2, epsilon);
            adam.lr = Some(adam_lr);
            self.adam_state = Some(adam);
        } else {
            self.adam_state = None;
        }

        let learning_seq = build_learning_sequence(
            &self.update_sequence.predictions,
            &self.update_sequence.updates,
            inputs_x_idxs,
            &self.edges,
        );

        let n_time = x.len();
        let time_step = 1.0;

        let mut node_trajectories = NodeTrajectories { nodes: Vec::new() };

        if record_trajectories {
            for _ in 0..self.attributes.states.len() {
                node_trajectories.nodes.push(NodeTrajectory::with_capacity(n_time));
            }
        }

        for t in 0..n_time {
            for (i, &node_idx) in inputs_x_idxs.iter().enumerate() {
                set_predictors(self, node_idx, x[t][i]);
            }

            for &(idx, step) in &learning_seq.prediction_steps {
                step.call(self, idx, time_step);
            }

            for (i, &node_idx) in inputs_y_idxs.iter().enumerate() {
                set_observation(self, node_idx, y[t][i]);
            }

            for &(idx, step) in &learning_seq.update_steps {
                step.call(self, idx, time_step);
            }

            // Increment Adam timestep once per iteration (before learning steps)
            if let Some(ref mut adam) = self.adam_state {
                adam.increment_timestep();
            }

            for &(idx, step) in &learning_seq.learning_steps {
                step.call(self, idx, time_step);
            }

            if record_trajectories {
                for (i, state) in self.attributes.states.iter().enumerate() {
                    node_trajectories.nodes[i].push_state(state);
                    node_trajectories.nodes[i].push_vectors(&self.attributes.vectors[i]);
                }
            }
        }

        if record_trajectories {
            self.node_trajectories = node_trajectories;
        }
    }

    pub fn predict(
        &self,
        x: &[Vec<f64>],
        inputs_x_idxs: &[usize],
        inputs_y_idxs: &[usize],
    ) -> Vec<Vec<f64>> {
        let time_step = 1.0;

        let prediction_steps: Vec<(usize, UpdateStep)> = self
            .update_sequence
            .predictions
            .iter()
            .filter(|(idx, _)| !inputs_x_idxs.contains(idx))
            .cloned()
            .collect();

        // Clone unchanging parts once (edges, vectors, fn_ptrs don't change
        // during prediction-only passes).
        let mut temp = Network {
            attributes: self.attributes.clone(),
            edges: self.edges.clone(),
            inputs: Vec::new(),
            volatility_updates: String::new(),
            mean_field_updates: false,
            update_sequence: UpdateSequence {
                predictions: Vec::new(),
                updates: Vec::new(),
            },
            node_trajectories: NodeTrajectories { nodes: Vec::new() },
            layers: Vec::new(),
            adam_state: None,
            roots: Vec::new(),
            leafs: Vec::new(),
            max_posterior_precision: self.max_posterior_precision,
        };

        x.iter()
            .map(|x_row| {
                // Only reset states per sample (plain f64 fields, no allocation).
                temp.attributes.states.clone_from(&self.attributes.states);

                for (i, &node_idx) in inputs_x_idxs.iter().enumerate() {
                    set_predictors(&mut temp, node_idx, x_row[i]);
                }

                for &(idx, step) in &prediction_steps {
                    step.call(&mut temp, idx, time_step);
                }

                inputs_y_idxs
                    .iter()
                    .map(|&idx| temp.attributes.states[idx].expected_mean)
                    .collect()
            })
            .collect()
    }

    pub fn weight_initialisation(
        &mut self,
        strategy: &str,
        seed: Option<u64>,
    ) -> Result<(), String> {
        if self.layers.len() < 2 {
            return Err(format!(
                "weight_initialisation requires at least 2 tracked layers. \
                 The network currently has {} layer(s).",
                self.layers.len()
            ));
        }

        // Collect the children of the first tracked layer that are NOT in any
        // tracked layer themselves (e.g. output nodes created via add_nodes).
        // These form an implicit "layer -1" whose weights also need initialisation.
        {
            let first_layer = &self.layers[0];
            let all_layer_nodes: std::collections::HashSet<usize> = self
                .layers
                .iter()
                .flat_map(|l| l.iter().copied())
                .collect();

            let mut pre_layer: Vec<usize> = Vec::new();
            for &node_idx in first_layer {
                if let Some(ref vc) = self.edges[node_idx].value_children {
                    for &child_idx in vc {
                        if !all_layer_nodes.contains(&child_idx)
                            && !pre_layer.contains(&child_idx)
                        {
                            let nt = &self.edges[child_idx].node_type;
                            if nt == "continuous-state"
                                || nt == "volatile-state"
                                || nt == "binary-state"
                                || nt == "constant-state"
                            {
                                pre_layer.push(child_idx);
                            }
                        }
                    }
                }
            }

            // Binary-state children always use 1.0 weights — skip initialisation.
            let pre_layer_has_binary = pre_layer
                .iter()
                .any(|&idx| self.edges[idx].node_type == "binary-state");

            if !pre_layer.is_empty() && !pre_layer_has_binary {
                let parent_nodes = first_layer.clone();
                let n_parents = parent_nodes.len();
                let n_children = pre_layer.len();

                if let Ok(weights) = weight_init_by_name(strategy, n_parents, n_children, seed) {
                    for (p_local, &parent_idx) in parent_nodes.iter().enumerate() {
                        for (c_local, &child_idx) in pre_layer.iter().enumerate() {
                            let w = weights[p_local * n_children + c_local];
                            crate::utils::set_coupling::set_coupling(
                                self, parent_idx, child_idx, w,
                            );
                        }
                    }
                }
            }
        }

        for layer_idx in 0..self.layers.len() - 1 {
            let current_nodes = self.layers[layer_idx].clone();
            let parent_nodes = self.layers[layer_idx + 1].clone();

            let all_eligible = current_nodes.iter().chain(parent_nodes.iter()).all(|&idx| {
                let nt = &self.edges[idx].node_type;
                nt == "continuous-state" || nt == "volatile-state" || nt == "constant-state"
            });
            if !all_eligible {
                continue;
            }

            let n_parents = parent_nodes.len();
            let n_current = current_nodes.len();

            let weights = weight_init_by_name(strategy, n_parents, n_current, seed)?;

            for (p_local, &parent_idx) in parent_nodes.iter().enumerate() {
                for (c_local, &child_idx) in current_nodes.iter().enumerate() {
                    let w = weights[p_local * n_current + c_local];
                    crate::utils::set_coupling::set_coupling(
                        self, parent_idx, child_idx, w,
                    );
                }
            }
        }
        Ok(())
    }
}

/// Apply parameter overrides for continuous-state nodes
fn apply_overrides_continuous(state: &mut NodeState, overrides: &HashMap<String, f64>) {
    for (key, &value) in overrides {
        match key.as_str() {
            "mean" => state.mean = value,
            "expected_mean" => state.expected_mean = value,
            "precision" => state.precision = value,
            "expected_precision" => state.expected_precision = value,
            "tonic_volatility" => state.tonic_volatility = value,
            "tonic_drift" => state.tonic_drift = value,
            "autoconnection_strength" => state.autoconnection_strength = value,
            "current_variance" => state.current_variance = value,
            _ => {}
        }
    }
}

/// Apply parameter overrides for volatile-state nodes
fn apply_overrides_volatile(state: &mut NodeState, overrides: &HashMap<String, f64>) {
    apply_overrides_continuous(state, overrides);
    for (key, &value) in overrides {
        match key.as_str() {
            "mean_vol" => state.mean_vol = value,
            "expected_mean_vol" => state.expected_mean_vol = value,
            "precision_vol" => state.precision_vol = value,
            "expected_precision_vol" => state.expected_precision_vol = value,
            "tonic_volatility_vol" => state.tonic_volatility_vol = value,
            "tonic_drift_vol" => state.tonic_drift_vol = value,
            "autoconnection_strength_vol" => state.autoconnection_strength_vol = value,
            "volatility_coupling_internal" => state.volatility_coupling_internal = value,
            _ => {}
        }
    }
}

// Python interface
#[pymethods]
impl Network {

    #[new]
    #[pyo3(signature = (volatility_updates="unbounded", max_posterior_precision=1e10, mean_field_updates=false))]
    fn py_new(volatility_updates: &str, max_posterior_precision: f64, mean_field_updates: bool) -> Self {
        let mut net = Network::new(volatility_updates);
        net.max_posterior_precision = max_posterior_precision;
        net.mean_field_updates = mean_field_updates;
        net
    }

    #[getter]
    fn get_max_posterior_precision(&self) -> f64 {
        self.max_posterior_precision
    }

    #[setter]
    fn set_max_posterior_precision(&mut self, value: f64) {
        self.max_posterior_precision = value;
    }

    #[pyo3(name = "add_nodes", signature = (kind="continuous-state", n_nodes=1, value_parents=None, value_children=None, volatility_parents=None, volatility_children=None, coupling_fn=None, **kwargs))]
    fn py_add_nodes<'py>(
        mut slf: PyRefMut<'py, Self>,
        kind: &str,
        n_nodes: usize,
        value_parents: Option<IntOrList>,
        value_children: Option<IntOrList>,
        volatility_parents: Option<IntOrList>,
        volatility_children: Option<IntOrList>,
        coupling_fn: Option<String>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let additional_parameters = match kwargs {
            Some(dict) => {
                let mut map = HashMap::new();
                for (key, value) in dict.iter() {
                    let key_str: String = key.extract()?;
                    if let Ok(val) = value.extract::<f64>() {
                        map.insert(key_str, val);
                    }
                }
                if map.is_empty() { None } else { Some(map) }
            }
            None => None,
        };
        slf.add_nodes(kind, n_nodes, value_parents, value_children, volatility_parents, volatility_children, coupling_fn, additional_parameters);
        Ok(slf)
    }

    #[pyo3(name = "set_update_sequence")]
    fn py_set_update_sequence<'py>(mut slf: PyRefMut<'py, Self>) -> PyResult<PyRefMut<'py, Self>> {
        slf.set_update_sequence();
        Ok(slf)
    }

    #[pyo3(name = "input_data", signature = (input_data, time_steps=None, record_trajectories=true))]
    fn py_input_data<'py>(
        mut slf: PyRefMut<'py, Self>,
        input_data: Bound<'py, PyAny>,
        time_steps: Option<Bound<'py, PyAny>>,
        record_trajectories: bool,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // Accept both 1D (Vec<f64>) and 2D (Vec<Vec<f64>>) input
        let data: Vec<Vec<f64>> = if let Ok(flat) = input_data.extract::<Vec<f64>>() {
            flat.into_iter().map(|v| vec![v]).collect()
        } else {
            input_data.extract::<Vec<Vec<f64>>>()?
        };
        let ts: Option<Vec<f64>> = match time_steps {
            Some(ref obj) => Some(obj.extract()?),
            None => None,
        };
        slf.input_data(data, ts, record_trajectories);
        Ok(slf)
    }

    #[getter]
    pub fn get_node_trajectories<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        let py_list = PyList::empty(py);

        for (i, traj) in self.node_trajectories.nodes.iter().enumerate() {
            let py_dict = PyDict::new(py);
            let node_type = &self.edges[i].node_type;
            let fields = trajectory_fields_for_type(node_type);

            for &field in fields {
                let data = trajectory_field_ref(traj, field);
                if !data.is_empty() {
                    py_dict.set_item(field, PyArray1::from_vec(py, data.clone()).to_owned())?;
                }
            }

            // Vector trajectories
            if !traj.xis.is_empty() {
                py_dict.set_item("xis", PyArray::from_vec2(py, &traj.xis).unwrap())?;
            }
            if !traj.value_coupling_parents.is_empty() {
                py_dict.set_item("value_coupling_parents", PyArray::from_vec2(py, &traj.value_coupling_parents).unwrap())?;
            }
            if !traj.value_coupling_children.is_empty() {
                py_dict.set_item("value_coupling_children", PyArray::from_vec2(py, &traj.value_coupling_children).unwrap())?;
            }
            if !traj.volatility_coupling_parents.is_empty() {
                py_dict.set_item("volatility_coupling_parents", PyArray::from_vec2(py, &traj.volatility_coupling_parents).unwrap())?;
            }
            if !traj.volatility_coupling_children.is_empty() {
                py_dict.set_item("volatility_coupling_children", PyArray::from_vec2(py, &traj.volatility_coupling_children).unwrap())?;
            }

            py_list.append(py_dict)?;
        }

        Ok(py_list.into())
    }

    #[getter]
    pub fn get_inputs<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        Ok(PyList::new(py, &self.inputs)?.into())
    }

    #[getter]
    pub fn get_edges<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        let py_list = PyList::empty(py);
        for edge in &self.edges {
            let py_dict = PyDict::new(py);
            py_dict.set_item("value_parents", &edge.value_parents)?;
            py_dict.set_item("value_children", &edge.value_children)?;
            py_dict.set_item("volatility_parents", &edge.volatility_parents)?;
            py_dict.set_item("volatility_children", &edge.volatility_children)?;
            py_list.append(py_dict)?;
        }
        Ok(py_list.into())
    }

    #[getter]
    pub fn get_update_sequence<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        let py_list = PyList::empty(py);

        for sequence in [&self.update_sequence.predictions, &self.update_sequence.updates] {
            for &(num, step) in sequence {
                let py_func_name = step.name()
                    .into_pyobject(py)?.into_any().unbind();
                let py_num = num.into_pyobject(py)?.into_any().unbind();
                py_list.append(PyTuple::new(py, &[py_num, py_func_name])?)?;
            }
        }

        Ok(py_list.into())
    }

    #[pyo3(name = "add_layer", signature = (size=1, kind="volatile-state", value_children=None, coupling_strengths=1.0, coupling_fn=None, add_constant_input=true, **kwargs))]
    fn py_add_layer<'py>(
        mut slf: PyRefMut<'py, Self>,
        size: usize,
        kind: &str,
        value_children: Option<Vec<usize>>,
        coupling_strengths: f64,
        coupling_fn: Option<String>,
        add_constant_input: bool,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let additional_parameters = match kwargs {
            Some(dict) => {
                let mut map = HashMap::new();
                for (key, value) in dict.iter() {
                    let key_str: String = key.extract()?;
                    if let Ok(val) = value.extract::<f64>() {
                        map.insert(key_str, val);
                    }
                }
                if map.is_empty() { None } else { Some(map) }
            }
            None => None,
        };
        slf.add_layer(size, kind, value_children, coupling_strengths, coupling_fn, additional_parameters, add_constant_input);
        Ok(slf)
    }

    #[pyo3(name = "add_layer_stack", signature = (layer_sizes, kind="volatile-state", value_children=None, coupling_strengths=1.0, coupling_fn=None, add_constant_input=true, **kwargs))]
    fn py_add_layer_stack<'py>(
        mut slf: PyRefMut<'py, Self>,
        layer_sizes: Vec<usize>,
        kind: &str,
        value_children: Option<Vec<usize>>,
        coupling_strengths: f64,
        coupling_fn: Option<String>,
        add_constant_input: bool,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let additional_parameters = match kwargs {
            Some(dict) => {
                let mut map = HashMap::new();
                for (key, value) in dict.iter() {
                    let key_str: String = key.extract()?;
                    if let Ok(val) = value.extract::<f64>() {
                        map.insert(key_str, val);
                    }
                }
                if map.is_empty() { None } else { Some(map) }
            }
            None => None,
        };
        slf.add_layer_stack(layer_sizes, kind, value_children, coupling_strengths, coupling_fn, additional_parameters, add_constant_input);
        Ok(slf)
    }

    #[pyo3(name = "fit", signature = (x, y, inputs_x_idxs=None, inputs_y_idxs=None, lr=None, record_trajectories=true, params=None, learning_kind="precision_weighted"))]
    fn py_fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: Bound<'py, PyAny>,
        y: Bound<'py, PyAny>,
        inputs_x_idxs: Option<Vec<usize>>,
        inputs_y_idxs: Option<Vec<usize>>,
        lr: Option<Bound<'py, PyAny>>,
        record_trajectories: bool,
        params: Option<&Bound<'py, PyDict>>,
        learning_kind: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // lr can be a non-negative float (fixed step size) or the string "adam"
        // (triggers the Adam optimiser).  When omitted, defaults to 0.2.
        // On the Rust side, None signals Adam, Some(f) is the fixed step size.
        let lr_option: Option<f64> = match lr {
            Some(ref obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    if s == "adam" {
                        None
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            format!("Invalid lr string '{}'. Expected a non-negative float or 'adam'.", s),
                        ));
                    }
                } else {
                    Some(obj.extract::<f64>()?)
                }
            }
            None => Some(0.2),
        };

        let x_idxs = match inputs_x_idxs {
            Some(v) => v,
            None => slf.leafs.clone(),
        };
        let y_idxs = match inputs_y_idxs {
            Some(v) => v,
            None => slf.roots.clone(),
        };

        let x_data: Vec<Vec<f64>> = if let Ok(outer) = x.extract::<Vec<Vec<f64>>>() {
            outer
        } else {
            let flat: Vec<f64> = x.extract()?;
            flat.into_iter().map(|v| vec![v]).collect()
        };

        let y_data: Vec<Vec<f64>> = if let Ok(outer) = y.extract::<Vec<Vec<f64>>>() {
            outer
        } else {
            let flat: Vec<f64> = y.extract()?;
            flat.into_iter().map(|v| vec![v]).collect()
        };

        let params_map: Option<HashMap<String, f64>> = match params {
            Some(dict) => {
                let mut map = HashMap::new();
                for (key, value) in dict.iter() {
                    let key_str: String = key.extract()?;
                    if let Ok(val) = value.extract::<f64>() {
                        map.insert(key_str, val);
                    }
                }
                if map.is_empty() { None } else { Some(map) }
            }
            None => None,
        };

        slf.fit(&x_data, &y_data, &x_idxs, &y_idxs, lr_option, record_trajectories, params_map.as_ref(), learning_kind);
        Ok(slf)
    }

    #[pyo3(name = "predict", signature = (x, inputs_x_idxs=None, inputs_y_idxs=None))]
    fn py_predict<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: Bound<'py, PyAny>,
        inputs_x_idxs: Option<Vec<usize>>,
        inputs_y_idxs: Option<Vec<usize>>,
    ) -> PyResult<Py<numpy::PyArray2<f64>>> {
        if slf.update_sequence.predictions.is_empty()
            && slf.update_sequence.updates.is_empty()
        {
            slf.set_update_sequence();
        }

        let x_idxs = match inputs_x_idxs {
            Some(v) => v,
            None => slf.leafs.clone(),
        };
        let y_idxs = match inputs_y_idxs {
            Some(v) => v,
            None => slf.roots.clone(),
        };

        let x_data: Vec<Vec<f64>> = if let Ok(outer) = x.extract::<Vec<Vec<f64>>>() {
            outer
        } else {
            let flat: Vec<f64> = x.extract()?;
            flat.into_iter().map(|v| vec![v]).collect()
        };

        let predictions = slf.predict(&x_data, &x_idxs, &y_idxs);

        let n_samples = predictions.len();
        let n_outputs = if n_samples > 0 { predictions[0].len() } else { 0 };
        let flat: Vec<f64> = predictions.into_iter().flatten().collect();
        let array = numpy::PyArray1::from_vec(py, flat)
            .reshape([n_samples, n_outputs])?;
        Ok(array.into())
    }

    #[pyo3(name = "weight_initialisation", signature = (strategy, seed=None))]
    fn py_weight_initialisation<'py>(
        mut slf: PyRefMut<'py, Self>,
        strategy: &str,
        seed: Option<u64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.weight_initialisation(strategy, seed)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(slf)
    }

    #[getter]
    pub fn get_layers<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        let py_list = PyList::empty(py);
        for layer in &self.layers {
            py_list.append(PyList::new(py, layer)?)?;
        }
        Ok(py_list.into())
    }

    #[getter]
    pub fn get_roots<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        Ok(PyList::new(py, &self.roots)?.into())
    }

    #[getter]
    pub fn get_leafs<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        Ok(PyList::new(py, &self.leafs)?.into())
    }
}

// Create a module to expose the class to Python
#[pymodule]
fn rshgf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Network>()?;
    Ok(())
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_family_gaussian() {
        let mut network = Network::new("eHGF");
        network.add_nodes("ef-state", 1, None, None, None, None, None, None);

        let input_data: Vec<Vec<f64>> = vec![vec![1.0], vec![1.3], vec![1.5], vec![1.7]];
        network.set_update_sequence();
        network.input_data(input_data, None, true);
    }

    #[test]
    fn test_volatile_node_ehgf_matches_explicit() {
        let mut volatile_net = Network::new("eHGF");
        volatile_net.add_nodes("continuous-state", 1, None, None, None, None, None, None);
        volatile_net.add_nodes("volatile-state", 1, None, Some(0.into()), None, None, None,
            Some(HashMap::from([("autoconnection_strength".into(), 1.0)])));
        volatile_net.set_update_sequence();

        let input_data: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64) * 0.1]).collect();
        volatile_net.input_data(input_data.clone(), None, true);

        let mut explicit_net = Network::new("eHGF");
        explicit_net.add_nodes("continuous-state", 1, None, None, None, None, None, None);
        explicit_net.add_nodes("continuous-state", 1, None, Some(0.into()), None, None, None, None);
        explicit_net.add_nodes("continuous-state", 1, None, None, None, Some(1.into()), None, None);
        explicit_net.set_update_sequence();
        explicit_net.input_data(input_data, None, true);

        assert_volatile_matches_explicit(&volatile_net, &explicit_net);
    }

    #[test]
    fn test_volatile_node_standard_matches_explicit() {
        let mut volatile_net = Network::new("standard");
        volatile_net.add_nodes("continuous-state", 1, None, None, None, None, None, None);
        volatile_net.add_nodes("volatile-state", 1, None, Some(0.into()), None, None, None,
            Some(HashMap::from([("autoconnection_strength".into(), 1.0)])));
        volatile_net.set_update_sequence();

        let input_data: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64) * 0.1]).collect();
        volatile_net.input_data(input_data.clone(), None, true);

        let mut explicit_net = Network::new("standard");
        explicit_net.add_nodes("continuous-state", 1, None, None, None, None, None, None);
        explicit_net.add_nodes("continuous-state", 1, None, Some(0.into()), None, None, None, None);
        explicit_net.add_nodes("continuous-state", 1, None, None, None, Some(1.into()), None, None);
        explicit_net.set_update_sequence();
        explicit_net.input_data(input_data, None, true);

        assert_volatile_matches_explicit(&volatile_net, &explicit_net);
    }

    #[test]
    fn test_volatile_node_unbounded_matches_explicit() {
        let mut volatile_net = Network::new("unbounded");
        volatile_net.add_nodes("continuous-state", 1, None, None, None, None, None, None);
        volatile_net.add_nodes("volatile-state", 1, None, Some(0.into()), None, None, None,
            Some(HashMap::from([("autoconnection_strength".into(), 1.0)])));
        volatile_net.set_update_sequence();

        let input_data: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64) * 0.1]).collect();
        volatile_net.input_data(input_data.clone(), None, true);

        let mut explicit_net = Network::new("unbounded");
        explicit_net.add_nodes("continuous-state", 1, None, None, None, None, None, None);
        explicit_net.add_nodes("continuous-state", 1, None, Some(0.into()), None, None, None, None);
        explicit_net.add_nodes("continuous-state", 1, None, None, None, Some(1.into()), None, None);
        explicit_net.set_update_sequence();
        explicit_net.input_data(input_data, None, true);

        assert_volatile_matches_explicit(&volatile_net, &explicit_net);
    }

    /// Helper: assert volatile node 1 trajectories match explicit nodes 1 & 2
    fn assert_volatile_matches_explicit(volatile_net: &Network, explicit_net: &Network) {
        let vol_traj = &volatile_net.node_trajectories.nodes[1];
        let exp_traj = &explicit_net.node_trajectories.nodes[1];

        // Value-level keys
        let vol_fields: Vec<(&Vec<f64>, &str)> = vec![
            (&vol_traj.mean, "mean"),
            (&vol_traj.expected_mean, "expected_mean"),
            (&vol_traj.precision, "precision"),
            (&vol_traj.expected_precision, "expected_precision"),
        ];
        let exp_fields: Vec<(&Vec<f64>, &str)> = vec![
            (&exp_traj.mean, "mean"),
            (&exp_traj.expected_mean, "expected_mean"),
            (&exp_traj.precision, "precision"),
            (&exp_traj.expected_precision, "expected_precision"),
        ];

        for ((vol, key), (exp, _)) in vol_fields.iter().zip(exp_fields.iter()) {
            for (t, (v, e)) in vol.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (v - e).abs() < 1e-6,
                    "Value-level key '{}' mismatch at t={}: volatile={}, explicit={}",
                    key, t, v, e
                );
            }
        }

        let exp2_traj = &explicit_net.node_trajectories.nodes[2];
        let vol_key_map: Vec<(&Vec<f64>, &Vec<f64>, &str, &str)> = vec![
            (&vol_traj.mean_vol, &exp2_traj.mean, "mean_vol", "mean"),
            (&vol_traj.expected_mean_vol, &exp2_traj.expected_mean, "expected_mean_vol", "expected_mean"),
            (&vol_traj.precision_vol, &exp2_traj.precision, "precision_vol", "precision"),
            (&vol_traj.expected_precision_vol, &exp2_traj.expected_precision, "expected_precision_vol", "expected_precision"),
        ];

        for (vol, exp, vol_key, exp_key) in vol_key_map {
            for (t, (v, e)) in vol.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (v - e).abs() < 1e-6,
                    "Vol-level key '{}' vs '{}' mismatch at t={}: volatile={}, explicit={}",
                    vol_key, exp_key, t, v, e
                );
            }
        }
    }
}
