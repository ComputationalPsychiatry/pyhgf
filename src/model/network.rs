use crate::utils::beliefs_propagation::belief_propagation;
use crate::utils::function_pointer::UpdateStep;
use crate::utils::set_sequence::set_update_sequence;
use numpy::{PyArray, PyArray1};
use pyo3::types::PyTuple;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use std::collections::HashMap;

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
    fn from(v: Vec<usize>) -> Self {
        IntOrList::List(v)
    }
}

impl From<usize> for IntOrList {
    fn from(v: usize) -> Self {
        IntOrList::Single(v)
    }
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
pub struct AdjacencyLists {
    #[pyo3(get, set)]
    pub node_type: String,
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
/// Unused fields default to 0.0.
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
    // EF-state
    pub nus: f64,
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
            nus: 0.0,
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
    pub nus: Vec<f64>,
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
            nus: Vec::with_capacity(n),
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
        self.volatility_prediction_error
            .push(s.volatility_prediction_error);
        self.nus.push(s.nus);
    }

    pub fn push_vectors(&mut self, v: &NodeVectors) {
        if !v.xis.is_empty() {
            self.xis.push(v.xis.clone());
        }
        if !v.value_coupling_parents.is_empty() {
            self.value_coupling_parents
                .push(v.value_coupling_parents.clone());
        }
        if !v.value_coupling_children.is_empty() {
            self.value_coupling_children
                .push(v.value_coupling_children.clone());
        }
        if !v.volatility_coupling_parents.is_empty() {
            self.volatility_coupling_parents
                .push(v.volatility_coupling_parents.clone());
        }
        if !v.volatility_coupling_children.is_empty() {
            self.volatility_coupling_children
                .push(v.volatility_coupling_children.clone());
        }
    }
}

#[derive(Debug)]
pub struct NodeTrajectories {
    pub nodes: Vec<NodeTrajectory>,
}

#[derive(Debug)]
#[pyclass]
pub struct Network {
    pub attributes: Attributes,
    pub edges: Vec<AdjacencyLists>,
    pub inputs: Vec<usize>,
    pub volatility_updates: String,
    pub mean_field_updates: bool,
    pub update_sequence: UpdateSequence,
    pub node_trajectories: NodeTrajectories,
    /// Upper bound applied to every posterior precision write. Defaults to
    /// ``1e10`` and is shared with the JAX backends.
    pub max_posterior_precision: f64,
    /// Bound applied to binary predicted means (`[v, 1 - v]`) so the implied binary
    /// precision never collapses. A larger value (e.g. 1e-3, matching TAPAS) stabilises
    /// the forward filter in high-volatility regimes; a very small value (default 1e-6)
    /// avoids flat, zero-gradient plateaus that hurt gradient-based inference. Shared
    /// with the JAX backends.
    pub precision_clipping_value: f64,
}

/// Helper: get the list of trajectory field names to export for a given node type.
fn trajectory_fields_for_type(node_type: &str) -> &'static [&'static str] {
    match node_type {
        "binary-state" => &[
            "observed",
            "mean",
            "expected_mean",
            "precision",
            "expected_precision",
            "value_prediction_error",
        ],
        "continuous-state" => &[
            "mean",
            "expected_mean",
            "precision",
            "expected_precision",
            "tonic_volatility",
            "tonic_drift",
            "autoconnection_strength",
            "current_variance",
            "effective_precision",
            "value_prediction_error",
            "volatility_prediction_error",
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
        "nus" => &traj.nus,
        _ => &traj.mean, // fallback
    }
}

// Core Rust methods (also callable from Python via chaining wrappers below)
impl Network {
    pub fn new(volatility_updates: &str) -> Self {
        Network {
            attributes: Attributes {
                states: Vec::new(),
                vectors: Vec::new(),
                fn_ptrs: Vec::new(),
            },
            edges: Vec::new(),
            inputs: Vec::new(),
            volatility_updates: String::from(volatility_updates),
            mean_field_updates: false,
            update_sequence: UpdateSequence {
                predictions: Vec::new(),
                updates: Vec::new(),
            },
            node_trajectories: NodeTrajectories { nodes: Vec::new() },
            max_posterior_precision: 1e10,
            precision_clipping_value: 1e-6,
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
        let coupling_fn_opt: Option<&'static crate::math::CouplingFn> =
            match coupling_fn.as_deref().unwrap_or("linear") {
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

            let is_input = !has_children;
            if is_input {
                self.inputs.push(node_id);
            }

            let edges = AdjacencyLists {
                node_type: String::from(kind),
                value_parents: value_parents.clone(),
                value_children: value_children.clone(),
                volatility_parents: volatility_parents.clone(),
                volatility_children: volatility_children.clone(),
            };

            match kind {
                "continuous-state" => {
                    let (autoconnection, tonic_vol) =
                        if is_input { (0.0, 0.0) } else { (1.0, -4.0) };

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
                    let fns = NodeFnPtrs {
                        coupling_fn: coupling_fn_opt,
                    };

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
                                self.attributes.vectors[child_idx]
                                    .value_coupling_parents
                                    .push(1.0);
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
                                self.attributes.vectors[child_idx]
                                    .volatility_coupling_parents
                                    .push(1.0);
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
                                self.attributes.vectors[child_idx]
                                    .value_coupling_parents
                                    .push(1.0);
                            }
                        }
                    }

                    self.attributes.vectors.push(vecs);
                    self.attributes.fn_ptrs.push(NodeFnPtrs {
                        coupling_fn: coupling_fn_opt,
                    });
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
                    // parent-precision vector for the constant column.
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
                                self.attributes.vectors[child_idx]
                                    .value_coupling_parents
                                    .push(1.0);
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
                                self.attributes.vectors[child_idx]
                                    .volatility_coupling_parents
                                    .push(1.0);
                            }
                        }
                    }

                    self.attributes.vectors.push(vecs);
                    // Force constant-state nodes to use no coupling (identity)
                    // regardless of what the caller passed.
                    self.attributes
                        .fn_ptrs
                        .push(NodeFnPtrs { coupling_fn: None });
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
                    if parent_idx >= self.edges.len() {
                        continue;
                    }
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
                        .value_children
                        .as_ref()
                        .map(|c| c.len())
                        .unwrap_or(0);
                    let parent_coupling_len = self.attributes.vectors[parent_idx]
                        .value_coupling_children
                        .len();
                    if parent_coupling_len < parent_n_children {
                        self.attributes.vectors[parent_idx]
                            .value_coupling_children
                            .push(1.0);
                    }
                }
            }
            if let Some(ref volp) = volp_clone {
                for &parent_idx in volp {
                    if parent_idx >= self.edges.len() {
                        continue;
                    }
                    match &mut self.edges[parent_idx].volatility_children {
                        Some(children) => {
                            if !children.contains(&node_id) {
                                children.push(node_id);
                            }
                        }
                        None => self.edges[parent_idx].volatility_children = Some(vec![node_id]),
                    }
                    let parent_n_children = self.edges[parent_idx]
                        .volatility_children
                        .as_ref()
                        .map(|c| c.len())
                        .unwrap_or(0);
                    let parent_coupling_len = self.attributes.vectors[parent_idx]
                        .volatility_coupling_children
                        .len();
                    if parent_coupling_len < parent_n_children {
                        self.attributes.vectors[parent_idx]
                            .volatility_coupling_children
                            .push(1.0);
                    }
                }
            }
        } // end for n_nodes
    }

    pub fn set_update_sequence(&mut self) {
        self.update_sequence = set_update_sequence(self);
    }

    pub fn input_data(
        &mut self,
        input_data: Vec<Vec<f64>>,
        time_steps: Option<Vec<f64>>,
        record_trajectories: bool,
    ) {
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
                node_trajectories
                    .nodes
                    .push(NodeTrajectory::with_capacity(n_time));
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

// Python interface
#[pymethods]
impl Network {
    #[new]
    #[pyo3(signature = (volatility_updates="unbounded", max_posterior_precision=1e10, mean_field_updates=false, precision_clipping_value=1e-6))]
    fn py_new(
        volatility_updates: &str,
        max_posterior_precision: f64,
        mean_field_updates: bool,
        precision_clipping_value: f64,
    ) -> Self {
        let mut net = Network::new(volatility_updates);
        net.max_posterior_precision = max_posterior_precision;
        net.mean_field_updates = mean_field_updates;
        net.precision_clipping_value = precision_clipping_value;
        net
    }

    #[getter]
    fn get_max_posterior_precision(&self) -> f64 {
        self.max_posterior_precision
    }

    #[getter]
    fn get_precision_clipping_value(&self) -> f64 {
        self.precision_clipping_value
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
                if map.is_empty() {
                    None
                } else {
                    Some(map)
                }
            }
            None => None,
        };
        slf.add_nodes(
            kind,
            n_nodes,
            value_parents,
            value_children,
            volatility_parents,
            volatility_children,
            coupling_fn,
            additional_parameters,
        );
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
                py_dict.set_item(
                    "value_coupling_parents",
                    PyArray::from_vec2(py, &traj.value_coupling_parents).unwrap(),
                )?;
            }
            if !traj.value_coupling_children.is_empty() {
                py_dict.set_item(
                    "value_coupling_children",
                    PyArray::from_vec2(py, &traj.value_coupling_children).unwrap(),
                )?;
            }
            if !traj.volatility_coupling_parents.is_empty() {
                py_dict.set_item(
                    "volatility_coupling_parents",
                    PyArray::from_vec2(py, &traj.volatility_coupling_parents).unwrap(),
                )?;
            }
            if !traj.volatility_coupling_children.is_empty() {
                py_dict.set_item(
                    "volatility_coupling_children",
                    PyArray::from_vec2(py, &traj.volatility_coupling_children).unwrap(),
                )?;
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

        for sequence in [
            &self.update_sequence.predictions,
            &self.update_sequence.updates,
        ] {
            for &(num, step) in sequence {
                let py_func_name = step.name().into_pyobject(py)?.into_any().unbind();
                let py_num = num.into_pyobject(py)?.into_any().unbind();
                py_list.append(PyTuple::new(py, &[py_num, py_func_name])?)?;
            }
        }

        Ok(py_list.into())
    }
}

// The Python module registration lives in `lib.rs`.

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
}
