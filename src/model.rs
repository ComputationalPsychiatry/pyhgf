use std::collections::HashMap;
use crate::utils::function_pointer::FnType;
use crate::utils::set_sequence::set_update_sequence;
use crate::utils::beliefs_propagation::belief_propagation;
use crate::utils::function_pointer::get_func_map;
use pyo3::types::PyTuple;
use pyo3::{prelude::*, types::{PyList, PyDict}};
use numpy::{PyArray1, PyArray};

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

#[derive(Debug)]
#[pyclass]
pub struct AdjacencyLists{
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
    pub predictions: Vec<(usize, FnType)>,
    pub updates: Vec<(usize, FnType)>,
}

#[derive(Debug)]
pub struct Attributes {
    pub floats: HashMap<usize, HashMap<String, f64>>,
    pub vectors: HashMap<usize, HashMap<String, Vec<f64>>>,
}

#[derive(Debug)]
pub struct NodeTrajectories {
    pub floats: HashMap<usize, HashMap<String, Vec<f64>>>,
    pub vectors: HashMap<usize, HashMap<String, Vec<Vec<f64>>>>,
}

#[derive(Debug)]
#[pyclass]
pub struct Network{
    pub attributes: Attributes,
    pub edges: HashMap<usize, AdjacencyLists>,
    pub inputs: Vec<usize>,
    pub update_type: String,
    pub update_sequence: UpdateSequence,
    pub node_trajectories: NodeTrajectories,
}

// Core Rust methods (also callable from Python via chaining wrappers below)
impl Network {

    pub fn new(update_type: &str) -> Self {
        Network {
            attributes: Attributes { floats: HashMap::new(), vectors: HashMap::new() },
            edges: HashMap::new(),
            inputs: Vec::new(),
            update_type: String::from(update_type),
            update_sequence: UpdateSequence { predictions: Vec::new(), updates: Vec::new() },
            node_trajectories: NodeTrajectories { floats: HashMap::new(), vectors: HashMap::new() },
        }
    }

    /// Add nodes to the network.
    ///
    /// # Arguments
    /// * `kind` - The type of node (`"continuous-state"` or `"ef-state"`).
    /// * `value_parents` - Index(es) of the node's value parents (int or list).
    /// * `value_children` - Index(es) of the node's value children (int or list).
    /// * `volatility_parents` - Index(es) of the node's volatility parents (int or list).
    /// * `volatility_children` - Index(es) of the node's volatility children (int or list).
    pub fn add_nodes(
        &mut self,
        kind: &str,
        value_parents: Option<IntOrList>,
        value_children: Option<IntOrList>,
        volatility_parents: Option<IntOrList>,
        volatility_children: Option<IntOrList>,
    ) {
        let value_parents = value_parents.map(|v| v.into_vec());
        let value_children = value_children.map(|v| v.into_vec());
        let volatility_parents = volatility_parents.map(|v| v.into_vec());
        let volatility_children = volatility_children.map(|v| v.into_vec());

        let node_id = self.edges.len();

        // Input nodes have no children
        let is_input = value_children.is_none() && volatility_children.is_none();
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
                // Input nodes: no autoconnection, no tonic volatility
                let (autoconnection, tonic_vol) = if is_input { (0.0, 0.0) } else { (1.0, -4.0) };

                self.attributes.floats.insert(node_id, HashMap::from([
                    ("mean".into(), 0.0),
                    ("expected_mean".into(), 0.0),
                    ("precision".into(), 1.0),
                    ("expected_precision".into(), 1.0),
                    ("tonic_volatility".into(), tonic_vol),
                    ("tonic_drift".into(), 0.0),
                    ("autoconnection_strength".into(), autoconnection),
                    ("current_variance".into(), 1.0),
                ]));
                self.edges.insert(node_id, edges);

                // Coupling strength vectors and reciprocal edges
                let mut vec_attrs: HashMap<String, Vec<f64>> = HashMap::new();

                if let Some(ref vp) = value_parents {
                    vec_attrs.insert("value_coupling_parents".into(), vec![1.0; vp.len()]);
                }
                if let Some(ref vc) = value_children {
                    vec_attrs.insert("value_coupling_children".into(), vec![1.0; vc.len()]);
                    for &child_idx in vc {
                        if let Some(child_edges) = self.edges.get_mut(&child_idx) {
                            match &mut child_edges.value_parents {
                                Some(parents) => parents.push(node_id),
                                None => child_edges.value_parents = Some(vec![node_id]),
                            }
                        }
                        let child_vecs = self.attributes.vectors.entry(child_idx).or_default();
                        child_vecs.entry("value_coupling_parents".into())
                            .and_modify(|cs| cs.push(1.0))
                            .or_insert_with(|| vec![1.0]);
                    }
                }
                if let Some(ref volp) = volatility_parents {
                    vec_attrs.insert("volatility_coupling_parents".into(), vec![1.0; volp.len()]);
                }
                if let Some(ref volc) = volatility_children {
                    vec_attrs.insert("volatility_coupling_children".into(), vec![1.0; volc.len()]);
                    for &child_idx in volc {
                        if let Some(child_edges) = self.edges.get_mut(&child_idx) {
                            match &mut child_edges.volatility_parents {
                                Some(parents) => parents.push(node_id),
                                None => child_edges.volatility_parents = Some(vec![node_id]),
                            }
                        }
                        let child_vecs = self.attributes.vectors.entry(child_idx).or_default();
                        child_vecs.entry("volatility_coupling_parents".into())
                            .and_modify(|cs| cs.push(1.0))
                            .or_insert_with(|| vec![1.0]);
                    }
                }

                if !vec_attrs.is_empty() {
                    self.attributes.vectors.insert(node_id, vec_attrs);
                }
            }
            "ef-state" => {
                self.attributes.floats.insert(node_id, HashMap::from([
                    ("mean".into(), 0.0),
                    ("nus".into(), 3.0),
                ]));
                self.attributes.vectors.insert(node_id, HashMap::from([
                    ("xis".into(), vec![0.0, 1.0]),
                ]));
                self.edges.insert(node_id, edges);
            }
            "volatile-state" => {
                // Volatile nodes have an implicit internal volatility parent.
                // They do NOT accept external volatility parents/children.
                let volatile_edges = AdjacencyLists {
                    node_type: String::from(kind),
                    value_parents: value_parents.clone(),
                    value_children: value_children.clone(),
                    volatility_parents: None,
                    volatility_children: None,
                };

                self.attributes.floats.insert(node_id, HashMap::from([
                    // Value level parameters (external facing)
                    ("mean".into(), 0.0),
                    ("expected_mean".into(), 0.0),
                    ("precision".into(), 1.0),
                    ("expected_precision".into(), 1.0),
                    ("tonic_volatility".into(), -4.0),
                    ("tonic_drift".into(), 0.0),
                    ("autoconnection_strength".into(), 1.0),
                    ("current_variance".into(), 1.0),
                    // Volatility level parameters (implicit internal)
                    ("mean_vol".into(), 0.0),
                    ("expected_mean_vol".into(), 0.0),
                    ("precision_vol".into(), 1.0),
                    ("expected_precision_vol".into(), 1.0),
                    ("tonic_volatility_vol".into(), -4.0),
                    ("tonic_drift_vol".into(), 0.0),
                    ("autoconnection_strength_vol".into(), 1.0),
                    // Internal coupling
                    ("volatility_coupling_internal".into(), 1.0),
                    // State
                    ("observed".into(), 1.0),
                    // Temp variables
                    ("effective_precision".into(), 0.0),
                    ("value_prediction_error".into(), 0.0),
                    ("volatility_prediction_error".into(), 0.0),
                    ("effective_precision_vol".into(), 0.0),
                ]));
                self.edges.insert(node_id, volatile_edges);

                // Coupling strength vectors and reciprocal edges (value only)
                let mut vec_attrs: HashMap<String, Vec<f64>> = HashMap::new();

                if let Some(ref vp) = value_parents {
                    vec_attrs.insert("value_coupling_parents".into(), vec![1.0; vp.len()]);
                }
                if let Some(ref vc) = value_children {
                    vec_attrs.insert("value_coupling_children".into(), vec![1.0; vc.len()]);
                    for &child_idx in vc {
                        if let Some(child_edges) = self.edges.get_mut(&child_idx) {
                            match &mut child_edges.value_parents {
                                Some(parents) => parents.push(node_id),
                                None => child_edges.value_parents = Some(vec![node_id]),
                            }
                        }
                        let child_vecs = self.attributes.vectors.entry(child_idx).or_default();
                        child_vecs.entry("value_coupling_parents".into())
                            .and_modify(|cs| cs.push(1.0))
                            .or_insert_with(|| vec![1.0]);
                    }
                }

                if !vec_attrs.is_empty() {
                    self.attributes.vectors.insert(node_id, vec_attrs);
                }
            }
            _ => {}
        }
    }

    pub fn set_update_sequence(&mut self) {
        self.update_sequence = set_update_sequence(self);
    }

    /// Add a sequence of observations.
    ///
    /// # Arguments
    /// * `input_data` - A vector of observations (one per time step).
    /// * `time_steps` - Optional time steps (defaults to ones).
    pub fn input_data(&mut self, input_data: Vec<f64>, time_steps: Option<Vec<f64>>) {
        // Automatically set the update sequence if not already done
        if self.update_sequence.predictions.is_empty() && self.update_sequence.updates.is_empty() {
            self.set_update_sequence();
        }

        let n_time = input_data.len();
        let time_steps = time_steps.unwrap_or_else(|| vec![1.0; n_time]);
        let predictions = self.update_sequence.predictions.clone();
        let updates = self.update_sequence.updates.clone();

        let mut node_trajectories = NodeTrajectories {
            floats: HashMap::new(),
            vectors: HashMap::new(),
        };

        // Preallocate float trajectories
        for (node_idx, node) in &self.attributes.floats {
            let mut map = HashMap::with_capacity(node.len());
            for key in node.keys() {
                map.insert(key.clone(), Vec::with_capacity(n_time));
            }
            node_trajectories.floats.insert(*node_idx, map);
        }

        // Preallocate vector trajectories
        for (node_idx, node) in &self.attributes.vectors {
            let mut map = HashMap::with_capacity(node.len());
            for key in node.keys() {
                map.insert(key.clone(), Vec::with_capacity(n_time));
            }
            node_trajectories.vectors.insert(*node_idx, map);
        }

        // Iterate over observations
        for (t, observation) in input_data.iter().enumerate() {
            belief_propagation(self, vec![*observation], &predictions, &updates, time_steps[t]);

            // Record float trajectories
            for (node_idx, node) in &self.attributes.floats {
                let traj = node_trajectories.floats.get_mut(node_idx).expect("node not found");
                for (key, value) in node {
                    traj.entry(key.clone())
                        .or_insert_with(|| Vec::with_capacity(n_time))
                        .push(*value);
                }
            }

            // Record vector trajectories
            for (node_idx, node) in &self.attributes.vectors {
                let traj = node_trajectories.vectors.entry(*node_idx).or_default();
                for (key, value) in node {
                    traj.entry(key.clone())
                        .or_insert_with(|| Vec::with_capacity(n_time))
                        .push(value.clone());
                }
            }
        }

        self.node_trajectories = node_trajectories;
    }
}

// Python interface — wrappers that return self for method chaining
#[pymethods]
impl Network {

    #[new]
    #[pyo3(signature = (update_type="eHGF"))]
    fn py_new(update_type: &str) -> Self {
        Network::new(update_type)
    }

    #[pyo3(name = "add_nodes", signature = (kind="continuous-state", value_parents=None, value_children=None, volatility_parents=None, volatility_children=None))]
    fn py_add_nodes<'py>(
        mut slf: PyRefMut<'py, Self>,
        kind: &str,
        value_parents: Option<IntOrList>,
        value_children: Option<IntOrList>,
        volatility_parents: Option<IntOrList>,
        volatility_children: Option<IntOrList>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.add_nodes(kind, value_parents, value_children, volatility_parents, volatility_children);
        Ok(slf)
    }

    #[pyo3(name = "set_update_sequence")]
    fn py_set_update_sequence<'py>(mut slf: PyRefMut<'py, Self>) -> PyResult<PyRefMut<'py, Self>> {
        slf.set_update_sequence();
        Ok(slf)
    }

    #[pyo3(name = "input_data", signature = (input_data, time_steps=None))]
    fn py_input_data<'py>(
        mut slf: PyRefMut<'py, Self>,
        input_data: Bound<'py, PyAny>,
        time_steps: Option<Bound<'py, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // Accept both plain lists and numpy arrays
        let data: Vec<f64> = input_data.extract()?;
        let ts: Option<Vec<f64>> = match time_steps {
            Some(ref obj) => Some(obj.extract()?),
            None => None,
        };
        slf.input_data(data, ts);
        Ok(slf)
    }

    // ---- Python getters --------------------------------------------------------

    #[getter]
    pub fn get_node_trajectories<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        let py_list = PyList::empty(py);

        let mut sorted_keys: Vec<&usize> = self.node_trajectories.floats.keys().collect();
        sorted_keys.sort();

        for node_idx in sorted_keys {
            let py_dict = PyDict::new(py);

            for (key, value) in &self.node_trajectories.floats[node_idx] {
                py_dict.set_item(key, PyArray1::from_vec(py, value.clone()).to_owned())?;
            }
            if let Some(vector_node) = self.node_trajectories.vectors.get(node_idx) {
                for (key, value) in vector_node {
                    py_dict.set_item(key, PyArray::from_vec2(py, value).unwrap())?;
                }
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
        for i in 0..self.edges.len() {
            let py_dict = PyDict::new(py);
            py_dict.set_item("value_parents", &self.edges[&i].value_parents)?;
            py_dict.set_item("value_children", &self.edges[&i].value_children)?;
            py_dict.set_item("volatility_parents", &self.edges[&i].volatility_parents)?;
            py_dict.set_item("volatility_children", &self.edges[&i].volatility_children)?;
            py_list.append(py_dict)?;
        }
        Ok(py_list.into())
    }

    #[getter]
    pub fn get_update_sequence<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        let func_map = get_func_map();
        let py_list = PyList::empty(py);

        for sequence in [&self.update_sequence.predictions, &self.update_sequence.updates] {
            for &(num, func) in sequence {
                let py_func_name = func_map.get(&func).unwrap_or(&"unknown")
                    .into_pyobject(py)?.into_any().unbind();
                let py_num = num.into_pyobject(py)?.into_any().unbind();
                py_list.append(PyTuple::new(py, &[py_num, py_func_name])?)?;
            }
        }

        Ok(py_list.into())
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
        network.add_nodes("ef-state", None, None, None, None);

        let input_data = vec![1.0, 1.3, 1.5, 1.7];
        network.set_update_sequence();
        network.input_data(input_data, None);
    }

    #[test]
    fn test_volatile_node_ehgf_matches_explicit() {
        // Both networks use eHGF (default)
        let mut volatile_net = Network::new("eHGF");
        volatile_net.add_nodes("continuous-state", None, None, None, None);
        volatile_net.add_nodes("volatile-state", None, Some(0.into()), None, None);
        volatile_net.set_update_sequence();

        let input_data: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
        volatile_net.input_data(input_data.clone(), None);

        let mut explicit_net = Network::new("eHGF");
        explicit_net.add_nodes("continuous-state", None, None, None, None);
        explicit_net.add_nodes("continuous-state", None, Some(0.into()), None, None);
        explicit_net.add_nodes("continuous-state", None, None, None, Some(1.into()));
        explicit_net.set_update_sequence();
        explicit_net.input_data(input_data, None);

        assert_volatile_matches_explicit(&volatile_net, &explicit_net);
    }

    #[test]
    fn test_volatile_node_standard_matches_explicit() {
        let mut volatile_net = Network::new("standard");
        volatile_net.add_nodes("continuous-state", None, None, None, None);
        volatile_net.add_nodes("volatile-state", None, Some(0.into()), None, None);
        volatile_net.set_update_sequence();

        let input_data: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
        volatile_net.input_data(input_data.clone(), None);

        let mut explicit_net = Network::new("standard");
        explicit_net.add_nodes("continuous-state", None, None, None, None);
        explicit_net.add_nodes("continuous-state", None, Some(0.into()), None, None);
        explicit_net.add_nodes("continuous-state", None, None, None, Some(1.into()));
        explicit_net.set_update_sequence();
        explicit_net.input_data(input_data, None);

        assert_volatile_matches_explicit(&volatile_net, &explicit_net);
    }

    #[test]
    fn test_volatile_node_unbounded_matches_explicit() {
        let mut volatile_net = Network::new("unbounded");
        volatile_net.add_nodes("continuous-state", None, None, None, None);
        volatile_net.add_nodes("volatile-state", None, Some(0.into()), None, None);
        volatile_net.set_update_sequence();

        let input_data: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
        volatile_net.input_data(input_data.clone(), None);

        let mut explicit_net = Network::new("unbounded");
        explicit_net.add_nodes("continuous-state", None, None, None, None);
        explicit_net.add_nodes("continuous-state", None, Some(0.into()), None, None);
        explicit_net.add_nodes("continuous-state", None, None, None, Some(1.into()));
        explicit_net.set_update_sequence();
        explicit_net.input_data(input_data, None);

        assert_volatile_matches_explicit(&volatile_net, &explicit_net);
    }

    /// Helper: assert volatile node 1 trajectories match explicit nodes 1 & 2
    fn assert_volatile_matches_explicit(volatile_net: &Network, explicit_net: &Network) {
        let vol_floats = &volatile_net.node_trajectories.floats[&1];
        let exp_floats = &explicit_net.node_trajectories.floats[&1];

        for key in ["mean", "expected_mean", "precision", "expected_precision"] {
            let vol = &vol_floats[key];
            let exp = &exp_floats[key];
            for (t, (v, e)) in vol.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (v - e).abs() < 1e-6,
                    "Value-level key '{}' mismatch at t={}: volatile={}, explicit={}",
                    key, t, v, e
                );
            }
        }

        let exp2_floats = &explicit_net.node_trajectories.floats[&2];
        let vol_key_map = [
            ("mean_vol", "mean"),
            ("expected_mean_vol", "expected_mean"),
            ("precision_vol", "precision"),
            ("expected_precision_vol", "expected_precision"),
        ];

        for (vol_key, exp_key) in vol_key_map {
            let vol = &vol_floats[vol_key];
            let exp = &exp2_floats[exp_key];
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
