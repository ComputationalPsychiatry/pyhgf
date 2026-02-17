use std::collections::HashMap;
use crate::utils::function_pointer::FnType;
use crate::utils::set_sequence::set_update_sequence;
use crate::utils::beliefs_propagation::belief_propagation;
use crate::utils::function_pointer::get_func_map;
use pyo3::types::PyTuple;
use pyo3::{prelude::*, types::{PyList, PyDict}};
use numpy::{PyArray1, PyArray};

/// A helper type that accepts either a single int or a list of ints from Python.
/// This allows the user to write `value_children=0` or `value_children=[0, 1]`.
#[derive(Debug, Clone)]
pub enum IntOrList {
    Single(usize),
    List(Vec<usize>),
}

impl<'py> FromPyObject<'py> for IntOrList {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(val) = ob.extract::<usize>() {
            Ok(IntOrList::Single(val))
        } else {
            Ok(IntOrList::List(ob.extract::<Vec<usize>>()?))
        }
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
    pub update_sequence: UpdateSequence,
    pub node_trajectories: NodeTrajectories,
}

#[pymethods]
impl Network {

    // Create a new graph
    #[new]  // Define the constructor accessible from Python
    pub fn new() -> Self {
        Network {
            attributes: Attributes {floats: HashMap::new(), vectors: HashMap::new()},
            edges: HashMap::new(),
            inputs: Vec::new(),
            update_sequence: UpdateSequence {predictions: Vec::new(), updates: Vec::new()},
            node_trajectories: NodeTrajectories {floats: HashMap::new(), vectors: HashMap::new()},
        }
        }

    /// Add nodes to the network.
    /// 
    /// # Arguments
    /// * `kind` - The type of node that should be added.
    /// * `value_parents` - The index(es) of the node's value parents (int or list).
    /// * `value_children` - The index(es) of the node's value children (int or list).
    /// * `volatility_parents` - The index(es) of the node's volatility parents (int or list).
    /// * `volatility_children` - The index(es) of the node's volatility children (int or list).
    #[pyo3(signature = (kind="continuous-state", value_parents=None, value_children=None, volatility_parents=None, volatility_children=None,))]
    pub fn add_nodes(&mut self, kind: &str, 
        value_parents: Option<IntOrList>, 
        value_children: Option<IntOrList>,
        volatility_parents: Option<IntOrList>, 
        volatility_children: Option<IntOrList>, )  {

        self.add_nodes_inner(
            kind,
            value_parents.map(|v| v.into_vec()),
            value_children.map(|v| v.into_vec()),
            volatility_parents.map(|v| v.into_vec()),
            volatility_children.map(|v| v.into_vec()),
        );
    }

    pub fn set_update_sequence(&mut self) {
        self.update_sequence = set_update_sequence(self);
    }

    /// Add a sequence of observations.
    /// 
    /// # Arguments
    /// * `input_data` - A vector of observations. Each value is one observation per time step.
    /// * `time_steps` - An optional vector of time steps of the same length as `input_data`.
    ///   If `None`, defaults to a vector of ones.
    #[pyo3(signature = (input_data, time_steps=None))]
    pub fn input_data(&mut self, input_data: Vec<f64>, time_steps: Option<Vec<f64>>) {

        let n_time = input_data.len();

        // Default to a vector of ones if no time steps are provided
        let time_steps = time_steps.unwrap_or_else(|| vec![1.0; n_time]);
        let predictions = self.update_sequence.predictions.clone();
        let updates = self.update_sequence.updates.clone();

        // initialize the belief trajectories result struture
        let mut node_trajectories = NodeTrajectories {floats: HashMap::new(), vectors: HashMap::new()};
        
        // preallocate empty vectors in the floats hashmap
        for (node_idx, node) in &self.attributes.floats {
            let new_map: HashMap<String, Vec<f64>> = HashMap::new();
            node_trajectories.floats.insert(*node_idx, new_map);
            let attr = node_trajectories.floats.get_mut(node_idx).expect("New map not found.");
            for key in node.keys() {
                attr.insert(key.clone(), Vec::with_capacity(n_time));
            }
            }

        // preallocate empty vectors in the vectors hashmap
        for (node_idx, node) in &self.attributes.vectors {
            let new_map: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
            node_trajectories.vectors.insert(*node_idx, new_map);
            let attr = node_trajectories.vectors.get_mut(node_idx).expect("New vector map not found.");
            for key in node.keys() {
                attr.insert(key.clone(), Vec::with_capacity(n_time));
            }
            }


        // iterate over the observations
        for (t, observation) in input_data.iter().enumerate() {

            // 1. belief propagation for one time slice
            belief_propagation(self, vec![*observation], &predictions, &updates, time_steps[t]);

            // 2. append the new beliefs in the trajectories structure
            // iterate over the float hashmap
            for (new_node_idx, new_node) in &self.attributes.floats {
                for (new_key, new_value) in new_node {
                    let old_node = node_trajectories.floats.get_mut(&new_node_idx).expect("Old node not found.");
                    // lazily create the vector if this key was added during belief propagation
                    let old_value = old_node.entry(new_key.clone()).or_insert_with(|| Vec::with_capacity(n_time));
                    old_value.push(*new_value);
                    }
                }

            // iterate over the vector hashmap
            for (new_node_idx, new_node) in &self.attributes.vectors {
                for (new_key, new_value) in new_node {
                    let old_node = node_trajectories.vectors
                        .entry(*new_node_idx)
                        .or_insert_with(HashMap::new);
                    let old_value = old_node.entry(new_key.clone()).or_insert_with(|| Vec::with_capacity(n_time));
                    old_value.push(new_value.clone());
                    }
                }
            }
        self.node_trajectories = node_trajectories;
    }

    #[getter]
    pub fn get_node_trajectories<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        let py_list = PyList::empty(py);

        // Sort node indices so py_list[i] corresponds to node i
        let mut sorted_keys: Vec<&usize> = self.node_trajectories.floats.keys().collect();
        sorted_keys.sort();

        for node_idx in sorted_keys {
            let node = &self.node_trajectories.floats[node_idx];
            let py_dict = PyDict::new(py);
            for (key, value) in node {
                py_dict.set_item(key, PyArray1::from_vec(py, value.clone()).to_owned()).expect("Failed to set item in PyDict");
            }

            // Iterate over the vector hashmap if any and insert key-value pairs into the list as PyDict
            if let Some(vector_node) = self.node_trajectories.vectors.get(node_idx) {
                for (vector_key, vector_value) in vector_node {
                    py_dict.set_item(vector_key, PyArray::from_vec2(py, &vector_value).unwrap()).expect("Failed to set item in PyDict");
                }
            }
            py_list.append(py_dict)?;
        }

        Ok(py_list.into())
    }

    #[getter]
    pub fn get_inputs<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        let py_list = PyList::new(py, &self.inputs)?;  // Create a PyList from Vec<usize>
        Ok(py_list.into())
    }

    #[getter]
    pub fn get_edges<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        // Create a new Python list
        let py_list = PyList::empty(py);
    
        // Convert each struct in the Vec to a Python object and add to PyList
        for i in 0..self.edges.len() {
            // Create a new Python dictionary for each MyStruct
            let py_dict = PyDict::new(py);
            py_dict.set_item("value_parents", &self.edges[&i].value_parents)?;
            py_dict.set_item("value_children", &self.edges[&i].value_children)?;
            py_dict.set_item("volatility_parents", &self.edges[&i].volatility_parents)?;
            py_dict.set_item("volatility_children", &self.edges[&i].volatility_children)?;
    
            // Add the dictionary to the list
            py_list.append(py_dict)?;
        }
    
        // Return the PyList object directly
        Ok(py_list.into())
    }


    #[getter]
    pub fn get_update_sequence<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
    
        let func_map = get_func_map();
        let py_list = PyList::empty(py);

        // Iterate over the Rust vector and convert each tuple
        for &(num, func) in self.update_sequence.predictions.iter() {
            
            // Resolve the Python objects from the Rust values
            let py_func_name = func_map.get(&func).unwrap_or(&"unknown").into_pyobject(py)?.into_any().unbind();
            let py_num = num.into_pyobject(py)?.into_any().unbind();         // Converts num to PyObject

            // Create a Python tuple
            let py_tuple = PyTuple::new(py, &[py_num, py_func_name])?;
            
            // Append the Python tuple to the Python list
            py_list.append(py_tuple)?;
        }        
        for &(num, func) in self.update_sequence.updates.iter() {

            // Resolve the Python objects from the Rust values
            let py_func_name = func_map.get(&func).unwrap_or(&"unknown").into_pyobject(py)?.into_any().unbind();
            let py_num = num.into_pyobject(py)?.into_any().unbind();         // Converts num to PyObject

            // Create a Python tuple
            let py_tuple = PyTuple::new(py, &[py_num, py_func_name])?;
            
            // Append the Python tuple to the Python list
            py_list.append(py_tuple)?;
        }        
        Ok(py_list.into())
    }
}

/// Non-PyO3 methods for internal Rust usage.
impl Network {
    /// Internal implementation of add_nodes that works with plain Vec<usize>.
    /// Use this from Rust code; from Python, use `add_nodes` which accepts int or list.
    pub fn add_nodes_inner(&mut self, kind: &str, 
        value_parents: Option<Vec<usize>>, 
        value_children: Option<Vec<usize>>,
        volatility_parents: Option<Vec<usize>>, 
        volatility_children: Option<Vec<usize>>, )  {

        // the node ID is equal to the number of nodes already in the network
        let node_id: usize = self.edges.len();

        // determine if this is an input node (no children)
        let is_input = value_children.is_none() && volatility_children.is_none();
        if is_input {
            self.inputs.push(node_id);
        }
        
        let edges = AdjacencyLists{
            node_type: String::from(kind),
            value_parents: value_parents.clone(),
            value_children: value_children.clone(), 
            volatility_parents: volatility_parents.clone(),
            volatility_children: volatility_children.clone(),
        };

        // add edges and attributes
        if kind == "continuous-state" {

            // input nodes have autoconnection_strength=0.0 and tonic_volatility=0.0
            let (autoconnection, tonic_vol) = if is_input {
                (0.0, 0.0)
            } else {
                (1.0, -4.0)
            };

            let attributes: HashMap<String, f64> = [
                (String::from("mean"), 0.0), 
                (String::from("expected_mean"), 0.0), 
                (String::from("precision"), 1.0), 
                (String::from("expected_precision"), 1.0), 
                (String::from("tonic_volatility"), tonic_vol), 
                (String::from("tonic_drift"), 0.0), 
                (String::from("autoconnection_strength"), autoconnection),
                (String::from("current_variance"), 1.0),
            ].into_iter().collect();

            self.attributes.floats.insert(node_id, attributes);
            self.edges.insert(node_id, edges);

            // Set coupling strength vectors and update reciprocal edges
            let mut vec_attrs: HashMap<String, Vec<f64>> = HashMap::new();

            if let Some(ref vp) = value_parents {
                vec_attrs.insert(String::from("value_coupling_parents"), vec![1.0; vp.len()]);
            }
            if let Some(ref vc) = value_children {
                vec_attrs.insert(String::from("value_coupling_children"), vec![1.0; vc.len()]);
                // Update reciprocal edges: for each child, add this node as a value parent
                for &child_idx in vc {
                    if let Some(child_edges) = self.edges.get_mut(&child_idx) {
                        match &mut child_edges.value_parents {
                            Some(parents) => parents.push(node_id),
                            None => child_edges.value_parents = Some(vec![node_id]),
                        }
                    }
                    // Add coupling strength to the child's value_coupling_parents
                    let child_vecs = self.attributes.vectors.entry(child_idx).or_insert_with(HashMap::new);
                    match child_vecs.get_mut("value_coupling_parents") {
                        Some(cs) => cs.push(1.0),
                        None => { child_vecs.insert(String::from("value_coupling_parents"), vec![1.0]); }
                    }
                }
            }
            if let Some(ref volp) = volatility_parents {
                vec_attrs.insert(String::from("volatility_coupling_parents"), vec![1.0; volp.len()]);
            }
            if let Some(ref volc) = volatility_children {
                vec_attrs.insert(String::from("volatility_coupling_children"), vec![1.0; volc.len()]);
                // Update reciprocal edges: for each child, add this node as a volatility parent
                for &child_idx in volc {
                    if let Some(child_edges) = self.edges.get_mut(&child_idx) {
                        match &mut child_edges.volatility_parents {
                            Some(parents) => parents.push(node_id),
                            None => child_edges.volatility_parents = Some(vec![node_id]),
                        }
                    }
                    // Add coupling strength to the child's volatility_coupling_parents
                    let child_vecs = self.attributes.vectors.entry(child_idx).or_insert_with(HashMap::new);
                    match child_vecs.get_mut("volatility_coupling_parents") {
                        Some(cs) => cs.push(1.0),
                        None => { child_vecs.insert(String::from("volatility_coupling_parents"), vec![1.0]); }
                    }
                }
            }

            if !vec_attrs.is_empty() {
                self.attributes.vectors.insert(node_id, vec_attrs);
            }

        } else if kind == "ef-state" {

            let floats_attributes =  [
                (String::from("mean"), 0.0), 
                (String::from("nus"), 3.0)].into_iter().collect();
            let vector_attributes =  [
                (String::from("xis"), vec![0.0, 1.0])].into_iter().collect();

            self.attributes.floats.insert(node_id, floats_attributes);
            self.attributes.vectors.insert(node_id, vector_attributes);
            self.edges.insert(node_id, edges);

        }
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
        // initialize network
        let mut network = Network::new();

        // create a network with two exponential family state nodes
        network.add_nodes_inner("ef-state", None, None, None, None);

        // belief propagation
        let input_data = vec![1.0, 1.3, 1.5, 1.7];
        network.set_update_sequence();
        network.input_data(input_data, None);

        println!("Update sequence: {:?}", network.update_sequence);
        println!("Node trajectories: {:?}", network.node_trajectories);
    }
}
