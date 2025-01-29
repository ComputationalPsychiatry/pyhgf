use std::collections::HashMap;
use crate::utils::function_pointer::FnType;
use crate::utils::set_sequence::set_update_sequence;
use crate::utils::beliefs_propagation::belief_propagation;
use crate::utils::function_pointer::get_func_map;
use pyo3::types::PyTuple;
use pyo3::{prelude::*, types::{PyList, PyDict}};
use numpy::{PyArray1, PyArray};

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
            node_trajectories: NodeTrajectories {floats: HashMap::new(), vectors: HashMap::new()}
        }
        }

    /// Add nodes to the network.
    /// 
    /// # Arguments
    /// * `kind` - The type of node that should be added.
    /// * `value_parents` - The indexes of the node's value parents.
    /// * `value_children` - The indexes of the node's value children.
    /// * `volatility_children` - The indexes of the node's volatility children.
    /// * `volatility_parents` - The indexes of the node's volatility parents.
    #[pyo3(signature = (kind="continuous-state", value_parents=None, value_children=None, volatility_parents=None, volatility_children=None,))]
    pub fn add_nodes(&mut self, kind: &str, value_parents: Option<Vec<usize>>, 
        value_children: Option<Vec<usize>>,
        volatility_parents: Option<Vec<usize>>, volatility_children: Option<Vec<usize>>, )  {

        // the node ID is equal to the number of nodes already in the network
        let node_id: usize = self.edges.len();

        // if this node has no children, this is an input node
        if (value_children == None) & (volatility_children == None) {
            self.inputs.push(node_id);
        }
        
        let edges = AdjacencyLists{
            node_type: String::from(kind),
            value_parents: value_parents,
            value_children: value_children, 
            volatility_parents: volatility_parents,
            volatility_children: volatility_children,
        };

        // add edges and attributes
        if kind == "continuous-state" {

            let attributes =  [
                (String::from("mean"), 0.0), 
                (String::from("expected_mean"), 0.0), 
                (String::from("precision"), 1.0), 
                (String::from("expected_precision"), 1.0), 
                (String::from("tonic_volatility"), -4.0), 
                (String::from("tonic_drift"), 0.0), 
                (String::from("autoconnection_strength"), 1.0)].into_iter().collect();

            self.attributes.floats.insert(node_id, attributes);
            self.edges.insert(node_id, edges);

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

    pub fn set_update_sequence(&mut self) {
        self.update_sequence = set_update_sequence(self);
    }

    /// Add a sequence of observations.
    /// 
    /// # Arguments
    /// * `input_data` - A vector of vectors. Each vector is a time series of observations
    /// associated with one node.
    pub fn input_data(&mut self, input_data: Vec<f64>) {

        let n_time = input_data.len();
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
        for observation in input_data {

            // 1. belief propagation for one time slice
            belief_propagation(self, vec![observation], &predictions, &updates);

            // 2. append the new beliefs in the trajectories structure
            // iterate over the float hashmap
            for (new_node_idx, new_node) in &self.attributes.floats {
                for (new_key, new_value) in new_node {
                    // If the key exists in map1, append the vector from map2
                    let old_node = node_trajectories.floats.get_mut(&new_node_idx).expect("Old node not found.");
                    let old_value = old_node.get_mut(new_key).expect("Old value not found");
                    old_value.push(*new_value);
                    }
                }

            // iterate over the vector hashmap
            for (new_node_idx, new_node) in &self.attributes.vectors {
                for (new_key, new_value) in new_node {
                    // If the key exists in map1, append the vector from map2
                    let old_node = node_trajectories.vectors.get_mut(&new_node_idx).expect("Old vector node not found.");
                    let old_value = old_node.get_mut(new_key).expect("Old vector value not found.");
                    old_value.push(new_value.clone());
                    }
                }
            }
        self.node_trajectories = node_trajectories;
    }

    #[getter]
    pub fn get_node_trajectories<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        let py_list = PyList::empty(py);
        
        
        // Iterate over the float hashmap and insert key-value pairs into the list as PyDict
        for (node_idx, node) in &self.node_trajectories.floats {
            let py_dict = PyDict::new(py);
            for (key, value) in node {
                // Create a new Python dictionary
                py_dict.set_item(key, PyArray1::from_vec(py, value.clone()).to_owned()).expect("Failed to set item in PyDict");
            }

            // Iterate over the vector hashmap if any and insert key-value pairs into the list as PyDict
            if let Some(vector_node) = self.node_trajectories.vectors.get(node_idx) {
                for (vector_key, vector_value) in vector_node {
                    // Create a new Python dictionary
                    py_dict.set_item(vector_key, PyArray::from_vec2(py, &vector_value).unwrap()).expect("Failed to set item in PyDict");
                }
            }
            py_list.append(py_dict)?;
        }

        // Create a PyList from Vec<usize>
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

// Create a module to expose the class to Python
#[pymodule]
fn rshgf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Network>()?;
    Ok(())
}

// Tests module for unit tests
#[cfg(test)] // Only compile and include this module when running tests
mod tests {
    use super::*; // Import the parent module's items to test them


    #[test]
    fn test_exponential_family_gaussian() {
    
        // initialize network
        let mut network = Network::new();
    
        // create a network with two exponential family state nodes
        network.add_nodes(
            "ef-state",
            None,
            None,
            None,
            None
        );
    
        // println!("Graph before belief propagation: {:?}", network);
    
        // belief propagation
        let input_data = vec![1.0, 1.3, 1.5, 1.7];
        network.set_update_sequence();
        network.input_data(input_data);
        
        println!("Update sequence: {:?}", network.update_sequence);
        println!("Node trajectories: {:?}", network.node_trajectories);
    
    }
}
