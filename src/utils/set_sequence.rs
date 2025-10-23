use crate::{model::{AdjacencyLists, Network, UpdateSequence}, updates::{posterior::continuous::posterior_update_continuous_state_node, prediction::continuous::prediction_continuous_state_node, prediction_error::{continuous::prediction_error_continuous_state_node, exponential::prediction_error_exponential_state_node}}};
use crate::utils::function_pointer::FnType;

pub fn set_update_sequence(network: &Network) -> UpdateSequence {
    let predictions = get_predictions_sequence(network);
    let updates = get_updates_sequence(network);

    // return the update sequence
    let update_sequence = UpdateSequence {predictions: predictions, updates: updates};
    update_sequence
}


pub fn get_predictions_sequence(network: &Network) -> Vec<(usize, FnType)> {

    let mut predictions : Vec<(usize, FnType)> = Vec::new();

    // 1. get prediction sequence ------------------------------------------------------

    // list all nodes availables in the network
    let mut nodes_idxs: Vec<usize> = network.edges.keys().cloned().collect();

    // iterate over all nodes and add the prediction step if all criteria are met
    let mut n_remaining = nodes_idxs.len();

    while n_remaining > 0 {
        
        // were we able to add an update step in the list on that iteration?
        let mut has_update = false;

        // loop over all the remaining nodes
        for i in 0..nodes_idxs.len() {

            let idx = nodes_idxs[i];

            // list the node's parents
            let value_parents_idxs = &network.edges[&idx].value_parents;
            let volatility_parents_idxs = &network.edges[&idx].volatility_parents;
                
            let parents_idxs = match (value_parents_idxs, volatility_parents_idxs) {
                // If both are Some, merge the vectors
                (Some(ref vec1), Some(ref vec2)) => {
                    // Create a new vector by merging the two
                    let vec: Vec<usize> = vec1.iter().chain(vec2.iter()).cloned().collect();
                    Some(vec) // Return the merged vector wrapped in Some
                }
                // If one is Some and the other is None, return the one that's Some
                (Some(vec), None) | (None, Some(vec)) => Some(vec.clone()),
                // If both are None, return None
                (None, None) => None,
            };


            // check if there is any parent node that is still found in the to-be-updated list 
            let contains_common = match parents_idxs {
                Some(vec) => vec.iter().any(|item| nodes_idxs.contains(item)),
                None => false
            };
            
            // if all parents have processed their prediction, this one can be added
            if !(contains_common) {
    
                // add the node in the update list
                match network.edges.get(&idx) {
                    Some(AdjacencyLists {node_type, ..}) if node_type == "continuous-state" => {
                        predictions.push((idx, prediction_continuous_state_node));
                    }
                    _ => ()

                }
    
                // remove the node from the to-be-updated list
                nodes_idxs.retain(|&x| x != idx);
                n_remaining -= 1;
                has_update = true;
                break;
            }
            }
        // 2. get update sequence ------------------------------------------------------
        
        if !(has_update) {
            break;
        }
    }
    predictions

}

pub fn get_updates_sequence(network: &Network) -> Vec<(usize, FnType)> {

    let mut updates : Vec<(usize, FnType)> = Vec::new();

    // 1. get update sequence ----------------------------------------------------------

    // list all nodes availables in the network
    let mut pe_nodes_idxs: Vec<usize> = network.edges.keys().cloned().collect();
    let mut po_nodes_idxs: Vec<usize> = network.edges.keys().cloned().collect();

    // remove the input nodes from the to-be-visited nodes for posterior updates
    po_nodes_idxs.retain(|x| !network.inputs.contains(x));

    // iterate over all nodes and add the prediction step if all criteria are met
    let mut n_remaining = po_nodes_idxs.len() + pe_nodes_idxs.len();  // posterior updates + prediction errors

    while n_remaining > 0 {
        
        // were we able to add an update step in the list on that iteration?
        let mut has_update = false;

        // loop over all the remaining nodes for prediction errors ---------------------
        // -----------------------------------------------------------------------------
        for i in 0..pe_nodes_idxs.len() {

            let idx = pe_nodes_idxs[i];

            // to send a prediction error, this node should have been updated first
            if !(po_nodes_idxs.contains(&idx)) {

                // only send a prediction error if this node has any parent
                let value_parents_idxs = &network.edges[&idx].value_parents;
                let volatility_parents_idxs = &network.edges[&idx].volatility_parents;
                
                let has_parents = match (value_parents_idxs, volatility_parents_idxs) {
                    // If both are None, return false
                    (None, None) => false,
                    _ => true,
                };

                // add the node in the update list
                match (network.edges.get(&idx), has_parents) {
                    (Some(AdjacencyLists {node_type, ..}), true) if node_type == "continuous-state" => {
                        updates.push((idx, prediction_error_continuous_state_node));
                        // remove the node from the to-be-updated list
                        pe_nodes_idxs.retain(|&x| x != idx);
                        n_remaining -= 1;
                        has_update = true;
                        break;
                    }
                    (Some(AdjacencyLists {node_type, ..}), _) if node_type == "ef-state" => {
                        updates.push((idx, prediction_error_exponential_state_node));
                        // remove the node from the to-be-updated list
                        pe_nodes_idxs.retain(|&x| x != idx);
                        n_remaining -= 1;
                        has_update = true;
                        break;
                    }
                    _ => ()

                }
            }
        }

        // loop over all the remaining nodes for posterior updates ---------------------
        // -----------------------------------------------------------------------------
        for i in 0..po_nodes_idxs.len() {

            let idx = po_nodes_idxs[i];

            // to start a posterior update, all children should have sent prediction errors
            // 1. get a list of all children
            let value_children_idxs = &network.edges[&idx].value_children;
            let volatility_children_idxs = &network.edges[&idx].volatility_children;
            
            let children_idxs = match (value_children_idxs, volatility_children_idxs) {
                // If both are Some, merge the vectors
                (Some(ref vec1), Some(ref vec2)) => {
                    // Create a new vector by merging the two
                    let merged_vec: Vec<usize> = vec1.iter().chain(vec2.iter()).cloned().collect();
                    Some(merged_vec) // Return the merged vector wrapped in Some
                }
                // If one is Some and the other is None, return the one that's Some
                (Some(vec), None) | (None, Some(vec)) => Some(vec.clone()),
                // If both are None, return None
                (None, None) => None,
            };

            // 2. check if any of the children is still on the to-be-visited list for prediction errors
            // check if there is any parent node that is still found in the to-be-updated list 
            let missing_pe = match children_idxs {
                Some(vec) => vec.iter().any(|item| pe_nodes_idxs.contains(item)),
                None => false
            };

            // 3. if false, add the posterior update to the list
            if !missing_pe {
    
                // add the node in the update list
                match network.edges.get(&idx) {
                    Some(AdjacencyLists {node_type, ..}) if node_type == "continuous-state" => {
                        updates.push((idx, posterior_update_continuous_state_node));
                    }
                    _ => ()

                }
    
                // remove the node from the to-be-updated list
                po_nodes_idxs.retain(|&x| x != idx);
                n_remaining -= 1;
                has_update = true;
                break;
            }
            }        
        if !(has_update) {
            break;
        }
    }
    updates

}

// Tests module for unit tests
#[cfg(test)] // Only compile and include this module when running tests
mod tests {
    use crate::utils::function_pointer::get_func_map;

    use super::*; // Import the parent module's items to test them

    #[test]
    fn test_get_update_order() {
    
        let func_map = get_func_map();

        // initialize network
        let mut hgf_network = Network::new();
    
        // create a network
        hgf_network.add_nodes(
            "continuous-state",
            Some(vec![1]),
            None,
            Some(vec![2]),
            None,
        );
        hgf_network.add_nodes(
            "continuous-state",
            None,
            Some(vec![0]),
            None,
            None,
        );
        hgf_network.add_nodes(
            "continuous-state",
            None,
            None,
            None,
            Some(vec![0]),
        );
        hgf_network.set_update_sequence();

        println!("Prediction sequence ----------");
        println!("Node: {} - Function name: {}", &hgf_network.update_sequence.predictions[0].0, func_map.get(&hgf_network.update_sequence.predictions[0].1).unwrap_or(&"unknown"));
        println!("Node: {} - Function name: {}", &hgf_network.update_sequence.predictions[1].0, func_map.get(&hgf_network.update_sequence.predictions[1].1).unwrap_or(&"unknown"));
        println!("Node: {} - Function name: {}", &hgf_network.update_sequence.predictions[2].0, func_map.get(&hgf_network.update_sequence.predictions[2].1).unwrap_or(&"unknown"));
        println!("Update sequence ----------");
        println!("Node: {} - Function name: {}", &hgf_network.update_sequence.updates[0].0, func_map.get(&hgf_network.update_sequence.updates[0].1).unwrap_or(&"unknown"));
        println!("Node: {} - Function name: {}", &hgf_network.update_sequence.updates[1].0, func_map.get(&hgf_network.update_sequence.updates[1].1).unwrap_or(&"unknown"));
        println!("Node: {} - Function name: {}", &hgf_network.update_sequence.updates[2].0, func_map.get(&hgf_network.update_sequence.updates[2].1).unwrap_or(&"unknown"));

        // initialize network
        let mut exp_network = Network::new();
        exp_network.add_nodes(
            "ef-state",
            None,
            None,
            None,
            None,
        );
        exp_network.set_update_sequence();
        println!("Node: {} - Function name: {}", &exp_network.update_sequence.updates[0].0, func_map.get(&exp_network.update_sequence.updates[0].1).unwrap_or(&"unknown"));

    }
}
