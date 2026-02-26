use crate::{model::{AdjacencyLists, Network, UpdateSequence}, updates::{posterior::continuous::{posterior_update_continuous_state_node, posterior_update_continuous_state_node_ehgf, posterior_update_continuous_state_node_unbounded}, posterior::volatile::{posterior_update_volatile_state_node, posterior_update_volatile_state_node_ehgf, posterior_update_volatile_state_node_unbounded}, prediction::continuous::prediction_continuous_state_node, prediction::volatile::prediction_volatile_state_node, prediction_error::{continuous::prediction_error_continuous_state_node, exponential::prediction_error_exponential_state_node, volatile::prediction_error_volatile_state_node}}};
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
    nodes_idxs.sort();

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
                    Some(AdjacencyLists {node_type, ..}) if node_type == "volatile-state" => {
                        predictions.push((idx, prediction_volatile_state_node));
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

    let mut updates: Vec<(usize, FnType)> = Vec::new();

    // List all nodes available in the network
    let mut pe_nodes_idxs: Vec<usize> = network.edges.keys().cloned().collect();
    let mut po_nodes_idxs: Vec<usize> = network.edges.keys().cloned().collect();
    pe_nodes_idxs.sort();
    po_nodes_idxs.sort();

    // Remove the input nodes from posterior updates (they have no children)
    po_nodes_idxs.retain(|x| !network.inputs.contains(x));

    // Iteratively resolve the topological order:
    //   1. Find ALL nodes eligible for posterior update (all children have sent PEs).
    //   2. Find ALL nodes eligible for prediction error (already have a posterior).
    // Process entire batches per iteration to match Python's intended semantics.
    loop {
        let mut has_update = false;

        // --- Batch: posterior updates ------------------------------------------------
        // Collect all currently‐eligible PO nodes, then process them all at once.
        let eligible_po: Vec<usize> = po_nodes_idxs
            .iter()
            .copied()
            .filter(|&idx| {
                let children = get_all_children(&network.edges[&idx]);
                children.iter().all(|c| !pe_nodes_idxs.contains(c))
            })
            .collect();

        for idx in &eligible_po {
            match network.edges.get(idx) {
                Some(AdjacencyLists { node_type, volatility_children, .. })
                    if node_type == "continuous-state" =>
                {
                    if volatility_children.is_some() {
                        match network.update_type.as_str() {
                            "eHGF" => updates.push((*idx, posterior_update_continuous_state_node_ehgf)),
                            "unbounded" => updates.push((*idx, posterior_update_continuous_state_node_unbounded)),
                            _ => updates.push((*idx, posterior_update_continuous_state_node)),
                        }
                    } else {
                        updates.push((*idx, posterior_update_continuous_state_node));
                    }
                }
                Some(AdjacencyLists { node_type, .. })
                    if node_type == "volatile-state" =>
                {
                    match network.update_type.as_str() {
                        "eHGF" => updates.push((*idx, posterior_update_volatile_state_node_ehgf)),
                        "unbounded" => updates.push((*idx, posterior_update_volatile_state_node_unbounded)),
                        _ => updates.push((*idx, posterior_update_volatile_state_node)),
                    }
                }
                _ => (),
            }
            has_update = true;
        }
        po_nodes_idxs.retain(|x| !eligible_po.contains(x));

        // --- Batch: prediction errors ------------------------------------------------
        // Collect all currently‐eligible PE nodes, then process them all at once.
        let eligible_pe: Vec<usize> = pe_nodes_idxs
            .iter()
            .copied()
            .filter(|&idx| {
                // Node must have completed its posterior update (or not need one)
                if po_nodes_idxs.contains(&idx) {
                    return false;
                }
                true
            })
            .collect();

        for idx in &eligible_pe {
            let has_parents = match (&network.edges[idx].value_parents, &network.edges[idx].volatility_parents) {
                (None, None) => false,
                _ => true,
            };

            match (network.edges.get(idx), has_parents) {
                (Some(AdjacencyLists { node_type, .. }), true)
                    if node_type == "continuous-state" =>
                {
                    updates.push((*idx, prediction_error_continuous_state_node));
                    has_update = true;
                }
                (Some(AdjacencyLists { node_type, .. }), true)
                    if node_type == "volatile-state" =>
                {
                    updates.push((*idx, prediction_error_volatile_state_node));
                    has_update = true;
                }
                (Some(AdjacencyLists { node_type, .. }), _)
                    if node_type == "ef-state" =>
                {
                    updates.push((*idx, prediction_error_exponential_state_node));
                    has_update = true;
                }
                _ => (),
            }
        }
        pe_nodes_idxs.retain(|x| !eligible_pe.contains(x));

        if pe_nodes_idxs.is_empty() && po_nodes_idxs.is_empty() {
            break;
        }
        if !has_update {
            break;
        }
    }
    updates
}

/// Collect all children (value + volatility) of a node's adjacency lists.
fn get_all_children(adj: &AdjacencyLists) -> Vec<usize> {
    match (&adj.value_children, &adj.volatility_children) {
        (Some(v), Some(vol)) => v.iter().chain(vol.iter()).copied().collect(),
        (Some(v), None) => v.clone(),
        (None, Some(vol)) => vol.clone(),
        (None, None) => vec![],
    }
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
        let mut hgf_network = Network::new("eHGF");
    
        // create a network
        hgf_network.add_nodes(
            "continuous-state",
            1,
            Some(vec![1].into()),
            None,
            Some(vec![2].into()),
            None,
            None,
            None,
        );
        hgf_network.add_nodes(
            "continuous-state",
            1,
            None,
            Some(vec![0].into()),
            None,
            None,
            None,
            None,
        );
        hgf_network.add_nodes(
            "continuous-state",
            1,
            None,
            None,
            None,
            Some(vec![0].into()),
            None,
            None,
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
        let mut exp_network = Network::new("eHGF");
        exp_network.add_nodes(
            "ef-state",
            1,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        exp_network.set_update_sequence();
        println!("Node: {} - Function name: {}", &exp_network.update_sequence.updates[0].0, func_map.get(&exp_network.update_sequence.updates[0].1).unwrap_or(&"unknown"));

    }
}
