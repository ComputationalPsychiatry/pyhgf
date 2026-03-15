use crate::{model::{AdjacencyLists, Network, UpdateSequence}, updates::{posterior::continuous::{posterior_update_continuous_state_node, posterior_update_continuous_state_node_ehgf, posterior_update_continuous_state_node_unbounded}, posterior::volatile::{posterior_update_volatile_state_node, posterior_update_volatile_state_node_ehgf, posterior_update_volatile_state_node_unbounded}, prediction::binary::prediction_binary_state_node, prediction::continuous::prediction_continuous_state_node, prediction::volatile::prediction_volatile_state_node, prediction_error::{binary::prediction_error_binary_state_node, continuous::prediction_error_continuous_state_node, exponential::prediction_error_exponential_state_node, volatile::prediction_error_volatile_state_node}}};
use crate::utils::function_pointer::FnType;

pub fn set_update_sequence(network: &Network) -> UpdateSequence {
    let predictions = get_predictions_sequence(network);
    let updates = get_updates_sequence(network);
    UpdateSequence { predictions, updates }
}

pub fn get_predictions_sequence(network: &Network) -> Vec<(usize, FnType)> {
    let mut predictions: Vec<(usize, FnType)> = Vec::new();

    let mut nodes_idxs: Vec<usize> = (0..network.edges.len()).collect();
    let mut n_remaining = nodes_idxs.len();

    while n_remaining > 0 {
        let mut has_update = false;

        for i in 0..nodes_idxs.len() {
            let idx = nodes_idxs[i];
            let edge = &network.edges[idx];

            let parents_idxs = match (&edge.value_parents, &edge.volatility_parents) {
                (Some(ref vec1), Some(ref vec2)) => {
                    Some(vec1.iter().chain(vec2.iter()).copied().collect::<Vec<usize>>())
                }
                (Some(vec), None) | (None, Some(vec)) => Some(vec.clone()),
                (None, None) => None,
            };

            let contains_common = match parents_idxs {
                Some(vec) => vec.iter().any(|item| nodes_idxs.contains(item)),
                None => false
            };

            if !contains_common {
                match edge.node_type.as_str() {
                    "continuous-state" => predictions.push((idx, prediction_continuous_state_node)),
                    "volatile-state" => predictions.push((idx, prediction_volatile_state_node)),
                    "binary-state" => predictions.push((idx, prediction_binary_state_node)),
                    _ => ()
                }

                nodes_idxs.retain(|&x| x != idx);
                n_remaining -= 1;
                has_update = true;
                break;
            }
        }

        if !has_update {
            break;
        }
    }
    predictions
}

pub fn get_updates_sequence(network: &Network) -> Vec<(usize, FnType)> {
    let mut updates: Vec<(usize, FnType)> = Vec::new();

    let mut pe_nodes_idxs: Vec<usize> = (0..network.edges.len()).collect();
    let mut po_nodes_idxs: Vec<usize> = (0..network.edges.len()).collect();

    po_nodes_idxs.retain(|x| !network.inputs.contains(x));
    po_nodes_idxs.retain(|&x| network.edges[x].node_type != "constant-state");

    loop {
        let mut has_update = false;

        // --- Batch: posterior updates ---
        let eligible_po: Vec<usize> = po_nodes_idxs
            .iter()
            .copied()
            .filter(|&idx| {
                let children = get_all_children(&network.edges[idx]);
                children.iter().all(|c| !pe_nodes_idxs.contains(c))
            })
            .collect();

        for &idx in &eligible_po {
            let edge = &network.edges[idx];
            match edge.node_type.as_str() {
                "continuous-state" => {
                    if edge.volatility_children.is_some() {
                        match network.update_type.as_str() {
                            "eHGF" => updates.push((idx, posterior_update_continuous_state_node_ehgf)),
                            "unbounded" => updates.push((idx, posterior_update_continuous_state_node_unbounded)),
                            _ => updates.push((idx, posterior_update_continuous_state_node)),
                        }
                    } else {
                        updates.push((idx, posterior_update_continuous_state_node));
                    }
                }
                "volatile-state" => {
                    match network.update_type.as_str() {
                        "eHGF" => updates.push((idx, posterior_update_volatile_state_node_ehgf)),
                        "unbounded" => updates.push((idx, posterior_update_volatile_state_node_unbounded)),
                        _ => updates.push((idx, posterior_update_volatile_state_node)),
                    }
                }
                _ => (),
            }
            has_update = true;
        }
        po_nodes_idxs.retain(|x| !eligible_po.contains(x));

        // --- Batch: prediction errors ---
        let eligible_pe: Vec<usize> = pe_nodes_idxs
            .iter()
            .copied()
            .filter(|&idx| !po_nodes_idxs.contains(&idx))
            .collect();

        for &idx in &eligible_pe {
            let edge = &network.edges[idx];
            let has_parents = edge.value_parents.is_some() || edge.volatility_parents.is_some();

            match (edge.node_type.as_str(), has_parents) {
                ("continuous-state", true) => {
                    updates.push((idx, prediction_error_continuous_state_node));
                    has_update = true;
                }
                ("volatile-state", true) => {
                    updates.push((idx, prediction_error_volatile_state_node));
                    has_update = true;
                }
                ("ef-state", _) => {
                    updates.push((idx, prediction_error_exponential_state_node));
                    has_update = true;
                }
                ("binary-state", true) => {
                    updates.push((idx, prediction_error_binary_state_node));
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

fn get_all_children(adj: &AdjacencyLists) -> Vec<usize> {
    match (&adj.value_children, &adj.volatility_children) {
        (Some(v), Some(vol)) => v.iter().chain(vol.iter()).copied().collect(),
        (Some(v), None) => v.clone(),
        (None, Some(vol)) => vol.clone(),
        (None, None) => vec![],
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::function_pointer::get_func_map;
    use super::*;

    #[test]
    fn test_get_update_order() {
        let func_map = get_func_map();

        let mut hgf_network = Network::new("eHGF");
        hgf_network.add_nodes("continuous-state", 1, Some(vec![1].into()), None, Some(vec![2].into()), None, None, None);
        hgf_network.add_nodes("continuous-state", 1, None, Some(vec![0].into()), None, None, None, None);
        hgf_network.add_nodes("continuous-state", 1, None, None, None, Some(vec![0].into()), None, None);
        hgf_network.set_update_sequence();

        println!("Prediction sequence ----------");
        for (_i, &(node, func)) in hgf_network.update_sequence.predictions.iter().enumerate() {
            println!("Node: {} - Function name: {}", node, func_map.get(&func).unwrap_or(&"unknown"));
        }
        println!("Update sequence ----------");
        for &(node, func) in &hgf_network.update_sequence.updates {
            println!("Node: {} - Function name: {}", node, func_map.get(&func).unwrap_or(&"unknown"));
        }

        let mut exp_network = Network::new("eHGF");
        exp_network.add_nodes("ef-state", 1, None, None, None, None, None, None);
        exp_network.set_update_sequence();
        println!("Node: {} - Function name: {}", &exp_network.update_sequence.updates[0].0, func_map.get(&exp_network.update_sequence.updates[0].1).unwrap_or(&"unknown"));
    }
}
