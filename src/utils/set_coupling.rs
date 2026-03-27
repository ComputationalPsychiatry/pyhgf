use crate::model::Network;

/// Update the value-coupling strength for a single `(parent, child)` pair.
pub fn set_coupling(
    network: &mut Network,
    parent_idx: usize,
    child_idx: usize,
    coupling: f64,
) {
    // 1. Child side: value_coupling_parents[pos of parent in child's value_parents]
    if let Some(pos) = network.edges[child_idx].value_parents.as_ref()
        .and_then(|vp| vp.iter().position(|&p| p == parent_idx))
    {
        let couplings = &mut network.attributes.vectors[child_idx].value_coupling_parents;
        if pos < couplings.len() {
            couplings[pos] = coupling;
        }
    }

    // 2. Parent side: value_coupling_children[pos of child in parent's value_children]
    if let Some(pos) = network.edges[parent_idx].value_children.as_ref()
        .and_then(|vc| vc.iter().position(|&c| c == child_idx))
    {
        let couplings = &mut network.attributes.vectors[parent_idx].value_coupling_children;
        if pos < couplings.len() {
            couplings[pos] = coupling;
        }
    }
}

/// Update the value-coupling strength for every combination of parents and children.
pub fn set_coupling_vec(
    network: &mut Network,
    parent_idxs: &[usize],
    child_idxs: &[usize],
    coupling: f64,
) {
    for &parent_idx in parent_idxs {
        for &child_idx in child_idxs {
            set_coupling(network, parent_idx, child_idx, coupling);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        AdjacencyLists, Attributes, Network, NodeState, NodeVectors, NodeFnPtrs,
        NodeTrajectories, UpdateSequence,
    };

    /// Build a minimal 3-node network:
    ///   node 0 (child)  — value_parents: [1, 2]
    ///   node 1 (parent) — value_children: [0]
    ///   node 2 (parent) — value_children: [0]
    fn make_test_network() -> Network {
        Network {
            attributes: Attributes {
                states: vec![NodeState::default(), NodeState::default(), NodeState::default()],
                vectors: vec![
                    NodeVectors { value_coupling_parents: vec![1.0, 1.0], ..Default::default() },
                    NodeVectors { value_coupling_children: vec![1.0], ..Default::default() },
                    NodeVectors { value_coupling_children: vec![1.0], ..Default::default() },
                ],
                fn_ptrs: vec![NodeFnPtrs::default(), NodeFnPtrs::default(), NodeFnPtrs::default()],
            },
            edges: vec![
                AdjacencyLists {
                    node_type: "continuous-state".into(),
                    value_parents: Some(vec![1, 2]),
                    value_children: None,
                    volatility_parents: None,
                    volatility_children: None,
                },
                AdjacencyLists {
                    node_type: "continuous-state".into(),
                    value_parents: None,
                    value_children: Some(vec![0]),
                    volatility_parents: None,
                    volatility_children: None,
                },
                AdjacencyLists {
                    node_type: "continuous-state".into(),
                    value_parents: None,
                    value_children: Some(vec![0]),
                    volatility_parents: None,
                    volatility_children: None,
                },
            ],
            inputs: vec![0],
            update_type: "standard".into(),
            update_sequence: UpdateSequence { predictions: Vec::new(), updates: Vec::new() },
            node_trajectories: NodeTrajectories { nodes: Vec::new() },
            layers: Vec::new(),
            adam_state: None,
            roots: vec![1],
            leafs: vec![0],
        }
    }

    #[test]
    fn test_set_coupling_updates_both_sides() {
        let mut net = make_test_network();
        set_coupling(&mut net, 1, 0, 0.42);

        assert_eq!(net.attributes.vectors[0].value_coupling_parents[0], 0.42);
        assert_eq!(net.attributes.vectors[0].value_coupling_parents[1], 1.0);
        assert_eq!(net.attributes.vectors[1].value_coupling_children[0], 0.42);
    }

    #[test]
    fn test_set_coupling_second_parent() {
        let mut net = make_test_network();
        set_coupling(&mut net, 2, 0, 3.5);

        assert_eq!(net.attributes.vectors[0].value_coupling_parents[0], 1.0);
        assert_eq!(net.attributes.vectors[0].value_coupling_parents[1], 3.5);
        assert_eq!(net.attributes.vectors[2].value_coupling_children[0], 3.5);
    }

    #[test]
    fn test_set_coupling_nonexistent_edge_is_noop() {
        let mut net = make_test_network();
        set_coupling(&mut net, 1, 2, 99.0);

        assert_eq!(net.attributes.vectors[0].value_coupling_parents, vec![1.0, 1.0]);
        assert_eq!(net.attributes.vectors[1].value_coupling_children, vec![1.0]);
        assert_eq!(net.attributes.vectors[2].value_coupling_children, vec![1.0]);
    }

    #[test]
    fn test_set_coupling_vec_all_combinations() {
        let mut net = make_test_network();
        set_coupling_vec(&mut net, &[1, 2], &[0], 0.7);

        assert_eq!(net.attributes.vectors[0].value_coupling_parents[0], 0.7);
        assert_eq!(net.attributes.vectors[0].value_coupling_parents[1], 0.7);
        assert_eq!(net.attributes.vectors[1].value_coupling_children[0], 0.7);
        assert_eq!(net.attributes.vectors[2].value_coupling_children[0], 0.7);
    }

    #[test]
    fn test_set_coupling_vec_single_parent() {
        let mut net = make_test_network();
        set_coupling_vec(&mut net, &[1], &[0], 2.0);

        assert_eq!(net.attributes.vectors[0].value_coupling_parents[0], 2.0);
        assert_eq!(net.attributes.vectors[0].value_coupling_parents[1], 1.0);
    }

    #[test]
    fn test_set_coupling_vec_empty_parents() {
        let mut net = make_test_network();
        set_coupling_vec(&mut net, &[], &[0], 5.0);
        assert_eq!(net.attributes.vectors[0].value_coupling_parents, vec![1.0, 1.0]);
    }

    #[test]
    fn test_set_coupling_vec_empty_children() {
        let mut net = make_test_network();
        set_coupling_vec(&mut net, &[1, 2], &[], 5.0);
        assert_eq!(net.attributes.vectors[1].value_coupling_children, vec![1.0]);
    }

    #[test]
    fn test_set_coupling_vec_ignores_invalid_pairs() {
        let mut net = make_test_network();
        set_coupling_vec(&mut net, &[1], &[0, 2], 0.3);

        assert_eq!(net.attributes.vectors[0].value_coupling_parents[0], 0.3);
        assert_eq!(net.attributes.vectors[2].value_coupling_children, vec![1.0]);
    }
}
