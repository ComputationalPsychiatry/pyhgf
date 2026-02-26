use crate::model::Network;

// =============================================================================
// set_coupling — mirror of Python pyhgf.utils.set_coupling
// =============================================================================

/// Update the value-coupling strength for a single `(parent, child)` pair.
///
/// The coupling value is written to both the parent's
/// `"value_coupling_children"` vector and the child's
/// `"value_coupling_parents"` vector so that the two stay in sync.
///
/// If the parent–child relationship does not exist in the network's edge
/// lists, or the corresponding coupling vector is missing/too short, that
/// side is silently skipped.
pub fn set_coupling(
    network: &mut Network,
    parent_idx: usize,
    child_idx: usize,
    coupling: f64,
) {
    // 1. Child side: value_coupling_parents[pos of parent in child's value_parents]
    if let Some(pos) = network
        .edges
        .get(&child_idx)
        .and_then(|e| e.value_parents.as_ref())
        .and_then(|vp| vp.iter().position(|&p| p == parent_idx))
    {
        if let Some(couplings) = network
            .attributes
            .vectors
            .get_mut(&child_idx)
            .and_then(|v| v.get_mut("value_coupling_parents"))
        {
            if pos < couplings.len() {
                couplings[pos] = coupling;
            }
        }
    }

    // 2. Parent side: value_coupling_children[pos of child in parent's value_children]
    if let Some(pos) = network
        .edges
        .get(&parent_idx)
        .and_then(|e| e.value_children.as_ref())
        .and_then(|vc| vc.iter().position(|&c| c == child_idx))
    {
        if let Some(couplings) = network
            .attributes
            .vectors
            .get_mut(&parent_idx)
            .and_then(|v| v.get_mut("value_coupling_children"))
        {
            if pos < couplings.len() {
                couplings[pos] = coupling;
            }
        }
    }
}

/// Update the value-coupling strength for every combination of parents and
/// children in the given index vectors.
///
/// For each `(p, c)` in the Cartesian product `parent_idxs × child_idxs`,
/// the same `coupling` value is written to both sides of the edge (parent's
/// `"value_coupling_children"` and child's `"value_coupling_parents"`).
///
/// # Example
/// ```ignore
/// // Set coupling to 0.5 for all (parent, child) pairs:
/// set_coupling_vec(&mut network, &[2, 3], &[0, 1], 0.5);
/// // equivalent to:
/// //   set_coupling(&mut network, 2, 0, 0.5);
/// //   set_coupling(&mut network, 2, 1, 0.5);
/// //   set_coupling(&mut network, 3, 0, 0.5);
/// //   set_coupling(&mut network, 3, 1, 0.5);
/// ```
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Network;

    /// Build a minimal 3-node network:
    ///   node 0 (child)  — value_parents: [1, 2]
    ///   node 1 (parent) — value_children: [0]
    ///   node 2 (parent) — value_children: [0]
    fn make_test_network() -> Network {
        let mut net = Network::new("standard");
        // parent 1
        net.add_nodes(
            "continuous-state",
            1,
            None,
            Some(vec![].into()), // will set value_children below
            None,
            None,
            None,
            None,
        );
        // parent 2
        net.add_nodes(
            "continuous-state",
            1,
            None,
            Some(vec![].into()),
            None,
            None,
            None,
            None,
        );
        // child 0 — added last so its id is 2, but we want 0 as child.
        // It's simpler to build the topology manually for a unit test.
        // Let's just start fresh with direct manipulation:
        drop(net);

        use crate::model::{AdjacencyLists, Attributes, Network, NodeTrajectories, UpdateSequence};
        use std::collections::HashMap;

        let mut net = Network {
            attributes: Attributes {
                floats: HashMap::new(),
                vectors: HashMap::new(),
                fn_ptrs: HashMap::new(),
            },
            edges: HashMap::new(),
            inputs: vec![0],
            update_type: "standard".into(),
            update_sequence: UpdateSequence {
                predictions: Vec::new(),
                updates: Vec::new(),
            },
            node_trajectories: NodeTrajectories {
                floats: HashMap::new(),
                vectors: HashMap::new(),
            },
            layers: Vec::new(),
        };

        // Node 0 (child): value_parents = [1, 2]
        net.edges.insert(0, AdjacencyLists {
            node_type: "continuous-state".into(),
            value_parents: Some(vec![1, 2]),
            value_children: None,
            volatility_parents: None,
            volatility_children: None,
        });
        net.attributes.vectors.insert(0, HashMap::from([
            ("value_coupling_parents".into(), vec![1.0, 1.0]),
        ]));

        // Node 1 (parent): value_children = [0]
        net.edges.insert(1, AdjacencyLists {
            node_type: "continuous-state".into(),
            value_parents: None,
            value_children: Some(vec![0]),
            volatility_parents: None,
            volatility_children: None,
        });
        net.attributes.vectors.insert(1, HashMap::from([
            ("value_coupling_children".into(), vec![1.0]),
        ]));

        // Node 2 (parent): value_children = [0]
        net.edges.insert(2, AdjacencyLists {
            node_type: "continuous-state".into(),
            value_parents: None,
            value_children: Some(vec![0]),
            volatility_parents: None,
            volatility_children: None,
        });
        net.attributes.vectors.insert(2, HashMap::from([
            ("value_coupling_children".into(), vec![1.0]),
        ]));

        net
    }

    // ── set_coupling ─────────────────────────────────────────────────────

    #[test]
    fn test_set_coupling_updates_both_sides() {
        let mut net = make_test_network();

        set_coupling(&mut net, 1, 0, 0.42);

        // Child side
        let child_couplings = net.attributes.vectors.get(&0)
            .unwrap().get("value_coupling_parents").unwrap();
        assert_eq!(child_couplings[0], 0.42); // parent 1 is at position 0
        assert_eq!(child_couplings[1], 1.0);  // parent 2 untouched

        // Parent side
        let parent_couplings = net.attributes.vectors.get(&1)
            .unwrap().get("value_coupling_children").unwrap();
        assert_eq!(parent_couplings[0], 0.42); // child 0 is the only child
    }

    #[test]
    fn test_set_coupling_second_parent() {
        let mut net = make_test_network();

        set_coupling(&mut net, 2, 0, 3.5);

        // Child side: parent 2 sits at position 1
        let child_couplings = net.attributes.vectors.get(&0)
            .unwrap().get("value_coupling_parents").unwrap();
        assert_eq!(child_couplings[0], 1.0);  // parent 1 untouched
        assert_eq!(child_couplings[1], 3.5);

        // Parent side
        let parent_couplings = net.attributes.vectors.get(&2)
            .unwrap().get("value_coupling_children").unwrap();
        assert_eq!(parent_couplings[0], 3.5);
    }

    #[test]
    fn test_set_coupling_nonexistent_edge_is_noop() {
        let mut net = make_test_network();

        // Node 1 is not a parent of node 2 — should be a silent no-op
        set_coupling(&mut net, 1, 2, 99.0);

        // Nothing should have changed
        let c0 = net.attributes.vectors.get(&0)
            .unwrap().get("value_coupling_parents").unwrap();
        assert_eq!(c0, &vec![1.0, 1.0]);

        let c1 = net.attributes.vectors.get(&1)
            .unwrap().get("value_coupling_children").unwrap();
        assert_eq!(c1, &vec![1.0]);

        let c2 = net.attributes.vectors.get(&2)
            .unwrap().get("value_coupling_children").unwrap();
        assert_eq!(c2, &vec![1.0]);
    }

    // ── set_coupling_vec ─────────────────────────────────────────────────

    #[test]
    fn test_set_coupling_vec_all_combinations() {
        let mut net = make_test_network();

        // Set coupling = 0.7 for every (parent, child) pair
        set_coupling_vec(&mut net, &[1, 2], &[0], 0.7);

        // Child side: both parents updated
        let child_couplings = net.attributes.vectors.get(&0)
            .unwrap().get("value_coupling_parents").unwrap();
        assert_eq!(child_couplings[0], 0.7);
        assert_eq!(child_couplings[1], 0.7);

        // Parent sides
        let p1 = net.attributes.vectors.get(&1)
            .unwrap().get("value_coupling_children").unwrap();
        assert_eq!(p1[0], 0.7);

        let p2 = net.attributes.vectors.get(&2)
            .unwrap().get("value_coupling_children").unwrap();
        assert_eq!(p2[0], 0.7);
    }

    #[test]
    fn test_set_coupling_vec_single_parent() {
        let mut net = make_test_network();

        set_coupling_vec(&mut net, &[1], &[0], 2.0);

        let child_couplings = net.attributes.vectors.get(&0)
            .unwrap().get("value_coupling_parents").unwrap();
        assert_eq!(child_couplings[0], 2.0); // parent 1
        assert_eq!(child_couplings[1], 1.0); // parent 2 untouched
    }

    #[test]
    fn test_set_coupling_vec_empty_parents() {
        let mut net = make_test_network();

        // Empty parents list — no-op
        set_coupling_vec(&mut net, &[], &[0], 5.0);

        let child_couplings = net.attributes.vectors.get(&0)
            .unwrap().get("value_coupling_parents").unwrap();
        assert_eq!(child_couplings, &vec![1.0, 1.0]);
    }

    #[test]
    fn test_set_coupling_vec_empty_children() {
        let mut net = make_test_network();

        // Empty children list — no-op
        set_coupling_vec(&mut net, &[1, 2], &[], 5.0);

        let c1 = net.attributes.vectors.get(&1)
            .unwrap().get("value_coupling_children").unwrap();
        assert_eq!(c1, &vec![1.0]);
    }

    #[test]
    fn test_set_coupling_vec_ignores_invalid_pairs() {
        let mut net = make_test_network();

        // Mix of valid (1→0) and invalid (1→2) pairs
        set_coupling_vec(&mut net, &[1], &[0, 2], 0.3);

        // Valid pair updated
        let child_couplings = net.attributes.vectors.get(&0)
            .unwrap().get("value_coupling_parents").unwrap();
        assert_eq!(child_couplings[0], 0.3);

        // Invalid pair: parent 1 has no edge to child 2 — no crash, no change
        let c2 = net.attributes.vectors.get(&2)
            .unwrap().get("value_coupling_children").unwrap();
        assert_eq!(c2, &vec![1.0]);
    }
}
