use crate::model::Network;
use crate::utils::set_coupling::set_coupling;

// =============================================================================
// Learning routines
// =============================================================================

/// Unified weights update.
///
/// Branches on the node's `"lr"` float attribute:
/// - **Fixed** (`lr` present): uses a fixed learning rate.
///   Δw_i = lr · (PE · π_child) · g(parent_i)
/// - **Dynamic** (`lr` absent): uses precision-based learning rate (Kalman gain).
///   K_i = π_parent_i / (π_parent_i + π_child).
///
/// Matches Python `pyhgf.updates.learning.learning_weights`.
pub fn learning_weights(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    // Gather the value parents and current coupling strengths for this child node.
    let value_parents = match network.edges.get(&node_idx)
        .and_then(|e| e.value_parents.clone())
    {
        Some(vp) => vp,
        None => return, // nothing to learn if no value parents
    };

    let value_couplings = network.attributes.vectors
        .get(&node_idx)
        .and_then(|v| v.get("value_coupling_parents").cloned())
        .unwrap_or_else(|| vec![1.0; value_parents.len()]);

    let child_mean = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("mean"))
        .unwrap_or(&0.0);

    let child_expected_mean = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("expected_mean"))
        .unwrap_or(&0.0);

    // Check whether a fixed learning rate is set on this node.
    let fixed_lr = network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("lr"))
        .copied();

    // Both paths need child precision.
    let child_precision = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("precision"))
        .unwrap_or(&1.0);

    let n_parents = value_parents.len();

    // 1. Compute prospective_activation g(posterior mean) for each parent,
    //    and (for the dynamic path) collect parent precisions.
    let mut prospective_activation = Vec::with_capacity(n_parents);
    let mut parent_precisions = Vec::with_capacity(n_parents);

    for (i, &parent_idx) in value_parents.iter().enumerate() {
        let parent_mean = *network.attributes.floats
            .get(&parent_idx)
            .and_then(|f| f.get("mean"))
            .unwrap_or(&0.0);

        if fixed_lr.is_none() {
            let parent_precision = *network.attributes.floats
                .get(&parent_idx)
                .and_then(|f| f.get("precision"))
                .unwrap_or(&1.0);
            parent_precisions.push(parent_precision);
        }

        // Find the coupling function for this parent–child pair
        let coupling_fn = network.attributes.fn_ptrs
            .get(&node_idx)
            .and_then(|fp| fp.get("value_coupling_fn_parents"))
            .and_then(|fns| fns.get(i).copied());

        let prosp_act = match coupling_fn {
            Some(cf) => (cf.f)(parent_mean),
            None => parent_mean,
        };

        prospective_activation.push(prosp_act);
    }

    // 2. Update coupling weights
    let pe = child_mean - child_expected_mean;

    for (i, &parent_idx) in value_parents.iter().enumerate() {
        let coupling = value_couplings[i];
        let prosp_act = prospective_activation[i];

        let new_value_coupling = match fixed_lr {
            // Fixed learning rate: Δw_i = lr · (PE · π_child) · g(parent_i)
            Some(lr) => coupling + lr * pe * child_precision * prosp_act,
            // Dynamic (precision-weighted)
            None => {
                let precision_weighting = parent_precisions[i]
                    / (parent_precisions[i] + child_precision);
                coupling + precision_weighting * pe * prosp_act
            }
        };

        // Guard against Inf/NaN
        let new_value_coupling = if new_value_coupling.is_infinite() || new_value_coupling.is_nan() {
            coupling
        } else {
            new_value_coupling
        };

        set_coupling(network, parent_idx, node_idx, new_value_coupling);
    }
}
