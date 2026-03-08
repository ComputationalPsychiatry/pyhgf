use crate::model::Network;
use crate::utils::set_coupling::set_coupling;

// =============================================================================
// Learning routines
// =============================================================================

/// Weights update using a fixed learning rate.
///
/// For every value parent of `node_idx`, this function:
/// 1. Reads the posterior `mean` and prior `expected_mean` of each parent.
/// 2. Applies the coupling function to get prospective and current activations.
/// 3. Computes the target coupling that would explain the child's observed mean
///    given the other parents' contributions.
/// 4. Moves the current coupling towards that target at rate `lr * weighting`.
///
/// The learning rate `lr` is read from the node's float attribute `"lr"` and
/// defaults to **0.01** when absent.
///
/// Matches Python `pyhgf.updates.learning.learning_weights_fixed`.
pub fn learning_weights_fixed(
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

    let lr = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("lr"))
        .unwrap_or(&0.01);

    let n_parents = value_parents.len();

    // 1. Compute prospective_activation g(posterior mean) for each parent
    let mut prospective_activation = Vec::with_capacity(n_parents);

    for (i, &parent_idx) in value_parents.iter().enumerate() {
        let parent_mean = *network.attributes.floats
            .get(&parent_idx)
            .and_then(|f| f.get("mean"))
            .unwrap_or(&0.0);

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

    // 2. Backprop-style delta rule: Δw_i = lr · PE · g(parent_i)
    let pe = child_mean - child_expected_mean;

    for (i, &parent_idx) in value_parents.iter().enumerate() {
        let coupling = value_couplings[i];
        let prosp_act = prospective_activation[i];

        let new_value_coupling = coupling + lr * pe * prosp_act;

        // Guard against Inf/NaN
        let new_value_coupling = if new_value_coupling.is_infinite() || new_value_coupling.is_nan() {
            coupling
        } else {
            new_value_coupling
        };

        set_coupling(network, parent_idx, node_idx, new_value_coupling);
    }
}

/// Dynamic weights update using precision-based learning rate.
///
/// Identical to [`learning_weights_fixed`] except the fixed `lr` is replaced by
/// a *precision weighting*:
///
///   w = π_child / (π_parent + π_child)
///
/// which makes the effective learning rate adaptive: the update is larger when
/// the child is relatively more precise than the parent.
///
/// Matches Python `pyhgf.updates.learning.learning_weights_dynamic`.
pub fn learning_weights_dynamic(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let value_parents = match network.edges.get(&node_idx)
        .and_then(|e| e.value_parents.clone())
    {
        Some(vp) => vp,
        None => return,
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

    let child_precision = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("precision"))
        .unwrap_or(&1.0);

    let n_parents = value_parents.len();

    // 1. Compute prospective_activation and parent_precisions for each parent
    let mut prospective_activation = Vec::with_capacity(n_parents);
    let mut parent_precisions = Vec::with_capacity(n_parents);

    for (i, &parent_idx) in value_parents.iter().enumerate() {
        let parent_mean = *network.attributes.floats
            .get(&parent_idx)
            .and_then(|f| f.get("mean"))
            .unwrap_or(&0.0);

        let parent_precision = *network.attributes.floats
            .get(&parent_idx)
            .and_then(|f| f.get("precision"))
            .unwrap_or(&1.0);

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
        parent_precisions.push(parent_precision);
    }

    // 2. Backprop-style delta rule: Δw_i = precision_weight_i · PE · g(parent_i)
    let pe = child_mean - child_expected_mean;

    for (i, &parent_idx) in value_parents.iter().enumerate() {
        let coupling = value_couplings[i];
        let prosp_act = prospective_activation[i];

        // Precision weighting: π_child / (π_parent + π_child)
        let precision_weighting = child_precision
            / (parent_precisions[i] + child_precision);

        let new_value_coupling = coupling + precision_weighting * pe * prosp_act;

        // Guard against Inf/NaN
        let new_value_coupling = if new_value_coupling.is_infinite() || new_value_coupling.is_nan() {
            coupling
        } else {
            new_value_coupling
        };

        set_coupling(network, parent_idx, node_idx, new_value_coupling);
    }
}
