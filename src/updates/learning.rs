use crate::model::Network;
use crate::utils::set_coupling::set_coupling;

/// Unified weights update.
pub fn learning_weights(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let value_parents = match network.edges[node_idx].value_parents.clone() {
        Some(vp) => vp,
        None => return,
    };

    let value_couplings = network.attributes.vectors[node_idx].value_coupling_parents.clone();
    let child_mean = network.attributes.states[node_idx].mean;
    let child_expected_mean = network.attributes.states[node_idx].expected_mean;
    let child_precision = network.attributes.states[node_idx].precision;

    // Check whether a fixed learning rate is set (NaN = not set)
    let lr_val = network.attributes.states[node_idx].lr;
    let fixed_lr = if lr_val.is_nan() { None } else { Some(lr_val) };

    let n_parents = value_parents.len();

    // 1. Compute prospective activation and parent precisions
    let mut prospective_activation = Vec::with_capacity(n_parents);
    let mut parent_precisions = Vec::with_capacity(n_parents);

    for &parent_idx in value_parents.iter() {
        let parent_mean = network.attributes.states[parent_idx].mean;

        if fixed_lr.is_none() {
            let parent_precision = network.attributes.states[parent_idx].precision;
            parent_precisions.push(parent_precision);
        }

        let coupling_fn = network.attributes.fn_ptrs[parent_idx].coupling_fn;

        let prosp_act = match coupling_fn {
            Some(cf) => (cf.f)(parent_mean),
            None => parent_mean,
        };

        prospective_activation.push(prosp_act);
    }

    // 2. Update coupling weights
    let pe = child_mean - child_expected_mean;

    for (i, &parent_idx) in value_parents.iter().enumerate() {
        let coupling = value_couplings.get(i).copied().unwrap_or(1.0);
        let prosp_act = prospective_activation[i];

        let new_value_coupling = match fixed_lr {
            Some(lr) => coupling + lr * pe * child_precision * prosp_act,
            None => {
                let precision_weighting = parent_precisions[i]
                    / (parent_precisions[i] + child_precision);
                coupling + precision_weighting * pe * prosp_act
            }
        };

        let new_value_coupling = if new_value_coupling.is_infinite() || new_value_coupling.is_nan() {
            coupling
        } else {
            new_value_coupling
        };

        set_coupling(network, parent_idx, node_idx, new_value_coupling);
    }
}
