use crate::model::Network;
use crate::utils::set_coupling::set_coupling;

/// Unified weights update.
pub fn learning_weights(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let n_parents = match &network.edges[node_idx].value_parents {
        Some(vp) => vp.len(),
        None => return,
    };

    // --- read-only phase: copy all the scalars we need ---------------
    let child_mean = network.attributes.states[node_idx].mean;
    let child_expected_mean = network.attributes.states[node_idx].expected_mean;
    let child_precision = network.attributes.states[node_idx].precision;

    let lr_val = network.attributes.states[node_idx].lr;
    let fixed_lr = if lr_val.is_nan() { None } else { Some(lr_val) };

    let pe = child_mean - child_expected_mean;

    // --- per-parent update (no temporary Vecs) -----------------------
    for i in 0..n_parents {
        // Index into the parent list without cloning the Vec.
        let parent_idx = network.edges[node_idx]
            .value_parents
            .as_ref()
            .unwrap()[i];

        let coupling = network.attributes.vectors[node_idx]
            .value_coupling_parents
            .get(i)
            .copied()
            .unwrap_or(1.0);

        let parent_mean = network.attributes.states[parent_idx].mean;

        let coupling_fn = network.attributes.fn_ptrs[parent_idx].coupling_fn;
        let prosp_act = match coupling_fn {
            Some(cf) => (cf.f)(parent_mean),
            None => parent_mean,
        };

        let new_value_coupling = match fixed_lr {
            Some(lr) => {
                let gradient = pe * child_precision * prosp_act;

                // If Adam state is available, filter the gradient through Adam
                if let Some(ref mut adam) = network.adam_state {
                    let effective_lr = adam.lr.unwrap_or(lr);
                    let adam_update = adam.step(node_idx, i, gradient, effective_lr);
                    coupling + adam_update
                } else {
                    coupling + lr * gradient
                }
            }
            None => {
                let parent_precision = network.attributes.states[parent_idx].precision;
                let precision_weighting =
                    parent_precision / (parent_precision + child_precision);
                coupling + precision_weighting * pe * prosp_act
            }
        };

        let new_value_coupling = if new_value_coupling.is_infinite() || new_value_coupling.is_nan()
        {
            coupling
        } else {
            new_value_coupling
        };

        set_coupling(network, parent_idx, node_idx, new_value_coupling);
    }
}
