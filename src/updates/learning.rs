use crate::model::Network;
use crate::utils::set_coupling::set_coupling;

/// Unified weights update.
///
/// Computes a gradient according to `learning_kind` (standard /
/// precision_weighted / precision_ratio), then scales it by `lr` uniformly.
/// When Adam state is present, the gradient is filtered through Adam instead.
pub fn learning_weights(network: &mut Network, node_idx: usize, _time_step: f64) {
    let is_binary = network.edges[node_idx].node_type == "binary-state";

    let n_parents = match &network.edges[node_idx].value_parents {
        Some(vp) => vp.len(),
        None => return,
    };

    // --- read-only phase: copy all the scalars we need ---------------
    let child_mean = network.attributes.states[node_idx].mean;
    let child_expected_mean = network.attributes.states[node_idx].expected_mean;
    let child_precision = network.attributes.states[node_idx].precision;

    let lr_val = network.attributes.states[node_idx].lr;
    // NaN lr means "no lr set" → skip update for this node.
    if lr_val.is_nan() {
        return;
    }

    let learning_kind = network.edges[node_idx].learning_kind.clone();

    let pe = child_mean - child_expected_mean;

    // --- per-parent update -------------------------------------------
    for i in 0..n_parents {
        let parent_idx = network.edges[node_idx].value_parents.as_ref().unwrap()[i];

        let coupling = network.attributes.vectors[node_idx]
            .value_coupling_parents
            .get(i)
            .copied()
            .unwrap_or(1.0);

        let parent_mean = network.attributes.states[parent_idx].mean;

        // Binary nodes use sigmoid coupling for the weight update.
        let prosp_act = if is_binary {
            crate::math::sigmoid(parent_mean)
        } else {
            let coupling_fn = network.attributes.fn_ptrs[parent_idx].coupling_fn;
            match coupling_fn {
                Some(cf) => (cf.f)(parent_mean),
                None => parent_mean,
            }
        };

        // Compute the gradient according to learning_kind.
        // Binary nodes skip precision multiplication — the Bernoulli
        // variance is already embedded in the binary prediction-error formula.
        let gradient = if learning_kind == "precision_ratio" {
            let parent_precision = network.attributes.states[parent_idx].precision;
            let kalman_gain = child_precision / (parent_precision + child_precision);
            kalman_gain * pe * prosp_act
        } else if learning_kind == "standard" || is_binary {
            pe * prosp_act
        } else {
            // "precision_weighted" (default)
            pe * child_precision * prosp_act
        };

        // Apply lr uniformly: Adam filter if state is present, otherwise
        // direct scaling by lr_val.
        let new_value_coupling = if let Some(ref mut adam) = network.adam_state {
            let effective_lr = adam.lr.unwrap_or(lr_val);
            let adam_update = adam.step(node_idx, i, gradient, effective_lr);
            coupling + adam_update
        } else {
            coupling + lr_val * gradient
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
