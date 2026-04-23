use crate::model::Network;

// =============================================================================
// Value level updates
// =============================================================================

fn precision_update_value_level(network: &Network, node_idx: usize) -> f64 {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let mut posterior_precision = expected_precision;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_expected_mean = network.attributes.states[node_idx].expected_mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_expected_mean);
                    let g_second = (cf.d2f)(parent_expected_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), kappa * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            posterior_precision += child_expected_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term);
        }
    }

    posterior_precision
}

fn mean_update_value_level(network: &Network, node_idx: usize, node_precision: f64) -> f64 {
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let mut value_pwpe = 0.0;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_expected_mean = network.attributes.states[node_idx].expected_mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let coupling_fn_prime = match coupling_fn {
                Some(cf) => (cf.df)(parent_expected_mean),
                None => 1.0,
            };

            value_pwpe += (kappa * coupling_fn_prime * child_expected_precision / node_precision) * child_state.value_prediction_error;
        }
    }

    expected_mean + value_pwpe
}

// =============================================================================
// Posterior update
// =============================================================================

pub fn posterior_update_volatile_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {
    // POSTERIOR UPDATE VALUE LEVEL
    let precision_value = precision_update_value_level(network, node_idx);
    network.attributes.states[node_idx].precision = precision_value;

    let mean_value = mean_update_value_level(network, node_idx, precision_value);
    network.attributes.states[node_idx].mean = mean_value;
}