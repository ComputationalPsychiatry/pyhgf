use crate::model::Network;

// =============================================================================
// Value level updates
// =============================================================================

/// Posterior-step (smoothing) correction of the relaxed HGF in its fully-corrected
/// form, paired with the prediction-step marginal-precision correction.
///
/// Lifting the mean-field assumption `q(x_a, x_b) = q(x_a) q(x_b)` on the
/// value-coupling edge to a structured Gaussian and applying the Schur complement to
/// the joint `(x_a, x_b)` precision matrix replaces the canonical child-precision
/// factor by the harmonic combination
///
/// ```text
/// π̂_a · π_y / (π̂_a + π_y),    π_y = π_a − π̃_a,
/// ```
///
/// where `π̂_a` is the child's *conditional* predicted precision
/// (`conditional_expected_precision`: own variance plus volatility, without the
/// parent-uncertainty bleed-through term) and `π̃_a` its *marginal* predicted
/// precision (`expected_precision`). The Schur complement carries the conditional;
/// substituting the marginal would double-count parent uncertainty. The same factor
/// scales both the `(κ g')²` and `κ g'' · δ_a` contributions to π_b. Non-Gaussian
/// children and Gaussian leaves fall back to the canonical predicted-precision
/// factor `π̃_a` — pyhgf keeps `precision = expected_precision` for clamped
/// observations, so π_y = 0 and the harmonic form would otherwise zero out their
/// contribution.
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

            // Effective child precision under the smoothing correction. The
            // Schur derivation assumes a Gaussian-Gaussian value-coupling edge,
            // so the correction only applies when the child carries a Gaussian
            // belief (continuous-state or volatile-state) AND is interior (has
            // children of its own). Binary, categorical, input/constant,
            // exponential-family, and Dirichlet children, plus any Gaussian leaf,
            // fall back to the canonical predicted-precision factor π̃_a
            // (= `child.expected_precision`).
            let child_node_type = network.edges[child_idx].node_type.as_str();
            let child_is_gaussian_interior =
                matches!(child_node_type, "continuous-state" | "volatile-state")
                    && (network.edges[child_idx].value_children.is_some()
                        || network.edges[child_idx].volatility_children.is_some());
            let effective_child_precision = if child_is_gaussian_interior {
                // Bottom-up evidence precision π_y = π_a − π̃_a, measured against the
                // child's *marginal* predicted precision π̃_a.
                let child_precision = child_state.precision;
                let pi_y = child_precision - child_expected_precision;
                // The Schur complement carries the *conditional* predicted precision
                // π̂_a. Both volatile- and continuous-state children store it (a
                // gaussian-interior child is necessarily one of those two types). Using
                // the marginal would double-count parent uncertainty.
                let child_cond = child_state.conditional_expected_precision;
                child_cond * pi_y / (child_cond + pi_y)
            } else {
                child_expected_precision
            };

            posterior_precision += effective_child_precision
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

            // Coupling gain precision g_a. From the joint (x_a, x_b) Gaussian the
            // exact marginal-mean gain is
            //     g_a = π̂_a · π_a / (π̂_a + π_y),    π_y = π_a − π̃_a,
            // summed over children and divided once by the parent posterior precision
            // (`node_precision`) — this is what makes the multi-child mean exact rather
            // than a sum of independent RTS gains. For leaves / non-Gaussian children
            // π_y = 0 and g_a collapses to the marginal π̃_a, recovering the canonical
            // gain.
            let child_node_type = network.edges[child_idx].node_type.as_str();
            let child_is_gaussian_interior =
                matches!(child_node_type, "continuous-state" | "volatile-state")
                    && (network.edges[child_idx].value_children.is_some()
                        || network.edges[child_idx].volatility_children.is_some());
            let gain_precision = if child_is_gaussian_interior {
                let child_precision = child_state.precision;
                let pi_y = child_precision - child_expected_precision;
                let child_cond = child_state.conditional_expected_precision;
                child_cond * child_precision / (child_cond + pi_y)
            } else {
                child_expected_precision
            };

            value_pwpe += (kappa * coupling_fn_prime * gain_precision / node_precision)
                * child_state.value_prediction_error;
        }
    }

    expected_mean + value_pwpe
}

// =============================================================================
// Posterior update
// =============================================================================

pub fn posterior_update_volatile_state_node(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    // POSTERIOR UPDATE VALUE LEVEL
    let precision_value =
        precision_update_value_level(network, node_idx).min(network.max_posterior_precision);
    network.attributes.states[node_idx].precision = precision_value;

    let mean_value = mean_update_value_level(network, node_idx, precision_value);
    network.attributes.states[node_idx].mean = mean_value;
}

// =============================================================================
// Mean-field value-level building blocks
// =============================================================================

fn precision_update_value_level_mean_field(network: &Network, node_idx: usize) -> f64 {
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

fn mean_update_value_level_mean_field(
    network: &Network,
    node_idx: usize,
    node_precision: f64,
) -> f64 {
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

            value_pwpe += (kappa * coupling_fn_prime * child_expected_precision / node_precision)
                * child_state.value_prediction_error;
        }
    }

    expected_mean + value_pwpe
}

// =============================================================================
// Mean-field volatile posterior update
// =============================================================================

pub fn posterior_update_volatile_state_node_mean_field(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let precision_value = precision_update_value_level_mean_field(network, node_idx)
        .min(network.max_posterior_precision);
    network.attributes.states[node_idx].precision = precision_value;

    let mean_value = mean_update_value_level_mean_field(network, node_idx, precision_value);
    network.attributes.states[node_idx].mean = mean_value;
}
