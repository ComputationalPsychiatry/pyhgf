use crate::math::{lambert_w0, logaddexp};
use crate::model::network::Network;

// =============================================================================
// Shared building blocks for precision and mean updates
// =============================================================================

/// Child-precision factors of the posterior-step relaxation for one value child.
///
/// Returns `(π_eff, g_a)`:
///
/// ```text
/// π_eff = π̄_a · π_y / (π̄_a + π_y),    g_a = π̄_a · π_a / (π̄_a + π_y),    π_y = π_a − π̂_a,
/// ```
///
/// where π̄_a is the child's conditional predicted precision
/// (`conditional_expected_precision`) and π̂_a its marginal predicted precision
/// (`expected_precision`). `π_eff` (the relaxed child precision) weights the
/// `(α g')²` term of the precision increment; `g_a` (the relaxed gain) weights its
/// `α g'' · δ_a` term and the mean increment.
///
/// The relaxation applies when the child carries a Gaussian belief
/// (continuous-state); the Schur derivation assumes a Gaussian-Gaussian
/// value-coupling edge. An interior child (one with children of its own) uses the
/// two factors above. A leaf is observed exactly: its stored `precision` is the
/// observation precision, not a posterior, so both factors take the fully observed
/// limit π_y → ∞, π_eff = g_a = π̄_a. For a leaf without a volatility parent the
/// prediction step sets π̄_a = π̂_a, which gives the canonical update. Binary,
/// categorical, input/constant, exponential-family and Dirichlet children use π̂_a
/// for both factors. With `mean_field`, every child uses π̂_a for both factors.
fn relaxed_factors(network: &Network, child_idx: usize, mean_field: bool) -> (f64, f64) {
    let child_state = &network.attributes.states[child_idx];
    let child_expected_precision = child_state.expected_precision;
    let child_is_gaussian = network.edges[child_idx].node_type.as_str() == "continuous-state";
    if mean_field || !child_is_gaussian {
        return (child_expected_precision, child_expected_precision);
    }

    // The Schur complement carries the *conditional* predicted precision π̄_a
    // (stored on the child). Using the marginal would double-count parent
    // uncertainty.
    let child_cond = child_state.conditional_expected_precision;
    let child_is_leaf = network.edges[child_idx].value_children.is_none()
        && network.edges[child_idx].volatility_children.is_none();
    if child_is_leaf {
        // Observed leaf: the fully observed limit π_y → ∞ of both factors.
        return (child_cond, child_cond);
    }

    let child_precision = child_state.precision;
    let pi_y = child_precision - child_expected_precision;
    let denominator = child_cond + pi_y;
    (
        child_cond * pi_y / denominator,
        child_cond * child_precision / denominator,
    )
}

/// Precision increment from the value children.
///
/// The relaxed update implements the posterior-step relaxation of the relaxed HGF
/// (rHGF). Each child contributes
///
/// ```text
/// π̄_a / (π̄_a + π_y) · (α² g'² · π_y − α g'' · π_a · δ_a),    π_y = π_a − π̂_a,
/// ```
///
/// with α the value-coupling strength: the relaxed child precision weights the
/// `(α g')²` term and the relaxed gain of the mean increment weights the
/// `α g'' · δ_a` term (see `relaxed_factors`). Reduces to the canonical formula
/// when the child is fully observed (π_y → ∞). When π̂_a = π̄_a, the relaxed gain
/// equals π̂_a and the g'' term is the canonical one.
fn value_coupling_precision_wpe(network: &Network, node_idx: usize, mean_field: bool) -> f64 {
    let mut precision_wpe = 0.0;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let observed = child_state.observed;
            let alpha = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), alpha * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            let (effective_child_precision, gain_precision) =
                relaxed_factors(network, child_idx, mean_field);

            precision_wpe += (effective_child_precision * alpha.powi(2) * coupling_fn_prime_sq
                - gain_precision * coupling_fn_second_term)
                * observed;
        }
    }

    precision_wpe
}

/// Mean increment from the value children.
///
/// The relaxed update uses the relaxed gain of the posterior-step relaxation (the
/// joint-Gaussian Rauch-Tung-Striebel smoother gain):
///
/// ```text
/// g_a = π̄_a · π_a / (π̄_a + π_y),    π_y = π_a − π̂_a,
/// ```
///
/// accumulated across children and divided once by `node_precision` (π_b), which
/// makes the multi-child mean exact rather than a sum of independent single-child
/// RTS gains. A Gaussian leaf takes the fully observed limit π_y → ∞, g_a = π̄_a;
/// non-Gaussian children use the canonical gain π̂_a (= `child.expected_precision`).
/// With `mean_field`, every child uses π̂_a.
fn value_coupling_mean_wpe(
    network: &Network,
    node_idx: usize,
    node_precision: f64,
    mean_field: bool,
) -> f64 {
    let mut value_pwpe = 0.0;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_vape = child_state.value_prediction_error * child_state.observed;
            let alpha = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let coupling_fn_prime = match coupling_fn {
                Some(cf) => (cf.df)(parent_mean),
                None => 1.0,
            };

            // Relaxed gain g_a, summed over children then divided once by
            // node_precision (exact multi-child marginal mean); see
            // `relaxed_factors` for leaves and non-Gaussian children.
            let (_, gain_precision) = relaxed_factors(network, child_idx, mean_field);

            value_pwpe +=
                (alpha * coupling_fn_prime * gain_precision / node_precision) * child_vape;
        }
    }

    value_pwpe
}

/// Compute the precision update contribution from value (relaxed) and volatility
/// children.
fn precision_update_from_children(network: &Network, node_idx: usize) -> f64 {
    let mut precision_wpe = value_coupling_precision_wpe(network, node_idx, false);

    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let effective_precision = child_state.effective_precision;
            let volatility_pe = child_state.volatility_prediction_error;
            let observed = child_state.observed;
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);

            precision_wpe += (0.5 * (kappa * effective_precision).powi(2)
                + (kappa * effective_precision).powi(2) * volatility_pe
                - 0.5 * kappa.powi(2) * effective_precision * volatility_pe)
                * observed;
        }
    }

    precision_wpe
}

/// Compute the mean update contribution from value (relaxed) and volatility children.
fn mean_update_from_children(network: &Network, node_idx: usize, node_precision: f64) -> f64 {
    let value_pwpe = value_coupling_mean_wpe(network, node_idx, node_precision, false);
    let mut volatility_pwpe = 0.0;

    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let effective_precision = child_state.effective_precision;
            let volatility_pe = child_state.volatility_prediction_error;
            let observed = child_state.observed;
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);

            volatility_pwpe +=
                (kappa * effective_precision * volatility_pe) / (2.0 * node_precision) * observed;
        }
    }

    value_pwpe + volatility_pwpe
}

/// Whether any value or volatility child of the node reported an observation (mirrors
/// `_has_observations` in the JAX backend). A continuous state node with children
/// counts as observed when at least one of its own children was (see
/// `prediction_error_continuous_state_node`).
pub(crate) fn has_observations(network: &Network, node_idx: usize) -> bool {
    let edges = &network.edges[node_idx];
    edges
        .value_children
        .iter()
        .chain(edges.volatility_children.iter())
        .flatten()
        .any(|&child_idx| network.attributes.states[child_idx].observed != 0.0)
}

/// Posterior precision: the predicted precision plus the children's increment or, when
/// no child was observed, the predicted precision alone: without evidence the
/// posterior is the prediction (mirrors `precision_update_missing_values` in the JAX
/// backend).
fn observed_or_predicted_precision(
    network: &Network,
    node_idx: usize,
    precision_wpe: impl FnOnce() -> f64,
) -> f64 {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    if has_observations(network, node_idx) {
        (expected_precision + precision_wpe()).max(1e-128)
    } else {
        expected_precision
    }
    .min(network.max_posterior_precision)
}

/// Effective tonic volatility ω̃_a = ω_a + Σ_{p≠b} κ_p μ̂_p of the volatility child
/// `child_idx`, seen from its volatility parent `node_idx` (b). The volatility updates
/// of b treat the predicted volatility of the child as a function of x_b alone; the
/// other volatility parents p of the child enter through their predicted means, as in
/// the prediction of the child (mirrors `_effective_tonic_volatility` in the JAX
/// backend).
fn effective_tonic_volatility(network: &Network, node_idx: usize, child_idx: usize) -> f64 {
    let mut tonic_volatility = network.attributes.states[child_idx].tonic_volatility;
    if let Some(ref parent_idxs) = network.edges[child_idx].volatility_parents {
        let parent_couplings = &network.attributes.vectors[child_idx].volatility_coupling_parents;
        for (j, &parent_idx) in parent_idxs.iter().enumerate() {
            if parent_idx != node_idx {
                tonic_volatility += parent_couplings.get(j).copied().unwrap_or(1.0)
                    * network.attributes.states[parent_idx].expected_mean;
            }
        }
    }
    tonic_volatility
}

// =============================================================================
// Standard posterior update
// =============================================================================

pub fn posterior_update_continuous_state_node(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let expected_mean = network.attributes.states[node_idx].expected_mean;

    let posterior_precision = observed_or_predicted_precision(network, node_idx, || {
        precision_update_from_children(network, node_idx)
    });

    let mean_wpe = mean_update_from_children(network, node_idx, posterior_precision);
    let posterior_mean = expected_mean + mean_wpe;

    let state = &mut network.attributes.states[node_idx];
    state.precision = posterior_precision;
    state.mean = posterior_mean;
}

// =============================================================================
// eHGF posterior update
// =============================================================================

pub fn posterior_update_continuous_state_node_ehgf(
    network: &mut Network,
    node_idx: usize,
    time_step: f64,
) {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let expected_mean = network.attributes.states[node_idx].expected_mean;

    let mean_wpe = mean_update_from_children(network, node_idx, expected_precision);
    let posterior_mean = expected_mean + mean_wpe;
    network.attributes.states[node_idx].mean = posterior_mean;

    // eHGF safe precision update: recompute the effective precision from the
    // posterior mean and floor the volatility increment at zero.
    let posterior_precision = observed_or_predicted_precision(network, node_idx, || {
        precision_update_from_children_ehgf(network, node_idx, time_step)
    });
    network.attributes.states[node_idx].precision = posterior_precision;
}

// =============================================================================
// Unbounded posterior update
// =============================================================================

/// Quantities of one volatility child entering the unbounded update.
struct UnboundedVolatilityChild {
    coupling: f64,
    /// Effective tonic volatility ω̃, which absorbs the other volatility parents.
    tonic_volatility: f64,
    /// Log of the carried variance λ²/π the child used in its prediction.
    log_previous_variance: f64,
    /// Total posterior uncertainty of the child, 1/π + (μ − μ̂)².
    be_aux: f64,
    /// Volatility prediction error of the child, evaluated at the prediction μ̂.
    volatility_prediction_error: f64,
}

impl UnboundedVolatilityChild {
    /// Log of the predicted volatility Ω(x) = t exp(κ x + ω̃) of the child.
    fn log_volatility(&self, log_time_step: f64, x: f64) -> f64 {
        log_time_step + self.coupling * x + self.tonic_volatility
    }
}

/// Unbounded posterior update of a continuous state node with one or several volatility
/// children and, optionally, value children (section 3.6 in Mathys et al., 2026,
/// arXiv:2605.00966). Port of `posterior_update_unbounded` in the JAX backend.
///
/// With `n` volatility children, the variational energy of the node is, up to an
/// additive constant,
///
/// ```text
/// I(x) = Σ_i [ −½ log(α_i + Ω_i(x)) − ½ β_i / (α_i + Ω_i(x)) ] − ½ π̃ (x − μ̃)²,
/// Ω_i(x) = t exp(κ_i x + ω̃_i),    ω̃_i = ω_i + Σ_p κ_p μ̂_p,
/// ```
///
/// where (π̃, μ̃) is the prediction of the node or, when the node also has value
/// children, the effective prior returned by [`unbounded_effective_prior`], and the
/// effective tonic volatility ω̃_i absorbs the other volatility parents p of the child.
/// Unobserved children are left out of the sum. The posterior is the moment-matched Gaussian of n + 1 quadratic
/// expansions of I: one at the prediction μ̂ (concave part only, eqs. 52-53), and one
/// at the Lambert W0 mode of each child, with the curvature and gradient of the full
/// energy (eqs. 54-56). The expansions are weighted by a softmax of I at their means
/// (eqs. 57-60). For a single volatility child and no value child, this is the
/// two-expansion update of section 3.4. If no child is observed, the posterior
/// precision is the predicted precision.
pub fn posterior_update_continuous_state_node_unbounded(
    network: &mut Network,
    node_idx: usize,
    time_step: f64,
) {
    let volatility_children_idxs = network.edges[node_idx]
        .volatility_children
        .as_ref()
        .expect("No volatility children found");
    let volatility_couplings = &network.attributes.vectors[node_idx].volatility_coupling_children;
    let log_time_step = time_step.ln();
    let expected_mean = network.attributes.states[node_idx].expected_mean;

    // Effective prior (π̃, μ̃): the prediction, combined with the value-coupling terms
    // when the node also has value children.
    let (prior_precision, prior_mean) = unbounded_effective_prior(network, node_idx);

    let children: Vec<UnboundedVolatilityChild> = volatility_children_idxs
        .iter()
        .enumerate()
        .filter(|&(_, &child_idx)| network.attributes.states[child_idx].observed != 0.0)
        .map(|(i, &child_idx)| {
            let child_state = &network.attributes.states[child_idx];
            let be_aux = (1.0 / child_state.precision)
                + (child_state.mean - child_state.expected_mean).powi(2);
            UnboundedVolatilityChild {
                coupling: volatility_couplings.get(i).copied().unwrap_or(1.0),
                tonic_volatility: effective_tonic_volatility(network, node_idx, child_idx),
                log_previous_variance: child_state.current_variance.max(1e-128).ln(),
                be_aux,
                volatility_prediction_error: child_state.expected_precision * be_aux - 1.0,
            }
        })
        .collect();

    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());

    // Variational energy I(x), in log-space so that it stays finite when Ω(x) overflows.
    let variational_energy = |x: f64| {
        let mut energy = -0.5 * prior_precision * (x - prior_mean).powi(2);
        for child in &children {
            let log_denom = logaddexp(
                child.log_previous_variance,
                child.log_volatility(log_time_step, x),
            );
            energy = energy - 0.5 * log_denom - 0.5 * child.be_aux * (-log_denom).exp();
        }
        energy
    };

    // Expansion 1: quadratic at the prediction, concave part only.
    let mut precision_1 = prior_precision;
    let mut mean_shift_1 = prior_precision * (prior_mean - expected_mean);
    for child in &children {
        let w = sigmoid(
            child.log_volatility(log_time_step, expected_mean) - child.log_previous_variance,
        );
        precision_1 = precision_1 + 0.5 * child.coupling.powi(2) * w * (1.0 - w);
        mean_shift_1 = mean_shift_1 + 0.5 * child.coupling * w * child.volatility_prediction_error;
    }
    let mean_1 = expected_mean + mean_shift_1 / precision_1;

    // Expansions 2, ..., n+1: quadratic at each child's Lambert W0 mode.
    let mut precisions = Vec::with_capacity(children.len() + 1);
    let mut means = Vec::with_capacity(children.len() + 1);
    precisions.push(precision_1);
    means.push(mean_1);
    for child in &children {
        // A zero coupling makes the child independent of the node: no expansion.
        if child.coupling == 0.0 {
            continue;
        }
        let pihat_y = prior_precision / child.coupling.powi(2);
        let gamma_c = log_time_step + child.coupling * prior_mean + child.tonic_volatility;
        let log_w_arg = child.be_aux.ln() - (2.0 * pihat_y).ln() + 0.5 / pihat_y - gamma_c;
        let w_arg = log_w_arg.min(f64::MAX.ln()).exp();
        let y_star = gamma_c + lambert_w0(w_arg) - 0.5 / pihat_y;
        let x_star = (y_star - log_time_step - child.tonic_volatility) / child.coupling;

        // Curvature and gradient of the full variational energy at x_star, which
        // includes all the children.
        let mut full_curvature = prior_precision;
        let mut concave_curvature = prior_precision;
        let mut gradient = -prior_precision * (x_star - prior_mean);
        for other in &children {
            let log_s = other.log_volatility(log_time_step, x_star);
            let w = sigmoid(log_s - other.log_previous_variance);
            let da = other.be_aux * (-logaddexp(other.log_previous_variance, log_s)).exp() - 1.0;
            full_curvature =
                full_curvature + 0.5 * other.coupling.powi(2) * w * (w + (2.0 * w - 1.0) * da);
            concave_curvature = concave_curvature + 0.5 * other.coupling.powi(2) * w * (1.0 - w);
            gradient = gradient + 0.5 * other.coupling * w * da;
        }

        let precision_2 = if full_curvature <= 0.0 {
            concave_curvature
        } else {
            full_curvature
        };
        let mean_2 = x_star + gradient / precision_2;

        // Drop this expansion if it is not finite (zero weight in the JAX backend).
        if precision_2.is_finite() && mean_2.is_finite() {
            precisions.push(precision_2);
            means.push(mean_2);
        }
    }

    // Softmax weights and Gaussian mixture moment matching.
    let energies: Vec<f64> = means.iter().map(|&m| variational_energy(m)).collect();
    let max_energy = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let unnormalised: Vec<f64> = energies.iter().map(|&e| (e - max_energy).exp()).collect();
    let total: f64 = unnormalised.iter().sum();
    let weights: Vec<f64> = unnormalised.iter().map(|&u| u / total).collect();

    let posterior_mean: f64 = weights.iter().zip(&means).map(|(w, m)| w * m).sum();
    let posterior_variance: f64 = weights
        .iter()
        .zip(precisions.iter().zip(&means))
        .map(|(w, (p, m))| w * (1.0 / p + (m - posterior_mean).powi(2)))
        .sum();

    let posterior_precision = if has_observations(network, node_idx) {
        1.0 / posterior_variance
    } else {
        network.attributes.states[node_idx].expected_precision
    }
    .min(network.max_posterior_precision);
    let state = &mut network.attributes.states[node_idx];
    state.precision = posterior_precision;
    state.mean = posterior_mean;
}

/// Effective prior (π̃, μ̃) of the unbounded update, in which section 3.6 of Mathys et
/// al. (2026) absorbs the value-coupling terms of the variational energy.
///
/// It is the posterior that the regular value-coupling update returns when the
/// volatility children are left out: relaxed (posterior-step relaxation for Gaussian
/// interior children) or mean-field, following `Network::mean_field_updates`, so that value
/// children are weighted as in the standard and eHGF updates. Without value children,
/// this is the prediction of the node. Port of `_value_children_effective_prior` in the
/// JAX backend.
fn unbounded_effective_prior(network: &Network, node_idx: usize) -> (f64, f64) {
    let state = &network.attributes.states[node_idx];
    if network.edges[node_idx].value_children.is_none() {
        return (state.expected_precision, state.expected_mean);
    }
    let mean_field = network.mean_field_updates;
    // Lower bound as in `posterior_update_continuous_state_node`; the upper bound
    // applies to the final posterior only.
    let precision = (state.expected_precision
        + value_coupling_precision_wpe(network, node_idx, mean_field))
    .max(1e-128);
    let mean =
        state.expected_mean + value_coupling_mean_wpe(network, node_idx, precision, mean_field);
    (precision, mean)
}

// =============================================================================
// Mean-field (v0.2.11) building blocks
// =============================================================================

/// Mean-field precision update from children.
///
/// Uses `expected_precision` directly as the child-precision factor.
fn precision_update_from_children_mean_field(network: &Network, node_idx: usize) -> f64 {
    let mut precision_wpe = value_coupling_precision_wpe(network, node_idx, true);

    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let effective_precision = child_state.effective_precision;
            let volatility_pe = child_state.volatility_prediction_error;
            let observed = child_state.observed;
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);

            precision_wpe += (0.5 * (kappa * effective_precision).powi(2)
                + (kappa * effective_precision).powi(2) * volatility_pe
                - 0.5 * kappa.powi(2) * effective_precision * volatility_pe)
                * observed;
        }
    }

    precision_wpe
}

/// Enhanced-HGF "safe" volatility-coupling precision increment for one child.
///
/// Recomputes the effective precision from the parent's just-updated posterior mean
/// (`mean`) and the elapsed time, then floors the increment at zero, matching the
/// enhanced HGF volatility update (TAPAS `hgf_volatility_update` `'ehgf'` branch). This
/// guarantees the posterior precision never drops below the predicted precision. The
/// re-predicted volatility is ν = t exp(κ μ + ω̃), with `tonic_volatility` the effective
/// tonic volatility ω̃ from [`effective_tonic_volatility`].
fn ehgf_volatility_increment(
    child_state: &crate::model::network::NodeState,
    volatility_coupling: f64,
    mean: f64,
    tonic_volatility: f64,
    time_step: f64,
) -> f64 {
    // Carried variance σ = λ²/π the child used; the re-prediction below must add
    // volatility to the same quantity the prediction step used.
    let previous_variance = child_state.current_variance;
    // Re-predict the child's volatility and precision from the parent posterior mean.
    let predicted_volatility = time_step * (volatility_coupling * mean + tonic_volatility).exp();
    let expected_precision = 1.0 / (previous_variance + predicted_volatility);
    let effective_precision = predicted_volatility * expected_precision;
    let volatility_error_weight = (predicted_volatility - previous_variance) * expected_precision;
    let volatility_prediction_error = (1.0 / child_state.precision
        + (child_state.mean - child_state.expected_mean).powi(2))
        * expected_precision
        - 1.0;
    (0.5 * volatility_coupling.powi(2)
        * effective_precision
        * (effective_precision + volatility_error_weight * volatility_prediction_error))
        .max(0.0)
        * child_state.observed
}

/// Enhanced-HGF precision update from children (relaxed value coupling).
fn precision_update_from_children_ehgf(network: &Network, node_idx: usize, time_step: f64) -> f64 {
    let mut precision_wpe = value_coupling_precision_wpe(network, node_idx, false);

    // --- Volatility coupling (eHGF safe update) ---
    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);
            let tonic_volatility = effective_tonic_volatility(network, node_idx, child_idx);
            precision_wpe += ehgf_volatility_increment(
                child_state,
                kappa,
                parent_mean,
                tonic_volatility,
                time_step,
            );
        }
    }

    precision_wpe
}

/// Enhanced-HGF precision update from children (mean-field value coupling).
fn precision_update_from_children_ehgf_mean_field(
    network: &Network,
    node_idx: usize,
    time_step: f64,
) -> f64 {
    let mut precision_wpe = value_coupling_precision_wpe(network, node_idx, true);

    // --- Volatility coupling (eHGF safe update) ---
    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);
            let tonic_volatility = effective_tonic_volatility(network, node_idx, child_idx);
            precision_wpe += ehgf_volatility_increment(
                child_state,
                kappa,
                parent_mean,
                tonic_volatility,
                time_step,
            );
        }
    }

    precision_wpe
}

/// Mean-field mean update from children.
///
/// Uses `expected_precision` directly as the value-coupling gain.
fn mean_update_from_children_mean_field(
    network: &Network,
    node_idx: usize,
    node_precision: f64,
) -> f64 {
    let value_pwpe = value_coupling_mean_wpe(network, node_idx, node_precision, true);
    let mut volatility_pwpe = 0.0;

    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let effective_precision = child_state.effective_precision;
            let volatility_pe = child_state.volatility_prediction_error;
            let observed = child_state.observed;
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);

            volatility_pwpe +=
                (kappa * effective_precision * volatility_pe) / (2.0 * node_precision) * observed;
        }
    }

    value_pwpe + volatility_pwpe
}

// =============================================================================
// Mean-field posterior updates
// =============================================================================

pub fn posterior_update_continuous_state_node_mean_field(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let expected_mean = network.attributes.states[node_idx].expected_mean;

    let posterior_precision = observed_or_predicted_precision(network, node_idx, || {
        precision_update_from_children_mean_field(network, node_idx)
    });

    let mean_wpe = mean_update_from_children_mean_field(network, node_idx, posterior_precision);
    let posterior_mean = expected_mean + mean_wpe;

    let state = &mut network.attributes.states[node_idx];
    state.precision = posterior_precision;
    state.mean = posterior_mean;
}

pub fn posterior_update_continuous_state_node_ehgf_mean_field(
    network: &mut Network,
    node_idx: usize,
    time_step: f64,
) {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let expected_mean = network.attributes.states[node_idx].expected_mean;

    let mean_wpe = mean_update_from_children_mean_field(network, node_idx, expected_precision);
    let posterior_mean = expected_mean + mean_wpe;
    network.attributes.states[node_idx].mean = posterior_mean;

    // eHGF safe precision update (mean-field value coupling).
    let posterior_precision = observed_or_predicted_precision(network, node_idx, || {
        precision_update_from_children_ehgf_mean_field(network, node_idx, time_step)
    });
    network.attributes.states[node_idx].precision = posterior_precision;
}
