use crate::math::logaddexp;
use crate::model::network::Network;

/// Principal branch of the Lambert W function for z >= 0.
/// Solves w * exp(w) = z via 6 Halley iterations.
fn lambert_w0(z: f64) -> f64 {
    let mut w = (z + 1.0).ln();
    for _ in 0..6 {
        let ew = w.exp();
        let f = w * ew - z;
        let f1 = (w + 1.0) * ew;
        let f2 = (w + 2.0) * ew;
        w -= (2.0 * f * f1) / (2.0 * f1 * f1 - f * f2);
    }
    w
}

// =============================================================================
// Shared building blocks for precision and mean updates
// =============================================================================

/// Compute the precision update contribution from value and volatility children.
///
/// The value-coupling branch implements the posterior-step (smoothing) correction
/// of the relaxed HGF: the canonical child-precision factor is replaced by the
/// harmonic combination
///
/// ```text
/// π̂_a · π_y / (π̂_a + π_y),    π_y = π_a − π̃_a,
/// ```
///
/// where π̂_a is the child's conditional predicted precision
/// (`conditional_expected_precision`) and π̃_a its marginal predicted precision
/// (`expected_precision`). The same factor scales both the `(κ g')²` and the
/// `κ g'' · δ_a` terms. Reduces to the canonical formula when the child is fully
/// observed (π_y → ∞); returns no contribution when the child gained no bottom-up
/// information (π_y = 0).
///
/// Non-Gaussian children and Gaussian leaves fall back to the canonical
/// predicted-precision factor π̃_a — pyhgf keeps `precision = expected_precision`
/// for clamped observations, so the harmonic form would otherwise zero out their
/// contribution.
///
/// Volatility coupling is unchanged.
fn precision_update_from_children(network: &Network, node_idx: usize) -> f64 {
    let mut precision_wpe = 0.0;

    // --- Value coupling ---
    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let observed = child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), kappa * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            // Effective child precision under the smoothing correction. The
            // Schur derivation assumes a Gaussian-Gaussian value-coupling edge,
            // so the correction only applies when the child carries a Gaussian
            // belief (continuous-state) AND is interior (has
            // children of its own). Binary, categorical, input/constant,
            // exponential-family, and Dirichlet children, plus any Gaussian
            // leaf, fall back to the canonical predicted-precision factor π̃_a
            // (paper's Limit 3, π_a → ∞).
            let child_node_type = network.edges[child_idx].node_type.as_str();
            let child_is_gaussian_interior = child_node_type == "continuous-state"
                && (network.edges[child_idx].value_children.is_some()
                    || network.edges[child_idx].volatility_children.is_some());
            let effective_child_precision = if child_is_gaussian_interior {
                // π_y = π_a − π̃_a; the Schur complement carries the *conditional*
                // predicted precision π̂_a (stored on the child). Using the
                // marginal would double-count parent uncertainty.
                let child_precision = child_state.precision;
                let pi_y = child_precision - child_expected_precision;
                let child_cond = child_state.conditional_expected_precision;
                child_cond * pi_y / (child_cond + pi_y)
            } else {
                child_expected_precision
            };

            precision_wpe += (effective_child_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term))
                * observed;
        }
    }

    // --- Volatility coupling ---
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

/// Compute the mean update contribution from value and volatility children.
///
/// The value-coupling branch uses the joint-Gaussian (RTS-smoother) gain
///
/// ```text
/// g_a = π̂_a · π_a / (π̂_a + π_y),    π_y = π_a − π̃_a,
/// ```
///
/// accumulated across children and divided once by `node_precision` (π_b) — this is
/// what makes the multi-child mean exact rather than a sum of independent
/// single-child RTS gains. For leaves and non-Gaussian children π_y = 0 and g_a
/// collapses to π̃_a (= `child.expected_precision`), recovering the canonical gain.
fn mean_update_from_children(network: &Network, node_idx: usize, node_precision: f64) -> f64 {
    let mut value_pwpe = 0.0;
    let mut volatility_pwpe = 0.0;

    // --- Value coupling mean update ---
    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let child_vape = child_state.value_prediction_error * child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let coupling_fn_prime = match coupling_fn {
                Some(cf) => (cf.df)(parent_mean),
                None => 1.0,
            };

            // Joint-Gaussian (RTS-smoother) gain g_a = π̂_a · π_a / (π̂_a + π_y),
            // π_y = π_a − π̃_a; summed over children then divided once by node_precision
            // (exact multi-child marginal mean). Leaves / non-Gaussian children have
            // π_y = 0 so g_a collapses to the marginal, recovering the canonical gain.
            let child_node_type = network.edges[child_idx].node_type.as_str();
            let child_is_gaussian_interior = child_node_type == "continuous-state"
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

            value_pwpe +=
                (kappa * coupling_fn_prime * gain_precision / node_precision) * child_vape;
        }
    }

    // --- Volatility coupling mean update ---
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
// Standard posterior update
// =============================================================================

pub fn posterior_update_continuous_state_node(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let max_posterior_precision = network.max_posterior_precision;

    let precision_wpe = precision_update_from_children(network, node_idx);
    let posterior_precision = (expected_precision + precision_wpe)
        .max(1e-128)
        .min(max_posterior_precision);

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
    let max_posterior_precision = network.max_posterior_precision;

    let mean_wpe = mean_update_from_children(network, node_idx, expected_precision);
    let posterior_mean = expected_mean + mean_wpe;
    network.attributes.states[node_idx].mean = posterior_mean;

    // eHGF safe precision update: recompute the effective precision from the
    // posterior mean and floor the volatility increment at zero.
    let precision_wpe = precision_update_from_children_ehgf(network, node_idx, time_step);
    let posterior_precision = (expected_precision + precision_wpe)
        .max(1e-128)
        .min(max_posterior_precision);
    network.attributes.states[node_idx].precision = posterior_precision;
}

// =============================================================================
// Unbounded posterior update
// =============================================================================

/// Quantities of one volatility child entering the unbounded update.
struct UnboundedVolatilityChild {
    coupling: f64,
    tonic_volatility: f64,
    /// Log of the carried variance λ²/π the child used in its prediction.
    log_previous_variance: f64,
    /// Total posterior uncertainty of the child, 1/π + (μ − μ̂)².
    be_aux: f64,
    /// Volatility prediction error of the child, evaluated at the prediction μ̂.
    volatility_prediction_error: f64,
}

impl UnboundedVolatilityChild {
    /// Log of the predicted volatility Ω(x) = t exp(κ x + ω) of the child.
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
/// Ω_i(x) = t exp(κ_i x + ω_i),
/// ```
///
/// where (π̃, μ̃) is the prediction of the node or, when the node also has value
/// children, the effective prior returned by [`unbounded_effective_prior`]. The other
/// volatility parents of a child are not included in Ω_i. The posterior is the
/// moment-matched Gaussian of n + 1 quadratic expansions of I: one at the prediction μ̂
/// (concave part only, eqs. 52-53), and one at the Lambert W0 mode of each child, with
/// the curvature and gradient of the full energy (eqs. 54-56). The expansions are
/// weighted by a softmax of I at their means (eqs. 57-60). For a single volatility
/// child and no value child, this is the two-expansion update of section 3.4.
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
        .map(|(i, &child_idx)| {
            let child_state = &network.attributes.states[child_idx];
            let be_aux = (1.0 / child_state.precision)
                + (child_state.mean - child_state.expected_mean).powi(2);
            UnboundedVolatilityChild {
                coupling: volatility_couplings.get(i).copied().unwrap_or(1.0),
                tonic_volatility: child_state.tonic_volatility,
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

    // Expansion 1: quadratic at the prediction, concave part only (eqs. 52-53).
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

    // Expansions 2, ..., n+1: quadratic at each child's Lambert W0 mode (eqs. 54-56).
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

    // Softmax weights (eq. 57) and Gaussian mixture moment matching (eqs. 58-60).
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

    let max_posterior_precision = network.max_posterior_precision;
    let state = &mut network.attributes.states[node_idx];
    state.precision = (1.0 / posterior_variance).min(max_posterior_precision);
    state.mean = posterior_mean;
}

/// Effective prior (π̃, μ̃) of the unbounded update, in which section 3.6 of Mathys et
/// al. (2026) absorbs the value-coupling terms of the variational energy.
///
/// It is the posterior that the regular value-coupling update returns when the
/// volatility children are left out: relaxed (smoothing factor for Gaussian interior
/// children) or mean-field, following `Network::mean_field_updates`, so that value
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

/// Precision increment from the value children only: the value-coupling branch of
/// `precision_update_from_children` (relaxed) or of
/// `precision_update_from_children_mean_field`.
fn value_coupling_precision_wpe(network: &Network, node_idx: usize, mean_field: bool) -> f64 {
    let mut precision_wpe = 0.0;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let observed = child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), kappa * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            let child_node_type = network.edges[child_idx].node_type.as_str();
            let child_is_gaussian_interior = child_node_type == "continuous-state"
                && (network.edges[child_idx].value_children.is_some()
                    || network.edges[child_idx].volatility_children.is_some());
            let effective_child_precision = if child_is_gaussian_interior && !mean_field {
                let child_precision = child_state.precision;
                let pi_y = child_precision - child_expected_precision;
                let child_cond = child_state.conditional_expected_precision;
                child_cond * pi_y / (child_cond + pi_y)
            } else {
                child_expected_precision
            };

            precision_wpe += (effective_child_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term))
                * observed;
        }
    }

    precision_wpe
}

/// Mean increment from the value children only: the value-coupling branch of
/// `mean_update_from_children` (relaxed) or of `mean_update_from_children_mean_field`.
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
            let child_expected_precision = child_state.expected_precision;
            let child_vape = child_state.value_prediction_error * child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let coupling_fn_prime = match coupling_fn {
                Some(cf) => (cf.df)(parent_mean),
                None => 1.0,
            };

            let child_node_type = network.edges[child_idx].node_type.as_str();
            let child_is_gaussian_interior = child_node_type == "continuous-state"
                && (network.edges[child_idx].value_children.is_some()
                    || network.edges[child_idx].volatility_children.is_some());
            let gain_precision = if child_is_gaussian_interior && !mean_field {
                let child_precision = child_state.precision;
                let pi_y = child_precision - child_expected_precision;
                let child_cond = child_state.conditional_expected_precision;
                child_cond * child_precision / (child_cond + pi_y)
            } else {
                child_expected_precision
            };

            value_pwpe +=
                (kappa * coupling_fn_prime * gain_precision / node_precision) * child_vape;
        }
    }

    value_pwpe
}

// =============================================================================
// Mean-field (v0.2.11) building blocks
// =============================================================================

/// Mean-field precision update from children.
///
/// Uses `expected_precision` directly as the child-precision factor.
fn precision_update_from_children_mean_field(network: &Network, node_idx: usize) -> f64 {
    let mut precision_wpe = 0.0;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let observed = child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), kappa * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            precision_wpe += (child_expected_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term))
                * observed;
        }
    }

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
/// guarantees the posterior precision never drops below the predicted precision.
fn ehgf_volatility_increment(
    child_state: &crate::model::network::NodeState,
    volatility_coupling: f64,
    mean: f64,
    time_step: f64,
) -> f64 {
    // Carried variance σ = λ²/π the child used; the re-prediction below must add
    // volatility to the same quantity the prediction step used.
    let previous_variance = child_state.current_variance;
    // Re-predict the child's volatility and precision from the parent posterior mean.
    let predicted_volatility =
        time_step * (volatility_coupling * mean + child_state.tonic_volatility).exp();
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
    let mut precision_wpe = 0.0;

    // --- Value coupling (identical to the relaxed standard update) ---
    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let observed = child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), kappa * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            let child_node_type = network.edges[child_idx].node_type.as_str();
            let child_is_gaussian_interior = child_node_type == "continuous-state"
                && (network.edges[child_idx].value_children.is_some()
                    || network.edges[child_idx].volatility_children.is_some());
            let effective_child_precision = if child_is_gaussian_interior {
                let child_precision = child_state.precision;
                let pi_y = child_precision - child_expected_precision;
                let child_cond = child_state.conditional_expected_precision;
                child_cond * pi_y / (child_cond + pi_y)
            } else {
                child_expected_precision
            };

            precision_wpe += (effective_child_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term))
                * observed;
        }
    }

    // --- Volatility coupling (eHGF safe update) ---
    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);
            precision_wpe += ehgf_volatility_increment(child_state, kappa, parent_mean, time_step);
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
    let mut precision_wpe = 0.0;

    // --- Value coupling (identical to the mean-field standard update) ---
    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let observed = child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), kappa * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            precision_wpe += (child_expected_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term))
                * observed;
        }
    }

    // --- Volatility coupling (eHGF safe update) ---
    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);
            precision_wpe += ehgf_volatility_increment(child_state, kappa, parent_mean, time_step);
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
    let mut value_pwpe = 0.0;
    let mut volatility_pwpe = 0.0;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let child_vape = child_state.value_prediction_error * child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let coupling_fn_prime = match coupling_fn {
                Some(cf) => (cf.df)(parent_mean),
                None => 1.0,
            };

            value_pwpe += (kappa * coupling_fn_prime * child_expected_precision / node_precision)
                * child_vape;
        }
    }

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
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let max_posterior_precision = network.max_posterior_precision;

    let precision_wpe = precision_update_from_children_mean_field(network, node_idx);
    let posterior_precision = (expected_precision + precision_wpe)
        .max(1e-128)
        .min(max_posterior_precision);

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
    let max_posterior_precision = network.max_posterior_precision;

    let mean_wpe = mean_update_from_children_mean_field(network, node_idx, expected_precision);
    let posterior_mean = expected_mean + mean_wpe;
    network.attributes.states[node_idx].mean = posterior_mean;

    // eHGF safe precision update (mean-field value coupling).
    let precision_wpe =
        precision_update_from_children_ehgf_mean_field(network, node_idx, time_step);
    let posterior_precision = (expected_precision + precision_wpe)
        .max(1e-128)
        .min(max_posterior_precision);
    network.attributes.states[node_idx].precision = posterior_precision;
}
