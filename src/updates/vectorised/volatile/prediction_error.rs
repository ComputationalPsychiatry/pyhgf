//! Prediction errors and the volatility-level posterior updates for
//! volatile-node layers, mirroring
//! `pyhgf/updates/vectorized/volatile/prediction_error.py` (which, like here,
//! hosts the per-update-type volatility posteriors and the combined driver).

use crate::math::{lambert_w0, logaddexp, sigmoid};
use crate::vectorised::layer::{LayerState, VolatilityUpdate};
use ndarray::Array1;

/// Compute prediction errors and apply the volatility-level posterior update.
///
/// Mirrors `vectorized_layer_prediction_error`: always sets the value PE
/// `δ = mean − expected_mean`; when the layer has a volatility parent it also
/// computes the volatility PE and dispatches the volatility-level posterior.
pub fn layer_prediction_error(
    layer: &mut LayerState,
    volatility_updates: VolatilityUpdate,
    time_step: f64,
    has_volatility_parent: bool,
    max_posterior_precision: f64,
) {
    // Value prediction error (always).
    layer.value_prediction_error = &layer.mean - &layer.expected_mean;

    if !has_volatility_parent {
        return;
    }

    // Volatility prediction error: π̃/π + π̃·δ² − 1 (fused, one allocation).
    let vol_pe = ndarray::Zip::from(&layer.expected_precision)
        .and(&layer.precision)
        .and(&layer.mean)
        .and(&layer.expected_mean)
        .map_collect(|&ep, &p, &m, &em| volatility_pe_node(ep, p, m, em));
    layer.volatility_prediction_error = Some(vol_pe);

    match volatility_updates {
        VolatilityUpdate::Standard => {
            volatility_posterior_standard(layer, max_posterior_precision)
        }
        VolatilityUpdate::EHgf => {
            volatility_posterior_ehgf(layer, time_step, max_posterior_precision)
        }
        VolatilityUpdate::Unbounded => {
            volatility_posterior_unbounded(layer, time_step, max_posterior_precision)
        }
    }
}

/// The value-level volatility prediction error of one node:
/// `Δ = π̃/π + π̃·δ² − 1`.
#[inline]
pub(crate) fn volatility_pe_node(ep: f64, p: f64, m: f64, em: f64) -> f64 {
    let d = m - em;
    ep / p + ep * (d * d) - 1.0
}

/// Standard volatility-level posterior of one node: precision first
/// (`clip(π̂_vol + 0.5·γ² + γ²·Δ − 0.5·γ·Δ)`), then the mean from the
/// just-updated precision (`μ_vol = μ̂_vol + γΔ / (2·π_vol)`). The volatility
/// coupling is fixed at 1. Returns `(precision_vol, mean_vol)`. The single source
/// shared by the per-sample and the batched kernels, so the two paths cannot drift.
#[inline]
pub(crate) fn standard_vol_node(
    g: f64,
    dpe: f64,
    ep: f64,
    em: f64,
    max_posterior_precision: f64,
) -> (f64, f64) {
    let g2 = g * g;
    let contrib = 0.5 * g2 + g2 * dpe - 0.5 * g * dpe;
    let pp = (ep + contrib).min(max_posterior_precision);
    let m = em + (g * dpe) / (pp * 2.0);
    (pp, m)
}

/// Standard volatility-level posterior: precision first, then mean.
fn volatility_posterior_standard(state: &mut LayerState, max_posterior_precision: f64) {
    let eff = &state.effective_precision;
    let vpe = state.volatility_prediction_error.as_ref().unwrap();
    let epv = state.expected_precision_vol.as_ref().unwrap();
    let emv = state.expected_mean_vol.as_ref().unwrap();

    let n = eff.len();
    let mut posterior_precision_vol = Array1::<f64>::zeros(n);
    let mut posterior_mean_vol = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (pp, m) = standard_vol_node(
            eff[i],
            vpe[i],
            epv[i],
            emv[i],
            max_posterior_precision,
        );
        posterior_precision_vol[i] = pp;
        posterior_mean_vol[i] = m;
    }

    state.precision_vol = Some(posterior_precision_vol);
    state.mean_vol = Some(posterior_mean_vol);
}

/// eHGF volatility-level posterior of one node: mean first (using the
/// expected precision as the approximation), then the "safe" precision
/// increment recomputed from the updated mean and floored at zero. Returns
/// `(precision_vol, mean_vol)`.
///
/// Plain exponentials: the JAX eHGF posterior applies no underflow guard here
/// (unlike the prediction kernels' `guarded_volatility`); the only floor is
/// the `1e-128` clamp on `previous_variance`. The per-node backend's eHGF
/// update follows the same convention.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ehgf_vol_node(
    em: f64,
    ep: f64,
    g: f64,
    dpe: f64,
    cond: f64,
    prec: f64,
    mean: f64,
    emean: f64,
    time_step: f64,
    max_posterior_precision: f64,
) -> (f64, f64) {
    // Mean first, using expected_precision_vol as the approximation (coupling = 1).
    let posterior_mean_vol = em + (g * dpe) / (ep * 2.0);

    let pvol = |exponent: f64| time_step * exponent.exp();
    // Reconstruct the value level's pre-prediction variance exactly. The value
    // level carries no tonic volatility of its own.
    let mgf = 1.0 / (ep * 2.0);
    let predicted_vol = pvol(em + mgf);
    let previous_variance = (1.0 / cond - predicted_vol).max(1e-128);
    // Re-predict volatility / precision from the posterior mean.
    let repredicted_vol = pvol(posterior_mean_vol);
    let expected_precision = 1.0 / (previous_variance + repredicted_vol);
    let eff2 = repredicted_vol * expected_precision;
    let vew = (repredicted_vol - previous_variance) * expected_precision;
    let d = mean - emean;
    let vpe2 = (1.0 / prec + d * d) * expected_precision - 1.0;
    let inner = eff2 + vew * vpe2;
    let contrib = (eff2 * inner * 0.5).max(0.0);
    ((ep + contrib).min(max_posterior_precision), posterior_mean_vol)
}

/// eHGF volatility-level posterior: mean first (using the expected precision),
/// then the "safe" precision increment recomputed from the updated mean and
/// floored at zero.
fn volatility_posterior_ehgf(state: &mut LayerState, time_step: f64, max_posterior_precision: f64) {
    let eff = &state.effective_precision;
    let vpe = state.volatility_prediction_error.as_ref().unwrap();
    let epv = state.expected_precision_vol.as_ref().unwrap();
    let emv = state.expected_mean_vol.as_ref().unwrap();
    let cond = &state.conditional_expected_precision;
    let prec = &state.precision;
    let mean = &state.mean;
    let emean = &state.expected_mean;

    let n = mean.len();
    let mut posterior_precision_vol = Array1::<f64>::zeros(n);
    let mut posterior_mean_vol = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (pp, m) = ehgf_vol_node(
            emv[i],
            epv[i],
            eff[i],
            vpe[i],
            cond[i],
            prec[i],
            mean[i],
            emean[i],
            time_step,
            max_posterior_precision,
        );
        posterior_precision_vol[i] = pp;
        posterior_mean_vol[i] = m;
    }

    state.precision_vol = Some(posterior_precision_vol);
    state.mean_vol = Some(posterior_mean_vol);
}

/// Unbounded (uHGF) volatility-level posterior: two quadratic expansions — one
/// at the prediction, one at the Lambert-W approximate mode — blended by a
/// variational energy-based softmax, with Gaussian mixture moment matching for
/// the posterior precision.
///
/// Mirrors `vectorized_layer_volatility_posterior_unbounded`
/// (`pyhgf/updates/vectorized/volatile/prediction_error.py`) line by line,
/// including the log-space rewrites: those exist there for gradient safety, but
/// are kept here so both backends compute the *same* forward values.
fn volatility_posterior_unbounded(
    state: &mut LayerState,
    time_step: f64,
    max_posterior_precision: f64,
) {
    let epv = state.expected_precision_vol.as_ref().unwrap();
    let emv = state.expected_mean_vol.as_ref().unwrap();
    let cond = &state.conditional_expected_precision;

    let n = state.mean.len();
    let mut posterior_mean_vol = Array1::<f64>::zeros(n);
    let mut posterior_precision_vol = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (pp, m) = unbounded_vol_node(
            emv[i],
            epv[i],
            cond[i],
            state.expected_precision[i],
            state.precision[i],
            state.mean[i],
            state.expected_mean[i],
            time_step,
            max_posterior_precision,
        );
        posterior_precision_vol[i] = pp;
        posterior_mean_vol[i] = m;
    }

    state.precision_vol = Some(posterior_precision_vol);
    state.mean_vol = Some(posterior_mean_vol);
}

/// Unbounded (uHGF) volatility-level posterior of one node; see
/// [`volatility_posterior_unbounded`] for the scheme. Returns
/// `(precision_vol, mean_vol)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn unbounded_vol_node(
    em: f64,
    ep: f64,
    cond: f64,
    marginal_ep: f64,
    prec: f64,
    mean: f64,
    emean: f64,
    time_step: f64,
    max_posterior_precision: f64,
) -> (f64, f64) {
    let log_time_step = time_step.ln();
    let log_float_max = f64::MAX.ln();

    // Reconstruct the pre-prediction variance exactly from the conditional
    // predicted precision (the prediction's full exponent, MGF included). The
    // volatility coupling is fixed at 1 and the value level carries no tonic
    // volatility of its own.
    let predicted_volatility = time_step * (em + 1.0 / (2.0 * ep)).exp();
    let previous_variance = (1.0 / cond - predicted_volatility).max(1e-128);
    let d = mean - emean;
    let be_aux = 1.0 / prec + d * d;

    let log_previous_variance = previous_variance.ln();
    // Canonical exponent at prediction.
    let gamma_c = log_time_step + em;
    // w_jm1 = 1/(1 + previous_variance/exp(γ)) = sigmoid(γ − log α).
    let w_jm1 = sigmoid(gamma_c - log_previous_variance);
    // Volatility PE with the *marginal* predicted precision.
    let da_jm1 = marginal_ep * be_aux - 1.0;

    // Expansion 1: quadratic at the prediction (prior mean).
    let pi1 = ep + 0.5 * w_jm1 * (1.0 - w_jm1);
    let mu1 = em + (w_jm1 / (2.0 * pi1)) * da_jm1;

    // Expansion 2: quadratic at the Lambert-W approximate mode.
    let pihat_y = ep;
    let log_w_arg = be_aux.ln() - (2.0 * pihat_y).ln() + 0.5 / pihat_y - gamma_c;
    let w_arg = log_w_arg.min(log_float_max).exp();
    let v_w = lambert_w0(w_arg);
    let y_star = gamma_c + v_w - 0.5 / pihat_y;
    let x_star = y_star - log_time_step;

    let log_s2 = log_time_step + x_star;
    let log_denom_s = logaddexp(log_previous_variance, log_s2);
    let w2 = sigmoid(log_s2 - log_previous_variance);
    let da2 = be_aux * (-log_denom_s).exp() - 1.0;

    let pi2_full = ep + 0.5 * w2 * (w2 + (2.0 * w2 - 1.0) * da2);
    let pi2_safe = if pi2_full <= 0.0 {
        ep + 0.5 * w2 * (1.0 - w2)
    } else {
        pi2_full
    };
    let mu2_safe = x_star + (0.5 * w2 * da2 - ep * (x_star - em)) / pi2_safe;

    // Fall back to Expansion 1 if Expansion 2 is non-finite.
    let (pi2, mu2) = if pi2_safe.is_finite() && mu2_safe.is_finite() {
        (pi2_safe, mu2_safe)
    } else {
        (pi1, mu1)
    };

    // Variational energy-based softmax blend (log-space form).
    let log_ey1 = log_time_step + mu1;
    let log_denom_1 = logaddexp(log_previous_variance, log_ey1);
    let i1 = -0.5 * log_denom_1
        - 0.5 * be_aux * (-log_denom_1).exp()
        - 0.5 * ep * (mu1 - em) * (mu1 - em);
    let log_ey2 = log_time_step + mu2;
    let log_denom_2 = logaddexp(log_previous_variance, log_ey2);
    let i2 = -0.5 * log_denom_2
        - 0.5 * be_aux * (-log_denom_2).exp()
        - 0.5 * ep * (mu2 - em) * (mu2 - em);
    let b = sigmoid(i2 - i1);

    // Gaussian mixture moment matching.
    let mu = (1.0 - b) * mu1 + b * mu2;
    let sig2 = (1.0 - b) / pi1 + b / pi2 + b * (1.0 - b) * (mu1 - mu2) * (mu1 - mu2);
    ((1.0 / sig2).min(max_posterior_precision), mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Value PE is the plain residual; without a volatility parent the vol PE
    /// stays `None`.
    #[test]
    fn test_value_prediction_error() {
        let mut layer = LayerState::create(3, false);
        layer.mean = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        layer.expected_mean = Array1::from_vec(vec![0.5, 2.0, 4.0]);

        layer_prediction_error(&mut layer, VolatilityUpdate::EHgf, 1.0, false, 1e10);
        assert!((layer.value_prediction_error[0] - 0.5).abs() < 1e-12);
        assert!((layer.value_prediction_error[1] - 0.0).abs() < 1e-12);
        assert!((layer.value_prediction_error[2] - (-1.0)).abs() < 1e-12);
        assert!(layer.volatility_prediction_error.is_none());
    }

    /// The standard volatility posterior populates mean_vol/precision_vol and
    /// respects the max-precision clip.
    #[test]
    fn test_volatility_posterior_standard_runs() {
        let mut layer = LayerState::create(2, true);
        layer.mean = Array1::from_vec(vec![1.0, -1.0]);
        layer.expected_mean = Array1::from_vec(vec![0.0, 0.0]);
        layer.expected_precision = Array1::from_vec(vec![2.0, 2.0]);
        layer.precision = Array1::from_vec(vec![1.0, 1.0]);
        layer.effective_precision = Array1::from_vec(vec![0.3, 0.3]);
        layer.expected_precision_vol = Some(Array1::from_vec(vec![1.0, 1.0]));
        layer.expected_mean_vol = Some(Array1::from_vec(vec![0.0, 0.0]));

        layer_prediction_error(&mut layer, VolatilityUpdate::Standard, 1.0, true, 1e10);
        assert!(layer.volatility_prediction_error.is_some());
        assert!(layer.precision_vol.is_some());
        assert!(layer.mean_vol.is_some());
        assert!(layer.precision_vol.unwrap().iter().all(|&x| x.is_finite()));
    }
}
