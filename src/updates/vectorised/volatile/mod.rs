//! Vectorised kernels for volatile-node layers (a value level plus an internal
//! volatility level), mirroring `pyhgf/updates/vectorized/volatile/`.

pub mod posterior;
pub mod prediction;
pub mod prediction_error;

/// Values of the log-volatility exponentials below this are treated as
/// numerically zero and mapped to NaN, matching the JAX
/// `jnp.where(v > 1e-128, v, nan)` guard.
pub(crate) const MIN_VOLATILITY: f64 = 1e-128;

/// `time_step · exp(exponent)`, mapped to NaN when it underflows past
/// [`MIN_VOLATILITY`] — the single source of the JAX guard above, shared by
/// the prediction and prediction-error kernels so the copies cannot drift.
#[inline]
pub(crate) fn guarded_volatility(exponent: f64, time_step: f64) -> f64 {
    let v = time_step * exponent.exp();
    if v > MIN_VOLATILITY {
        v
    } else {
        f64::NAN
    }
}
