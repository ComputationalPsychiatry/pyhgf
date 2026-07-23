//! Vectorised kernels for volatile-node layers (a value level plus an internal
//! volatility level), mirroring `pyhgf/updates/vectorized/volatile/`.

use crate::vectorised::mat::Float;
pub mod posterior;
pub mod prediction;
pub mod prediction_error;

/// Values of the log-volatility exponentials below this are treated as
/// numerically zero and mapped to NaN, matching the JAX
/// `jnp.where(v > 1e-128, v, nan)` guard. `1e-128` underflows `f32`
/// (min normal ≈ 1.2e-38), so the `f32` engine floors at `1e-30` instead —
/// the same "numerically zero" role, expressed in the narrower type's range.
#[cfg(feature = "f64")]
pub(crate) const MIN_VOLATILITY: Float = 1e-128;
#[cfg(not(feature = "f64"))]
pub(crate) const MIN_VOLATILITY: Float = 1e-30;

/// `time_step · exp(exponent)`, mapped to NaN when it underflows past
/// [`MIN_VOLATILITY`] — the single source of the JAX guard above, shared by
/// the prediction and prediction-error kernels so the copies cannot drift.
#[inline]
pub(crate) fn guarded_volatility(exponent: Float, time_step: Float) -> Float {
    let v = time_step * exponent.exp();
    if v > MIN_VOLATILITY {
        v
    } else {
        Float::NAN
    }
}
