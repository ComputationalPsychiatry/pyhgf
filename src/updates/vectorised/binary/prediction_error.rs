//! Prediction error for binary leaf layers, mirroring
//! `pyhgf/updates/vectorized/binary/prediction_error.py`.

use crate::vectorised::layer::LayerState;

/// Binary leaf prediction error: `δ = (mean − μ̂) / π̂`, with the posterior
/// precision set equal to the expected precision (no precision update).
///
/// Mirrors `vectorized_binary_prediction_error`.
pub fn binary_prediction_error(layer: &mut LayerState) {
    layer.value_prediction_error =
        &(&layer.mean - &layer.expected_mean) / &layer.expected_precision;
    layer.precision = layer.expected_precision.clone();
}
