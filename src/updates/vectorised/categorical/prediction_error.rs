//! Prediction error for categorical leaf layers, mirroring
//! `pyhgf/updates/vectorized/categorical/prediction_error.py`.

use crate::vectorised::layer::LayerState;

/// Categorical leaf prediction error: the raw residual `δ = one_hot − softmax`
/// (the logit-space cross-entropy gradient), posterior precision left as the
/// expected precision.
///
/// Mirrors `vectorized_categorical_prediction_error`.
pub fn categorical_prediction_error(layer: &mut LayerState) {
    layer.value_prediction_error = &layer.mean - &layer.expected_mean;
    layer.precision = layer.expected_precision.clone();
}
