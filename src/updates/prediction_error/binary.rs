use crate::model::Network;

/// Prediction error for a binary state node
///
/// The value prediction error is:
///
/// ```text
/// δ_b = (μ − μ̂) · observed / π̂
/// ```
///
/// The scaling by 1/π̂ compensates for the uncertainty‑based convention
/// used in prediction (see `prediction_binary_state_node`).
///
/// After computing the PE, the precision is set equal to expected_precision.
pub fn prediction_error_binary_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {

    let floats = network.attributes.floats
        .get(&node_idx)
        .expect("No floats attributes found for binary node");

    let mean = *floats.get("mean").expect("mean not found");
    let expected_mean = *floats.get("expected_mean").expect("expected_mean not found");
    let expected_precision = *floats.get("expected_precision").expect("expected_precision not found");
    let observed = *floats.get("observed").expect("observed not found");

    // Value prediction error, scaled by 1 / expected_precision
    let value_prediction_error = (mean - expected_mean) * observed / expected_precision;

    let floats = network.attributes.floats
        .get_mut(&node_idx)
        .expect("No floats attributes found for binary node");
    floats.insert("value_prediction_error".into(), value_prediction_error);

    // Set precision = expected_precision
    floats.insert("precision".into(), expected_precision);
}
