use crate::model::Network;

/// Prediction error from a volatile state node
///
/// Computes both value and volatility prediction errors:
///
/// 1. Value PE (for external value parents):
///     δ = μ - μ̂
///
/// 2. Volatility PE (for internal volatility level):
///     Δ = (π̂ / π) + π̂ · δ² - 1
///
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// * `_time_step` - The time step (unused).
pub fn prediction_error_volatile_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {

    // Get the number of value parents from the edges
    let n_value_parents = network.edges.get(&node_idx)
        .and_then(|e| e.value_parents.as_ref())
        .map(|vp| vp.len());

    // Read the required attributes
    let floats = network.attributes.floats.get(&node_idx)
        .expect("No floats attributes found for node");
    let mean = *floats.get("mean").expect("mean not found");
    let expected_mean = *floats.get("expected_mean").expect("expected_mean not found");
    let precision = *floats.get("precision").expect("precision not found");
    let expected_precision = *floats.get("expected_precision")
        .expect("expected_precision not found");

    // 1. Value prediction error: δ = μ - μ̂
    let mut value_prediction_error = mean - expected_mean;

    // Divide by the number of value parents if any
    if let Some(n) = n_value_parents {
        value_prediction_error /= n as f64;
    }

    // 2. Volatility prediction error: Δ = (π̂ / π) + π̂ · δ² - 1
    // This is the internal coupling (always 1 implicit volatility "parent"), no division needed
    let volatility_prediction_error =
        (expected_precision / precision)
        + expected_precision * value_prediction_error.powi(2)
        - 1.0;

    // Store the prediction errors
    let floats_mut = network.attributes.floats.get_mut(&node_idx)
        .expect("No floats attributes found for node");
    floats_mut.insert(String::from("value_prediction_error"), value_prediction_error);
    floats_mut.insert(String::from("volatility_prediction_error"), volatility_prediction_error);
}
