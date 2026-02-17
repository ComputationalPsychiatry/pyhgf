use crate::model::Network;

/// Prediction error from a continuous state node
///
/// Compute the value prediction error and the volatility prediction error of a
/// continuous state node.
///
/// The value prediction error δ is given by:
///     δ = μ - μ̂
///
/// The volatility prediction error Δ is given by:
///     Δ = (π̂ / π) + π̂ · δ² - 1
///
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
///
/// # Returns
/// * `network` - The network after message passing.
pub fn prediction_error_continuous_state_node(network: &mut Network, node_idx: usize) {

    // Get the number of value parents and volatility parents from the edges
    let n_value_parents = network.edges.get(&node_idx)
        .and_then(|e| e.value_parents.as_ref())
        .map(|vp| vp.len());

    let n_volatility_parents = network.edges.get(&node_idx)
        .and_then(|e| e.volatility_parents.as_ref())
        .map(|vp| vp.len());

    // Read the required attributes (copy values to avoid borrow conflicts)
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
    let mut volatility_prediction_error =
        (expected_precision / precision)
        + expected_precision * value_prediction_error.powi(2)
        - 1.0;

    // Divide by the number of volatility parents if any
    if let Some(n) = n_volatility_parents {
        volatility_prediction_error /= n as f64;
    }

    // Store the prediction errors in the node's float attributes
    let floats_mut = network.attributes.floats.get_mut(&node_idx)
        .expect("No floats attributes found for node");
    floats_mut.insert(String::from("value_prediction_error"), value_prediction_error);
    floats_mut.insert(String::from("volatility_prediction_error"), volatility_prediction_error);
}