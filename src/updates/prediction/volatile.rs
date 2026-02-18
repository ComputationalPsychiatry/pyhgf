use crate::model::Network;

/// Prediction from a volatile state node
///
/// A volatile node has two internal levels:
/// 1. Volatility level (implicit, internal) — predicted first
/// 2. Value level (external facing) — its precision depends on the volatility level
///
/// ## Volatility level prediction
///
/// The volatility level's expected mean:
///     μ̂_vol = λ_vol · μ_vol + Δt · ρ_vol
///
/// The volatility level's expected precision:
///     π̂_vol = 1 / (1/π_vol + Δt · exp(ω_vol))
///
/// ## Value level prediction
///
/// The value level's expected mean (including value parents if any):
///     μ̂ = λ · μ + Δt · driftrate
///
/// The value level's expected precision (modulated by volatility level):
///     Ω = Δt · exp(ω + κ_internal · μ̂_vol)
///     π̂ = 1 / (1/π + Ω)
///
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// * `time_step` - The time step.
pub fn prediction_volatile_state_node(network: &mut Network, node_idx: usize, time_step: f64) {

    // Store current variance for potential unbounded updates
    let precision = *network.attributes.floats.get(&node_idx)
        .expect("No floats attributes found for node")
        .get("precision")
        .expect("precision not found");
    let current_variance = 1.0 / precision;
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("current_variance"), current_variance);

    // ===================================================================
    // 1. PREDICT VOLATILITY LEVEL (implicit internal state)
    // ===================================================================

    let floats = network.attributes.floats.get(&node_idx)
        .expect("No floats attributes found for node");
    let mean_vol = *floats.get("mean_vol").expect("mean_vol not found");
    let precision_vol = *floats.get("precision_vol").expect("precision_vol not found");
    let tonic_drift_vol = *floats.get("tonic_drift_vol").expect("tonic_drift_vol not found");
    let autoconnection_strength_vol = *floats.get("autoconnection_strength_vol")
        .expect("autoconnection_strength_vol not found");
    let tonic_volatility_vol = *floats.get("tonic_volatility_vol")
        .expect("tonic_volatility_vol not found");

    // Expected mean of the volatility level
    let expected_mean_vol = autoconnection_strength_vol * mean_vol + time_step * tonic_drift_vol;

    // Expected precision of the volatility level
    let predicted_volatility_vol = (time_step * tonic_volatility_vol.clamp(-80.0, 80.0).exp()).max(1e-128);
    let expected_precision_vol = 1.0 / ((1.0 / precision_vol) + predicted_volatility_vol);
    let effective_precision_vol = predicted_volatility_vol * expected_precision_vol;

    // Store volatility level predictions
    let floats_mut = network.attributes.floats.get_mut(&node_idx).unwrap();
    floats_mut.insert(String::from("expected_mean_vol"), expected_mean_vol);
    floats_mut.insert(String::from("expected_precision_vol"), expected_precision_vol);
    floats_mut.insert(String::from("effective_precision_vol"), effective_precision_vol);

    // ===================================================================
    // 2. PREDICT VALUE LEVEL (external facing)
    // ===================================================================

    // --- 2a. Predict precision (depends on volatility level) ---
    let floats = network.attributes.floats.get(&node_idx).unwrap();
    let tonic_volatility = *floats.get("tonic_volatility").expect("tonic_volatility not found");
    let volatility_coupling_internal = *floats.get("volatility_coupling_internal")
        .expect("volatility_coupling_internal not found");

    let total_volatility = tonic_volatility + volatility_coupling_internal * expected_mean_vol;
    let predicted_volatility = (time_step * total_volatility.clamp(-80.0, 80.0).exp()).max(1e-128);

    let expected_precision = 1.0 / ((1.0 / precision) + predicted_volatility);
    let effective_precision = predicted_volatility * expected_precision;

    // --- 2b. Predict mean (including value parents if any) ---
    let floats = network.attributes.floats.get(&node_idx).unwrap();
    let mean = *floats.get("mean").expect("mean not found");
    let tonic_drift = *floats.get("tonic_drift").expect("tonic_drift not found");
    let autoconnection_strength = *floats.get("autoconnection_strength")
        .expect("autoconnection_strength not found");

    let mut driftrate = tonic_drift;

    // Add phasic drift from value parents (if any)
    let value_parents = network.edges.get(&node_idx)
        .and_then(|e| e.value_parents.clone());

    if let Some(ref vp_idxs) = value_parents {
        let coupling_strengths = network.attributes.vectors
            .get(&node_idx)
            .and_then(|v| v.get("value_coupling_parents").cloned());

        for (i, &parent_idx) in vp_idxs.iter().enumerate() {
            let parent_expected_mean = *network.attributes.floats
                .get(&parent_idx)
                .expect("No floats for value parent")
                .get("expected_mean")
                .expect("expected_mean not found for value parent");

            let psi = coupling_strengths.as_ref().map(|cs| cs[i]).unwrap_or(1.0);
            driftrate += psi * parent_expected_mean;
        }
    }

    let expected_mean = autoconnection_strength * mean + time_step * driftrate;

    // Store value level predictions
    let floats_mut = network.attributes.floats.get_mut(&node_idx).unwrap();
    floats_mut.insert(String::from("expected_mean"), expected_mean);
    floats_mut.insert(String::from("expected_precision"), expected_precision);
    floats_mut.insert(String::from("effective_precision"), effective_precision);
}
