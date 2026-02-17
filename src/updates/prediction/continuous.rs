use crate::model::Network;

/// Prediction from a continuous state node
///
/// Compute the expected mean and expected precision of a continuous state node.
///
/// The expected mean is given by:
///     μ̂ = λ · μ + Δt · driftrate
///
/// where driftrate = ρ + Σ(ψ_j · μ̂_parent_j) over value parents.
///
/// The expected precision is given by:
///     π̂ = 1 / (1/π + Ω)
///
/// where the predicted volatility Ω is:
///     Ω = Δt · exp(ω + Σ(κ_j · μ_parent_j))
///
/// The effective precision γ is:
///     γ = Ω · π̂
///
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
///
/// # Returns
/// * `network` - The network after message passing.
pub fn prediction_continuous_state_node(network: &mut Network, node_idx: usize, time_step: f64) {

    // -------------------------------------------------------
    // 1. Predict the mean: μ̂ = λ · μ + Δt · driftrate
    // -------------------------------------------------------

    // Read the node's own attributes
    let floats = network.attributes.floats.get(&node_idx)
        .expect("No floats attributes found for node");
    let mean = *floats.get("mean").expect("mean not found");
    let tonic_drift = *floats.get("tonic_drift").expect("tonic_drift not found");
    let autoconnection_strength = *floats.get("autoconnection_strength")
        .expect("autoconnection_strength not found");

    // Start the drift rate from tonic drift
    let mut driftrate = tonic_drift;

    // Add phasic drift from value parents (if any)
    // driftrate += Σ(ψ_j · expected_mean_parent_j)
    let value_parents = network.edges.get(&node_idx)
        .and_then(|e| e.value_parents.clone());

    if let Some(ref vp_idxs) = value_parents {
        // Read coupling strengths from vector attributes
        let coupling_strengths = network.attributes.vectors
            .get(&node_idx)
            .and_then(|v| v.get("value_coupling_parents").cloned());

        for (i, &parent_idx) in vp_idxs.iter().enumerate() {
            let parent_expected_mean = *network.attributes.floats
                .get(&parent_idx)
                .expect("No floats attributes found for value parent")
                .get("expected_mean")
                .expect("expected_mean not found for value parent");

            let psi = coupling_strengths.as_ref()
                .map(|cs| cs[i])
                .unwrap_or(1.0);

            driftrate += psi * parent_expected_mean;
        }
    }

    // Compute the expected mean
    let expected_mean = autoconnection_strength * mean + time_step * driftrate;

    // -------------------------------------------------------
    // 2. Predict the precision: π̂ = 1 / (1/π + Ω)
    // -------------------------------------------------------

    let floats = network.attributes.floats.get(&node_idx)
        .expect("No floats attributes found for node");
    let precision = *floats.get("precision").expect("precision not found");
    let tonic_volatility = *floats.get("tonic_volatility")
        .expect("tonic_volatility not found");

    // Start with tonic volatility
    let mut total_volatility = tonic_volatility;

    // Add phasic volatility from volatility parents (if any)
    // total_volatility += Σ(κ_j · μ_parent_j)
    let volatility_parents = network.edges.get(&node_idx)
        .and_then(|e| e.volatility_parents.clone());

    if let Some(ref vol_parent_idxs) = volatility_parents {
        let vol_coupling_strengths = network.attributes.vectors
            .get(&node_idx)
            .and_then(|v| v.get("volatility_coupling_parents").cloned());

        for (i, &parent_idx) in vol_parent_idxs.iter().enumerate() {
            let parent_mean = *network.attributes.floats
                .get(&parent_idx)
                .expect("No floats attributes found for volatility parent")
                .get("mean")
                .expect("mean not found for volatility parent");

            let kappa = vol_coupling_strengths.as_ref()
                .map(|cs| cs[i])
                .unwrap_or(1.0);

            total_volatility += kappa * parent_mean;
        }
    }

    // Compute predicted volatility: Ω = Δt · exp(clamp(total_volatility, -80, 80))
    let predicted_volatility = (time_step
        * total_volatility.clamp(-80.0, 80.0).exp())
        .max(1e-128);

    // Expected precision: π̂ = 1 / (1/π + Ω)
    let expected_precision = 1.0 / ((1.0 / precision) + predicted_volatility);

    // Effective precision: γ = Ω · π̂
    let effective_precision = predicted_volatility * expected_precision;

    // -------------------------------------------------------
    // 3. Store results
    // -------------------------------------------------------

    // Check if this is an input node without volatility parents
    let edges = network.edges.get(&node_idx);
    let is_input = edges.map_or(true, |e| {
        e.value_children.is_none() && e.volatility_children.is_none()
    });
    let has_volatility_parents = edges.map_or(false, |e| e.volatility_parents.is_some());

    let floats_mut = network.attributes.floats.get_mut(&node_idx)
        .expect("No floats attributes found for node");

    // For input nodes without volatility parents, keep the current precision as expected
    if is_input && !has_volatility_parents {
        // expected_precision stays as the current precision (unchanged)
    } else {
        floats_mut.insert(String::from("expected_precision"), expected_precision);
    }

    // Store the current variance (1/π) for use by the posterior update
    let current_variance = 1.0 / floats_mut.get("precision").copied().unwrap_or(1.0);
    floats_mut.insert(String::from("current_variance"), current_variance);

    floats_mut.insert(String::from("expected_mean"), expected_mean);
    floats_mut.insert(String::from("effective_precision"), effective_precision);
}