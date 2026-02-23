use crate::model::Network;
use crate::utils::set_coupling::set_coupling;
use crate::utils::prospective::{prospective_precision, prospective_mean};

// =============================================================================
// Learning routines
// =============================================================================

/// Weights update using a fixed learning rate.
///
/// For every value parent of `node_idx`, this function:
/// 1. Computes a *prospective* posterior for the parent (precision, then mean)
///    — i.e. what the parent's belief *would* look like after the current update.
/// 2. Evaluates the coupling function at the prospective mean to derive an
///    *expected* coupling strength that would explain the observed child mean.
/// 3. Moves the current coupling towards that target at rate `lr`.
///
/// The learning rate `lr` is read from the node's float attribute `"lr"` and
/// defaults to **0.01** when absent.
///
/// Matches Python `pyhgf.updates.learning.learning_weights_fixed`.
pub fn learning_weights_fixed(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    // Gather the value parents and current coupling strengths for this child node.
    let value_parents = match network.edges.get(&node_idx)
        .and_then(|e| e.value_parents.clone())
    {
        Some(vp) => vp,
        None => return, // nothing to learn if no value parents
    };

    let value_couplings = network.attributes.vectors
        .get(&node_idx)
        .and_then(|v| v.get("value_coupling_parents").cloned())
        .unwrap_or_else(|| vec![1.0; value_parents.len()]);

    let child_mean = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("mean"))
        .unwrap_or(&0.0);

    let lr = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("lr"))
        .unwrap_or(&0.01);

    let weighting = 1.0 / value_parents.len() as f64;

    for (parent_idx, value_coupling) in value_parents.iter().zip(value_couplings.iter())
    {
        let parent_idx = *parent_idx;
        let value_coupling = *value_coupling;

        // ----- Prospective reconfiguration step -----
        // Infer the latent state that would explain the observation.
        let prosp_precision = prospective_precision(network, parent_idx);
        let prosp_mean = prospective_mean(network, parent_idx, prosp_precision);

        // ----- Find the coupling function for this parent–child pair -----
        // It is stored on the child, indexed by the parent's position in the
        // child's value_parents list.
        let coupling_fn = network.edges.get(&node_idx)
            .and_then(|e| e.value_parents.as_ref())
            .and_then(|vp| vp.iter().position(|&p| p == parent_idx))
            .and_then(|pos| {
                network.attributes.fn_ptrs
                    .get(&node_idx)
                    .and_then(|fp| fp.get("value_coupling_fn_parents"))
                    .and_then(|fns| fns.get(pos).copied())
            });

        // g(prospective_mean)
        let g_value = match coupling_fn {
            Some(cf) => (cf.f)(prosp_mean),
            None => prosp_mean, // linear identity
        };

        // expected_coupling = child_mean / g(prospective_mean)
        let mut expected_coupling = if g_value.abs() < 1e-128 {
            value_coupling // avoid division by zero
        } else {
            child_mean / g_value
        };

        // Guard against NaN / Inf
        if expected_coupling.is_nan() || expected_coupling.is_infinite() {
            expected_coupling = value_coupling;
        }

        // Move the coupling towards the target
        let mut new_value_coupling =
            value_coupling + (expected_coupling - value_coupling) * lr * weighting;

        // Guard against Inf
        if new_value_coupling.is_infinite() {
            new_value_coupling = value_coupling;
        }

        // Write back to both parent and child attribute vectors
        set_coupling(network, parent_idx, node_idx, new_value_coupling);
    }
}

/// Dynamic weights update using precision-based learning rate.
///
/// Identical to [`learning_weights_fixed`] except the fixed `lr` is replaced by
/// a *precision weighting*:
///
///   w = π̂_child / (π̂_parent + π̂_child)
///
/// which makes the effective learning rate adaptive: the update is larger when
/// the child is relatively more precise than the parent.
///
/// Matches Python `pyhgf.updates.learning.learning_weights_dynamic`.
pub fn learning_weights_dynamic(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let value_parents = match network.edges.get(&node_idx)
        .and_then(|e| e.value_parents.clone())
    {
        Some(vp) => vp,
        None => return,
    };

    let value_couplings = network.attributes.vectors
        .get(&node_idx)
        .and_then(|v| v.get("value_coupling_parents").cloned())
        .unwrap_or_else(|| vec![1.0; value_parents.len()]);

    let child_mean = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("mean"))
        .unwrap_or(&0.0);

    let child_expected_precision = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("expected_precision"))
        .unwrap_or(&1.0);

    let weighting = 1.0 / value_parents.len() as f64;

    for (parent_idx, value_coupling) in value_parents.iter().zip(value_couplings.iter())
    {
        let parent_idx = *parent_idx;
        let value_coupling = *value_coupling;

        // ----- Prospective reconfiguration step -----
        let prosp_precision = prospective_precision(network, parent_idx);
        let prosp_mean = prospective_mean(network, parent_idx, prosp_precision);

        // ----- Coupling function -----
        let coupling_fn = network.edges.get(&node_idx)
            .and_then(|e| e.value_parents.as_ref())
            .and_then(|vp| vp.iter().position(|&p| p == parent_idx))
            .and_then(|pos| {
                network.attributes.fn_ptrs
                    .get(&node_idx)
                    .and_then(|fp| fp.get("value_coupling_fn_parents"))
                    .and_then(|fns| fns.get(pos).copied())
            });

        let g_value = match coupling_fn {
            Some(cf) => (cf.f)(prosp_mean),
            None => prosp_mean,
        };

        let mut expected_coupling = if g_value.abs() < 1e-128 {
            value_coupling
        } else {
            child_mean / g_value
        };

        if expected_coupling.is_nan() || expected_coupling.is_infinite() {
            expected_coupling = value_coupling;
        }

        // ----- Precision weighting -----
        // Use expected_precision (prediction-time) for both child and parent
        // to avoid asymmetry from update ordering.
        let parent_expected_precision = *network.attributes.floats
            .get(&parent_idx)
            .and_then(|f| f.get("expected_precision"))
            .unwrap_or(&1.0);

        let precision_weighting = child_expected_precision
            / (parent_expected_precision + child_expected_precision);

        let mut new_value_coupling = value_coupling
            + (expected_coupling - value_coupling) * precision_weighting * weighting;

        if new_value_coupling.is_infinite() {
            new_value_coupling = value_coupling;
        }

        set_coupling(network, parent_idx, node_idx, new_value_coupling);
    }
}
