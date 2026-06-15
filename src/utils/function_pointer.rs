use std::collections::HashMap;

use crate::{model::Network, updates::{posterior::continuous::{posterior_update_continuous_state_node, posterior_update_continuous_state_node_ehgf, posterior_update_continuous_state_node_unbounded, posterior_update_continuous_state_node_mean_field, posterior_update_continuous_state_node_ehgf_mean_field}, posterior::volatile::{posterior_update_volatile_state_node, posterior_update_volatile_state_node_mean_field}, prediction::binary::prediction_binary_state_node, prediction::continuous::{prediction_continuous_state_node, prediction_continuous_state_node_mean_field}, prediction::volatile::{prediction_volatile_state_node, prediction_volatile_state_node_mean_field}, prediction_error::{binary::prediction_error_binary_state_node, continuous::prediction_error_continuous_state_node, exponential::prediction_error_exponential_state_node, volatile::{prediction_error_volatile_state_node, prediction_error_volatile_state_node_ehgf, prediction_error_volatile_state_node_unbounded}}}};
use crate::updates::learning::learning_weights;

// Create a default signature for update functions
pub type FnType = for<'a> fn(&'a mut Network, usize, f64);

/// Enum-based dispatch for update steps.
/// Unlike function pointers, enum variants allow the compiler to inline
/// the actual update functions through the `match` in `call()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateStep {
    PredictionContinuous,
    PredictionContinuousMeanField,
    PredictionVolatile,
    PredictionVolatileMeanField,
    PredictionBinary,
    PosteriorContinuous,
    PosteriorContinuousMeanField,
    PosteriorContinuousEhgf,
    PosteriorContinuousEhgfMeanField,
    PosteriorContinuousUnbounded,
    PosteriorVolatile,
    PosteriorVolatileMeanField,
    PredictionErrorContinuous,
    PredictionErrorVolatile,
    PredictionErrorVolatileEhgf,
    PredictionErrorVolatileUnbounded,
    PredictionErrorExponential,
    PredictionErrorBinary,
    LearningWeights,
}

impl UpdateStep {
    #[inline(always)]
    pub fn call(self, network: &mut Network, node_idx: usize, time_step: f64) {
        match self {
            Self::PredictionContinuous => prediction_continuous_state_node(network, node_idx, time_step),
            Self::PredictionContinuousMeanField => prediction_continuous_state_node_mean_field(network, node_idx, time_step),
            Self::PredictionVolatile => prediction_volatile_state_node(network, node_idx, time_step),
            Self::PredictionVolatileMeanField => prediction_volatile_state_node_mean_field(network, node_idx, time_step),
            Self::PredictionBinary => prediction_binary_state_node(network, node_idx, time_step),
            Self::PosteriorContinuous => posterior_update_continuous_state_node(network, node_idx, time_step),
            Self::PosteriorContinuousMeanField => posterior_update_continuous_state_node_mean_field(network, node_idx, time_step),
            Self::PosteriorContinuousEhgf => posterior_update_continuous_state_node_ehgf(network, node_idx, time_step),
            Self::PosteriorContinuousEhgfMeanField => posterior_update_continuous_state_node_ehgf_mean_field(network, node_idx, time_step),
            Self::PosteriorContinuousUnbounded => posterior_update_continuous_state_node_unbounded(network, node_idx, time_step),
            Self::PosteriorVolatile => posterior_update_volatile_state_node(network, node_idx, time_step),
            Self::PosteriorVolatileMeanField => posterior_update_volatile_state_node_mean_field(network, node_idx, time_step),
            Self::PredictionErrorContinuous => prediction_error_continuous_state_node(network, node_idx, time_step),
            Self::PredictionErrorVolatile => prediction_error_volatile_state_node(network, node_idx, time_step),
            Self::PredictionErrorVolatileEhgf => prediction_error_volatile_state_node_ehgf(network, node_idx, time_step),
            Self::PredictionErrorVolatileUnbounded => prediction_error_volatile_state_node_unbounded(network, node_idx, time_step),
            Self::PredictionErrorExponential => prediction_error_exponential_state_node(network, node_idx, time_step),
            Self::PredictionErrorBinary => prediction_error_binary_state_node(network, node_idx, time_step),
            Self::LearningWeights => learning_weights(network, node_idx, time_step),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::PredictionContinuous => "prediction_continuous_state_node",
            Self::PredictionContinuousMeanField => "prediction_continuous_state_node_mean_field",
            Self::PredictionVolatile => "prediction_volatile_state_node",
            Self::PredictionVolatileMeanField => "prediction_volatile_state_node_mean_field",
            Self::PredictionBinary => "prediction_binary_state_node",
            Self::PosteriorContinuous => "posterior_update_continuous_state_node",
            Self::PosteriorContinuousMeanField => "posterior_update_continuous_state_node_mean_field",
            Self::PosteriorContinuousEhgf => "posterior_update_continuous_state_node_ehgf",
            Self::PosteriorContinuousEhgfMeanField => "posterior_update_continuous_state_node_ehgf_mean_field",
            Self::PosteriorContinuousUnbounded => "posterior_update_continuous_state_node_unbounded",
            Self::PosteriorVolatile => "posterior_update_volatile_state_node",
            Self::PosteriorVolatileMeanField => "posterior_update_volatile_state_node_mean_field",
            Self::PredictionErrorContinuous => "prediction_error_continuous_state_node",
            Self::PredictionErrorVolatile => "prediction_error_volatile_state_node",
            Self::PredictionErrorVolatileEhgf => "prediction_error_volatile_state_node_ehgf",
            Self::PredictionErrorVolatileUnbounded => "prediction_error_volatile_state_node_unbounded",
            Self::PredictionErrorExponential => "prediction_error_exponential_state_node",
            Self::PredictionErrorBinary => "prediction_error_binary_state_node",
            Self::LearningWeights => "learning_weights",
        }
    }
}

pub fn get_func_map() -> HashMap<FnType, &'static str> {
    let function_map: HashMap<FnType, &str> = [
        (posterior_update_continuous_state_node as FnType, "posterior_update_continuous_state_node"),
        (posterior_update_continuous_state_node_ehgf as FnType, "posterior_update_continuous_state_node_ehgf"),
        (posterior_update_continuous_state_node_unbounded as FnType, "posterior_update_continuous_state_node_unbounded"),
        (prediction_continuous_state_node as FnType, "prediction_continuous_state_node"),
        (prediction_error_continuous_state_node as FnType, "prediction_error_continuous_state_node"),
        (prediction_error_exponential_state_node as FnType, "prediction_error_exponential_state_node"),
        (prediction_volatile_state_node as FnType, "prediction_volatile_state_node"),
        (posterior_update_volatile_state_node as FnType, "posterior_update_volatile_state_node"),
        (prediction_error_volatile_state_node as FnType, "prediction_error_volatile_state_node"),
        (prediction_error_volatile_state_node_ehgf as FnType, "prediction_error_volatile_state_node_ehgf"),
        (prediction_error_volatile_state_node_unbounded as FnType, "prediction_error_volatile_state_node_unbounded"),
        (learning_weights as FnType, "learning_weights"),
        (prediction_binary_state_node as FnType, "prediction_binary_state_node"),
        (prediction_error_binary_state_node as FnType, "prediction_error_binary_state_node"),
    ]
    .into_iter()
    .collect();
    function_map
}