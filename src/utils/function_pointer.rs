use crate::{
    model::network::Network,
    updates::nodalised::{
        posterior::continuous::{
            posterior_update_continuous_state_node, posterior_update_continuous_state_node_ehgf,
            posterior_update_continuous_state_node_ehgf_mean_field,
            posterior_update_continuous_state_node_mean_field,
            posterior_update_continuous_state_node_unbounded,
        },
        prediction::binary::prediction_binary_state_node,
        prediction::continuous::{
            prediction_continuous_state_node, prediction_continuous_state_node_mean_field,
        },
        prediction_error::{
            binary::prediction_error_binary_state_node,
            continuous::prediction_error_continuous_state_node,
            exponential::prediction_error_exponential_state_node,
        },
    },
};

/// Enum-based dispatch for update steps.
/// Unlike function pointers, enum variants allow the compiler to inline
/// the actual update functions through the `match` in `call()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateStep {
    PredictionContinuous,
    PredictionContinuousMeanField,
    PredictionBinary,
    PosteriorContinuous,
    PosteriorContinuousMeanField,
    PosteriorContinuousEhgf,
    PosteriorContinuousEhgfMeanField,
    PosteriorContinuousUnbounded,
    PredictionErrorContinuous,
    PredictionErrorExponential,
    PredictionErrorBinary,
}

impl UpdateStep {
    #[inline(always)]
    pub fn call(self, network: &mut Network, node_idx: usize, time_step: f64) {
        match self {
            Self::PredictionContinuous => {
                prediction_continuous_state_node(network, node_idx, time_step)
            }
            Self::PredictionContinuousMeanField => {
                prediction_continuous_state_node_mean_field(network, node_idx, time_step)
            }
            Self::PredictionBinary => prediction_binary_state_node(network, node_idx, time_step),
            Self::PosteriorContinuous => {
                posterior_update_continuous_state_node(network, node_idx, time_step)
            }
            Self::PosteriorContinuousMeanField => {
                posterior_update_continuous_state_node_mean_field(network, node_idx, time_step)
            }
            Self::PosteriorContinuousEhgf => {
                posterior_update_continuous_state_node_ehgf(network, node_idx, time_step)
            }
            Self::PosteriorContinuousEhgfMeanField => {
                posterior_update_continuous_state_node_ehgf_mean_field(network, node_idx, time_step)
            }
            Self::PosteriorContinuousUnbounded => {
                posterior_update_continuous_state_node_unbounded(network, node_idx, time_step)
            }
            Self::PredictionErrorContinuous => {
                prediction_error_continuous_state_node(network, node_idx, time_step)
            }
            Self::PredictionErrorExponential => {
                prediction_error_exponential_state_node(network, node_idx, time_step)
            }
            Self::PredictionErrorBinary => {
                prediction_error_binary_state_node(network, node_idx, time_step)
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::PredictionContinuous => "prediction_continuous_state_node",
            Self::PredictionContinuousMeanField => "prediction_continuous_state_node_mean_field",
            Self::PredictionBinary => "prediction_binary_state_node",
            Self::PosteriorContinuous => "posterior_update_continuous_state_node",
            Self::PosteriorContinuousMeanField => {
                "posterior_update_continuous_state_node_mean_field"
            }
            Self::PosteriorContinuousEhgf => "posterior_update_continuous_state_node_ehgf",
            Self::PosteriorContinuousEhgfMeanField => {
                "posterior_update_continuous_state_node_ehgf_mean_field"
            }
            Self::PosteriorContinuousUnbounded => {
                "posterior_update_continuous_state_node_unbounded"
            }
            Self::PredictionErrorContinuous => "prediction_error_continuous_state_node",
            Self::PredictionErrorExponential => "prediction_error_exponential_state_node",
            Self::PredictionErrorBinary => "prediction_error_binary_state_node",
        }
    }
}
