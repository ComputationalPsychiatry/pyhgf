use std::collections::HashMap;

use crate::{model::Network, updates::{posterior::continuous::posterior_update_continuous_state_node, prediction::continuous::prediction_continuous_state_node, prediction_error::{continuous::prediction_error_continuous_state_node, exponential::prediction_error_exponential_state_node}}};

// Create a default signature for update functions
pub type FnType = for<'a> fn(&'a mut Network, usize);

pub fn get_func_map() -> HashMap<FnType, &'static str> {
    let function_map: HashMap<FnType, &str> = [
        (posterior_update_continuous_state_node as FnType, "posterior_update_continuous_state_node"),
        (prediction_continuous_state_node as FnType, "prediction_continuous_state_node"),
        (prediction_error_continuous_state_node as FnType, "prediction_error_continuous_state_node"),
        (prediction_error_exponential_state_node as FnType, "prediction_error_exponential_state_node"),
    ]
    .into_iter()
    .collect();
    function_map
}