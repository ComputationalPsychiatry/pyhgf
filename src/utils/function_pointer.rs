use std::collections::HashMap;

use crate::{model::Network, updates::{posterior::continuous::{posterior_update_continuous_state_node, posterior_update_continuous_state_node_ehgf, posterior_update_continuous_state_node_unbounded}, posterior::volatile::{posterior_update_volatile_state_node, posterior_update_volatile_state_node_ehgf, posterior_update_volatile_state_node_unbounded}, prediction::continuous::prediction_continuous_state_node, prediction::volatile::prediction_volatile_state_node, prediction_error::{continuous::prediction_error_continuous_state_node, exponential::prediction_error_exponential_state_node, volatile::prediction_error_volatile_state_node}}};

// Create a default signature for update functions
pub type FnType = for<'a> fn(&'a mut Network, usize, f64);

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
        (posterior_update_volatile_state_node_ehgf as FnType, "posterior_update_volatile_state_node_ehgf"),
        (posterior_update_volatile_state_node_unbounded as FnType, "posterior_update_volatile_state_node_unbounded"),
        (prediction_error_volatile_state_node as FnType, "prediction_error_volatile_state_node"),
    ]
    .into_iter()
    .collect();
    function_map
}