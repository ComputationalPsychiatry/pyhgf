use rshgf::model::Network;

fn main() {

    // initialize network
    let mut network = Network::new();

    // create a network with two exponential family state nodes
    network.add_nodes(
        "exponential-state",
        None,
        None,
        None,
        None
    );

    // belief propagation
    let input_data = vec![1.0, 1.3, 1.5, 1.7];
    network.set_update_sequence();
    network.input_data(input_data);

}
