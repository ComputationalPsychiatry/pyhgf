pub fn sufficient_statistics(x: &f64) -> Vec<f64> {
    vec![*x, x.powf(2.0)]
}