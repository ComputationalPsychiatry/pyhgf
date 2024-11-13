pub fn normal(x: &f64) -> Vec<f64> {
    vec![*x, x.powf(2.0)]
}

pub fn multivariate_normal(x: &Vec<f64>) -> Vec<f64> {
    vec![*x, x.powf(2.0)]
}

pub fn get_sufficient_statistics_fn(distribution: String) {
    if distribution == "normal" {
        normal;
    } else if distribution == "multivariate_normal" {
        multivariate_normal;
    }
}