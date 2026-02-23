pub fn sufficient_statistics(x: &f64) -> Vec<f64> {
    vec![*x, x.powf(2.0)]
}

// ── Activation functions ──────────────────────────────────────────────────────

/// Rectified Linear Unit (ReLU).
///
/// $$f(x) = \max(0, x)$$
///
/// Returns `x` for positive inputs and `0` otherwise.
#[inline]
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Sigmoid (logistic) function.
///
/// $$f(x) = \frac{1}{1 + e^{-x}}$$
///
/// Maps any real value to the open interval (0, 1).
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Hyperbolic tangent activation.
///
/// $$f(x) = \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$
///
/// Maps any real value to the open interval (-1, 1).
#[inline]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Leaky Rectified Linear Unit (Leaky ReLU).
///
/// $$f(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{otherwise} \end{cases}$$
///
/// Uses a fixed small slope `alpha = 0.01` for negative inputs, preventing
/// the "dying ReLU" problem.
#[inline]
pub fn leaky_relu(x: f64) -> f64 {
    const ALPHA: f64 = 0.01;
    if x >= 0.0 { x } else { ALPHA * x }
}

/// Parametric Rectified Linear Unit (PReLU).
///
/// $$f(x, \alpha) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{otherwise} \end{cases}$$
///
/// Generalises [`leaky_relu`] by exposing the negative-slope coefficient `alpha`
/// as a learnable (or user-supplied) parameter.
#[inline]
pub fn prelu(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 { x } else { alpha * x }
}

/// Gaussian Error Linear Unit (GELU).
///
/// $$f(x) = x \cdot \Phi(x) = \frac{x}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$
///
/// where $\Phi$ is the standard-normal CDF. Uses the accurate series
/// approximation via the complementary error function (`erfc`).
///
/// Implemented via the identity
/// $\Phi(x) = \tfrac{1}{2}\,\mathrm{erfc}(-x / \sqrt{2})$.
#[inline]
pub fn gelu(x: f64) -> f64 {
    // Φ(x) = 0.5 * erfc(-x / √2)
    x * 0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Complementary error function used internally by [`gelu`].
///
/// Computed with the Horner-form rational approximation from Abramowitz &
/// Stegun §7.1.26 (maximum error < 1.5 × 10⁻⁷).
fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    let approx = poly * (-(x * x)).exp();
    if x >= 0.0 { approx } else { 2.0 - approx }
}

/// Identity (linear) coupling — passes the value through unchanged.
///
/// This is the default coupling function used when no `coupling_fn` is
/// specified in [`Network::add_nodes`].
pub fn linear(x: f64) -> f64 {
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for floating-point comparisons.
    const TOL: f64 = 1e-6;

    fn assert_close(actual: f64, expected: f64, label: &str) {
        assert!(
            (actual - expected).abs() < TOL,
            "{}: expected {:.10}, got {:.10} (diff = {:.2e})",
            label, expected, actual, (actual - expected).abs()
        );
    }

    // ── sufficient_statistics ─────────────────────────────────────────────────

    #[test]
    fn test_sufficient_statistics_positive() {
        let s = sufficient_statistics(&3.0);
        assert_close(s[0], 3.0,  "sufficient_statistics[0] for x=3");
        assert_close(s[1], 9.0,  "sufficient_statistics[1] for x=3");
    }

    #[test]
    fn test_sufficient_statistics_negative() {
        let s = sufficient_statistics(&-2.5);
        assert_close(s[0], -2.5, "sufficient_statistics[0] for x=-2.5");
        assert_close(s[1],  6.25,"sufficient_statistics[1] for x=-2.5");
    }

    #[test]
    fn test_sufficient_statistics_zero() {
        let s = sufficient_statistics(&0.0);
        assert_close(s[0], 0.0, "sufficient_statistics[0] for x=0");
        assert_close(s[1], 0.0, "sufficient_statistics[1] for x=0");
    }

    // ── relu ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_relu_positive() {
        assert_close(relu(3.5), 3.5, "relu(3.5)");
    }

    #[test]
    fn test_relu_negative() {
        assert_close(relu(-2.0), 0.0, "relu(-2.0)");
    }

    #[test]
    fn test_relu_zero() {
        assert_close(relu(0.0), 0.0, "relu(0.0)");
    }

    // ── sigmoid ───────────────────────────────────────────────────────────────

    #[test]
    fn test_sigmoid_zero() {
        // σ(0) = 0.5 exactly
        assert_close(sigmoid(0.0), 0.5, "sigmoid(0)");
    }

    #[test]
    fn test_sigmoid_positive_large() {
        // σ(+∞) → 1; σ(10) ≈ 0.9999546
        assert!(sigmoid(10.0) > 0.999, "sigmoid(10) should be > 0.999");
        assert!(sigmoid(10.0) < 1.0,   "sigmoid(10) should be < 1");
    }

    #[test]
    fn test_sigmoid_negative_large() {
        // σ(−∞) → 0; σ(−10) ≈ 4.54e-5
        assert!(sigmoid(-10.0) < 0.001, "sigmoid(-10) should be < 0.001");
        assert!(sigmoid(-10.0) > 0.0,   "sigmoid(-10) should be > 0");
    }

    #[test]
    fn test_sigmoid_known_value() {
        // σ(1) = 1 / (1 + e^{-1}) ≈ 0.7310585786
        assert_close(sigmoid(1.0), 0.731_058_578_6, "sigmoid(1)");
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // σ(x) + σ(−x) = 1
        let x = 2.3;
        assert_close(sigmoid(x) + sigmoid(-x), 1.0, "sigmoid symmetry");
    }

    // ── tanh ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_tanh_zero() {
        assert_close(tanh(0.0), 0.0, "tanh(0)");
    }

    #[test]
    fn test_tanh_known_value() {
        // tanh(1) ≈ 0.7615941559
        assert_close(tanh(1.0), 0.761_594_155_9, "tanh(1)");
    }

    #[test]
    fn test_tanh_antisymmetry() {
        // tanh is odd: tanh(−x) = −tanh(x)
        let x = 1.7;
        assert_close(tanh(-x), -tanh(x), "tanh antisymmetry");
    }

    #[test]
    fn test_tanh_bounds() {
        assert!(tanh(20.0) <= 1.0,  "tanh(20) <= 1");
        assert!(tanh(20.0) > 0.99,  "tanh(20) > 0.99");
        assert!(tanh(-20.0) >= -1.0,  "tanh(-20) >= -1");
        assert!(tanh(-20.0) < -0.99, "tanh(-20) < -0.99");
    }

    // ── leaky_relu ────────────────────────────────────────────────────────────

    #[test]
    fn test_leaky_relu_positive() {
        assert_close(leaky_relu(4.0), 4.0, "leaky_relu(4.0)");
    }

    #[test]
    fn test_leaky_relu_zero() {
        assert_close(leaky_relu(0.0), 0.0, "leaky_relu(0.0)");
    }

    #[test]
    fn test_leaky_relu_negative() {
        // α = 0.01, so leaky_relu(−3) = 0.01 × (−3) = −0.03
        assert_close(leaky_relu(-3.0), -0.03, "leaky_relu(-3.0)");
    }

    // ── prelu ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_prelu_positive() {
        assert_close(prelu(5.0, 0.25), 5.0, "prelu(5, α=0.25)");
    }

    #[test]
    fn test_prelu_zero() {
        assert_close(prelu(0.0, 0.25), 0.0, "prelu(0, α=0.25)");
    }

    #[test]
    fn test_prelu_negative() {
        // prelu(−4, 0.25) = 0.25 × (−4) = −1.0
        assert_close(prelu(-4.0, 0.25), -1.0, "prelu(-4, α=0.25)");
    }

    #[test]
    fn test_prelu_alpha_zero() {
        // α=0 collapses to standard ReLU
        assert_close(prelu(-2.0, 0.0), 0.0, "prelu(-2, α=0) == relu");
        assert_close(prelu(2.0, 0.0),  2.0, "prelu(2,  α=0) == relu");
    }

    #[test]
    fn test_prelu_matches_leaky_relu() {
        // prelu with α=0.01 must equal leaky_relu
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            assert_close(prelu(x, 0.01), leaky_relu(x),
                &format!("prelu(α=0.01) == leaky_relu at x={}", x));
        }
    }

    // ── gelu ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_gelu_zero() {
        // GELU(0) = 0 × Φ(0) = 0
        assert_close(gelu(0.0), 0.0, "gelu(0)");
    }

    #[test]
    fn test_gelu_positive_large() {
        // For large x, Φ(x) → 1, so GELU(x) → x
        let x = 10.0_f64;
        assert_close(gelu(x), x, "gelu(10) ≈ 10");
    }

    #[test]
    fn test_gelu_negative_large() {
        // For large negative x, Φ(x) → 0, so GELU(x) → 0
        assert!( gelu(-10.0).abs() < 1e-4, "gelu(-10) ≈ 0" );
    }

    #[test]
    fn test_gelu_known_value() {
        // GELU(1) ≈ 0.8413447  (x × Φ(1), Φ(1) ≈ 0.8413447)
        let expected = 0.841_344_7;
        assert!(
            (gelu(1.0) - expected).abs() < 1e-5,
            "gelu(1) expected ≈ {}, got {}", expected, gelu(1.0)
        );
    }

    #[test]
    fn test_gelu_negative_value() {
        // GELU(−1) = −1 × Φ(−1) ≈ −1 × 0.1586553 ≈ −0.1586553
        let expected = -0.158_655_3;
        assert!(
            (gelu(-1.0) - expected).abs() < 1e-5,
            "gelu(-1) expected ≈ {}, got {}", expected, gelu(-1.0)
        );
    }
}