pub fn sufficient_statistics(x: &f64) -> Vec<f64> {
    vec![*x, x.powf(2.0)]
}
/// A coupling (activation) function together with its first and second derivatives.
///
/// Use the module-level constants ([`LINEAR`], [`RELU`], [`SIGMOID`], [`TANH`],
/// [`LEAKY_RELU`], [`GELU`]) to obtain a `&'static CouplingFn`, or call
/// [`resolve_coupling_fn`] to resolve from a string name at node-creation time.
///
/// # Example
/// ```ignore
/// let c = crate::math::resolve_coupling_fn("sigmoid");
/// let y     = (c.f)(x);
/// let dy    = (c.df)(x);
/// let d2y   = (c.d2f)(x);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CouplingFn {
    /// The activation function $f(x)$.
    pub f:   fn(f64) -> f64,
    /// The first derivative $f'(x)$.
    pub df:  fn(f64) -> f64,
    /// The second derivative $f''(x)$.
    pub d2f: fn(f64) -> f64,
}

// ── Activation functions ──────────────────────────────────────────────────────

// ─── Linear ──────────────────────────────────────────────────────────────────

/// Linear (identity) activation: $f(x) = x$.
pub fn linear(x: f64) -> f64 { x }
/// First derivative of linear: $f'(x) = 1$.
pub fn linear_d1(_x: f64) -> f64 { 1.0 }
/// Second derivative of linear: $f''(x) = 0$.
pub fn linear_d2(_x: f64) -> f64 { 0.0 }
/// [`CouplingFn`] constant for the linear (identity) activation.
pub const LINEAR: CouplingFn = CouplingFn { f: linear, df: linear_d1, d2f: linear_d2 };

// ─── ReLU ────────────────────────────────────────────────────────────────────

/// Rectified Linear Unit: $f(x) = \max(0, x)$.
pub fn relu(x: f64) -> f64 { x.max(0.0) }
/// First derivative of ReLU: $f'(x) = 1$ if $x > 0$, else $0$ (zero at $x = 0$).
pub fn relu_d1(x: f64) -> f64 { if x > 0.0 { 1.0 } else { 0.0 } }
/// Second derivative of ReLU: $f''(x) = 0$ (almost everywhere).
pub fn relu_d2(_x: f64) -> f64 { 0.0 }
/// [`CouplingFn`] constant for the ReLU activation.
pub const RELU: CouplingFn = CouplingFn { f: relu, df: relu_d1, d2f: relu_d2 };

// ─── Sigmoid ─────────────────────────────────────────────────────────────────

/// Sigmoid: $f(x) = 1 / (1 + e^{-x})$.
pub fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
/// First derivative of sigmoid: $f'(x) = f(x)(1 - f(x))$.
pub fn sigmoid_d1(x: f64) -> f64 { let s = sigmoid(x); s * (1.0 - s) }
/// Second derivative of sigmoid: $f''(x) = f'(x)(1 - 2f(x))$.
pub fn sigmoid_d2(x: f64) -> f64 { let s = sigmoid(x); sigmoid_d1(x) * (1.0 - 2.0 * s) }
/// [`CouplingFn`] constant for the sigmoid activation.
pub const SIGMOID: CouplingFn = CouplingFn { f: sigmoid, df: sigmoid_d1, d2f: sigmoid_d2 };

// ─── Tanh ─────────────────────────────────────────────────────────────────────

/// Hyperbolic tangent: $f(x) = \tanh(x)$.
pub fn tanh(x: f64) -> f64 { x.tanh() }
/// First derivative of tanh: $f'(x) = 1 - \tanh^2(x)$.
pub fn tanh_d1(x: f64) -> f64 { 1.0 - x.tanh().powi(2) }
/// Second derivative of tanh: $f''(x) = -2\tanh(x)(1 - \tanh^2(x))$.
pub fn tanh_d2(x: f64) -> f64 { let t = x.tanh(); -2.0 * t * (1.0 - t * t) }
/// [`CouplingFn`] constant for the tanh activation.
pub const TANH: CouplingFn = CouplingFn { f: tanh, df: tanh_d1, d2f: tanh_d2 };

// ─── Leaky ReLU ──────────────────────────────────────────────────────────────

/// Leaky ReLU with fixed slope $\alpha = 0.01$: $f(x) = x$ if $x \ge 0$, else $0.01x$.
pub fn leaky_relu(x: f64) -> f64 { if x >= 0.0 { x } else { 0.01 * x } }
/// First derivative of Leaky ReLU: $f'(x) = 1$ if $x \ge 0$, else $0.01$.
pub fn leaky_relu_d1(x: f64) -> f64 { if x >= 0.0 { 1.0 } else { 0.01 } }
/// Second derivative of Leaky ReLU: $f''(x) = 0$ (almost everywhere).
pub fn leaky_relu_d2(_x: f64) -> f64 { 0.0 }
/// [`CouplingFn`] constant for the Leaky ReLU activation.
pub const LEAKY_RELU: CouplingFn = CouplingFn { f: leaky_relu, df: leaky_relu_d1, d2f: leaky_relu_d2 };

// ─── PReLU ───────────────────────────────────────────────────────────────────

/// Parametric ReLU: $f(x, \alpha) = x$ if $x \ge 0$, else $\alpha x$.
///
/// Note: PReLU requires a free `alpha` parameter, so it cannot be stored in a
/// `CouplingFn` constant. Use [`leaky_relu`] (fixed $\alpha = 0.01$) or
/// [`LEAKY_RELU`] as the bundled variant.
pub fn prelu(x: f64, alpha: f64) -> f64 { if x >= 0.0 { x } else { alpha * x } }
/// First derivative of PReLU.
pub fn prelu_d1(x: f64, alpha: f64) -> f64 { if x >= 0.0 { 1.0 } else { alpha } }
/// Second derivative of PReLU: $f''(x) = 0$ (almost everywhere).
pub fn prelu_d2(_x: f64, _alpha: f64) -> f64 { 0.0 }

// ─── GELU ────────────────────────────────────────────────────────────────────

/// Complementary error function (internal helper for [`gelu`] and [`gelu_d1`]).
///
/// Abramowitz & Stegun §7.1.26 rational approximation; max error < 1.5 × 10⁻⁷.
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

/// GELU: $f(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard-normal CDF.
pub fn gelu(x: f64) -> f64 {
    x * 0.5 * erfc(-x / std::f64::consts::SQRT_2)
}
/// First derivative of GELU: $f'(x) = \Phi(x) + x\,\phi(x)$.
pub fn gelu_d1(x: f64) -> f64 {
    let phi = 0.5 * (1.0 - erfc(x / std::f64::consts::SQRT_2));
    let pdf = (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    phi + x * pdf
}
/// Second derivative of GELU: $f''(x) = \phi(x)(2 - x^2)$.
pub fn gelu_d2(x: f64) -> f64 {
    let pdf = (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    pdf * (2.0 - x * x)
}
/// [`CouplingFn`] constant for the GELU activation.
pub const GELU: CouplingFn = CouplingFn { f: gelu, df: gelu_d1, d2f: gelu_d2 };

// ─── Resolver ────────────────────────────────────────────────────────────────

/// Resolve an activation name to its [`CouplingFn`] constant.
///
/// Called once at node-creation time in [`Network::add_nodes`]; the resulting
/// `&'static CouplingFn` is stored directly in `Attributes::fn_ptrs` so that
/// prediction code only needs to call `.f`, `.df`, or `.d2f`.
///
/// | Name | Constant |
/// |------|----------|
/// | `"linear"` | [`LINEAR`] |
/// | `"relu"` | [`RELU`] |
/// | `"sigmoid"` | [`SIGMOID`] |
/// | `"tanh"` | [`TANH`] |
/// | `"leaky_relu"` | [`LEAKY_RELU`] |
/// | `"gelu"` | [`GELU`] |
///
/// Any unrecognised name falls back to [`LINEAR`].
pub fn resolve_coupling_fn(name: &str) -> &'static CouplingFn {
    match name {
        "relu"       => &RELU,
        "sigmoid"    => &SIGMOID,
        "tanh"       => &TANH,
        "leaky_relu" => &LEAKY_RELU,
        "gelu"       => &GELU,
        _            => &LINEAR,
    }
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