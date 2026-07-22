/// Sufficient statistics `[x, x²]` of a univariate Gaussian observation.
pub fn sufficient_statistics(x: f64) -> Vec<f64> {
    vec![x, x * x]
}

/// Principal branch of the Lambert W function for `z >= 0`.
///
/// Solves `w * exp(w) = z` via 6 Halley iterations — the same scheme (and the
/// same initial guess `ln(z + 1)`) as `pyhgf.math.lambert_w0`, so the two
/// backends agree to machine precision. Used by the `unbounded` volatility
/// updates.
pub fn lambert_w0(z: f64) -> f64 {
    let mut w = (z + 1.0).ln();
    for _ in 0..6 {
        let ew = w.exp();
        let f = w * ew - z;
        let f1 = (w + 1.0) * ew;
        let f2 = (w + 2.0) * ew;
        w -= (2.0 * f * f1) / (2.0 * f1 * f1 - f * f2);
    }
    w
}

/// `ln(exp(a) + exp(b))`, computed stably (mirrors `jnp.logaddexp`).
#[inline]
pub fn logaddexp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    // With an infinite m the general formula would produce ∞ − ∞ = NaN; the
    // answer is m itself (−∞ only when both arguments are −∞).
    if m.is_infinite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Resolve a coupling-function name, erroring on unknown names.
///
/// This is the validating counterpart of [`resolve_coupling_fn`], used at the
/// Python boundary (mirroring the JAX resolver, which raises `ValueError`).
/// `"identity"` is accepted as an alias of `"linear"` for JAX-name parity.
pub fn parse_coupling_fn(name: &str) -> Result<&'static CouplingFn, String> {
    match name {
        "linear" | "identity" => Ok(&LINEAR),
        "relu" => Ok(&RELU),
        "sigmoid" => Ok(&SIGMOID),
        "tanh" => Ok(&TANH),
        "leaky_relu" => Ok(&LEAKY_RELU),
        "gelu" => Ok(&GELU),
        other => Err(format!(
            "Unknown coupling function '{other}'. Choose from [\"linear\", \
             \"identity\", \"relu\", \"sigmoid\", \"tanh\", \"leaky_relu\", \
             \"gelu\"]."
        )),
    }
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
    /// Which named coupling function this is (used by the `with_coupling!`
    /// macro to dispatch hot loops onto monomorphised, inlinable function
    /// items instead of the `fn` pointers below).
    pub kind: CouplingKind,
    /// The activation function $f(x)$.
    pub f: fn(f64) -> f64,
    /// The first derivative $f'(x)$.
    pub df: fn(f64) -> f64,
    /// The second derivative $f''(x)$.
    pub d2f: fn(f64) -> f64,
}

/// Tag identifying one of the named coupling functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CouplingKind {
    /// Identity.
    Linear,
    /// Rectified linear unit.
    Relu,
    /// Logistic sigmoid.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Leaky ReLU (slope 0.01 for negative inputs).
    LeakyRelu,
    /// Gaussian error linear unit.
    Gelu,
}

/// Run `$body` with `$f`/`$df`/`$d2f` bound to the *function items* of the
/// coupling in `$cf`.
///
/// Calling through the `fn` pointers stored on [`CouplingFn`] is an indirect
/// call per element, which blocks inlining and SIMD vectorisation in the
/// per-node hot loops. This macro matches on [`CouplingFn::kind`] once and
/// duplicates the body per arm with the concrete (zero-sized, inlinable)
/// function items, so the compiler monomorphises each loop. Use underscore
/// names (e.g. `|f, _df, _d2f|`) for the derivatives a body does not need.
macro_rules! with_coupling {
    ($cf:expr, |$f:ident, $df:ident, $d2f:ident| $body:expr) => {{
        match $cf.kind {
            $crate::math::CouplingKind::Linear => {
                let $f = $crate::math::linear;
                let $df = $crate::math::linear_d1;
                let $d2f = $crate::math::linear_d2;
                $body
            }
            $crate::math::CouplingKind::Relu => {
                let $f = $crate::math::relu;
                let $df = $crate::math::relu_d1;
                let $d2f = $crate::math::relu_d2;
                $body
            }
            $crate::math::CouplingKind::Sigmoid => {
                let $f = $crate::math::sigmoid;
                let $df = $crate::math::sigmoid_d1;
                let $d2f = $crate::math::sigmoid_d2;
                $body
            }
            $crate::math::CouplingKind::Tanh => {
                let $f = $crate::math::tanh;
                let $df = $crate::math::tanh_d1;
                let $d2f = $crate::math::tanh_d2;
                $body
            }
            $crate::math::CouplingKind::LeakyRelu => {
                let $f = $crate::math::leaky_relu;
                let $df = $crate::math::leaky_relu_d1;
                let $d2f = $crate::math::leaky_relu_d2;
                $body
            }
            $crate::math::CouplingKind::Gelu => {
                let $f = $crate::math::gelu;
                let $df = $crate::math::gelu_d1;
                let $d2f = $crate::math::gelu_d2;
                $body
            }
        }
    }};
}
pub(crate) use with_coupling;

// ── Coupling functions ────────────────────────────────────────────────────────

// ─── Linear ──────────────────────────────────────────────────────────────────

/// Linear (identity) coupling function: $f(x) = x$.
pub fn linear(x: f64) -> f64 {
    x
}
/// First derivative of linear: $f'(x) = 1$.
pub fn linear_d1(_x: f64) -> f64 {
    1.0
}
/// Second derivative of linear: $f''(x) = 0$.
pub fn linear_d2(_x: f64) -> f64 {
    0.0
}
/// [`CouplingFn`] constant for the linear (identity) coupling function.
pub const LINEAR: CouplingFn = CouplingFn {
    kind: CouplingKind::Linear,
    f: linear,
    df: linear_d1,
    d2f: linear_d2,
};

// ─── ReLU ────────────────────────────────────────────────────────────────────

/// Rectified Linear Unit: $f(x) = \max(0, x)$.
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}
/// First derivative of ReLU: $f'(x) = 1$ if $x > 0$, else $0$ (zero at $x = 0$).
pub fn relu_d1(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}
/// Second derivative of ReLU: $f''(x) = 0$ (almost everywhere).
pub fn relu_d2(_x: f64) -> f64 {
    0.0
}
/// [`CouplingFn`] constant for the ReLU coupling function.
pub const RELU: CouplingFn = CouplingFn {
    kind: CouplingKind::Relu,
    f: relu,
    df: relu_d1,
    d2f: relu_d2,
};

// ─── Sigmoid ─────────────────────────────────────────────────────────────────

/// Sigmoid: $f(x) = 1 / (1 + e^{-x})$.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
/// First derivative of sigmoid: $f'(x) = f(x)(1 - f(x))$.
pub fn sigmoid_d1(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}
/// Second derivative of sigmoid: $f''(x) = f'(x)(1 - 2f(x))$.
pub fn sigmoid_d2(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s) * (1.0 - 2.0 * s)
}
/// [`CouplingFn`] constant for the sigmoid coupling function.
pub const SIGMOID: CouplingFn = CouplingFn {
    kind: CouplingKind::Sigmoid,
    f: sigmoid,
    df: sigmoid_d1,
    d2f: sigmoid_d2,
};

// ─── Tanh ─────────────────────────────────────────────────────────────────────

/// Hyperbolic tangent: $f(x) = \tanh(x)$.
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}
/// First derivative of tanh: $f'(x) = 1 - \tanh^2(x)$.
pub fn tanh_d1(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}
/// Second derivative of tanh: $f''(x) = -2\tanh(x)(1 - \tanh^2(x))$.
pub fn tanh_d2(x: f64) -> f64 {
    let t = x.tanh();
    -2.0 * t * (1.0 - t * t)
}
/// [`CouplingFn`] constant for the tanh coupling function.
pub const TANH: CouplingFn = CouplingFn {
    kind: CouplingKind::Tanh,
    f: tanh,
    df: tanh_d1,
    d2f: tanh_d2,
};

// ─── Leaky ReLU ──────────────────────────────────────────────────────────────

/// Leaky ReLU with fixed slope $\alpha = 0.01$: $f(x) = x$ if $x \ge 0$, else $0.01x$.
pub fn leaky_relu(x: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        0.01 * x
    }
}
/// First derivative of Leaky ReLU: $f'(x) = 1$ if $x \ge 0$, else $0.01$.
pub fn leaky_relu_d1(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.01
    }
}
/// Second derivative of Leaky ReLU: $f''(x) = 0$ (almost everywhere).
pub fn leaky_relu_d2(_x: f64) -> f64 {
    0.0
}
/// [`CouplingFn`] constant for the Leaky ReLU coupling function.
pub const LEAKY_RELU: CouplingFn = CouplingFn {
    kind: CouplingKind::LeakyRelu,
    f: leaky_relu,
    df: leaky_relu_d1,
    d2f: leaky_relu_d2,
};

// ─── PReLU ───────────────────────────────────────────────────────────────────

/// Parametric ReLU: $f(x, \alpha) = x$ if $x \ge 0$, else $\alpha x$.
///
/// Note: PReLU requires a free `alpha` parameter, so it cannot be stored in a
/// `CouplingFn` constant. Use [`leaky_relu`] (fixed $\alpha = 0.01$) or
/// [`LEAKY_RELU`] as the bundled variant.
pub fn prelu(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        alpha * x
    }
}
/// First derivative of PReLU.
pub fn prelu_d1(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        alpha
    }
}
/// Second derivative of PReLU: $f''(x) = 0$ (almost everywhere).
pub fn prelu_d2(_x: f64, _alpha: f64) -> f64 {
    0.0
}

// ─── GELU ────────────────────────────────────────────────────────────────────

/// Complementary error function (internal helper for [`gelu`] and [`gelu_d1`]).
///
/// Abramowitz & Stegun §7.1.26 rational approximation; max error < 1.5 × 10⁻⁷.
fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let approx = poly * (-(x * x)).exp();
    if x >= 0.0 {
        approx
    } else {
        2.0 - approx
    }
}

/// GELU: $f(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard-normal CDF.
///
/// Implements the exact (erf-based) GELU through the `erfc` rational
/// approximation below, with a maximum absolute error of ~2 × 10⁻⁷ over
/// $x \in [-10, 10]$.
///
/// Cross-backend note: the JAX backend resolves `"gelu"` to `jax.nn.gelu`,
/// whose default is the *tanh* approximation — up to ~5 × 10⁻⁴ from the exact
/// GELU — so GELU-coupled networks agree across backends only to that
/// tolerance. Passing `approximate=False` on the JAX side tightens the
/// agreement to the ~2 × 10⁻⁷ of this approximation. Every other coupling
/// function evaluates the same closed form in both backends and matches to
/// machine precision.
pub fn gelu(x: f64) -> f64 {
    x * 0.5 * erfc(-x / std::f64::consts::SQRT_2)
}
/// First derivative of GELU: $f'(x) = \Phi(x) + x\,\phi(x)$.
pub fn gelu_d1(x: f64) -> f64 {
    let phi = 0.5 * erfc(-x / std::f64::consts::SQRT_2);
    let pdf = (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    phi + x * pdf
}
/// Second derivative of GELU: $f''(x) = \phi(x)(2 - x^2)$.
pub fn gelu_d2(x: f64) -> f64 {
    let pdf = (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    pdf * (2.0 - x * x)
}
/// [`CouplingFn`] constant for the GELU coupling function.
pub const GELU: CouplingFn = CouplingFn {
    kind: CouplingKind::Gelu,
    f: gelu,
    df: gelu_d1,
    d2f: gelu_d2,
};

// ─── Resolver ────────────────────────────────────────────────────────────────

/// Resolve a coupling-function name to its [`CouplingFn`] constant.
///
/// Called once at node-creation time in
/// [`Network::add_nodes`](crate::model::network::Network::add_nodes); the resulting
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
/// Any unrecognised name falls back to [`LINEAR`]; use [`parse_coupling_fn`]
/// where an unknown name should instead be an error.
pub fn resolve_coupling_fn(name: &str) -> &'static CouplingFn {
    parse_coupling_fn(name).unwrap_or(&LINEAR)
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
            label,
            expected,
            actual,
            (actual - expected).abs()
        );
    }

    // ── sufficient_statistics ─────────────────────────────────────────────────

    #[test]
    fn test_sufficient_statistics_positive() {
        let s = sufficient_statistics(3.0);
        assert_close(s[0], 3.0, "sufficient_statistics[0] for x=3");
        assert_close(s[1], 9.0, "sufficient_statistics[1] for x=3");
    }

    #[test]
    fn test_sufficient_statistics_negative() {
        let s = sufficient_statistics(-2.5);
        assert_close(s[0], -2.5, "sufficient_statistics[0] for x=-2.5");
        assert_close(s[1], 6.25, "sufficient_statistics[1] for x=-2.5");
    }

    #[test]
    fn test_sufficient_statistics_zero() {
        let s = sufficient_statistics(0.0);
        assert_close(s[0], 0.0, "sufficient_statistics[0] for x=0");
        assert_close(s[1], 0.0, "sufficient_statistics[1] for x=0");
    }

    // ── lambert_w0 ────────────────────────────────────────────────────────────

    #[test]
    fn test_lambert_w0_known_values() {
        assert_close(lambert_w0(0.0), 0.0, "W0(0)");
        assert_close(lambert_w0(std::f64::consts::E), 1.0, "W0(e)");
    }

    #[test]
    fn test_lambert_w0_round_trip() {
        // w · e^w must recover z across the range the volatility updates use.
        for &z in &[0.0, 0.1, 0.5, 1.0, 10.0, 100.0, 1e4] {
            let w = lambert_w0(z);
            let back = w * w.exp();
            assert!(
                (back - z).abs() <= 1e-9 * z.max(1.0),
                "W0 round trip at z={z}: w·e^w = {back}"
            );
        }
    }

    // ── logaddexp ─────────────────────────────────────────────────────────────

    #[test]
    fn test_logaddexp_matches_direct() {
        for &(a, b) in &[(0.0, 0.0), (1.0, 2.0), (-3.0, 4.0)] {
            assert_close(
                logaddexp(a, b),
                (a.exp() + b.exp()).ln(),
                &format!("logaddexp({a}, {b})"),
            );
        }
    }

    #[test]
    fn test_logaddexp_extremes() {
        // Large arguments must not overflow: ln(2·e^1000) = 1000 + ln 2.
        assert_close(
            logaddexp(1000.0, 1000.0),
            1000.0 + 2.0_f64.ln(),
            "logaddexp(1000, 1000)",
        );
        assert_close(logaddexp(f64::NEG_INFINITY, 5.0), 5.0, "logaddexp(-inf, 5)");
        assert_eq!(
            logaddexp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        assert_eq!(logaddexp(f64::INFINITY, 5.0), f64::INFINITY);
        assert_eq!(logaddexp(f64::INFINITY, f64::INFINITY), f64::INFINITY);
    }

    // ── coupling-function resolvers ───────────────────────────────────────────

    #[test]
    fn test_parse_coupling_fn_known_names() {
        for name in [
            "linear",
            "identity",
            "relu",
            "sigmoid",
            "tanh",
            "leaky_relu",
            "gelu",
        ] {
            assert!(parse_coupling_fn(name).is_ok(), "'{name}' should parse");
        }
        assert_eq!(
            parse_coupling_fn("identity").unwrap().kind,
            CouplingKind::Linear
        );
    }

    #[test]
    fn test_parse_coupling_fn_unknown_name() {
        let err = parse_coupling_fn("sigmiod").unwrap_err();
        assert!(
            err.contains("sigmiod"),
            "error should name the offender: {err}"
        );
    }

    #[test]
    fn test_resolve_coupling_fn_fallback() {
        assert_eq!(resolve_coupling_fn("tanh").kind, CouplingKind::Tanh);
        assert_eq!(resolve_coupling_fn("sigmiod").kind, CouplingKind::Linear);
    }

    // ── derivatives vs central differences ────────────────────────────────────

    /// Check `d1`/`d2` against central differences of `f` at off-kink points.
    fn check_derivatives(f: fn(f64) -> f64, d1: fn(f64) -> f64, d2: fn(f64) -> f64, name: &str) {
        let h = 1e-5;
        for &x in &[-2.0, -0.5, 0.3, 1.7] {
            let num_d1 = (f(x + h) - f(x - h)) / (2.0 * h);
            assert!(
                (d1(x) - num_d1).abs() < 1e-4,
                "{name}' at {x}: analytic {}, numeric {num_d1}",
                d1(x)
            );
            let num_d2 = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
            assert!(
                (d2(x) - num_d2).abs() < 1e-4,
                "{name}'' at {x}: analytic {}, numeric {num_d2}",
                d2(x)
            );
        }
    }

    #[test]
    fn test_derivatives_match_finite_differences() {
        check_derivatives(linear, linear_d1, linear_d2, "linear");
        check_derivatives(relu, relu_d1, relu_d2, "relu");
        check_derivatives(sigmoid, sigmoid_d1, sigmoid_d2, "sigmoid");
        check_derivatives(tanh, tanh_d1, tanh_d2, "tanh");
        check_derivatives(leaky_relu, leaky_relu_d1, leaky_relu_d2, "leaky_relu");
        check_derivatives(gelu, gelu_d1, gelu_d2, "gelu");
    }

    #[test]
    fn test_with_coupling_matches_fn_pointers() {
        // The macro must bind the same functions the constants store as pointers.
        for cf in [&LINEAR, &RELU, &SIGMOID, &TANH, &LEAKY_RELU, &GELU] {
            let x = 0.7;
            let (fv, dfv, d2fv) = with_coupling!(cf, |f, df, d2f| (f(x), df(x), d2f(x)));
            assert_close(fv, (cf.f)(x), "with_coupling f");
            assert_close(dfv, (cf.df)(x), "with_coupling df");
            assert_close(d2fv, (cf.d2f)(x), "with_coupling d2f");
        }
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
        assert!(sigmoid(10.0) < 1.0, "sigmoid(10) should be < 1");
    }

    #[test]
    fn test_sigmoid_negative_large() {
        // σ(−∞) → 0; σ(−10) ≈ 4.54e-5
        assert!(sigmoid(-10.0) < 0.001, "sigmoid(-10) should be < 0.001");
        assert!(sigmoid(-10.0) > 0.0, "sigmoid(-10) should be > 0");
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
        assert!(tanh(20.0) <= 1.0, "tanh(20) <= 1");
        assert!(tanh(20.0) > 0.99, "tanh(20) > 0.99");
        assert!(tanh(-20.0) >= -1.0, "tanh(-20) >= -1");
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
        assert_close(prelu(2.0, 0.0), 2.0, "prelu(2,  α=0) == relu");
    }

    #[test]
    fn test_prelu_matches_leaky_relu() {
        // prelu with α=0.01 must equal leaky_relu
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            assert_close(
                prelu(x, 0.01),
                leaky_relu(x),
                &format!("prelu(α=0.01) == leaky_relu at x={}", x),
            );
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
        assert!(gelu(-10.0).abs() < 1e-4, "gelu(-10) ≈ 0");
    }

    #[test]
    fn test_gelu_known_value() {
        // GELU(1) ≈ 0.8413447  (x × Φ(1), Φ(1) ≈ 0.8413447)
        let expected = 0.841_344_7;
        assert!(
            (gelu(1.0) - expected).abs() < 1e-5,
            "gelu(1) expected ≈ {}, got {}",
            expected,
            gelu(1.0)
        );
    }

    #[test]
    fn test_gelu_negative_value() {
        // GELU(−1) = −1 × Φ(−1) ≈ −1 × 0.1586553 ≈ −0.1586553
        let expected = -0.158_655_3;
        assert!(
            (gelu(-1.0) - expected).abs() < 1e-5,
            "gelu(-1) expected ≈ {}, got {}",
            expected,
            gelu(-1.0)
        );
    }
}
