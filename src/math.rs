/// Sufficient statistics `[x, x²]` of a univariate Gaussian observation.
pub fn sufficient_statistics(x: f64) -> Vec<f64> {
    vec![x, x * x]
}

/// Principal branch of the Lambert W function for `z >= 0`; see [`lambert_w0_log`].
pub fn lambert_w0(z: f64) -> f64 {
    lambert_w0_log(z.ln())
}

/// Principal branch of the Lambert W function of `z = exp(log_z)`, evaluated from
/// `log_z` without forming `z`. Used by the `unbounded` volatility updates.
///
/// Solves `w * exp(w) = z` via 6 Halley iterations from `w = ln(1 + z)`, the same
/// scheme as `pyhgf.math.lambert_w0_log`, so the two backends agree to machine
/// precision. `F`, `F'` and `F''` are scaled by `exp(-w)`, a common factor that
/// cancels from Halley's ratio, so only `z * exp(-w)` is formed.
pub fn lambert_w0_log(log_z: f64) -> f64 {
    let mut w = logaddexp(0.0, log_z);
    for _ in 0..6 {
        let f = w - (log_z - w).exp();
        w -= 2.0 * f * (w + 1.0) / (2.0 * (w + 1.0) * (w + 1.0) - f * (w + 2.0));
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

/// Parse the `coupling_moments` argument of a network: `Ok(false)` for `"taylor"`
/// (first-order Taylor expansion of the coupling functions) and `Ok(true)` for
/// `"exact"` (their moments under the belief of the parent), erroring on any other
/// name. Used at the Python boundary, as [`parse_coupling_fn`].
pub fn parse_coupling_moments(name: &str) -> Result<bool, String> {
    match name {
        "taylor" => Ok(false),
        "exact" => Ok(true),
        other => Err(format!(
            "Invalid coupling_moments '{other}'. Choose from 'taylor' or 'exact'."
        )),
    }
}

/// A coupling (activation) function together with its first and second derivatives
/// and its moments under a Gaussian.
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
/// let (mean, variance, slope) = (c.moments)(mu, sigma2);
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
    /// The moments of $f(x)$ for $x \sim \mathcal{N}(\mu, \sigma^2)$, as a function
    /// of $(\mu, \sigma^2)$: the mean $E[f(x)]$, the variance $\mathrm{Var}[f(x)]$
    /// and the mean slope $E[f'(x)] = \mathrm{Cov}[x, f(x)] / \sigma^2$. Mirrors
    /// `pyhgf.coupling.Coupling.moments`.
    pub moments: fn(f64, f64) -> (f64, f64, f64),
}

impl CouplingFn {
    /// Statistical linearisation of the function under $x \sim \mathcal{N}(\mu,
    /// \sigma^2)$: the linear function $c + A (x - \mu)$ closest to $f$ in mean
    /// squared error, with a residual of variance $R$. Returns `(c, A, R)` with
    /// $c = E[f(x)]$, $A = E[f'(x)]$ and $R = \mathrm{Var}[f(x)] - A^2 \sigma^2$,
    /// floored at zero against rounding. Mirrors `pyhgf.coupling.Coupling.linearise`
    /// with `coupling_moments="exact"`.
    #[inline]
    pub fn linearise(&self, mean: f64, variance: f64) -> (f64, f64, f64) {
        let (value, coupling_variance, slope) = (self.moments)(mean, variance);
        (
            value,
            slope,
            (coupling_variance - slope * slope * variance).max(0.0),
        )
    }

    /// [`Self::linearise`] for the vectorised engine. In single precision (the
    /// default engine), the rectified linear units take the distribution function of
    /// the normal distribution from a rational approximation of absolute error below
    /// $10^{-7}$ ([`normal_tail_single`]), which is below the resolution of the
    /// engine and several times cheaper than [`normal_cdf`]. With the `f64` feature
    /// this is [`Self::linearise`].
    #[inline]
    pub fn linearise_engine(&self, mean: f64, variance: f64) -> (f64, f64, f64) {
        #[cfg(not(feature = "f64"))]
        let (value, coupling_variance, slope) = match self.kind {
            CouplingKind::Relu => rectifier_moments_with(mean, variance, 0.0, normal_tail_single),
            CouplingKind::LeakyRelu => {
                rectifier_moments_with(mean, variance, 0.01, normal_tail_single)
            }
            _ => (self.moments)(mean, variance),
        };
        #[cfg(feature = "f64")]
        let (value, coupling_variance, slope) = (self.moments)(mean, variance);
        (
            value,
            slope,
            (coupling_variance - slope * slope * variance).max(0.0),
        )
    }
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

// ── Gaussian moments of coupling functions ───────────────────────────────────

/// Distribution function of the standard normal distribution, $\Phi(z)$.
///
/// $\Phi(z) = \tfrac12 \mathrm{erfc}(-z / \sqrt{2})$, with the complementary error
/// function computed from the series
/// $\mathrm{erf}(a) = \tfrac{2}{\sqrt{\pi}} e^{-a^2} \sum_n 2^n a^{2n+1} / (2n+1)!!$
/// for $a < 3$ and from its continued fraction for $a \ge 3$. The absolute error is
/// below $2 \times 10^{-15}$, and the relative error of the tail is below
/// $10^{-13}$ for $|z| \ge 3 \sqrt{2}$.
pub fn normal_cdf(z: f64) -> f64 {
    let a = z.abs() / std::f64::consts::SQRT_2;
    let erfc = if a < 3.0 {
        // Series of erf, whose terms are all positive.
        let mut term = a;
        let mut sum = a;
        for n in 1..200 {
            term *= 2.0 * a * a / (2.0 * n as f64 + 1.0);
            sum += term;
            if term <= 1e-17 * sum {
                break;
            }
        }
        1.0 - std::f64::consts::FRAC_2_SQRT_PI * (-a * a).exp() * sum
    } else {
        // Continued fraction of erfc, evaluated from its 40th term.
        let mut fraction = a;
        for k in (1..=40).rev() {
            fraction = a + 0.5 * k as f64 / fraction;
        }
        (-a * a).exp() / (std::f64::consts::PI.sqrt() * fraction)
    };
    if z >= 0.0 {
        1.0 - 0.5 * erfc
    } else {
        0.5 * erfc
    }
}

/// Moments of a leaky rectified linear unit under a Gaussian, in closed form.
///
/// For $x \sim \mathcal{N}(\mu, \sigma^2)$ and $g(x) = \max(x, 0) + s \min(x, 0)$,
/// with $z = \mu / \sigma$:
///
/// ```text
/// E[g]   = s μ + (1 − s) σ (φ(z) + z Φ(z)),
/// E[g']  = s + (1 − s) Φ(z),
/// Var[g] = σ² (s² + 2 s (1 − s) Φ(z) + (1 − s)² r(z)),
/// r(z)   = z² Φ(z) Φ(−z) + Φ(z) + z φ(z) (Φ(−z) − Φ(z)) − φ(z)².
/// ```
///
/// Returns `(E[g], Var[g], E[g'])`. The negative slope `s = 0` gives the rectified
/// linear unit and `s = 1` the identity. Mirrors `pyhgf.coupling.leaky_relu_moments`.
pub fn rectifier_moments(mean: f64, variance: f64, negative_slope: f64) -> (f64, f64, f64) {
    rectifier_moments_with(mean, variance, negative_slope, |a| normal_cdf(-a))
}

/// Mass of the standard normal distribution beyond `a >= 0`, $\Phi(-a)$, from the
/// rational approximation of the complementary error function used by [`gelu`]
/// (absolute error below $10^{-7}$). For the single precision engine.
#[cfg_attr(feature = "f64", allow(dead_code))]
#[inline]
fn normal_tail_single(a: f64) -> f64 {
    0.5 * erfc(a / std::f64::consts::SQRT_2)
}

/// [`rectifier_moments`] with the mass beyond `|z|`, $\Phi(-|z|)$, given by `tail`.
#[inline]
fn rectifier_moments_with(
    mean: f64,
    variance: f64,
    negative_slope: f64,
    tail: impl Fn(f64) -> f64,
) -> (f64, f64, f64) {
    let variance = if variance.is_nan() {
        f64::NAN
    } else {
        variance.max(0.0)
    };
    // The clipping of z gives the limits of a point belief without forming 0 · ∞.
    let sigma = variance.max(f64::MIN_POSITIVE).sqrt();
    let z = (mean / sigma).clamp(-40.0, 40.0);
    // Φ(z) and Φ(−z) from one evaluation of the tail beyond |z|.
    let tail = tail(z.abs());
    let (cdf, survival) = if z >= 0.0 {
        (1.0 - tail, tail)
    } else {
        (tail, 1.0 - tail)
    };
    let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();

    let slope = negative_slope + (1.0 - negative_slope) * cdf;
    let expected_value =
        negative_slope * mean + (1.0 - negative_slope) * (mean * cdf + sigma * pdf);
    // Variance of the rectified linear unit for a unit variance, written without the
    // cancellation of E[g²] − E[g]².
    let relu_variance = z * z * cdf * survival + cdf + z * pdf * (survival - cdf) - pdf * pdf;
    let expected_variance = variance
        * (negative_slope * negative_slope
            + 2.0 * negative_slope * (1.0 - negative_slope) * cdf
            + (1.0 - negative_slope) * (1.0 - negative_slope) * relu_variance);
    (expected_value, expected_variance.max(0.0), slope)
}

/// Positive nodes of the 20-point probabilists' Gauss-Hermite rule. The rule is
/// symmetric: the other 10 nodes are their opposites.
const GAUSS_HERMITE_NODES: [f64; 10] = [
    0.346964157081356,
    1.042945348802751,
    1.7452473208141268,
    2.458663611172368,
    3.1890148165533896,
    3.9439673506573163,
    4.734581334046055,
    5.5787388058932015,
    6.510590157013654,
    7.619048541679758,
];

/// Weights of the nodes of [`GAUSS_HERMITE_NODES`] (and of their opposites),
/// normalised so that the 20 weights sum to 1.
const GAUSS_HERMITE_WEIGHTS: [f64; 10] = [
    0.2607930634495548,
    0.16173933398400003,
    0.06150637206397696,
    0.013997837447100996,
    0.0018301031310804924,
    0.00012882627996192942,
    4.4021210902308646e-06,
    6.127490259982928e-08,
    2.4820623623151797e-10,
    1.2578006724379264e-13,
];

/// Moments of a function under a Gaussian, by 20-point Gauss-Hermite quadrature.
///
/// For $x \sim \mathcal{N}(\mu, \sigma^2)$, $E[h(x)] \approx \sum_i w_i
/// h(\mu + \sigma \xi_i)$ with the nodes $\xi_i$ and weights $w_i$ of the
/// probabilists' Gauss-Hermite rule, applied to $f$, to $(f - E[f])^2$ and to $f'$.
/// Returns `(E[f], Var[f], E[f'])`. Mirrors `pyhgf.coupling.gauss_hermite_moments`
/// with its default number of nodes.
///
/// `value_and_slope` returns $(f(x), f'(x))$ in one call, so that what the two share
/// (the hyperbolic tangent, the exponential, the error function) is evaluated once
/// per node.
pub fn gauss_hermite_moments(
    value_and_slope: impl Fn(f64) -> (f64, f64),
    mean: f64,
    variance: f64,
) -> (f64, f64, f64) {
    let sigma = if variance.is_nan() {
        f64::NAN
    } else {
        variance.max(f64::MIN_POSITIVE).sqrt()
    };
    let mut values = [0.0_f64; 20];
    let mut expected_value = 0.0;
    let mut slope = 0.0;
    for (i, (&node, &weight)) in GAUSS_HERMITE_NODES
        .iter()
        .zip(GAUSS_HERMITE_WEIGHTS.iter())
        .enumerate()
    {
        for (j, sign) in [-1.0, 1.0].into_iter().enumerate() {
            let (value, derivative) = value_and_slope(mean + sign * sigma * node);
            values[2 * i + j] = value;
            expected_value += weight * value;
            slope += weight * derivative;
        }
    }
    let mut expected_variance = 0.0;
    for (i, &weight) in GAUSS_HERMITE_WEIGHTS.iter().enumerate() {
        for j in 0..2 {
            let deviation = values[2 * i + j] - expected_value;
            expected_variance += weight * deviation * deviation;
        }
    }
    (expected_value, expected_variance, slope)
}

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
/// Moments of linear under a Gaussian: $(\mu, \sigma^2, 1)$.
pub fn linear_moments(mean: f64, variance: f64) -> (f64, f64, f64) {
    (mean, variance, 1.0)
}
/// [`CouplingFn`] constant for the linear (identity) coupling function.
pub const LINEAR: CouplingFn = CouplingFn {
    kind: CouplingKind::Linear,
    f: linear,
    df: linear_d1,
    d2f: linear_d2,
    moments: linear_moments,
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
/// Moments of ReLU under a Gaussian, in closed form (see [`rectifier_moments`]).
pub fn relu_moments(mean: f64, variance: f64) -> (f64, f64, f64) {
    rectifier_moments(mean, variance, 0.0)
}
/// [`CouplingFn`] constant for the ReLU coupling function.
pub const RELU: CouplingFn = CouplingFn {
    kind: CouplingKind::Relu,
    f: relu,
    df: relu_d1,
    d2f: relu_d2,
    moments: relu_moments,
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
/// Moments of sigmoid under a Gaussian, by quadrature (see
/// [`gauss_hermite_moments`]).
pub fn sigmoid_moments(mean: f64, variance: f64) -> (f64, f64, f64) {
    gauss_hermite_moments(
        |x| {
            let s = sigmoid(x);
            (s, s * (1.0 - s))
        },
        mean,
        variance,
    )
}
/// [`CouplingFn`] constant for the sigmoid coupling function.
pub const SIGMOID: CouplingFn = CouplingFn {
    kind: CouplingKind::Sigmoid,
    f: sigmoid,
    df: sigmoid_d1,
    d2f: sigmoid_d2,
    moments: sigmoid_moments,
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
/// Moments of tanh under a Gaussian, by quadrature (see [`gauss_hermite_moments`]).
pub fn tanh_moments(mean: f64, variance: f64) -> (f64, f64, f64) {
    gauss_hermite_moments(
        |x| {
            let t = x.tanh();
            (t, 1.0 - t.powi(2))
        },
        mean,
        variance,
    )
}
/// [`CouplingFn`] constant for the tanh coupling function.
pub const TANH: CouplingFn = CouplingFn {
    kind: CouplingKind::Tanh,
    f: tanh,
    df: tanh_d1,
    d2f: tanh_d2,
    moments: tanh_moments,
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
/// Moments of Leaky ReLU under a Gaussian, in closed form (see
/// [`rectifier_moments`]).
pub fn leaky_relu_moments(mean: f64, variance: f64) -> (f64, f64, f64) {
    rectifier_moments(mean, variance, 0.01)
}
/// [`CouplingFn`] constant for the Leaky ReLU coupling function.
pub const LEAKY_RELU: CouplingFn = CouplingFn {
    kind: CouplingKind::LeakyRelu,
    f: leaky_relu,
    df: leaky_relu_d1,
    d2f: leaky_relu_d2,
    moments: leaky_relu_moments,
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
/// Moments of GELU under a Gaussian, by quadrature (see [`gauss_hermite_moments`]).
pub fn gelu_moments(mean: f64, variance: f64) -> (f64, f64, f64) {
    gauss_hermite_moments(
        |x| {
            let phi = 0.5 * erfc(-x / std::f64::consts::SQRT_2);
            let pdf = (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
            (x * phi, phi + x * pdf)
        },
        mean,
        variance,
    )
}
/// [`CouplingFn`] constant for the GELU coupling function.
pub const GELU: CouplingFn = CouplingFn {
    kind: CouplingKind::Gelu,
    f: gelu,
    df: gelu_d1,
    d2f: gelu_d2,
    moments: gelu_moments,
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

    #[test]
    fn test_lambert_w0_large_finite_log_residual() {
        for &z in &[1e18, 1e50, 1e150, 1e250, 1e300] {
            let w = lambert_w0(z);
            assert!(w.is_finite() && w > 0.0, "non-finite W0({z}) = {w}");
            let residual = w.ln() + w - z.ln();
            assert!(residual.abs() < 2e-11, "W0 log residual at {z}: {residual}");
        }
        // Independent principal-branch reference from scipy.special.lambertw.
        assert_close(lambert_w0(1e250), 569.3018624411017, "W0(1e250)");
    }

    #[test]
    fn test_lambert_w0_small_positive_round_trip() {
        for &z in &[1e-12, 1e-8, 1e-4] {
            let w = lambert_w0(z);
            let relative_error = (w * w.exp() - z).abs() / z;
            assert!(
                relative_error < 1e-12,
                "W0 relative error at {z}: {relative_error}"
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

    // ── Gaussian moments ──────────────────────────────────────────────────────

    #[test]
    fn test_normal_cdf_known_values() {
        // Reference values of the standard normal distribution function, on both
        // sides of the switch between the series and the continued fraction.
        for (z, expected) in [
            (-6.5, 4.016_000_583_859_087_4e-11),
            (-1.3, 0.096_800_484_585_610_36),
            (0.0, 0.5),
            (0.4, 0.655_421_741_610_324_2),
            (2.2, 0.986_096_552_486_501_4),
            (5.0, 0.999_999_713_348_428_1),
        ] {
            let actual = normal_cdf(z);
            assert!(
                (actual - expected).abs() < 1e-14 + 1e-12 * expected,
                "normal_cdf({z}): expected {expected:e}, got {actual:e}"
            );
        }
        assert!(normal_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn test_coupling_moments_match_the_jax_backend() {
        // (coupling, mean, variance, [E[g], Var[g], E[g']]) from
        // `pyhgf.coupling.Coupling.moments` in double precision.
        let cases: [(&CouplingFn, f64, f64, [f64; 3]); 5] = [
            (&LINEAR, 0.3, 2.0, [0.3, 2.0, 1.0]),
            (
                &RELU,
                -0.75,
                1.0,
                [
                    0.131_166_917_872_153_3,
                    0.111_047_403_628_673_03,
                    0.226_627_352_376_868_2,
                ],
            ),
            (
                &LEAKY_RELU,
                0.4,
                2.25,
                [
                    0.815_369_496_231_326_8,
                    1.024_629_156_703_726,
                    0.609_085_718_640_615_1,
                ],
            ),
            (
                &TANH,
                0.3,
                0.64,
                [
                    0.203_113_845_084_327_3,
                    0.297_448_860_437_458_2,
                    0.661_295_905_497_601_8,
                ],
            ),
            (
                &SIGMOID,
                -1.0,
                1.0,
                [
                    0.303_265_329_874_443_8,
                    0.033_352_090_185_591_25,
                    0.177_943_379_384_997_35,
                ],
            ),
        ];
        for (coupling, mean, variance, expected) in cases {
            let (expected_value, expected_variance, slope) = (coupling.moments)(mean, variance);
            for (actual, expected) in [expected_value, expected_variance, slope]
                .into_iter()
                .zip(expected)
            {
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "{:?} moments at ({mean}, {variance}): expected {expected}, got {actual}",
                    coupling.kind
                );
            }
        }
    }

    #[test]
    fn test_engine_linearisation_matches_the_precise_one() {
        // The vectorised engine takes the distribution function of the rectified
        // linear units from a rational approximation: its linearisation stays
        // within the resolution of single precision of the precise one.
        for coupling in [&RELU, &LEAKY_RELU, &TANH] {
            for (mean, variance) in [
                (-2.0, 0.5),
                (-0.75, 1.0),
                (0.0, 1.0),
                (0.4, 2.25),
                (3.0, 0.3),
            ] {
                let precise = coupling.linearise(mean, variance);
                let engine = coupling.linearise_engine(mean, variance);
                for (a, b) in [precise.0, precise.1, precise.2]
                    .into_iter()
                    .zip([engine.0, engine.1, engine.2])
                {
                    assert!(
                        (a - b).abs() < 2e-6,
                        "{:?} at ({mean}, {variance}): precise {a}, engine {b}",
                        coupling.kind
                    );
                }
            }
        }
    }

    #[test]
    fn test_coupling_moments_of_a_point_belief() {
        // With a variance of 0, the moments are g(μ), 0 and g'(μ).
        for coupling in [&LINEAR, &RELU, &LEAKY_RELU, &TANH, &SIGMOID, &GELU] {
            for mean in [-1.5, 0.4, 2.0] {
                let (expected_value, expected_variance, slope) = (coupling.moments)(mean, 0.0);
                assert_close(expected_value, (coupling.f)(mean), "point belief mean");
                assert_close(expected_variance, 0.0, "point belief variance");
                assert_close(slope, (coupling.df)(mean), "point belief slope");
            }
        }
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
