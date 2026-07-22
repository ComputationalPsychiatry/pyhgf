//! Adam optimiser for the per-node backend's coupling-weight updates.

/// Adam optimiser state for filtering coupling weight updates.
///
/// Maintains per-weight first moment (m) and second moment (v) estimates,
/// following Kingma & Ba (2015).
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment vectors, indexed `[node_idx][parent_coupling_idx]`.
    pub m: Vec<Vec<f64>>,
    /// Second moment vectors, indexed `[node_idx][parent_coupling_idx]`.
    pub v: Vec<Vec<f64>>,
    /// Global timestep counter (incremented once per fit iteration).
    pub t: usize,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    /// Optional Adam-specific learning rate.  When `Some`, overrides the
    /// per-node `lr` used by `learning_weights`.
    pub lr: Option<f64>,
}

impl AdamState {
    /// Create a new Adam state sized to match the network's coupling structure.
    ///
    /// `coupling_sizes[i]` is the number of value-coupling parents for node `i`.
    pub fn new(coupling_sizes: &[usize], beta1: f64, beta2: f64, epsilon: f64) -> Self {
        let m: Vec<Vec<f64>> = coupling_sizes.iter().map(|&n| vec![0.0; n]).collect();
        let v: Vec<Vec<f64>> = coupling_sizes.iter().map(|&n| vec![0.0; n]).collect();
        AdamState {
            m,
            v,
            t: 0,
            beta1,
            beta2,
            epsilon,
            lr: None,
        }
    }

    /// Increment the global timestep.  Call once at the start of each fit iteration.
    pub fn increment_timestep(&mut self) {
        self.t += 1;
    }

    /// Compute the Adam-filtered update for a single coupling weight.
    ///
    /// Returns `lr · m̂ / (√v̂ + ε)` — already scaled by the learning rate and
    /// carrying the sign of the bias-corrected first moment — for the caller
    /// to *add* to the current coupling value.
    ///
    /// * `node_idx` / `parent_pos` — locate the m/v slot.
    /// * `gradient` — the raw gradient (before any learning-rate scaling).
    /// * `lr` — base learning rate.
    pub fn step(&mut self, node_idx: usize, parent_pos: usize, gradient: f64, lr: f64) -> f64 {
        debug_assert!(
            self.t > 0,
            "AdamState::step requires increment_timestep to have been called \
             first: the bias correction 1 - beta^t is zero at t = 0"
        );
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        adam_step(
            &mut self.m[node_idx][parent_pos],
            &mut self.v[node_idx][parent_pos],
            gradient,
            self.beta1,
            self.beta2,
            bc1,
            bc2,
            lr,
            self.epsilon,
        )
    }
}

/// One Adam step (Kingma & Ba, 2015) for a single weight, shared by the
/// per-node [`AdamState`] and the vectorised
/// [`crate::vectorised::optimiser::Optimizer`] so the update formula has a
/// single source of truth.
///
/// Updates the moment estimates `m`/`v` in place and returns
/// `lr · m̂ / (√v̂ + ε)`, which carries the sign of the bias-corrected first
/// moment. Each caller applies its own gradient convention: the per-node
/// backend *adds* the result to the coupling weight, the vectorised backend
/// *subtracts* it. `bc1`/`bc2` are the precomputed bias corrections `1 − βᵢᵗ`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn adam_step(
    m: &mut f64,
    v: &mut f64,
    gradient: f64,
    beta1: f64,
    beta2: f64,
    bc1: f64,
    bc2: f64,
    lr: f64,
    epsilon: f64,
) -> f64 {
    *m = beta1 * *m + (1.0 - beta1) * gradient;
    *v = beta2 * *v + (1.0 - beta2) * gradient * gradient;
    let m_hat = *m / bc1;
    let v_hat = *v / bc2;
    lr * m_hat / (v_hat.sqrt() + epsilon)
}
