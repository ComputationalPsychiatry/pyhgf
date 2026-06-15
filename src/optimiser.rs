/// Adam optimiser state for filtering coupling weight updates.
///
/// Maintains per-weight first moment (m) and second moment (v) estimates,
/// following Kingma & Ba (2015).

#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment vectors, indexed [node_idx][parent_coupling_idx].
    pub m: Vec<Vec<f64>>,
    /// Second moment vectors, indexed [node_idx][parent_coupling_idx].
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
    /// Returns the **signed step** to *add* to the current coupling value
    /// (i.e. `lr` is already factored in by the caller).
    ///
    /// * `node_idx` / `parent_pos` — locate the m/v slot.
    /// * `gradient` — the raw gradient (before any learning-rate scaling).
    /// * `lr` — base learning rate.
    pub fn step(&mut self, node_idx: usize, parent_pos: usize, gradient: f64, lr: f64) -> f64 {
        // Update biased first moment estimate
        self.m[node_idx][parent_pos] =
            self.beta1 * self.m[node_idx][parent_pos] + (1.0 - self.beta1) * gradient;

        // Update biased second raw moment estimate
        self.v[node_idx][parent_pos] =
            self.beta2 * self.v[node_idx][parent_pos] + (1.0 - self.beta2) * gradient * gradient;

        // Bias-corrected estimates
        let m_hat = self.m[node_idx][parent_pos] / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = self.v[node_idx][parent_pos] / (1.0 - self.beta2.powi(self.t as i32));

        lr * m_hat / (v_hat.sqrt() + self.epsilon)
    }
}
