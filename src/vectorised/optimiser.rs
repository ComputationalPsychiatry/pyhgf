//! Gradient-descent optimisers over the network's weight matrices, mirroring
//! the optax transforms the Python `fit` accepts (`optax.sgd`, `optax.adam`).
//! They consume the rank-one gradient *factors* from
//! [`crate::updates::vectorised::learning`] and form each element `u[i]·v[j]`
//! inline, so no gradient matrix is ever materialised.

use crate::vectorised::layer::{DeepNet, Layer};
use crate::vectorised::mat::{Matrix, Vector};

/// A gradient-descent optimizer over the network's weight matrices, mirroring
/// the optax transforms `fit` accepts. `PartialEq` lets the Python `fit`
/// detect an optimizer change between calls and reset the state, matching the
/// JAX class's re-initialisation on a new optax object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Optimizer {
    /// Plain SGD: `w ← w − lr · grad`.
    Sgd { lr: f64 },
    /// Adam (Kingma & Ba, 2015), matching `optax.adam` defaults.
    Adam { lr: f64, b1: f64, b2: f64, eps: f64 },
}

impl Optimizer {
    /// `optax.sgd(lr)`.
    pub fn sgd(lr: f64) -> Self {
        Optimizer::Sgd { lr }
    }

    /// `optax.adam(lr)` with the standard `b1=0.9, b2=0.999, eps=1e-8`.
    pub fn adam(lr: f64) -> Self {
        Optimizer::Adam {
            lr,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
        }
    }
}

/// Mutable optimizer state: a global step counter plus per-layer Adam moment
/// pairs (`None` for layers without incoming weights, e.g. the bottom layer).
#[derive(Debug, Clone)]
pub struct OptState {
    /// Global step counter (Adam bias correction). Shared across all weights.
    pub t: i32,
    /// Per-layer `(m, v)` moment matrices; `None` where the layer has no weights.
    pub moments: Vec<Option<(Matrix, Matrix)>>,
}

impl OptState {
    /// Allocate zeroed moments matching each layer's `weights_in`.
    pub fn init(net: &DeepNet) -> Self {
        let moments = net
            .layers
            .iter()
            .map(|layer| {
                layer.weights_in.as_ref().map(|w| {
                    let z = Matrix::zeros(w.raw_dim());
                    (z.clone(), z)
                })
            })
            .collect();
        OptState { t: 0, moments }
    }
}

impl Optimizer {
    /// Apply one optimizer step to every layer's `weights_in` from the matched
    /// per-layer descent-gradient factors (`grads[i]` is `None` where the layer
    /// has no weights; otherwise `(u, v)` with `u[i]·v[j]` the gradient
    /// element). Updates `state` and the weight matrices in place — the
    /// gradient is consumed factor-wise, never materialised as a matrix.
    pub fn apply(
        &self,
        state: &mut OptState,
        layers: &mut [Layer],
        grads: &[Option<(Vector, Vector)>],
    ) {
        // The step counter only feeds Adam's bias correction; SGD leaves it
        // untouched so a later switch to Adam starts from a fresh schedule.
        if let Optimizer::Adam { .. } = self {
            state.t += 1;
        }
        for (i, layer) in layers.iter_mut().enumerate() {
            let (u, v) = match &grads[i] {
                Some(f) => f,
                None => continue,
            };
            let weights = layer
                .weights_in
                .as_mut()
                .expect("gradient present but layer has no weights");

            match *self {
                Optimizer::Sgd { lr } => {
                    // w[i][j] += -lr · u[i]·v[j], in place (no temporaries).
                    ndarray::Zip::from(weights.rows_mut())
                        .and(u)
                        .for_each(|mut wrow, &ui| {
                            ndarray::Zip::from(&mut wrow)
                                .and(v)
                                .for_each(|w, &vj| *w -= lr * (ui * vj));
                        });
                }
                Optimizer::Adam { lr, b1, b2, eps } => {
                    let (m, mv) = state.moments[i]
                        .as_mut()
                        .expect("Adam moments missing for a weighted layer");
                    let bc1 = 1.0 - b1.powi(state.t);
                    let bc2 = 1.0 - b2.powi(state.t);
                    // Fused moment update + parameter step, all in place; the
                    // descent sign is applied to the shared adam_step magnitude.
                    ndarray::Zip::from(weights.rows_mut())
                        .and(m.rows_mut())
                        .and(mv.rows_mut())
                        .and(u)
                        .for_each(|mut wrow, mut mrow, mut vrow, &ui| {
                            ndarray::Zip::from(&mut wrow)
                                .and(&mut mrow)
                                .and(&mut vrow)
                                .and(v)
                                .for_each(|w, m_, v_, &vj| {
                                    *w -= crate::optimiser::adam_step(
                                        m_,
                                        v_,
                                        ui * vj,
                                        b1,
                                        b2,
                                        bc1,
                                        bc2,
                                        lr,
                                        eps,
                                    );
                                });
                        });
                }
            }
        }
    }

    /// Apply one optimizer step to every layer's `weights_in` from matched
    /// per-layer dense descent gradients (`grads[i]` is `None` where the layer
    /// has no weights). The dense twin of [`Self::apply`], for gradients that
    /// are not rank-one; [`crate::vectorised::network::DeepNet::batch_update`]
    /// uses it with the batch-mean gradient, which is a sum of outer products.
    pub fn apply_dense(
        &self,
        state: &mut OptState,
        layers: &mut [Layer],
        grads: &[Option<Matrix>],
    ) {
        // Same step-counter semantics as `apply`: only Adam advances it.
        if let Optimizer::Adam { .. } = self {
            state.t += 1;
        }
        for (i, layer) in layers.iter_mut().enumerate() {
            let grad = match &grads[i] {
                Some(g) => g,
                None => continue,
            };
            let weights = layer
                .weights_in
                .as_mut()
                .expect("gradient present but layer has no weights");

            match *self {
                Optimizer::Sgd { lr } => {
                    ndarray::Zip::from(&mut *weights)
                        .and(grad)
                        .for_each(|w, &g| *w -= lr * g);
                }
                Optimizer::Adam { lr, b1, b2, eps } => {
                    let (m, mv) = state.moments[i]
                        .as_mut()
                        .expect("Adam moments missing for a weighted layer");
                    let bc1 = 1.0 - b1.powi(state.t);
                    let bc2 = 1.0 - b2.powi(state.t);
                    ndarray::Zip::from(&mut *weights)
                        .and(&mut *m)
                        .and(&mut *mv)
                        .and(grad)
                        .for_each(|w, m_, v_, &g| {
                            *w -= crate::optimiser::adam_step(
                                m_, v_, g, b1, b2, bc1, bc2, lr, eps,
                            );
                        });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorised::layer::{DeepNet, LayerConfig};

    /// `apply` and `apply_dense` produce the same step from the same gradient,
    /// whether given as rank-one factors or as the materialised matrix.
    #[test]
    fn test_apply_dense_matches_factor_apply() {
        let mut net_a = DeepNet::from_configs(&[LayerConfig::new(2), LayerConfig::new(2)]).unwrap();
        let mut net_b = DeepNet::from_configs(&[LayerConfig::new(2), LayerConfig::new(2)]).unwrap();
        let mut state_a = OptState::init(&net_a);
        let mut state_b = OptState::init(&net_b);

        let u = Vector::from_vec(vec![0.5, -1.5]);
        let v = Vector::from_vec(vec![2.0, 1.0, -0.5]);
        let dense = {
            let mut g = Matrix::zeros((2, 3));
            for i in 0..2 {
                for j in 0..3 {
                    g[[i, j]] = u[i] * v[j];
                }
            }
            g
        };
        let opt = Optimizer::adam(0.01);
        for _ in 0..3 {
            opt.apply(
                &mut state_a,
                &mut net_a.layers,
                &[None, Some((u.clone(), v.clone()))],
            );
            opt.apply_dense(&mut state_b, &mut net_b.layers, &[None, Some(dense.clone())]);
        }
        let wa = net_a.layers[1].weights_in.as_ref().unwrap();
        let wb = net_b.layers[1].weights_in.as_ref().unwrap();
        for (a, b) in wa.iter().zip(wb.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sgd_step_descends() {
        let mut net = DeepNet::from_configs(&[LayerConfig::new(1), LayerConfig::new(1)]).unwrap();
        let mut state = OptState::init(&net);
        // Weight on layer 1 is (1, 1+bias) = (1,2), all ones; factors give a
        // uniform descent gradient of 2.0 per element.
        let grads = vec![
            None,
            Some((Vector::from_elem(1, 2.0), Vector::from_elem(2, 1.0))),
        ];

        Optimizer::sgd(0.1).apply(&mut state, &mut net.layers, &grads);
        let w = net.layers[1].weights_in.as_ref().unwrap();
        // w ← 1 − 0.1*2 = 0.8.
        assert!((w[[0, 0]] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_adam_first_step_matches_lr_signed() {
        let mut net = DeepNet::from_configs(&[LayerConfig::new(1), LayerConfig::new(1)]).unwrap();
        let mut state = OptState::init(&net);
        let grads = vec![
            None,
            Some((Vector::from_elem(1, 3.0), Vector::from_elem(2, 1.0))),
        ];

        Optimizer::adam(0.01).apply(&mut state, &mut net.layers, &grads);
        // First Adam step: m_hat/√v_hat ≈ sign(grad); update ≈ -lr = -0.01.
        let w = net.layers[1].weights_in.as_ref().unwrap();
        assert!((w[[0, 0]] - (1.0 - 0.01)).abs() < 1e-6);
        assert_eq!(state.t, 1);
    }
}
