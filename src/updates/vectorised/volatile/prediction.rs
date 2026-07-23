//! Top-down prediction for volatile-node layers, mirroring
//! `pyhgf/updates/vectorized/volatile/prediction.py`.

use super::guarded_volatility;
use crate::math::{with_coupling, CouplingFn};
use crate::vectorised::layer::{LayerParams, LayerState};
use crate::vectorised::mat::Matrix;
use ndarray::Array1;

/// Predict the value- and volatility-level expectations for every node in a
/// child layer from its parent.
///
/// Mirrors `vectorized_layer_prediction`. `weights` is the parent's
/// `weights_in` (shape `(n_child, n_parent[+1])`), `coupling_fn` is the
/// parent's coupling function, and `params` are the *child's* layer parameters.
///
/// Writes the predicted fields into `child` in place; the carried belief fields
/// (`mean`, `precision`, prediction errors, the posterior volatility level) are
/// left untouched, so no full-state clone is allocated.
#[allow(clippy::too_many_arguments)]
pub fn layer_prediction(
    child: &mut LayerState,
    child_params: &LayerParams,
    parent: &LayerState,
    weights: &Matrix,
    coupling_fn: &CouplingFn,
    time_step: f64,
    parent_has_constant: bool,
    has_volatility_parent: bool,
    is_input_layer: bool,
) {
    let n = child.mean.len();
    let n_parent = parent.expected_mean.len();
    let p = weights.ncols(); // n_parent (+1 for the bias column)

    // 1. Volatility level (internal); computed into locals, written at the end.
    // Frozen (no volatility parent) → the fields stay as they are (`None`).
    let (expected_mean_vol, expected_precision_vol, effective_precision_vol) =
        if has_volatility_parent {
            let mean_vol = child.mean_vol.as_ref().expect("volatility mean");
            let precision_vol = child.precision_vol.as_ref().expect("volatility precision");

            // Autoconnection strength is fixed at 1.
            let expected_mean_vol = mean_vol.clone();
            let mut epv = Array1::<f64>::zeros(n);
            let mut effv = Array1::<f64>::zeros(n);
            ndarray::Zip::from(&mut epv)
                .and(&mut effv)
                .and(precision_vol)
                .and(&child_params.tonic_volatility_vol)
                .for_each(|e, ev, &pv_prec, &tvv| {
                    let pvv = guarded_volatility(tvv, time_step);
                    let epval = 1.0 / (1.0 / pv_prec + pvv);
                    *e = epval;
                    *ev = pvv * epval;
                });
            (Some(expected_mean_vol), Some(epv), Some(effv))
        } else {
            (None, None, None)
        };

    // 2. Value level.
    // Coupled parent activations (bias node = 1) and the per-parent Laplace
    // variance factor (t·g'(μ̂))² / π̃_parent, built together in one pass. The
    // bias parent has zero derivative and infinite precision → factor 0.
    let mut coupled = Array1::<f64>::zeros(p);
    let mut ppv = Array1::<f64>::zeros(p);
    with_coupling!(coupling_fn, |f, df, _d2f| {
        for j in 0..n_parent {
            let mu = parent.expected_mean[j];
            coupled[j] = f(mu);
            let num = time_step * df(mu);
            ppv[j] = (num * num) / parent.expected_precision[j];
        }
    });
    if parent_has_constant {
        coupled[n_parent] = 1.0; // ppv[n_parent] stays 0
    }

    // Mean prediction: W @ g(parent means).
    child.expected_mean = weights.dot(&coupled);

    // Predicted volatility Ω = t·exp(μ_vol + 1/(2·π̂_vol)), fused. The volatility
    // coupling is fixed at 1 and the value level carries no tonic volatility of
    // its own.
    let predicted_vol = if has_volatility_parent {
        let emv = expected_mean_vol.as_ref().unwrap();
        let epv = expected_precision_vol.as_ref().unwrap();
        ndarray::Zip::from(emv)
            .and(epv)
            .map_collect(|&m, &pv| guarded_volatility(m + 1.0 / (pv * 2.0), time_step))
    } else {
        // No volatility parent and no tonic volatility: the value level has no
        // volatility source, so it does not undergo a Gaussian random walk. The
        // diffusion term is zero, leaving the conditional predicted precision
        // equal to the prior precision.
        Array1::<f64>::zeros(n)
    };

    // Laplace value-coupling variance vcv[i] = Σ_j W[i,j]²·ppv[j] (no W² matrix).
    let mut value_coupling_variance = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..p {
            let w = weights[[i, j]];
            s += w * w * ppv[j];
        }
        value_coupling_variance[i] = s;
    }

    // Conditional (π̂) and marginal (π̃) predicted precisions, written in place.
    if is_input_layer {
        // Input/leaf override: no random walk between observations.
        let prior = child.precision.clone();
        child.conditional_expected_precision.assign(&prior);
        child.expected_precision.assign(&prior);
        child.effective_precision.fill(0.0);
    } else {
        // π̂ = 1/(1/π + Ω)
        ndarray::Zip::from(&mut child.conditional_expected_precision)
            .and(&child.precision)
            .and(&predicted_vol)
            .for_each(|c, &p, &pv| *c = 1.0 / (1.0 / p + pv));
        // π̃ = 1/(1/π̂ + value_coupling_variance)
        ndarray::Zip::from(&mut child.expected_precision)
            .and(&child.conditional_expected_precision)
            .and(&value_coupling_variance)
            .for_each(|e, &c, &v| *e = 1.0 / (1.0 / c + v));
        // γ = Ω · π̃
        ndarray::Zip::from(&mut child.effective_precision)
            .and(&predicted_vol)
            .and(&child.expected_precision)
            .for_each(|ef, &pv, &e| *ef = pv * e);
    }

    if has_volatility_parent {
        child.expected_mean_vol = expected_mean_vol;
        child.expected_precision_vol = expected_precision_vol;
        child.effective_precision_vol = effective_precision_vol;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::resolve_coupling_fn;
    use crate::vectorised::layer::LayerParams;

    fn linear() -> &'static CouplingFn {
        resolve_coupling_fn("linear")
    }

    /// With linear coupling and no bias, the predicted mean is exactly
    /// `W @ parent_mean`.
    #[test]
    fn test_prediction_mean_is_matvec() {
        let mut parent = LayerState::create(2, true);
        parent.expected_mean = Array1::from_vec(vec![1.0, 2.0]);
        let mut child = LayerState::create(3, true);
        let child_params = LayerParams::create(3);
        let weights = Matrix::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        layer_prediction(
            &mut child,
            &child_params,
            &parent,
            &weights,
            linear(),
            1.0,
            false,
            true,
            false,
        );
        // rows: [1,0]·[1,2]=1, [0,1]·[1,2]=2, [1,1]·[1,2]=3
        assert!((child.expected_mean[0] - 1.0).abs() < 1e-12);
        assert!((child.expected_mean[1] - 2.0).abs() < 1e-12);
        assert!((child.expected_mean[2] - 3.0).abs() < 1e-12);
    }

    /// The trailing bias column adds a constant `1` activation.
    #[test]
    fn test_prediction_bias_column() {
        let mut parent = LayerState::create(1, true);
        parent.expected_mean = Array1::from_vec(vec![2.0]);
        let mut child = LayerState::create(1, true);
        let child_params = LayerParams::create(1);
        // (n_child=1, n_parent=1 + bias) => weight 3 on parent, bias 5.
        let weights = Matrix::from_shape_vec((1, 2), vec![3.0, 5.0]).unwrap();

        layer_prediction(
            &mut child,
            &child_params,
            &parent,
            &weights,
            linear(),
            1.0,
            true,
            true,
            false,
        );
        // 3*2 + 5*1 = 11
        assert!((child.expected_mean[0] - 11.0).abs() < 1e-12);
    }

    /// An input/leaf layer keeps its prior precision and zeroes γ.
    #[test]
    fn test_prediction_input_layer_override() {
        let parent = LayerState::create(2, true);
        let mut child = LayerState::create(2, true);
        child.precision = Array1::from_vec(vec![7.0, 9.0]);
        let prior_precision = child.precision.clone();
        let child_params = LayerParams::create(2);
        let weights = Matrix::from_elem((2, 2), 1.0);

        layer_prediction(
            &mut child,
            &child_params,
            &parent,
            &weights,
            linear(),
            1.0,
            false,
            true,
            true, // is_input_layer
        );
        assert_eq!(child.expected_precision, prior_precision);
        assert_eq!(child.conditional_expected_precision, prior_precision);
        assert!(child.effective_precision.iter().all(|&x| x == 0.0));
    }
}
