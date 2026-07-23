//! Matrix primitives for the vectorised backend.
//!
//! The backend works directly on [`ndarray`] types: matrix products go through
//! `ndarray`'s `.dot()`, which the default `blas` feature routes through a
//! linked BLAS (`SGEMM`/`DGEMM`); building with `--no-default-features` falls
//! back to the pure-Rust (threaded) `matrixmultiply` kernel without any code
//! change. This module only adds the scalar/type aliases and the two
//! constructions `ndarray` does not provide directly ([`eye`]).
//!
//! The engine's scalar is [`Float`]: `f32` by default (half the memory
//! traffic of `f64`, which dominates the elementwise sweeps), switched to
//! `f64` by the `f64` cargo feature for bit-compatibility with the JAX
//! backend under `jax_enable_x64`. The nodalised (per-node) backend is
//! unaffected: it computes in `f64` under either feature.

use ndarray::{Array1, Array2};

/// Scalar type of the vectorised engine: `f32` unless the `f64` feature is on.
#[cfg(not(feature = "f64"))]
pub type Float = f32;

/// Scalar type of the vectorised engine: `f64` (the `f64` feature is on).
#[cfg(feature = "f64")]
pub type Float = f64;

/// Row-major 2D matrix of [`Float`] values backed by [`ndarray::Array2`].
pub type Matrix = Array2<Float>;

/// 1D vector of [`Float`] values.
pub type Vector = Array1<Float>;

/// Create an identity matrix of size `n × n`.
pub fn eye(n: usize) -> Matrix {
    let mut m = Matrix::zeros((n, n));
    for i in 0..n {
        m[[i, i]] = 1.0;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye() {
        let m = eye(3);
        let s: &[usize] = &[3, 3];
        assert_eq!(m.shape(), s);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((m[[i, j]] - 1.0).abs() < 1e-10);
                } else {
                    assert!((m[[i, j]] - 0.0).abs() < 1e-10);
                }
            }
        }
    }
}
