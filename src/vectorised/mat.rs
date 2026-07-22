//! Matrix primitives for the vectorised backend.
//!
//! The backend works directly on [`ndarray`] types: matrix products go through
//! `ndarray`'s `.dot()`, which uses the pure-Rust (threaded) `matrixmultiply`
//! kernel by default; enabling the crate's `blas` feature re-routes the same
//! calls through a linked BLAS (`DGEMM`/`DGEMV`) without any code change. This
//! module only adds the type aliases and the two constructions `ndarray` does
//! not provide directly ([`eye`]).

use ndarray::{Array1, Array2};

/// Row-major 2D matrix of f64 values backed by [`ndarray::Array2`].
pub type Matrix = Array2<f64>;

/// 1D vector of f64 values.
pub type Vector = Array1<f64>;

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
