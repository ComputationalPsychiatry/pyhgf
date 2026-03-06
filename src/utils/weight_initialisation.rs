//! Weight initialisation strategies for predictive-coding neural networks.
//!
//! Each function takes a fan-in (`n_parents`) and fan-out (`n_children`) and returns a
//! flat `Vec<f64>` of length `n_parents * n_children` (row-major) that can be used as
//! initial coupling weights.
//!
//! Available strategies:
//! - **Xavier / Glorot** — [`xavier_init`]
//! - **He / Kaiming** — [`he_init`]
//! - **Orthogonal** — [`orthogonal_init`]
//! - **Sparse** — [`sparse_init`]

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Uniform};

/// Build a [`SmallRng`] from an optional seed.
fn make_rng(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_os_rng(),
    }
}

// ---------------------------------------------------------------------------
// Xavier / Glorot uniform
// ---------------------------------------------------------------------------

/// Xavier / Glorot uniform initialisation.
///
/// Draws weights from U(-a, a) where a = sqrt(6 / (fan_in + fan_out)).
pub fn xavier_init(n_parents: usize, n_children: usize, seed: Option<u64>) -> Vec<f64> {
    let mut rng = make_rng(seed);
    let limit = (6.0_f64 / (n_parents + n_children) as f64).sqrt();
    let dist = Uniform::new(-limit, limit).unwrap();
    (0..n_parents * n_children)
        .map(|_| rng.sample(&dist))
        .collect()
}

// ---------------------------------------------------------------------------
// He / Kaiming normal
// ---------------------------------------------------------------------------

/// He / Kaiming normal initialisation.
///
/// Draws weights from N(0, σ²) where σ = sqrt(2 / fan_in).
/// Designed for layers followed by ReLU activations.
pub fn he_init(n_parents: usize, n_children: usize, seed: Option<u64>) -> Vec<f64> {
    let mut rng = make_rng(seed);
    let std = (2.0_f64 / n_parents as f64).sqrt();
    let dist = Normal::new(0.0, std).unwrap();
    (0..n_parents * n_children)
        .map(|_| rng.sample(&dist))
        .collect()
}

// ---------------------------------------------------------------------------
// Orthogonal
// ---------------------------------------------------------------------------

/// Orthogonal initialisation.
///
/// Generates a random matrix, computes a thin QR decomposition equivalent via
/// Gram–Schmidt, and returns the (semi-)orthogonal factor scaled by `gain`.
pub fn orthogonal_init(
    n_parents: usize,
    n_children: usize,
    gain: f64,
    seed: Option<u64>,
) -> Vec<f64> {
    let mut rng = make_rng(seed);
    let rows = n_parents;
    let cols = n_children;
    let dist = Normal::new(0.0, 1.0).unwrap();

    if rows >= cols {
        // Tall or square: fill (rows, cols) matrix, orthogonalise columns.
        let mut a: Vec<Vec<f64>> = (0..rows)
            .map(|_| (0..cols).map(|_| rng.sample(&dist)).collect())
            .collect();
        gram_schmidt_columns(&mut a, rows, cols);
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                out.push(gain * a[r][c]);
            }
        }
        out
    } else {
        // Wide: build (cols, rows) tall matrix, orthogonalise, then transpose.
        let mut a: Vec<Vec<f64>> = (0..cols)
            .map(|_| (0..rows).map(|_| rng.sample(&dist)).collect())
            .collect();
        gram_schmidt_columns(&mut a, cols, rows);
        // Transpose (cols, rows) → (rows, cols)
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                out.push(gain * a[c][r]);
            }
        }
        out
    }
}

/// In-place modified Gram–Schmidt on the columns of a row-major matrix `a`
/// with shape `(nrows, ncols)`.
fn gram_schmidt_columns(a: &mut [Vec<f64>], nrows: usize, ncols: usize) {
    for j in 0..ncols {
        // Normalise column j
        let mut norm = 0.0_f64;
        for i in 0..nrows {
            norm += a[i][j] * a[i][j];
        }
        norm = norm.sqrt();
        if norm > 1e-15 {
            for i in 0..nrows {
                a[i][j] /= norm;
            }
        }
        // Subtract projection of column j from all subsequent columns
        for k in (j + 1)..ncols {
            let mut dot = 0.0_f64;
            for i in 0..nrows {
                dot += a[i][j] * a[i][k];
            }
            for i in 0..nrows {
                a[i][k] -= dot * a[i][j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse
// ---------------------------------------------------------------------------

/// Sparse initialisation.
///
/// Most weights are zero; only a fraction `1 - sparsity` of entries are drawn
/// from N(0, std²).
pub fn sparse_init(
    n_parents: usize,
    n_children: usize,
    sparsity: f64,
    std: f64,
    seed: Option<u64>,
) -> Vec<f64> {
    let mut rng = make_rng(seed);
    let size = n_parents * n_children;
    let n_nonzero = ((1.0 - sparsity) * size as f64).round() as usize;
    let n_nonzero = n_nonzero.max(1);
    let mut weights = vec![0.0_f64; size];

    // Fisher–Yates-style partial shuffle to pick `n_nonzero` unique indices.
    let mut indices: Vec<usize> = (0..size).collect();
    for i in 0..n_nonzero {
        let j = rng.random_range(i..size);
        indices.swap(i, j);
    }

    let dist = Normal::new(0.0, std).unwrap();
    for &idx in &indices[..n_nonzero] {
        weights[idx] = rng.sample(&dist);
    }
    weights
}

// ---------------------------------------------------------------------------
// Dispatch by strategy name
// ---------------------------------------------------------------------------

/// Dispatch to a weight initialisation strategy by name.
///
/// Valid names: `"xavier"`, `"he"`, `"orthogonal"`, `"sparse"`.
///
/// For `"orthogonal"` the gain defaults to 1.0, and for `"sparse"` the
/// sparsity defaults to 0.9 with std 0.01.
pub fn weight_init_by_name(
    strategy: &str,
    n_parents: usize,
    n_children: usize,
    seed: Option<u64>,
) -> Result<Vec<f64>, String> {
    match strategy {
        "xavier" => Ok(xavier_init(n_parents, n_children, seed)),
        "he" => Ok(he_init(n_parents, n_children, seed)),
        "orthogonal" => Ok(orthogonal_init(n_parents, n_children, 1.0, seed)),
        "sparse" => Ok(sparse_init(n_parents, n_children, 0.9, 0.01, seed)),
        _ => Err(format!(
            "Unknown weight initialisation strategy '{}'. \
             Choose from: xavier, he, orthogonal, sparse.",
            strategy
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_shape_and_range() {
        let w = xavier_init(4, 3, Some(42));
        assert_eq!(w.len(), 12);
        let limit = (6.0_f64 / 7.0).sqrt();
        for &v in &w {
            assert!(v >= -limit && v <= limit, "value {v} out of range");
        }
    }

    #[test]
    fn test_he_shape() {
        let w = he_init(8, 4, Some(0));
        assert_eq!(w.len(), 32);
    }

    #[test]
    fn test_orthogonal_shape_tall() {
        let w = orthogonal_init(8, 4, 1.0, Some(0));
        assert_eq!(w.len(), 32);
    }

    #[test]
    fn test_orthogonal_shape_wide() {
        let w = orthogonal_init(3, 6, 1.0, Some(0));
        assert_eq!(w.len(), 18);
    }

    #[test]
    fn test_orthogonal_columns_are_unit() {
        let n = 6;
        let m = 4;
        let w = orthogonal_init(n, m, 1.0, Some(42));
        // Check each column has unit norm
        for c in 0..m {
            let mut norm_sq = 0.0;
            for r in 0..n {
                norm_sq += w[r * m + c] * w[r * m + c];
            }
            assert!((norm_sq - 1.0).abs() < 1e-10, "column {c} norm² = {norm_sq}");
        }
    }

    #[test]
    fn test_sparse_sparsity() {
        let w = sparse_init(10, 10, 0.9, 0.01, Some(42));
        assert_eq!(w.len(), 100);
        let zeros = w.iter().filter(|&&v| v == 0.0).count();
        // ~90 zeros for sparsity=0.9 on 100 weights
        assert_eq!(zeros, 90);
    }

    #[test]
    fn test_weight_init_by_name() {
        for name in &["xavier", "he", "orthogonal", "sparse"] {
            let w = weight_init_by_name(name, 4, 3, Some(0));
            assert!(w.is_ok());
            assert_eq!(w.unwrap().len(), 12);
        }
        assert!(weight_init_by_name("unknown", 4, 3, None).is_err());
    }

    #[test]
    fn test_deterministic_with_seed() {
        let a = xavier_init(4, 3, Some(123));
        let b = xavier_init(4, 3, Some(123));
        assert_eq!(a, b);
    }
}
