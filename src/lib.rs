// Force-link the BLAS provider when the `blas` feature routes ndarray's
// `.dot()` through DGEMM; nothing references it directly otherwise.
#[cfg(feature = "blas")]
extern crate blas_src;

pub mod math;
pub mod model;
pub mod optimiser;
pub mod updates;
pub mod utils;
pub mod vectorised;

use pyo3::prelude::*;

/// The `rshgf` Python extension module, exposing the two model classes
/// (mirroring `pyhgf.model`): the per-node `Network` and the vectorised
/// `DeepNetwork`.
#[pymodule]
fn rshgf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<model::network::Network>()?;
    m.add_class::<model::deep_network::DeepNetwork>()?;
    Ok(())
}
