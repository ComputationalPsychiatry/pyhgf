//! Belief-update kernels, split by backend: `nodalised` (per-node scalar
//! updates) and `vectorised` (matrix updates over whole layers), mirroring
//! `pyhgf/updates/` and `pyhgf/updates/vectorized/`.

pub mod nodalised;
pub mod vectorised;
