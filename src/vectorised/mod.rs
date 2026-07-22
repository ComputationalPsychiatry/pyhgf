//! The vectorised deep-network backend: columnar layer types ([`layer`]),
//! matrix primitives ([`mat`]), the whole-network sweep driver ([`network`]),
//! and the weight optimisers ([`optimiser`]). The per-layer update kernels
//! live in [`crate::updates::vectorised`], mirroring the JAX package split
//! (`pyhgf/typing/vectorised.py` + `pyhgf/utils/vectorized_belief_propagation.py`
//! vs `pyhgf/updates/vectorized/`).

pub mod layer;
pub mod mat;
pub mod network;
pub mod optimiser;
