//! Array creation helpers mirroring common NumPy constructors.
//!
//! Functions here allocate new C-contiguous buffers — unlike view modules,
//! nothing shares storage unless noted (e.g. [`meshgrid`](grids::meshgrid)
//! broadcast views). Shapes are validated and allocation sizes checked
//! before any `Vec` is grown.
//!
//! Range helpers in [`ranges`] use widened integer arithmetic where needed;
//! triangular helpers in [`triangular`] follow NumPy diagonal offset `k`.

mod factories;
mod grids;
mod ranges;
mod triangular;

pub use factories::{full, ones, zeros};
pub use grids::{meshgrid, MeshgridIndexing};
pub use ranges::{arange, arange_stop, geomspace, linspace, logspace};
pub use triangular::{diag, eye, eye_with, tri, tri_with, tril, triu};
