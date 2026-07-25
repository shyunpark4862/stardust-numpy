//! C-order iteration over all coordinates in a shape.
//!
//! [`NdIndex`] is the coordinate counterpart to flat iteration: it yields
//! multi-indices `(i, j, …)` in the same order NumPy uses when flattening.

use crate::error::Result;
use crate::index::advance_multi_index;
use crate::shape::checked_size_of_shape;

/// C-order iterator over every coordinate in a shape.
///
/// Yields `Vec<usize>` multi-indices in row-major order — the same traversal
/// order used by [`FlatIter`] and [`ndenumerate`]. A zero-length dimension
/// yields no coordinates. The 0-D shape `[]` yields one empty coordinate
/// vector, matching NumPy's single scalar element.
///
/// Created by [`ndindex`].
#[derive(Clone, Debug)]
pub struct NdIndex {
    shape: Vec<usize>,
    indices: Vec<usize>,
    remaining: usize,
}

/// Return every coordinate in `shape` in logical C-order.
///
/// Each step of the returned [`NdIndex`] yields one coordinate vector
/// `(i, j, …)` visiting the last axis fastest. Empty shapes and zero-length
/// dimensions follow NumPy conventions (see [`NdIndex`]).
///
/// # Arguments
///
/// * `shape` - Shape whose coordinates to enumerate.
///
/// # Returns
///
/// An [`NdIndex`] iterator over all valid multi-indices.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`](crate::error::Error::InvalidArgument)
/// when the shape product overflows allocation limits.
///
/// # Examples
///
/// ```rust
/// use sdnp::ndindex;
///
/// let coords: Vec<_> = ndindex(&[2, 2]).unwrap().collect();
/// assert_eq!(coords.len(), 4);
/// assert_eq!(coords[0], vec![0, 0]);
/// assert_eq!(coords[3], vec![1, 1]);
/// ```
pub fn ndindex(shape: &[usize]) -> Result<NdIndex> {
    let remaining = checked_size_of_shape(shape)?;
    Ok(NdIndex {
        shape: shape.to_vec(),
        indices: vec![0; shape.len()],
        remaining,
    })
}

impl Iterator for NdIndex {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let current = self.indices.clone();
        self.remaining -= 1;
        if self.remaining > 0 {
            advance_multi_index(&mut self.indices, &self.shape);
        }
        Some(current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for NdIndex {}
