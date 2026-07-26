//! Coordinate grid construction via reshape and broadcast views.
//!
//! [`meshgrid`] builds N output arrays that share a common output shape from
//! N one-dimensional inputs — the same pattern as `numpy.meshgrid`. Outputs
//! are read-only broadcast views; contiguous inputs may alias their source
//! buffer without copying.

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};

/// Selects Cartesian (`xy`) or matrix (`ij`) axis ordering for [`meshgrid`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshgridIndexing {
    /// Cartesian indexing: swap the first two output dimensions.
    Xy,
    /// Matrix indexing: preserve input dimension order.
    Ij,
}

/// Build coordinate grids from same-dtype one-dimensional input arrays.
///
/// Each input must be one-dimensional. The output shape is the broadcast
/// product of all input lengths. One output array is returned per input;
/// each output varies along one axis of the shared grid while holding the
/// others fixed.
///
/// **Indexing modes:**
/// * [`MeshgridIndexing::Ij`] (matrix indexing) — output `i` varies along
///   axis `i`. The output shape is `(len_0, len_1, …)`.
/// * [`MeshgridIndexing::Xy`] (Cartesian indexing) — when at least two
///   inputs are supplied, the first two output axes are swapped so that
///   the first input varies along the x-axis (columns) and the second
///   along the y-axis (rows), matching NumPy's default `indexing='xy'`.
///
/// An empty input slice yields an empty `Vec` of outputs.
///
/// # Arguments
///
/// * `arrays` - One-dimensional inputs of equal element type.
/// * `indexing` - Axis ordering mode ([`MeshgridIndexing::Xy`] or
///   [`MeshgridIndexing::Ij`]).
///
/// # Returns
///
/// A vector of broadcast views, one per input, all sharing the same output
/// shape. Views may alias contiguous input buffers without copying.
///
/// # Errors
///
/// * [`Error::InvalidRank`](crate::Error::InvalidRank) - Any input is not
///   one-dimensional.
/// * [`Error::Broadcast`](crate::Error::Broadcast) - Reshape or broadcast
///   to the grid shape fails.
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Reshape
///   dimensions are invalid.
///
/// # Examples
///
/// ```rust
/// use sdnp::{meshgrid, Array, MeshgridIndexing};
///
/// let x = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
/// let y = Array::from_slice(&[10_i64, 20], &[2]).unwrap();
/// let grids = meshgrid(&[&x, &y], MeshgridIndexing::Xy).unwrap();
/// assert_eq!(grids.len(), 2);
/// assert_eq!(grids[0].shape(), &[2, 3]);
/// ```
pub fn meshgrid<T: Scalar>(
    arrays: &[&Array<T>],
    indexing: MeshgridIndexing,
) -> Result<Vec<Array<T>>> {
    if let Some(array) = arrays.iter().find(|array| array.ndim() != 1) {
        return Err(Error::InvalidRank {
            op: "meshgrid",
            expected: "1-D input arrays",
            actual: array.ndim(),
        });
    }
    let ndim = arrays.len();
    let mut output_shape: Vec<usize> =
        arrays.iter().map(|array| array.shape()[0]).collect();
    if indexing == MeshgridIndexing::Xy && ndim >= 2 {
        output_shape.swap(0, 1);
    }

    arrays
        .iter()
        .enumerate()
        .map(|(input_axis, array)| {
            // Place this input's values along one axis of the output grid.
            let output_axis = match (indexing, input_axis) {
                (MeshgridIndexing::Xy, 0) if ndim >= 2 => 1,
                (MeshgridIndexing::Xy, 1) if ndim >= 2 => 0,
                _ => input_axis,
            };
            let mut reshape = vec![1_isize; ndim];
            reshape[output_axis] = array.shape()[0] as isize;
            array.reshape(&reshape)?.broadcast_to(&output_shape)
        })
        .collect()
}
