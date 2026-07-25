use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::Result;

/// Axis ordering used by [`meshgrid`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshgridIndexing {
    /// Cartesian (`xy`) indexing: swap the first two output dimensions.
    Xy,
    /// Matrix (`ij`) indexing: preserve input dimension order.
    Ij,
}

/// Build coordinate grids from same-dtype one-dimensional input arrays.
///
/// One output is returned for each input. Every output has the common shape
/// formed from the input lengths; [`MeshgridIndexing::Xy`] swaps the first two
/// dimensions when at least two arrays are supplied, while
/// [`MeshgridIndexing::Ij`] preserves their order. Outputs are read-only
/// broadcast views; contiguous inputs share their corresponding input buffer,
/// while a non-contiguous input may first be copied by reshape. An empty input
/// slice returns an empty vector.
pub fn meshgrid<T: Scalar>(
    arrays: &[&Array<T>],
    indexing: MeshgridIndexing,
) -> Result<Vec<Array<T>>> {
    for array in arrays {
        debug_assert_eq!(array.ndim(), 1, "meshgrid inputs must be 1-D");
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
