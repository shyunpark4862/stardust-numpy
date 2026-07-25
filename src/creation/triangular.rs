use num_traits::{One, Zero};

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::linalg::diagonal_geometry::{
    diagonal_geometry, is_lower_triangle, is_upper_triangle,
};
use crate::shape::checked_size_of_shape;

/// 2-D identity-like matrix with ones on diagonal `k` (default main diagonal).
pub fn eye<T: Scalar + Zero + One>(n: usize) -> Result<Array<T>> {
    eye_with(n, n, 0)
}

/// Like [`eye`], with optional column count `m` and diagonal offset `k`.
///
/// `k > 0` is above the main diagonal; `k < 0` is below.
pub fn eye_with<T: Scalar + Zero + One>(
    n: usize,
    m: usize,
    k: isize,
) -> Result<Array<T>> {
    let mut data = vec![T::zero(); checked_size_of_shape(&[n, m])?];
    let diagonal = diagonal_geometry(n, m, k);
    for index in 0..diagonal.len {
        let row = diagonal.row_start + index;
        let column = diagonal.column_start + index;
        data[row * m + column] = T::one();
    }
    Array::from_vec(data, &[n, m])
}

/// Return an `n × n` lower-triangular array of ones.
pub fn tri<T: Scalar + Zero + One>(n: usize) -> Result<Array<T>> {
    tri_with(n, n, 0)
}

/// Return an `n × m` array with ones on and below diagonal `k`.
pub fn tri_with<T: Scalar + Zero + One>(
    n: usize,
    m: usize,
    k: isize,
) -> Result<Array<T>> {
    let len = checked_size_of_shape(&[n, m])?;
    let mut data = Vec::with_capacity(len);
    for row in 0..n {
        for column in 0..m {
            data.push(if is_lower_triangle(row, column, k) {
                T::one()
            } else {
                T::zero()
            });
        }
    }
    Array::from_vec(data, &[n, m])
}

/// Return a C-order copy with elements above diagonal `k` replaced by zero.
///
/// One-dimensional inputs are tiled row-wise into a square matrix. For
/// higher-dimensional inputs the mask applies to the last two axes.
pub fn tril<T: Scalar + Zero>(array: &Array<T>, k: isize) -> Result<Array<T>> {
    triangle_copy(array, k, true)
}

/// Return a C-order copy with elements below diagonal `k` replaced by zero.
///
/// One-dimensional inputs are tiled row-wise into a square matrix. For
/// higher-dimensional inputs the mask applies to the last two axes.
pub fn triu<T: Scalar + Zero>(array: &Array<T>, k: isize) -> Result<Array<T>> {
    triangle_copy(array, k, false)
}

fn triangle_copy<T: Scalar + Zero>(
    array: &Array<T>,
    k: isize,
    lower: bool,
) -> Result<Array<T>> {
    if array.ndim() == 0 {
        return Err(Error::InvalidArgument(
            "tril and triu require at least 1 dimension".into(),
        ));
    }

    if array.ndim() == 1 {
        let values = array.to_vec_c_order();
        let n = values.len();
        let len = checked_size_of_shape(&[n, n])?;
        let mut output = Vec::with_capacity(len);
        for row in 0..n {
            for (column, &value) in values.iter().enumerate() {
                let keep = if lower {
                    is_lower_triangle(row, column, k)
                } else {
                    is_upper_triangle(row, column, k)
                };
                output.push(if keep { value } else { T::zero() });
            }
        }
        return Array::from_vec(output, &[n, n]);
    }

    let rows = array.shape()[array.ndim() - 2];
    let columns = array.shape()[array.ndim() - 1];
    let mut output = array.to_vec_c_order();
    let matrix_len = rows * columns;
    if matrix_len > 0 {
        for matrix in output.chunks_exact_mut(matrix_len) {
            for (row, row_values) in
                matrix.chunks_exact_mut(columns).enumerate()
            {
                for (column, value) in row_values.iter_mut().enumerate() {
                    let keep = if lower {
                        is_lower_triangle(row, column, k)
                    } else {
                        is_upper_triangle(row, column, k)
                    };
                    if !keep {
                        *value = T::zero();
                    }
                }
            }
        }
    }
    Array::from_vec(output, array.shape())
}

/// Construct a diagonal matrix from a vector, or extract one from a matrix.
///
/// Only one- and two-dimensional inputs are accepted.
pub fn diag<T: Scalar + Zero>(array: &Array<T>, k: isize) -> Result<Array<T>> {
    match array.ndim() {
        1 => {
            let side = array.shape()[0]
                .checked_add(k.unsigned_abs())
                .ok_or_else(|| {
                    Error::InvalidArgument(
                        "diag output dimension overflows usize".into(),
                    )
                })?;
            let mut output =
                vec![T::zero(); checked_size_of_shape(&[side, side])?];
            let diagonal = diagonal_geometry(side, side, k);
            let values = array.to_vec_c_order();
            debug_assert_eq!(diagonal.len, values.len());
            for (index, value) in values.into_iter().enumerate() {
                let row = diagonal.row_start + index;
                let column = diagonal.column_start + index;
                output[row * side + column] = value;
            }
            Array::from_vec(output, &[side, side])
        }
        2 => crate::linalg::diagonal(array, k, 0, 1),
        ndim => Err(Error::InvalidArgument(format!(
            "diag requires a 1-D or 2-D array, got ndim={ndim}"
        ))),
    }
}
