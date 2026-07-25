//! Identity, triangular, and diagonal matrix constructors.
//!
//! These mirror `numpy.eye`, `numpy.tri`, `numpy.tril`/`triu`, and
//! `numpy.diag`. Diagonal offset `k` follows NumPy: positive values shift
//! above the main diagonal; negative values shift below.

use num_traits::{One, Zero};

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::linalg::diagonal_geometry::{
    diagonal_geometry, is_lower_triangle, is_upper_triangle,
};
use crate::shape::{checked_allocation_len, checked_size_of_shape};

/// Return an `n × n` identity matrix (ones on the main diagonal).
///
/// Equivalent to [`eye_with`] with `m = n` and `k = 0`.
///
/// # Arguments
///
/// * `n` - Number of rows and columns.
///
/// # Returns
///
/// A square 2-D [`Array`] with ones on the main diagonal and zeros
/// elsewhere.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - The
///   allocation size overflows platform limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::eye;
///
/// let a = eye::<i64>(2).unwrap();
/// assert_eq!(a.to_vec(), vec![1, 0, 0, 1]);
/// ```
pub fn eye<T: Scalar + Zero + One>(n: usize) -> Result<Array<T>> {
    eye_with(n, n, 0)
}

/// Return an identity-like matrix with shape `(n, m)` and diagonal offset `k`.
///
/// Ones are placed on the diagonal shifted by `k`. Positive `k` moves the
/// diagonal above the main diagonal; negative `k` moves it below.
///
/// # Arguments
///
/// * `n` - Number of rows.
/// * `m` - Number of columns.
/// * `k` - Diagonal offset (NumPy convention).
///
/// # Returns
///
/// A 2-D [`Array`] of shape `(n, m)` with ones on the shifted diagonal.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Allocation
///   overflows platform limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::eye_with;
///
/// let a = eye_with::<i64>(2, 3, 1).unwrap();
/// assert_eq!(a.to_vec(), vec![0, 1, 0, 0, 0, 1]);
/// ```
pub fn eye_with<T: Scalar + Zero + One>(
    n: usize,
    m: usize,
    k: isize,
) -> Result<Array<T>> {
    let len = checked_size_of_shape(&[n, m])?;
    checked_allocation_len::<T>(len)?;
    let mut data = vec![T::zero(); len];
    let diagonal = diagonal_geometry(n, m, k);
    for index in 0..diagonal.len {
        let row = diagonal.row_start + index;
        let column = diagonal.column_start + index;
        data[row * m + column] = T::one();
    }
    Array::from_vec(data, &[n, m])
}

/// Return an `n × n` lower-triangular matrix of ones.
///
/// Equivalent to [`tri_with`] with `m = n` and `k = 0`.
///
/// # Arguments
///
/// * `n` - Side length of the square matrix.
///
/// # Returns
///
/// A square 2-D [`Array`] with ones on and below the main diagonal.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Allocation
///   overflows platform limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::tri;
///
/// let a = tri::<i64>(3).unwrap();
/// assert_eq!(a.to_vec(), vec![1, 0, 0, 1, 1, 0, 1, 1, 1]);
/// ```
pub fn tri<T: Scalar + Zero + One>(n: usize) -> Result<Array<T>> {
    tri_with(n, n, 0)
}

/// Return an `n × m` matrix with ones on and below diagonal `k`.
///
/// Elements above the shifted diagonal are zero.
///
/// # Arguments
///
/// * `n` - Number of rows.
/// * `m` - Number of columns.
/// * `k` - Diagonal offset (see [`eye_with`]).
///
/// # Returns
///
/// A 2-D [`Array`] of shape `(n, m)`.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Allocation
///   overflows platform limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::tri_with;
///
/// let a = tri_with::<i64>(2, 3, 0).unwrap();
/// assert_eq!(a.to_vec(), vec![1, 0, 0, 1, 1, 0]);
/// ```
pub fn tri_with<T: Scalar + Zero + One>(
    n: usize,
    m: usize,
    k: isize,
) -> Result<Array<T>> {
    let len = checked_size_of_shape(&[n, m])?;
    checked_allocation_len::<T>(len)?;
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

/// Return a copy with elements above diagonal `k` zeroed out.
///
/// Keeps the lower triangle (on and below diagonal `k`). One-dimensional
/// inputs are tiled row-wise into a square matrix. For higher ranks the
/// mask applies to the trailing two axes only.
///
/// # Arguments
///
/// * `array` - Input array (any rank ≥ 1).
/// * `k` - Diagonal offset (see [`eye_with`]).
///
/// # Returns
///
/// A new [`Array`] with the same shape as the input.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Allocation
///   overflows platform limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::{tril, Array};
///
/// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
/// assert_eq!(tril(&a, 0).unwrap().to_vec(), vec![1, 0, 3, 4]);
/// ```
pub fn tril<T: Scalar + Zero>(array: &Array<T>, k: isize) -> Result<Array<T>> {
    triangle_copy(array, k, true)
}

/// Return a copy with elements below diagonal `k` zeroed out.
///
/// Keeps the upper triangle (on and above diagonal `k`). One-dimensional
/// inputs are tiled row-wise into a square matrix. For higher ranks the
/// mask applies to the trailing two axes only.
///
/// # Arguments
///
/// * `array` - Input array (any rank ≥ 1).
/// * `k` - Diagonal offset (see [`eye_with`]).
///
/// # Returns
///
/// A new [`Array`] with the same shape as the input.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Allocation
///   overflows platform limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::{triu, Array};
///
/// let a = Array::from_slice(&[1_i64, 2, 3, 4], &[2, 2]).unwrap();
/// assert_eq!(triu(&a, 0).unwrap().to_vec(), vec![1, 2, 0, 4]);
/// ```
pub fn triu<T: Scalar + Zero>(array: &Array<T>, k: isize) -> Result<Array<T>> {
    triangle_copy(array, k, false)
}

/// Apply a lower or upper triangle mask to the trailing 2-D slice(s).
///
/// Materializes C-order data, then zeroes elements outside the requested
/// triangle on the trailing two axes. One-dimensional inputs are tiled into
/// a square matrix first.
///
/// # Arguments
///
/// * `array` — input array (any rank ≥ 1)
/// * `k` — diagonal offset (see [`eye_with`])
/// * `lower` — `true` for lower triangle (`tril`); `false` for upper
///   (`triu`)
///
/// # Returns
///
/// A new C-contiguous [`Array`] with the same shape as the input (or
/// `(n, n)` when the input is 1-D).
///
/// # Errors
///
/// * [`Error::InvalidArgument`] — allocation overflows platform limits
/// * [`Error::BufferSizeMismatch`] — internal buffer length mismatch
fn triangle_copy<T: Scalar + Zero>(
    array: &Array<T>,
    k: isize,
    lower: bool,
) -> Result<Array<T>> {
    if array.ndim() == 1 {
        let values = array.to_vec_c_order();
        let n = values.len();
        let len = checked_size_of_shape(&[n, n])?;
        checked_allocation_len::<T>(len)?;
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
    let matrix_len = checked_size_of_shape(&[rows, columns])?;
    if matrix_len > 0 {
        // Process each trailing matrix independently in C-order layout.
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

/// Build a diagonal matrix from a vector, or extract a diagonal from a matrix.
///
/// For a 1-D input, returns a square matrix with the vector on diagonal `k`.
/// For a 2-D input, extracts the diagonal as a 1-D array (delegates to
/// [`crate::linalg::diagonal`]). Higher-rank inputs also extract the diagonal
/// from the trailing 2-D slice.
///
/// # Arguments
///
/// * `array` - 1-D vector or 2-D (or higher) matrix input.
/// * `k` - Diagonal offset (see [`eye_with`]).
///
/// # Returns
///
/// A new [`Array`]: square matrix for 1-D input, 1-D diagonal otherwise.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - Output
///   dimension or allocation overflows limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) -
///   Internal buffer length mismatch.
///
/// # Examples
///
/// ```rust
/// use sdnp::{diag, Array};
///
/// let v = Array::from_slice(&[1_i64, 2, 3], &[3]).unwrap();
/// assert_eq!(diag(&v, 0).unwrap().to_vec(), vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
/// ```
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
            let len = checked_size_of_shape(&[side, side])?;
            checked_allocation_len::<T>(len)?;
            let mut output = vec![T::zero(); len];
            let diagonal = diagonal_geometry(side, side, k);
            let values = array.to_vec_c_order();
            for (index, value) in values.into_iter().enumerate() {
                let row = diagonal.row_start + index;
                let column = diagonal.column_start + index;
                output[row * side + column] = value;
            }
            Array::from_vec(output, &[side, side])
        }
        2 => crate::linalg::diagonal(array, k, 0, 1),
        _ => crate::linalg::diagonal(array, k, 0, 1),
    }
}
