//! Constant-fill array constructors (`zeros`, `ones`, `full`).
//!
//! Each function validates the requested shape, checks that the allocation
//! fits platform limits, and builds a dense C-contiguous [`Array`] backed
//! by a freshly allocated `Vec`.

use num_traits::{One, Zero};

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::shape::{checked_allocation_len, checked_size_of_shape};

/// Return a C-contiguous array filled with `value`.
///
/// Every element of the output equals `value`. The buffer is freshly
/// allocated; no views or aliasing with existing arrays are created.
///
/// # Arguments
///
/// * `shape` - Output dimensions. A zero in any dimension yields a
///   zero-length buffer while preserving rank.
/// * `value` - Scalar replicated into every element.
///
/// # Returns
///
/// A new writable [`Array`] with C-order strides.
///
/// # Errors
///
/// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) - The shape
///   product overflows `usize`, or the allocation exceeds platform limits.
/// * [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch) - Internal
///   buffer length does not match the shape (should not occur for valid input).
///
/// # Examples
///
/// ```rust
/// use sdnp::full;
///
/// let a = full(&[2, 2], 7_i64).unwrap();
/// assert_eq!(a.to_vec(), vec![7, 7, 7, 7]);
/// ```
pub fn full<T: Scalar>(shape: &[usize], value: T) -> Result<Array<T>> {
    let size = checked_size_of_shape(shape)?;
    checked_allocation_len::<T>(size)?;
    Array::from_vec(vec![value; size], shape)
}

/// Return a C-contiguous array whose elements are zero.
///
/// Equivalent to [`full`] with [`Zero::zero`]. Useful for allocating
/// accumulators or masks before in-place updates.
///
/// # Arguments
///
/// * `shape` - Output dimensions (see [`full`]).
///
/// # Returns
///
/// A new [`Array`] filled with `T::zero()`.
///
/// # Errors
///
/// Same as [`full`]: [`Error::InvalidArgument`](crate::Error::InvalidArgument)
/// or [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch).
///
/// # Examples
///
/// ```rust
/// use sdnp::zeros;
///
/// let a = zeros::<f64>(&[3]).unwrap();
/// assert_eq!(a.to_vec(), vec![0.0, 0.0, 0.0]);
/// ```
pub fn zeros<T: Scalar + Zero>(shape: &[usize]) -> Result<Array<T>> {
    full(shape, T::zero())
}

/// Return a C-contiguous array whose elements are one.
///
/// Equivalent to [`full`] with [`One::one`].
///
/// # Arguments
///
/// * `shape` - Output dimensions (see [`full`]).
///
/// # Returns
///
/// A new [`Array`] filled with `T::one()`.
///
/// # Errors
///
/// Same as [`full`]: [`Error::InvalidArgument`](crate::Error::InvalidArgument)
/// or [`Error::BufferSizeMismatch`](crate::Error::BufferSizeMismatch).
///
/// # Examples
///
/// ```rust
/// use sdnp::ones;
///
/// let a = ones::<i64>(&[2, 2]).unwrap();
/// assert_eq!(a.to_vec(), vec![1, 1, 1, 1]);
/// ```
pub fn ones<T: Scalar + One>(shape: &[usize]) -> Result<Array<T>> {
    full(shape, T::one())
}
