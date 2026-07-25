use num_traits::{One, Zero};

use crate::array::Array;
use crate::dtype::Scalar;
use crate::error::Result;
use crate::shape::checked_size_of_shape;

/// Return a C-contiguous array filled with `value`.
pub fn full<T: Scalar>(shape: &[usize], value: T) -> Result<Array<T>> {
    let size = checked_size_of_shape(shape)?;
    Array::from_vec(vec![value; size], shape)
}

/// Return a C-contiguous array of zeros.
pub fn zeros<T: Scalar + Zero>(shape: &[usize]) -> Result<Array<T>> {
    full(shape, T::zero())
}

/// Return a C-contiguous array of ones.
pub fn ones<T: Scalar + One>(shape: &[usize]) -> Result<Array<T>> {
    full(shape, T::one())
}
