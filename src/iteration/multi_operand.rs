use crate::array::Array;
use crate::broadcast::broadcast_arrays;
use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::index::advance_multi_index;
use crate::shape::{checked_size_of_shape, offset_at};

/// Read-only C-order iterator over broadcast-compatible operands.
///
/// Every step yields one value per operand. All operands have the same scalar
/// type in the Rust core; runtime dtype dispatch belongs to the Python layer.
pub struct NdIter<T: Scalar> {
    operands: Vec<Array<T>>,
    shape: Vec<usize>,
    indices: Vec<usize>,
    linear: usize,
    remaining: usize,
    all_contiguous: bool,
}

/// Iterate over one or more broadcast-compatible arrays in lockstep.
pub fn nditer<T: Scalar>(operands: &[&Array<T>]) -> Result<NdIter<T>> {
    if operands.is_empty() {
        return Err(Error::InvalidArgument(
            "nditer requires at least one operand".into(),
        ));
    }

    let operands = broadcast_arrays(operands)?;
    let shape = operands[0].shape().to_vec();
    let remaining = checked_size_of_shape(&shape)?;
    let all_contiguous = operands.iter().all(Array::is_c_contiguous);

    Ok(NdIter {
        operands,
        indices: vec![0; shape.len()],
        shape,
        linear: 0,
        remaining,
        all_contiguous,
    })
}

impl<T: Scalar> Iterator for NdIter<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let values = if self.all_contiguous {
            self.operands
                .iter()
                .map(|operand| {
                    operand.as_buffer()[operand.offset() + self.linear]
                })
                .collect()
        } else {
            self.operands
                .iter()
                .map(|operand| {
                    let offset = offset_at(
                        &self.indices,
                        operand.strides(),
                        operand.offset(),
                    );
                    operand.as_buffer()[offset]
                })
                .collect()
        };

        self.remaining -= 1;
        self.linear += 1;
        if self.remaining > 0 && !self.all_contiguous {
            advance_multi_index(&mut self.indices, &self.shape);
        }
        Some(values)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Scalar> ExactSizeIterator for NdIter<T> {}
