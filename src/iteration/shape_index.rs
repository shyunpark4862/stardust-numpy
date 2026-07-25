use crate::error::Result;
use crate::index::advance_multi_index;
use crate::shape::checked_size_of_shape;

/// C-order iterator over all coordinates in a shape.
#[derive(Clone, Debug)]
pub struct NdIndex {
    shape: Vec<usize>,
    indices: Vec<usize>,
    remaining: usize,
}

/// Return every coordinate in `shape` in logical C-order.
///
/// A shape containing a zero-length dimension yields no coordinates. The
/// zero-dimensional shape `[]` yields one empty coordinate.
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
