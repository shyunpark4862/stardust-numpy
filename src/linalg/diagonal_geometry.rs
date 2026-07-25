//! Two-dimensional diagonal indexing shared by `diagonal` and `trace`.
//!
//! Given matrix dimensions and a NumPy-style offset, this module resolves
//! where a diagonal starts and how many elements it contains. Triangle
//! predicates support future masked linear algebra without duplicating
//! offset arithmetic.

/// Inclusive start and length of a diagonal in a 2-D face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DiagonalGeometry {
    /// First row index on the diagonal.
    pub(crate) row_start: usize,
    /// First column index on the diagonal.
    pub(crate) column_start: usize,
    /// Number of elements along the diagonal.
    pub(crate) len: usize,
}

/// Compute diagonal geometry for a `rows × columns` matrix face.
///
/// Non-negative offsets shift the start column right; negative offsets shift
/// the start row down. Length is clipped by both dimensions so the walk stays
/// inside the face.
///
/// # Arguments
///
/// * `rows` — height of the 2-D face
/// * `columns` — width of the 2-D face
/// * `offset` — diagonal offset (`0` = main diagonal, `k>0` = upper, `k<0`
///   = lower)
///
/// # Returns
///
/// Inclusive `(row_start, column_start)` and the number of diagonal elements.
pub(crate) fn diagonal_geometry(
    rows: usize,
    columns: usize,
    offset: isize,
) -> DiagonalGeometry {
    let (row_start, column_start) = if offset >= 0 {
        (0, offset as usize)
    } else {
        (offset.unsigned_abs(), 0)
    };
    let len = rows
        .saturating_sub(row_start)
        .min(columns.saturating_sub(column_start));
    DiagonalGeometry {
        row_start,
        column_start,
        len,
    }
}

/// True when `(row, column)` is on or below diagonal `offset`.
///
/// Uses signed arithmetic so large indices compare correctly relative to the
/// shifted main diagonal.
///
/// # Arguments
///
/// * `row` — row index on the 2-D face
/// * `column` — column index on the 2-D face
/// * `offset` — same convention as [`diagonal_geometry`]
///
/// # Returns
///
/// `true` when the cell lies in the lower triangle (including the diagonal).
#[inline]
pub(crate) fn is_lower_triangle(
    row: usize,
    column: usize,
    offset: isize,
) -> bool {
    column as i128 <= row as i128 + offset as i128
}

/// True when `(row, column)` is on or above diagonal `offset`.
///
/// Complement of the lower triangle for the same offset convention.
///
/// # Arguments
///
/// * `row` — row index on the 2-D face
/// * `column` — column index on the 2-D face
/// * `offset` — same convention as [`diagonal_geometry`]
///
/// # Returns
///
/// `true` when the cell lies in the upper triangle (including the diagonal).
#[inline]
pub(crate) fn is_upper_triangle(
    row: usize,
    column: usize,
    offset: isize,
) -> bool {
    column as i128 >= row as i128 + offset as i128
}
