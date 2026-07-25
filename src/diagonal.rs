//! Shared two-dimensional diagonal geometry.

/// Start coordinates and length of diagonal `offset` in a matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DiagonalGeometry {
    /// Starting row.
    pub(crate) row_start: usize,
    /// Starting column.
    pub(crate) column_start: usize,
    /// Number of diagonal elements.
    pub(crate) len: usize,
}

/// Resolve a NumPy-style diagonal offset for a `rows × columns` matrix.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_positive_negative_and_empty_diagonals() {
        assert_eq!(
            diagonal_geometry(3, 4, 1),
            DiagonalGeometry {
                row_start: 0,
                column_start: 1,
                len: 3,
            }
        );
        assert_eq!(
            diagonal_geometry(3, 4, -1),
            DiagonalGeometry {
                row_start: 1,
                column_start: 0,
                len: 2,
            }
        );
        assert_eq!(diagonal_geometry(3, 4, 8).len, 0);
    }
}
