//! Width-bounded `repr` and R-style `str` formatting for arrays.
//!
//! This module owns presentation logic that is independent of Python. The
//! diagnostic representation is intentionally compact and reads only the
//! boundary elements needed for display, including for strided views.
//!
//! Formatting is split into planning and rendering:
//!
//! 1. [`EdgeSelection`] decides how many items survive at each edge.
//! 2. Width planners use prefix sums, so adding a candidate is O(1).
//! 3. Renderers materialize only the selected values and a possible `...`.
//!
//! Consequently, formatting work is bounded by the amount of visible output,
//! rather than by the total array size. Logical flat access still costs
//! O(ndim) per selected value because it must unravel a C-order index.

use crate::{Array, Complex64, Scalar};

/// Maximum width of every line emitted by [`format_array_repr`].
pub const ARRAY_REPR_WIDTH: usize = 80;

/// Maximum values inspected at either end of a flattened sequence.
///
/// The width planner never needs more candidates than can plausibly fit in
/// an 80-column line. Keeping this bound explicit prevents formatting a large
/// array from accidentally becoming O(size).
const MAX_EDGE_CANDIDATES: usize = 32;
/// Rows retained at both ends of a tall matrix.
const MATRIX_EDGE_ROWS: usize = 3;
/// Columns considered at both ends of a wide matrix.
const MATRIX_EDGE_COLUMNS: usize = 6;
/// Pages retained at both ends of a high-dimensional array.
const PAGE_EDGE_ITEMS: usize = 3;
/// Maximum printed width assigned to one matrix cell.
///
/// Wider scalar strings are shortened in the middle, preserving both their
/// leading and trailing characters.
const MAX_CELL_WIDTH: usize = 20;

/// A compact plan for retaining both ends of an ordered sequence.
///
/// `front + back <= length` is the central invariant. When the sum is smaller
/// than `length`, rendering inserts exactly one omission marker between the
/// two retained ranges. The struct stores counts rather than index vectors so
/// width fitting can adjust the plan without allocating.
#[derive(Clone, Copy)]
struct EdgeSelection {
    /// Total logical number of items before abbreviation.
    length: usize,
    /// Number of consecutive items retained from index zero.
    front: usize,
    /// Number of consecutive items retained through the final index.
    back: usize,
}

impl EdgeSelection {
    /// Build a size-driven selection with a fixed number of edge items.
    ///
    /// Short sequences are retained completely. Longer sequences retain
    /// `edge` items on both sides with one omitted middle region.
    fn fixed(length: usize, edge: usize) -> Self {
        if length <= edge.saturating_mul(2).saturating_add(1) {
            Self {
                length,
                front: length,
                back: 0,
            }
        } else {
            Self {
                length,
                front: edge,
                back: edge,
            }
        }
    }

    /// Return whether the plan hides at least one logical item.
    fn omitted(self) -> bool {
        self.front + self.back < self.length
    }

    /// Expand the compact plan into indices consumed by row/page renderers.
    ///
    /// `Some(index)` identifies a real item and `None` is the unique ellipsis
    /// position. Value renderers use the counts directly to avoid this small
    /// allocation on their hotter path.
    fn indices(self) -> Vec<Option<usize>> {
        let mut indices = Vec::with_capacity(
            self.front + self.back + usize::from(self.omitted()),
        );
        indices.extend((0..self.front).map(Some));
        if self.omitted() {
            indices.push(None);
        }
        indices.extend((self.length - self.back..self.length).map(Some));
        indices
    }
}

/// Scalar formatting shared by diagnostic `repr` and R-style `str`.
///
/// Implementations are provided for every scalar supported by [`Array`].
/// Values are ASCII-only, which makes byte length equal terminal column width
/// and permits safe byte-indexed middle truncation.
pub trait ArrayReprElement: Scalar {
    /// Fixed-width dtype name shown in the metadata block.
    const DTYPE_NAME: &'static str;

    /// Format one value for a flattened field or matrix cell.
    ///
    /// Complex numbers deliberately omit surrounding parentheses; non-finite
    /// floating-point values use the stable spellings `nan`, `inf`, and
    /// `-inf`.
    fn format_repr_value(self) -> String;
}

impl ArrayReprElement for bool {
    const DTYPE_NAME: &'static str = "bool";

    fn format_repr_value(self) -> String {
        if self {
            "True".to_string()
        } else {
            "False".to_string()
        }
    }
}

impl ArrayReprElement for i64 {
    const DTYPE_NAME: &'static str = "int64";

    fn format_repr_value(self) -> String {
        self.to_string()
    }
}

impl ArrayReprElement for f64 {
    const DTYPE_NAME: &'static str = "float64";

    fn format_repr_value(self) -> String {
        format_float(self)
    }
}

impl ArrayReprElement for Complex64 {
    const DTYPE_NAME: &'static str = "complex128";

    fn format_repr_value(self) -> String {
        if self.im == 0.0 {
            format!("{}+0j", format_float(self.re))
        } else if self.im.is_sign_positive() {
            format!("{}+{}j", format_float(self.re), format_float(self.im))
        } else {
            format!("{}{}j", format_float(self.re), format_float(self.im))
        }
    }
}

/// Format an array as a width-bounded diagnostic metadata block.
///
/// `address` is supplied by the caller because the core does not own the
/// surrounding language object. Python passes its `id(array)` address.
///
/// # Arguments
///
/// * `array` — typed array whose logical C-order values are displayed
/// * `address` — external object address rendered in hexadecimal
///
/// # Returns
///
/// Six lines containing the address, flattened data, shape, rank, size, and
/// dtype. Every line is at most [`ARRAY_REPR_WIDTH`] ASCII columns.
///
/// # Complexity
///
/// At most `2 * MAX_EDGE_CANDIDATES` values are read. Each logical read costs
/// O(ndim), so work is bounded by visible output rather than element count.
pub fn format_array_repr<T: ArrayReprElement>(
    array: &Array<T>,
    address: usize,
) -> String {
    let data = format_data(array, ARRAY_REPR_WIDTH - "  @ data: ".len());
    let shape = truncate_middle(
        &format!("{:?}", array.shape()),
        ARRAY_REPR_WIDTH - "  @ shape: ".len(),
    );

    [
        format!("sdnp-array at 0x{address:x}"),
        format!("  @ data: {data}"),
        format!("  @ shape: {shape}"),
        format!("  @ ndim: {}", array.ndim()),
        format!("  @ size: {}", array.size()),
        format!("  @ dtype: {}", T::DTYPE_NAME),
    ]
    .join("\n")
}

/// Format an array using zero-based R-style vector, matrix, and page labels.
///
/// One-dimensional arrays use a `[0]` row label. Two-dimensional arrays use
/// `[,0]` column and `[0,]` row labels. Higher-dimensional arrays paginate
/// over the leading axes, leaving the final two axes as each displayed
/// matrix; for example, a 3-D page starts with `[0, ,]`. Wide matrices and
/// long row/page sequences retain both edges with `...` between them.
///
/// # Arguments
///
/// * `array` — typed array to format in logical C-order
///
/// # Returns
///
/// R-style text whose every line is at most [`ARRAY_REPR_WIDTH`] columns.
///
/// # Complexity
///
/// Work is proportional to visible pages, rows, and columns. Hidden regions
/// are represented by sentinels and are never traversed.
pub fn format_array_str<T: ArrayReprElement>(array: &Array<T>) -> String {
    match array.ndim() {
        0 => truncate_middle(
            &array
                .item()
                .expect("valid scalar array")
                .format_repr_value(),
            ARRAY_REPR_WIDTH,
        ),
        1 => format_vector(array),
        2 => format_matrix(array, &[]),
        _ => format_pages(array),
    }
}

/// Format a rank-1 array as one width-bounded, zero-based R-style row.
///
/// Values are gathered directly from logical edge indices. The shared width
/// planner balances front and back counts, then rendering occurs once.
fn format_vector<T: ArrayReprElement>(array: &Array<T>) -> String {
    let (front, back) = edge_values(array);
    let selection = fit_edge_selection(
        array.size(),
        &front,
        &back,
        ARRAY_REPR_WIDTH - "[0] ".len(),
        0,
        1,
    );
    truncate_middle(
        &format!(
            "[0] {}",
            render_edge_values(&front, &back, selection, " ", "", "")
        ),
        ARRAY_REPR_WIDTH,
    )
}

/// Format rank-3+ arrays as matrices selected by leading-axis coordinates.
///
/// The final two axes form each matrix. Leading axes are flattened only for
/// page selection, then converted back to coordinates by [`unravel`].
fn format_pages<T: ArrayReprElement>(array: &Array<T>) -> String {
    let page_shape = &array.shape()[..array.ndim() - 2];
    let page_count = page_shape.iter().copied().product::<usize>();
    let pages = EdgeSelection::fixed(page_count, PAGE_EDGE_ITEMS).indices();
    let mut output = Vec::new();

    for page in pages {
        if let Some(flat_page) = page {
            let coordinates = unravel(flat_page, page_shape);
            output.push(format!(
                "{}\n{}",
                format_page_label(&coordinates),
                format_matrix(array, &coordinates)
            ));
        } else {
            output.push("...".to_string());
        }
    }
    output.join("\n\n")
}

/// Render a leading-axis coordinate in modified, zero-based R notation.
///
/// Empty row and column slots are appended so `[0]` for a 3-D input becomes
/// `[0, ,]`. Very high-rank labels are shortened to the line budget.
fn format_page_label(coordinates: &[usize]) -> String {
    let leading = coordinates
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    truncate_middle(&format!("[{leading}, ,]"), ARRAY_REPR_WIDTH)
}

/// Plan and render the final two dimensions as an R-style matrix.
///
/// Rows use a fixed edge count. Columns additionally account for actual
/// header and visible-cell widths; those widths are measured once before the
/// common edge planner chooses a balanced set that fits beside row labels.
///
/// # Arguments
///
/// * `array` — source array with rank at least two
/// * `page` — fixed coordinates for every leading axis
fn format_matrix<T: ArrayReprElement>(
    array: &Array<T>,
    page: &[usize],
) -> String {
    let rows = array.shape()[array.ndim() - 2];
    let columns = array.shape()[array.ndim() - 1];
    let selected_rows = EdgeSelection::fixed(rows, MATRIX_EDGE_ROWS).indices();
    let edge_columns = columns.min(MATRIX_EDGE_COLUMNS);
    let front_widths: Vec<_> = (0..edge_columns)
        .map(|column| matrix_column_width(array, page, &selected_rows, column))
        .collect();
    let back_widths: Vec<_> = (columns - edge_columns..columns)
        .map(|column| matrix_column_width(array, page, &selected_rows, column))
        .collect();
    let row_label_width = selected_rows
        .iter()
        .flatten()
        .map(|row| format!("[{row},]").len())
        .max()
        .unwrap_or(0);
    let selected_columns = fit_edge_widths(
        columns,
        &front_widths,
        &back_widths,
        ARRAY_REPR_WIDTH.saturating_sub(row_label_width + 1),
        0,
        1,
    )
    .indices();
    matrix_lines(array, page, &selected_rows, &selected_columns).join("\n")
}

/// Measure the display width required by one visible matrix column.
///
/// The maximum of the column header and visible row values is capped at
/// [`MAX_CELL_WIDTH`]. Omitted rows are ignored because they render as a
/// standalone omission line instead of matrix cells.
fn matrix_column_width<T: ArrayReprElement>(
    array: &Array<T>,
    page: &[usize],
    rows: &[Option<usize>],
    column: usize,
) -> usize {
    let header = format!("[,{column}]").len();
    let value = rows
        .iter()
        .flatten()
        .map(|row| format_matrix_value(array, page, *row, column).len())
        .max()
        .unwrap_or(0);
    header.max(value).min(MAX_CELL_WIDTH)
}

/// Materialize aligned header and row lines from completed edge selections.
///
/// `None` rows become standalone omission lines; `None` columns become
/// aligned omission cells. Values wider than their planned cell are shortened
/// in the middle while preserving both ends.
fn matrix_lines<T: ArrayReprElement>(
    array: &Array<T>,
    page: &[usize],
    rows: &[Option<usize>],
    columns: &[Option<usize>],
) -> Vec<String> {
    let row_label_width = rows
        .iter()
        .flatten()
        .map(|row| format!("[{row},]").len())
        .max()
        .unwrap_or(0);
    let widths: Vec<_> = columns
        .iter()
        .map(|column| match column {
            None => 3,
            Some(column) => matrix_column_width(array, page, rows, *column),
        })
        .collect();

    let mut lines = vec![format!(
        "{} {}",
        " ".repeat(row_label_width),
        columns
            .iter()
            .zip(&widths)
            .map(|(column, width)| match column {
                Some(column) => format!("{:>width$}", format!("[,{column}]")),
                None => format!("{:>width$}", "..."),
            })
            .collect::<Vec<_>>()
            .join(" ")
    )
    .trim_end()
    .to_string()];

    for row in rows {
        match row {
            None => lines.push("...".to_string()),
            Some(row) => {
                let label = format!("[{row},]");
                let cells = columns
                    .iter()
                    .zip(&widths)
                    .map(|(column, width)| match column {
                        Some(column) => {
                            let value =
                                format_matrix_value(array, page, *row, *column);
                            let value = truncate_middle(&value, *width);
                            format!("{value:>width$}")
                        }
                        None => format!("{:>width$}", "..."),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                lines.push(format!("{label:>row_label_width$} {cells}"));
            }
        }
    }
    lines
}

/// Read and format one cell at a fixed leading-axis page.
///
/// Coordinates are assembled as `page + [row, column]`, so the same path
/// correctly handles contiguous arrays and arbitrary strided views.
fn format_matrix_value<T: ArrayReprElement>(
    array: &Array<T>,
    page: &[usize],
    row: usize,
    column: usize,
) -> String {
    let mut indices = page.to_vec();
    indices.extend([row, column]);
    array
        .get(&indices)
        .expect("matrix coordinates derived from valid array geometry")
        .format_repr_value()
}

/// Convert one C-order flat index into coordinates for `shape`.
///
/// This is used only for the small set of visible pages. Division proceeds
/// from the final axis toward the first, matching row-major traversal.
fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        indices[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    indices
}

/// Format the flattened `repr` data field within an exact width budget.
///
/// Only edge values are gathered. Planning includes the two brackets and
/// comma-space separators; middle truncation remains as a guard for one huge
/// scalar that cannot fit even when displayed alone.
fn format_data<T: ArrayReprElement>(array: &Array<T>, width: usize) -> String {
    if array.size() == 0 {
        return "[]".to_string();
    }
    let (front, back) = edge_values(array);
    let selection =
        fit_edge_selection(array.size(), &front, &back, width, 2, 2);
    truncate_middle(
        &render_edge_values(&front, &back, selection, ", ", "[", "]"),
        width,
    )
}

/// Gather formatted candidates from both logical ends of an array.
///
/// Logical flat indices are resolved independently, avoiding `Iterator::skip`
/// and an O(size) scan to reach the trailing edge. The vectors may overlap for
/// short arrays; [`EdgeSelection`] prevents overlapping output ranges.
fn edge_values<T: ArrayReprElement>(
    array: &Array<T>,
) -> (Vec<String>, Vec<String>) {
    let edge = array.size().min(MAX_EDGE_CANDIDATES);
    let front = (0..edge)
        .map(|index| value_at_flat(array, index).format_repr_value())
        .collect();
    let back = (array.size() - edge..array.size())
        .map(|index| value_at_flat(array, index).format_repr_value())
        .collect();
    (front, back)
}

/// Convert formatted candidate strings into widths for the common planner.
///
/// Keeping this adapter separate lets flattened values and pre-measured
/// matrix columns share the same [`fit_edge_widths`] implementation.
fn fit_edge_selection(
    length: usize,
    front: &[String],
    back: &[String],
    width: usize,
    wrapper_width: usize,
    separator_width: usize,
) -> EdgeSelection {
    let front_widths: Vec<_> = front.iter().map(String::len).collect();
    let back_widths: Vec<_> = back.iter().map(String::len).collect();
    fit_edge_widths(
        length,
        &front_widths,
        &back_widths,
        width,
        wrapper_width,
        separator_width,
    )
}

/// Choose balanced front and back counts that fit a width budget.
///
/// `front` and `back` contain candidate widths in logical order; selected
/// back values come from the suffix of `back`. `wrapper_width` accounts for
/// delimiters such as `[]`, and `separator_width` covers spacing between all
/// visible tokens, including `...`.
///
/// Prefix sums make each width query O(1). The loop adds at most one item per
/// iteration and prefers the less-populated edge, yielding linear planning
/// time without constructing trial strings.
fn fit_edge_widths(
    length: usize,
    front: &[usize],
    back: &[usize],
    width: usize,
    wrapper_width: usize,
    separator_width: usize,
) -> EdgeSelection {
    if length == 0 {
        return EdgeSelection {
            length,
            front: 0,
            back: 0,
        };
    }

    let front_prefix = prefix_sums(front);
    let back_prefix = prefix_sums(back);
    // Back widths use a suffix subtraction because the final `back_count`
    // candidates, rather than the first ones, are rendered.
    let measured_width = |front_count: usize, back_count: usize| {
        let omitted = front_count + back_count < length;
        let token_count = front_count + back_count + usize::from(omitted);
        let back_start = back.len() - back_count;
        wrapper_width
            + front_prefix[front_count]
            + (back_prefix[back.len()] - back_prefix[back_start])
            + if omitted { 3 } else { 0 }
            + separator_width.saturating_mul(token_count.saturating_sub(1))
    };

    if length <= front.len() && measured_width(length, 0) <= width {
        return EdgeSelection {
            length,
            front: length,
            back: 0,
        };
    }

    // Seed both edges before growing the shorter side. If one scalar alone is
    // wider than the budget, final rendering truncates it instead of dropping
    // an endpoint and losing boundary context.
    let mut selection = EdgeSelection {
        length,
        front: 1,
        back: usize::from(length > 1),
    };
    loop {
        let can_front = selection.front < front.len()
            && selection.front + selection.back + 1 < length
            && measured_width(selection.front + 1, selection.back) <= width;
        let can_back = selection.back < back.len()
            && selection.front + selection.back + 1 < length
            && measured_width(selection.front, selection.back + 1) <= width;
        if !can_front && !can_back {
            break;
        }
        if can_front && (!can_back || selection.front <= selection.back) {
            selection.front += 1;
        } else {
            selection.back += 1;
        }
    }
    selection
}

/// Build a prefix table for constant-time range-width queries.
///
/// The result starts with zero and has one more entry than `widths`, so the
/// sum of `widths[a..b]` is `sums[b] - sums[a]`.
fn prefix_sums(widths: &[usize]) -> Vec<usize> {
    let mut sums = Vec::with_capacity(widths.len() + 1);
    sums.push(0);
    for width in widths {
        sums.push(sums.last().copied().unwrap_or(0) + width);
    }
    sums
}

/// Render a completed edge plan exactly once.
///
/// `repr` supplies brackets and comma-space separators; vector `str` supplies
/// spaces and no wrapper. One ellipsis is inserted exactly when the plan hides
/// logical items.
fn render_edge_values(
    front: &[String],
    back: &[String],
    selection: EdgeSelection,
    separator: &str,
    open: &str,
    close: &str,
) -> String {
    let mut parts = Vec::with_capacity(selection.front + selection.back + 1);
    parts.extend(front[..selection.front].iter().map(String::as_str));
    if selection.omitted() {
        parts.push("...");
    }
    parts.extend(
        back[back.len() - selection.back..]
            .iter()
            .map(String::as_str),
    );
    format!("{open}{}{close}", parts.join(separator))
}

/// Read a logical C-order value without traversing preceding elements.
///
/// The flat index is unraveled and passed to [`Array::get`], which applies the
/// actual offset and strides. Edge access is therefore O(ndim) for contiguous
/// arrays and arbitrary views alike.
fn value_at_flat<T: ArrayReprElement>(
    array: &Array<T>,
    flat_index: usize,
) -> T {
    let mut remainder = flat_index;
    let mut indices = vec![0; array.ndim()];
    for axis in (0..array.ndim()).rev() {
        let length = array.shape()[axis];
        indices[axis] = remainder % length;
        remainder /= length;
    }
    array
        .get(&indices)
        .expect("flat index derived from valid array geometry")
}

/// Shorten an ASCII token while retaining both its leading and trailing text.
///
/// All formatters in this module emit ASCII, so byte indices equal terminal
/// columns. Widths of three or less become dots because no content can fit
/// around a complete ellipsis.
fn truncate_middle(value: &str, width: usize) -> String {
    if value.len() <= width {
        return value.to_string();
    }
    if width <= 3 {
        return ".".repeat(width);
    }
    let available = width - 3;
    let left = (available + 1) / 2;
    let right = available / 2;
    format!("{}...{}", &value[..left], &value[value.len() - right..])
}

/// Format a float with stable NumPy-style non-finite spellings.
///
/// Finite values use Rust's shortest round-trippable decimal output; signed
/// zero remains distinguishable through `f64::to_string`.
fn format_float(value: f64) -> String {
    if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else {
        value.to_string()
    }
}
