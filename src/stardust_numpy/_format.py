"""repr() and str() of arrays.

Kept in Python on purpose (ARCHITECTURE.md §1): printing is a cold path, and
string layout is far easier to read here than in C++. The C++ side only hands
over the (possibly summarized) values via ``ndarray._summary``.

Layout rules, numpy-like:
  * every element is right-aligned to the widest element of the whole array;
  * rows of a 2-D block go on separate lines, 2-D blocks are separated by a
    blank line, and so on for higher dimensions;
  * arrays with more than ``_THRESHOLD`` elements show only ``_EDGE`` items at
    each end of every axis, with ``...`` in between.
"""

_THRESHOLD = 1000
_EDGE = 3
_LINEWIDTH = 75


def _fmt(v):
    if v is True or v is False:
        return "True" if v else "False"
    return repr(v)  # int: '12', float: '1.5' / 'nan' / '1e+300', complex: '(1+2j)'


def _leaves(node, depth, ndim, out):
    if node is Ellipsis:
        return
    if depth == ndim:
        out.append(_fmt(node))
        return
    for child in node:
        _leaves(child, depth + 1, ndim, out)


def _render(node, depth, ndim, width, sep, indent):
    """Render one bracketed level; `indent` is the column of its '['."""
    if depth == ndim - 1:
        items = ["..." if v is Ellipsis else _fmt(v).rjust(width) for v in node]
        lines, line = [], ""
        for i, item in enumerate(items):
            piece = item + (sep if i < len(items) - 1 else "")
            # +1 for the '[' (or the indentation that replaces it on wrapped lines)
            if line and indent + 1 + len(line) + len(piece.rstrip()) > _LINEWIDTH:
                lines.append(line.rstrip())
                line = ""
            line += piece
        lines.append(line)
        return "[" + ("\n" + " " * (indent + 1)).join(lines) + "]"
    row_sep = sep.rstrip() + "\n" * (ndim - depth - 1) + " " * (indent + 1)
    rows = ["..." if child is Ellipsis else _render(child, depth + 1, ndim, width, sep, indent + 1)
            for child in node]
    return "[" + row_sep.join(rows) + "]"


def _body(arr, sep, indent):
    edge = _EDGE if arr.size > _THRESHOLD else 0
    values = arr._summary(edge)
    leaves = []
    _leaves(values, 0, arr.ndim, leaves)
    width = max((len(s) for s in leaves), default=0)
    return _render(values, 0, arr.ndim, width, sep, indent)


def array_repr(arr):
    if arr.size == 0:
        shape = "" if arr.ndim == 1 else f"shape={arr.shape}, "
        return f"array([], {shape}dtype={arr.dtype})"
    prefix = "array("
    return f"{prefix}{_body(arr, ', ', len(prefix))}, dtype={arr.dtype})"


def array_str(arr):
    if arr.size == 0:
        return "[]"
    return _body(arr, " ", 0)
