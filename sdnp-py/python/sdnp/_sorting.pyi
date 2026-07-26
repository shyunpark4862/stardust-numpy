"""Sorting, indirect ordering, and unique-value extraction.

Ordering functions operate on bool, integer, and float storage; complex arrays
are rejected.  ``sort`` and ``argsort`` process independent slices along the
selected axis and allocate new output, leaving the input unchanged.  Their
typical cost is O(n * log(n)) per sorted region, plus O(n) output storage.

``unique`` scans and orders values into a newly allocated one-dimensional
array of distinct elements.  No operation returns a writable view, and NumPy
options such as ``kind=``, ``order=``, ``stable=``, or unique inverse/count
outputs are not exposed.
"""

from ._array import Array, RealScalar, ScalarT

def sort(a: Array[ScalarT], axis: int | None = None) -> Array[ScalarT]:
    """Return a sorted copy along one axis.

    Sorting copies the input along the chosen axis in Rust.  Complex arrays are not supported for order-statistics operations.

    Parameters
    ----------
    a : Array
        Boolean, integer, or float input.
    axis : int or None, optional
        Sort axis. The implementation uses the last axis when ``None``.

    Returns
    -------
    Array
        Sorted copy with unchanged shape and dtype.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        For complex input or an invalid axis.

    Examples
    --------
    >>> sdnp.sort(sdnp.array([3, 1, 2])).to_list()
    [1, 2, 3]
    """

def argsort(a: Array[RealScalar], axis: int | None = None) -> Array[int]:
    """Return indices that would sort an array.

    Sorting copies the input along the chosen axis in Rust.  Complex arrays are not supported for order-statistics operations.

    Parameters
    ----------
    a : Array
        Boolean, integer, or float input.
    axis : int or None, optional
        Sort axis. The implementation uses the last axis when ``None``.

    Returns
    -------
    Array of int
        Int64 indices with the same shape as ``a``.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        For complex input or an invalid axis.

    Examples
    --------
    >>> sdnp.argsort(sdnp.array([3, 1, 2])).to_list()
    [1, 2, 0]
    """

def unique(a: Array[ScalarT]) -> Array[ScalarT]:
    """Return sorted unique values from a flattened array.

    Sorting copies the input along the chosen axis in Rust.  Complex arrays are not supported for order-statistics operations.

    Parameters
    ----------
    a : Array
        Input of any supported dtype.

    Returns
    -------
    Array
        One-dimensional sorted unique values.

    Raises
    ------
    TypeError
        If input is zero-dimensional.
    ValueError
        If uniqueness computation fails.

    Examples
    --------
    >>> sdnp.unique(sdnp.array([2, 1, 2, 3])).to_list()
    [1, 2, 3]
    """
