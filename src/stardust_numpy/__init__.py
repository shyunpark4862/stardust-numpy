"""stardust-numpy: an educational rebuild of a NumPy subset.

Usage: ``import stardust_numpy as sdnp``. The public API is specified in API.md.
"""

import math as _math

from . import _core
from ._core import (
    arange,
    asarray,
    astype,
    bool,
    broadcast_arrays,
    broadcast_to,
    complex128,
    concat,
    empty,
    empty_like,
    expand_dims,
    eye,
    float64,
    full,
    full_like,
    imag,
    int64,
    linspace,
    matrix_transpose,
    moveaxis,
    ndarray,
    ones,
    ones_like,
    permute_dims,
    ravel,
    real,
    reshape,
    squeeze,
    stack,
    zeros,
    zeros_like,
)

e = _math.e
pi = _math.pi
inf = _math.inf
nan = _math.nan
newaxis = None

__all__ = [
    # dtypes and constants
    "bool", "int64", "float64", "complex128", "e", "pi", "inf", "nan", "newaxis",
    # the array type
    "ndarray",
    # creation
    "asarray", "zeros", "ones", "empty", "full",
    "zeros_like", "ones_like", "empty_like", "full_like",
    "arange", "linspace", "eye",
    # shape manipulation
    "reshape", "ravel", "permute_dims", "matrix_transpose", "moveaxis",
    "squeeze", "expand_dims", "broadcast_to", "broadcast_arrays", "concat", "stack",
    # conversion
    "astype",
    # elementwise (more in M4)
    "real", "imag",
]
