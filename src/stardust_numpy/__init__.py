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
    complex128,
    empty,
    empty_like,
    eye,
    float64,
    full,
    full_like,
    int64,
    linspace,
    ndarray,
    ones,
    ones_like,
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
    # conversion
    "astype",
]
