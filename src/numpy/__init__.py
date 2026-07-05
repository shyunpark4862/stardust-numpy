################################################################################
#### MIT License                                                            ####
################################################################################
#
# Copyright (c) 2026 shyunpark4862@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
################################################################################

"""Public package surface for the educational NumPy reimplementation.

Re-exports :class:`ndarray`, array creation helpers, ufuncs, linear algebra,
broadcasting, manipulation, selection, sorting, and iteration utilities.
"""

from .broadcasting import broadcast_arrays, broadcast_shape, broadcast_to
from .creation import (
    arange,
    array,
    diag,
    eye,
    full,
    ones,
    tri,
    tril,
    triu,
    zeros,
)
from .iteration import ndenumerate, ndindex, nditer
from .linalg import diagonal, outer, trace, vdot
from .manipulation import concatenate, expand_dims, stack
from .ndarray import ndarray
from .selection import clip, nonzero, take, where
from .sorting import sort, unique
from .ufuncs import (
    absolute,
    add,
    angle,
    ceil,
    conj,
    conjugate,
    cos,
    divide,
    exp,
    floor,
    floor_divide,
    imag,
    isfinite,
    isinf,
    isnan,
    log,
    logical_and,
    logical_not,
    logical_or,
    mod,
    multiply,
    power,
    real,
    round,
    sin,
    sqrt,
    square,
    subtract,
    tan,
)

__all__ = [
    "arange",
    "array",
    "broadcast_arrays",
    "broadcast_shape",
    "broadcast_to",
    "concatenate",
    "expand_dims",
    "eye",
    "diag",
    "diagonal",
    "tri",
    "tril",
    "triu",
    "full",
    "clip",
    "nonzero",
    "take",
    "where",
    "ndarray",
    "ndenumerate",
    "ndindex",
    "nditer",
    "ones",
    "sort",
    "unique",
    "stack",
    "trace",
    "outer",
    "vdot",
    "zeros",
    "absolute",
    "add",
    "angle",
    "conjugate",
    "conj",
    "real",
    "imag",
    "subtract",
    "multiply",
    "divide",
    "floor_divide",
    "mod",
    "power",
    "square",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "floor",
    "ceil",
    "logical_and",
    "logical_not",
    "logical_or",
    "round",
    "isnan",
    "isinf",
    "isfinite",
]
