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

"""Runtime type identification for ``ndarray`` instances.

Lazy import of :class:`ndarray` avoids circular dependencies while
supporting ``is_ndarray`` checks throughout the package.
"""

from __future__ import annotations

from typing import Any


def ndarray_type() -> type:
    """
    Return the :class:`ndarray` class with deferred import.

    Internal helper used by :func:`is_ndarray` to avoid a circular import
    between ``_types`` and ``ndarray``.

    Returns
    -------
    type
        The :class:`numpy.ndarray` class object.
    """
    from .ndarray import ndarray

    return ndarray


def is_ndarray(obj: Any) -> bool:
    """
    Return whether ``obj`` is an instance of this library's ``ndarray``.

    Internal helper used by validation, iteration, and public API entry
    points that need a fast type check.

    Parameters
    ----------
    obj : any
        Object to test.

    Returns
    -------
    bool
        ``True`` if ``obj`` is an :class:`ndarray`, otherwise ``False``.

    Notes
    -----
    Unlike :func:`numpy.isscalar`, this helper only recognizes the local
    ``ndarray`` type and does not treat NumPy arrays or other array
    libraries as ndarray instances.
    """
    return isinstance(obj, ndarray_type())
