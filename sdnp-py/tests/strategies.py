import math

import numpy as np
from hypothesis import strategies as st

PY_DTYPES = (bool, int, float, complex)


def finite_scalar(dtype):
    if dtype is bool:
        return st.booleans()
    if dtype is int:
        return st.integers(min_value=-10_000, max_value=10_000)
    finite = st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )
    if dtype is float:
        return finite
    if dtype is complex:
        return st.builds(complex, finite, finite)
    raise AssertionError(f"unsupported dtype: {dtype}")


@st.composite
def shaped_values(
    draw,
    *,
    dtypes=PY_DTYPES,
    min_dims=1,
    max_dims=3,
    max_side=4,
    allow_empty=True,
):
    dtype = draw(st.sampled_from(dtypes))
    minimum = 0 if allow_empty else 1
    ndim = draw(st.integers(min_value=min_dims, max_value=max_dims))
    shape = tuple(
        draw(
            st.lists(
                st.integers(min_value=minimum, max_value=max_side),
                min_size=ndim,
                max_size=ndim,
            )
        )
    )
    size = math.prod(shape)
    flat = draw(st.lists(finite_scalar(dtype), min_size=size, max_size=size))
    nested = np.asarray(flat, dtype=_numpy_dtype(dtype)).reshape(shape).tolist()
    return dtype, shape, nested


@st.composite
def nonempty_shaped_values(
    draw, *, dtypes=PY_DTYPES, min_dims=1, max_dims=3, max_side=4
):
    return draw(
        shaped_values(
            dtypes=dtypes,
            min_dims=min_dims,
            max_dims=max_dims,
            max_side=max_side,
            allow_empty=False,
        )
    )


@st.composite
def broadcast_pair(draw, *, dtypes=(int, float), max_side=4):
    dtype = draw(st.sampled_from(dtypes))
    rows = draw(st.integers(min_value=1, max_value=max_side))
    columns = draw(st.integers(min_value=1, max_value=max_side))
    left = draw(st.lists(finite_scalar(dtype), min_size=rows, max_size=rows))
    right = draw(
        st.lists(finite_scalar(dtype), min_size=columns, max_size=columns)
    )
    return dtype, [[value] for value in left], [right]


def _numpy_dtype(dtype):
    return {
        bool: np.bool_,
        int: np.int64,
        float: np.float64,
        complex: np.complex128,
    }[dtype]
