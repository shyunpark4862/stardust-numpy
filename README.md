# stardust-numpy

Educational NumPy reimplementation in pure Python—`ndarray`, ufuncs,
indexing, broadcasting, and linear algebra, built for learning by reading
the source.

## Overview

StarDust-Numpy is a from-scratch array library for learning how NumPy-style
numerical computing works under the hood. The entire core is readable Python
with module docstrings, numpydoc-style API docs, and 496 automated tests.

**Supported dtypes:** `bool`, `int`, `float`, `complex`, `str`

**Highlights:**

- `ndarray` with views, broadcasting, fancy indexing, and reductions
- Ufuncs, linear algebra, shape manipulation, sorting, and selection
- R-style `repr` / `str` formatting
- Integration tests for real-world workflows

This is **not** a drop-in replacement for [NumPy](https://numpy.org). It
covers a focused subset of the API for teaching and experimentation.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended)

## Setup

```bash
git clone https://github.com/shyunpark4862/stardust-numpy.git
cd stardust-numpy
uv sync
```

## Run tests

```bash
uv run pytest tests/ -q
```

## Quick example

```python
from numpy import array, dot, arange

a = array([[1.0, 2.0], [3.0, 4.0]])
b = arange(2, dtype=float)
print(dot(a, b))   # array([5., 11.])
print(a.T)
print(a.mean(axis=0))
```

## Project layout

```
src/numpy/     package source (ndarray, ufuncs, mixins, helpers)
tests/         pytest suite
main.py        formatting and edge-case demos
```

## License

MIT License — see source file headers and [LICENSE](LICENSE).
