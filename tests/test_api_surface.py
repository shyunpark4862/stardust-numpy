"""M10: API.md is the contract -- every name it lists must exist, and the
package must export nothing it does not list."""

import re
from pathlib import Path

import stardust_numpy as sdnp

API = (Path(__file__).parent.parent / "API.md").read_text()
METHODS = {"reshape", "astype", "copy", "tolist", "sum", "prod", "min", "max", "mean", "var", "std",
           "argmin", "argmax", "any", "all"}
ATTRIBUTES = {"dtype", "shape", "ndim", "size", "strides", "itemsize", "nbytes", "base", "T", "mT"}
CONSTANTS = {"e", "pi", "inf", "nan", "newaxis", "bool", "int64", "float64", "complex128", "ndarray"}


def api_names():
    names = set()
    for block in re.findall(r"```\n(.*?)```", API, re.S):
        for line in block.splitlines():
            code = line.split("#")[0]
            names |= set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", code))
            if "(" not in code:
                names |= {w for w in code.split() if re.fullmatch(r"[a-z_][a-z0-9_]*", w)}
    # Most methods are also module functions (A2); copy and tolist are methods only.
    return names - {"copy", "tolist"} - ATTRIBUTES


def test_every_documented_function_exists():
    missing = sorted(n for n in api_names() if not hasattr(sdnp, n))
    assert missing == []


def test_every_documented_ndarray_member_exists():
    x = sdnp.zeros((2, 2))
    assert sorted(m for m in METHODS | ATTRIBUTES if not hasattr(x, m)) == []


def test_nothing_undocumented_is_exported():
    assert sorted(set(sdnp.__all__) - api_names() - CONSTANTS) == []
    assert set(sdnp.__all__) <= set(dir(sdnp))
