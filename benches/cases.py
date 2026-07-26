"""Declarative sdnp-py versus NumPy benchmark case registry."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
import sdnp

from benches.matrix import DTYPES, SIZE_ELEMENTS, points
from benches.measure import BackendTask

PY_DTYPES = {
    "bool": bool,
    "int64": int,
    "float64": float,
    "complex128": complex,
}
NP_DTYPES = {
    "bool": np.bool_,
    "int64": np.int64,
    "float64": np.float64,
    "complex128": np.complex128,
}

ALL_DTYPES = frozenset(DTYPES)
REAL_DTYPES = frozenset({"bool", "int64", "float64"})
NUMERIC_DTYPES = frozenset({"int64", "float64", "complex128"})
ORDERED_DTYPES = frozenset({"int64", "float64"})
FLOAT_DTYPES = frozenset({"float64", "complex128"})


@dataclass(frozen=True)
class Operation:
    name: str
    category: str
    dtypes: frozenset[str] = ALL_DTYPES
    ndims: frozenset[int] = frozenset({1, 2, 3, 6})
    sizes: frozenset[str] = frozenset({"small", "medium", "large"})
    variant: str = "default"


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    function: str
    category: str
    dtype: str
    size: str
    ndim: int
    shape: tuple[int, ...]
    variant: str

    def prepare(self) -> tuple[BackendTask, BackendTask]:
        return prepare_tasks(self)

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "function": self.function,
            "category": self.category,
            "dtype": self.dtype,
            "size": self.size,
            "ndim": self.ndim,
            "shape": list(self.shape),
            "variant": self.variant,
        }


def _ops(names: list[str], category: str, **kwargs: Any) -> list[Operation]:
    return [Operation(name, category, **kwargs) for name in names]


OPERATIONS: tuple[Operation, ...] = tuple(
    _ops(["array", "zeros", "ones", "full"], "creation")
    + _ops(
        ["arange"],
        "creation",
        dtypes=frozenset({"int64"}),
        ndims=frozenset({1}),
    )
    + _ops(
        ["linspace", "logspace", "geomspace"],
        "creation",
        dtypes=frozenset({"float64"}),
        ndims=frozenset({1}),
    )
    + _ops(
        ["eye", "eye_with", "tri", "tri_with"],
        "creation",
        dtypes=NUMERIC_DTYPES,
        ndims=frozenset({2}),
    )
    + _ops(["tril", "triu"], "creation", dtypes=NUMERIC_DTYPES)
    + _ops(
        ["diag"],
        "creation",
        dtypes=NUMERIC_DTYPES,
        ndims=frozenset({1, 2}),
    )
    + _ops(
        ["meshgrid"], "creation", dtypes=NUMERIC_DTYPES, ndims=frozenset({1})
    )
    + _ops(
        ["add", "subtract", "multiply", "divide", "power"],
        "ufunc",
        dtypes=NUMERIC_DTYPES,
    )
    + _ops(
        ["trunc_divide", "remainder"],
        "ufunc",
        dtypes=ORDERED_DTYPES,
    )
    + _ops(["negative", "absolute"], "ufunc", dtypes=NUMERIC_DTYPES)
    + _ops(["equal", "not_equal"], "ufunc")
    + _ops(
        ["less", "less_equal", "greater", "greater_equal"],
        "ufunc",
        dtypes=ORDERED_DTYPES,
    )
    + _ops(["logical_and", "logical_or", "logical_not"], "ufunc")
    + _ops(["isnan", "isinf", "isfinite"], "ufunc", dtypes=FLOAT_DTYPES)
    + _ops(["conj", "real", "imag"], "ufunc", dtypes=frozenset({"complex128"}))
    + _ops(
        ["sum", "prod", "mean", "any", "all", "cumsum", "cumprod"], "reduction"
    )
    + _ops(
        ["min", "max", "var", "std", "argmin", "argmax"],
        "reduction",
        dtypes=REAL_DTYPES,
    )
    + _ops(["concatenate", "stack", "vstack", "hstack"], "manipulation")
    + _ops(["where", "nonzero"], "selection")
    + _ops(["clip"], "selection", dtypes=REAL_DTYPES)
    + _ops(["sort", "argsort"], "sorting", dtypes=REAL_DTYPES)
    + _ops(["unique"], "sorting")
    + _ops(
        ["dot"],
        "linalg",
        dtypes=NUMERIC_DTYPES,
        ndims=frozenset({1, 2}),
    )
    + _ops(
        ["matmul"], "linalg", dtypes=NUMERIC_DTYPES, ndims=frozenset({1, 2, 3})
    )
    + _ops(["vdot", "outer"], "linalg", dtypes=NUMERIC_DTYPES)
    + _ops(["diagonal", "trace"], "linalg", ndims=frozenset({2, 3, 6}))
    + _ops(["ndindex"], "iteration", dtypes=frozenset({"int64"}))
    + _ops(["ndenumerate", "nditer"], "iteration")
    + _ops(
        [
            "Array.copy",
            "Array.astype",
            "Array.reshape",
            "Array.squeeze",
            "Array.transpose",
            "Array.permute_axes",
            "Array.to_list",
            "Array.__getitem__",
            "Array.__setitem__",
            "Array.__len__",
            "Array.__iter__",
            "Array.__repr__",
            "Array.__str__",
            "Array.flat",
            "Array.shape",
            "Array.ndim",
            "Array.size",
            "Array.dtype",
            "Array.T",
        ],
        "array",
    )
    + _ops(
        [
            "operator.add",
            "operator.subtract",
            "operator.multiply",
            "operator.truediv",
            "operator.power",
            "operator.negative",
            "operator.absolute",
            "operator.matmul",
        ],
        "dunder",
        dtypes=NUMERIC_DTYPES,
    )
    + _ops(
        [
            "operator.floordiv",
            "operator.mod",
            "operator.less",
            "operator.less_equal",
            "operator.greater",
            "operator.greater_equal",
        ],
        "dunder",
        dtypes=ORDERED_DTYPES,
    )
    + _ops(
        [
            "operator.equal",
            "operator.not_equal",
        ],
        "dunder",
    )
)


def registered_functions() -> tuple[str, ...]:
    return tuple(sorted({operation.name for operation in OPERATIONS}))


def exported_function_gaps(exports: list[str] | tuple[str, ...]) -> set[str]:
    expected = set(exports) - {"Array"}
    registered = {
        operation.name for operation in OPERATIONS if "." not in operation.name
    }
    return expected - registered


def build_cases(profile: str) -> list[BenchmarkCase]:
    matrix_points = points(profile)
    cases: list[BenchmarkCase] = []
    for operation in OPERATIONS:
        for point in matrix_points:
            if point.dtype not in operation.dtypes:
                continue
            if point.ndim not in operation.ndims:
                continue
            if point.size not in operation.sizes:
                continue
            if profile == "smoke" and point.dtype != "float64":
                continue
            safe_name = (
                operation.name.replace("Array.", "array-")
                .replace("operator.", "operator-")
                .replace("__", "")
                .replace("_", "-")
            )
            case_id = (
                f"{operation.category}.{safe_name}.{point.dtype}."
                f"{point.size}.{point.ndim}d.{operation.variant}"
            )
            cases.append(
                BenchmarkCase(
                    id=case_id,
                    function=operation.name,
                    category=operation.category,
                    dtype=point.dtype,
                    size=point.size,
                    ndim=point.ndim,
                    shape=point.shape,
                    variant=operation.variant,
                )
            )
    return sorted(cases, key=lambda case: case.id)


def select_cases(
    cases: list[BenchmarkCase],
    *,
    functions: list[str] | None = None,
    case_ids: list[str] | None = None,
    matches: list[str] | None = None,
    categories: list[str] | None = None,
    dtypes: list[str] | None = None,
    sizes: list[str] | None = None,
    ndims: list[int] | None = None,
) -> list[BenchmarkCase]:
    functions = functions or []
    case_ids = case_ids or []
    matches = matches or []
    categories = categories or []
    dtypes = dtypes or []
    sizes = sizes or []
    ndims = ndims or []

    known_functions = set(registered_functions())
    unknown_functions = sorted(set(functions) - known_functions)
    if unknown_functions:
        raise ValueError(
            f"unknown function(s): {', '.join(unknown_functions)}; "
            "use --list-functions"
        )

    selected = [
        case
        for case in cases
        if (not functions or case.function in functions)
        and (not case_ids or case.id in case_ids)
        and (not matches or any(text in case.id for text in matches))
        and (not categories or case.category in categories)
        and (not dtypes or case.dtype in dtypes)
        and (not sizes or case.size in sizes)
        and (not ndims or case.ndim in ndims)
    ]
    if not selected:
        raise ValueError("no benchmark cases matched the requested filters")
    return selected


def _np_values(
    shape: tuple[int, ...], dtype: str, offset: int = 0
) -> np.ndarray:
    count = math.prod(shape)
    values = np.arange(count, dtype=np.int64) + offset
    if dtype == "bool":
        return (values % 3 != 0).reshape(shape)
    if dtype == "int64":
        return ((values % 97) + 1).astype(np.int64).reshape(shape)
    if dtype == "float64":
        return (((values % 97) + 1) * 0.125).astype(np.float64).reshape(shape)
    real = (((values % 97) + 1) * 0.125).astype(np.float64)
    imag = ((values % 11) * 0.0625).astype(np.float64)
    return (real + 1j * imag).astype(np.complex128).reshape(shape)


def _sd_array(array: np.ndarray, dtype: str) -> Any:
    return sdnp.array(array.tolist(), dtype=PY_DTYPES[dtype])


def _task(operation: Any) -> BackendTask:
    return BackendTask(lambda: operation)


def _pair_tasks(
    sd_operation: Any, np_operation: Any
) -> tuple[BackendTask, BackendTask]:
    return _task(sd_operation), _task(np_operation)


def _linalg_side(size: str) -> int:
    return {"small": 8, "medium": 32, "large": 128}[size]


def _matrix_side(size: str) -> int:
    return math.isqrt(SIZE_ELEMENTS[size])


def prepare_tasks(case: BenchmarkCase) -> tuple[BackendTask, BackendTask]:
    name = case.function
    dtype = case.dtype
    shape = case.shape
    np_a = _np_values(shape, dtype)
    np_b = _np_values(shape, dtype, 7)
    sd_a = _sd_array(np_a, dtype)
    sd_b = _sd_array(np_b, dtype)
    py_dtype = PY_DTYPES[dtype]
    np_dtype = NP_DTYPES[dtype]
    count = math.prod(shape)

    if name == "array":
        values = np_a.tolist()
        return _pair_tasks(
            lambda: sdnp.array(values, dtype=py_dtype),
            lambda: np.array(values, dtype=np_dtype),
        )
    if name in {"zeros", "ones"}:
        return _pair_tasks(
            lambda: getattr(sdnp, name)(shape, dtype=py_dtype),
            lambda: getattr(np, name)(shape, dtype=np_dtype),
        )
    if name == "full":
        fill = py_dtype(1)
        return _pair_tasks(
            lambda: sdnp.full(shape, fill),
            lambda: np.full(shape, fill, dtype=np_dtype),
        )
    if name == "arange":
        return _pair_tasks(
            lambda: sdnp.arange(count),
            lambda: np.arange(count, dtype=np.int64),
        )
    if name in {"linspace", "logspace", "geomspace"}:
        start, stop = (1.0, 4.0) if name == "geomspace" else (0.0, 1.0)
        return _pair_tasks(
            lambda: getattr(sdnp, name)(start, stop, count),
            lambda: getattr(np, name)(start, stop, count),
        )
    if name in {"eye", "tri"}:
        side = _matrix_side(case.size)
        return _pair_tasks(
            lambda: getattr(sdnp, name)(side, dtype=py_dtype),
            lambda: getattr(np, name)(side, dtype=np_dtype),
        )
    if name in {"eye_with", "tri_with"}:
        side = _matrix_side(case.size)
        sd_name = name
        np_name = name.removesuffix("_with")
        return _pair_tasks(
            lambda: getattr(sdnp, sd_name)(side, side + 1, dtype=py_dtype),
            lambda: getattr(np, np_name)(side, side + 1, dtype=np_dtype),
        )
    if name in {"tril", "triu"}:
        if case.ndim == 1:
            vector_np = _np_values((_matrix_side(case.size),), dtype)
            vector_sd = _sd_array(vector_np, dtype)
            return _pair_tasks(
                lambda: getattr(sdnp, name)(vector_sd),
                lambda: getattr(np, name)(vector_np),
            )
        return _pair_tasks(
            lambda: getattr(sdnp, name)(sd_a),
            lambda: getattr(np, name)(np_a),
        )
    if name == "diag":
        if case.ndim == 1:
            vector_np = _np_values((_matrix_side(case.size),), dtype)
            vector_sd = _sd_array(vector_np, dtype)
            return _pair_tasks(
                lambda: sdnp.diag(vector_sd),
                lambda: np.diag(vector_np),
            )
        return _pair_tasks(lambda: sdnp.diag(sd_a), lambda: np.diag(np_a))
    if name == "meshgrid":
        vector_shape = (_matrix_side(case.size),)
        left_np = _np_values(vector_shape, dtype)
        right_np = _np_values(vector_shape, dtype, 3)
        left_sd = _sd_array(left_np, dtype)
        right_sd = _sd_array(right_np, dtype)
        return _pair_tasks(
            lambda: sdnp.meshgrid(left_sd, right_sd, indexing="ij"),
            lambda: np.meshgrid(left_np, right_np, indexing="ij", copy=False),
        )

    binary = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.divide,
        "remainder": np.remainder,
        "power": np.power,
        "equal": np.equal,
        "not_equal": np.not_equal,
        "less": np.less,
        "less_equal": np.less_equal,
        "greater": np.greater,
        "greater_equal": np.greater_equal,
        "logical_and": np.logical_and,
        "logical_or": np.logical_or,
    }
    if name in binary:
        right_np = (
            np.full(shape, 2, dtype=np_dtype) if name == "power" else np_b
        )
        right_sd = _sd_array(right_np, dtype)
        np_binary = (
            np.floor_divide
            if name == "divide" and dtype == "int64"
            else binary[name]
        )
        return _pair_tasks(
            lambda: getattr(sdnp, name)(sd_a, right_sd),
            lambda: np_binary(np_a, right_np),
        )
    if name == "trunc_divide":
        np_trunc_divide = (
            (lambda: np.floor_divide(np_a, np_b))
            if dtype == "int64"
            else (lambda: np.trunc(np.divide(np_a, np_b)))
        )
        return _pair_tasks(
            lambda: sdnp.trunc_divide(sd_a, sd_b),
            np_trunc_divide,
        )
    unary = {
        "negative": np.negative,
        "absolute": np.absolute,
        "logical_not": np.logical_not,
        "isnan": np.isnan,
        "isinf": np.isinf,
        "isfinite": np.isfinite,
        "conj": np.conj,
        "real": np.real,
        "imag": np.imag,
    }
    if name in unary:
        np_unary = unary[name]
        if name in {"real", "imag"}:
            # sdnp intentionally returns a new array, while NumPy exposes
            # complex components as views. Materialize NumPy's result so this
            # benchmark compares the same copy contract.
            return _pair_tasks(
                lambda: getattr(sdnp, name)(sd_a),
                lambda: np_unary(np_a).copy(),
            )
        return _pair_tasks(
            lambda: getattr(sdnp, name)(sd_a),
            lambda: np_unary(np_a),
        )

    if name in {
        "sum",
        "prod",
        "min",
        "max",
        "mean",
        "var",
        "std",
        "any",
        "all",
    }:
        sd_call = getattr(sdnp, name)
        np_call = getattr(np, name)
        return _pair_tasks(
            lambda: sd_call(sd_a, axis=-1),
            lambda: np_call(np_a, axis=-1),
        )
    if name in {"argmin", "argmax", "cumsum", "cumprod"}:
        return _pair_tasks(
            lambda: getattr(sdnp, name)(sd_a, axis=-1),
            lambda: getattr(np, name)(np_a, axis=-1),
        )
    if name in {"concatenate", "stack", "vstack", "hstack"}:
        return _pair_tasks(
            lambda: getattr(sdnp, name)((sd_a, sd_b)),
            lambda: getattr(np, name)((np_a, np_b)),
        )
    if name == "where":
        condition_np = np.arange(count).reshape(shape) % 2 == 0
        condition_sd = _sd_array(condition_np, "bool")
        return _pair_tasks(
            lambda: sdnp.where(condition_sd, sd_a, sd_b),
            lambda: np.where(condition_np, np_a, np_b),
        )
    if name == "nonzero":
        return _pair_tasks(lambda: sdnp.nonzero(sd_a), lambda: np.nonzero(np_a))
    if name == "clip":
        return _pair_tasks(
            lambda: sdnp.clip(sd_a, 2, 8),
            lambda: np.clip(np_a, 2, 8),
        )
    if name in {"sort", "argsort"}:
        return _pair_tasks(
            lambda: getattr(sdnp, name)(sd_a, axis=-1),
            lambda: getattr(np, name)(np_a, axis=-1, kind="stable"),
        )
    if name == "unique":
        return _pair_tasks(lambda: sdnp.unique(sd_a), lambda: np.unique(np_a))

    if name in {"dot", "matmul"}:
        side = _linalg_side(case.size)
        if case.ndim == 1:
            left_np = _np_values((side,), dtype)
            right_np = _np_values((side,), dtype, 2)
        elif case.ndim == 2:
            left_np = _np_values((side, side), dtype)
            right_np = _np_values((side, side), dtype, 2)
        else:
            left_np = _np_values((2, side, side), dtype)
            right_np = _np_values((2, side, side), dtype, 2)
        left_sd = _sd_array(left_np, dtype)
        right_sd = _sd_array(right_np, dtype)
        return _pair_tasks(
            lambda: getattr(sdnp, name)(left_sd, right_sd),
            lambda: getattr(np, name)(left_np, right_np),
        )
    if name == "vdot":
        return _pair_tasks(
            lambda: sdnp.vdot(sd_a, sd_b),
            lambda: np.vdot(np_a, np_b),
        )
    if name == "outer":
        vector_shape = (_matrix_side(case.size),)
        left_np = _np_values(vector_shape, dtype)
        right_np = _np_values(vector_shape, dtype, 2)
        left_sd = _sd_array(left_np, dtype)
        right_sd = _sd_array(right_np, dtype)
        return _pair_tasks(
            lambda: sdnp.outer(left_sd, right_sd),
            lambda: np.outer(left_np, right_np),
        )
    if name in {"diagonal", "trace"}:
        return _pair_tasks(
            lambda: getattr(sdnp, name)(sd_a, axis1=-2, axis2=-1),
            lambda: getattr(np, name)(np_a, axis1=-2, axis2=-1),
        )

    if name == "ndindex":
        return _pair_tasks(
            lambda: sum(1 for _ in sdnp.ndindex(shape)),
            lambda: sum(1 for _ in np.ndindex(shape)),
        )
    if name == "ndenumerate":
        return _pair_tasks(
            lambda: sum(1 for _ in sdnp.ndenumerate(sd_a)),
            lambda: sum(1 for _ in np.ndenumerate(np_a)),
        )
    if name == "nditer":
        return _pair_tasks(
            lambda: sum(1 for _ in sdnp.nditer((sd_a,))),
            lambda: sum(1 for _ in np.nditer((np_a,))),
        )

    if name == "Array.copy":
        return _pair_tasks(lambda: sd_a.copy(), lambda: np_a.copy())
    if name == "Array.astype":
        target_py = float if dtype != "float64" else int
        target_np = np.float64 if dtype != "float64" else np.int64
        np_cast_source = np_a.real if dtype == "complex128" else np_a
        return _pair_tasks(
            lambda: sd_a.astype(target_py),
            lambda: np_cast_source.astype(target_np),
        )
    if name == "Array.reshape":
        target = (count,)
        return _pair_tasks(
            lambda: sd_a.reshape(target), lambda: np_a.reshape(target)
        )
    if name == "Array.squeeze":
        np_squeezable = np_a.reshape((1, *shape, 1))
        sd_squeezable = _sd_array(np_squeezable, dtype)
        return _pair_tasks(
            lambda: sd_squeezable.squeeze((0, -1)),
            lambda: np_squeezable.squeeze((0, -1)),
        )
    if name in {"Array.transpose", "Array.T"}:
        return _pair_tasks(
            (lambda: sd_a.transpose())
            if name.endswith("transpose")
            else (lambda: sd_a.T),
            (lambda: np_a.transpose())
            if name.endswith("transpose")
            else (lambda: np_a.T),
        )
    if name == "Array.permute_axes":
        axes = tuple(reversed(range(case.ndim)))
        return _pair_tasks(
            lambda: sd_a.permute_axes(axes),
            lambda: np_a.transpose(axes),
        )
    if name == "Array.to_list":
        return _pair_tasks(lambda: sd_a.to_list(), lambda: np_a.tolist())
    if name == "Array.__getitem__":
        index = (slice(None, None, 2),) + (slice(None),) * (case.ndim - 1)
        return _pair_tasks(lambda: sd_a[index], lambda: np_a[index])
    if name == "Array.__setitem__":

        def sd_factory() -> Any:
            target = sd_a.copy()
            return lambda: target.__setitem__((0,) * case.ndim, py_dtype(1))

        def np_factory() -> Any:
            target = np_a.copy()
            return lambda: target.__setitem__((0,) * case.ndim, np_dtype(1))

        # Reassigning the same scalar to the same element is idempotent, so one
        # target per sample run keeps array construction outside the timed loop.
        return BackendTask(sd_factory), BackendTask(np_factory)
    if name == "Array.__len__":
        return _pair_tasks(lambda: len(sd_a), lambda: len(np_a))
    if name == "Array.__iter__":
        return _pair_tasks(
            lambda: sum(1 for _ in sd_a),
            lambda: sum(1 for _ in np_a),
        )
    if name == "Array.__repr__":
        return _pair_tasks(lambda: repr(sd_a), lambda: repr(np_a))
    if name == "Array.__str__":
        return _pair_tasks(lambda: str(sd_a), lambda: str(np_a))
    if name == "Array.flat":
        return _pair_tasks(
            lambda: sum(1 for _ in sd_a.flat),
            lambda: sum(1 for _ in np_a.flat),
        )
    if name.startswith("Array."):
        attribute = name.split(".", 1)[1]
        return _pair_tasks(
            lambda: getattr(sd_a, attribute),
            lambda: getattr(np_a, attribute),
        )

    operator_map = {
        "operator.add": operator.add,
        "operator.subtract": operator.sub,
        "operator.multiply": operator.mul,
        "operator.truediv": operator.truediv,
        "operator.floordiv": operator.floordiv,
        "operator.mod": operator.mod,
        "operator.power": operator.pow,
        "operator.negative": operator.neg,
        "operator.absolute": operator.abs,
        "operator.matmul": operator.matmul,
        "operator.equal": operator.eq,
        "operator.not_equal": operator.ne,
        "operator.less": operator.lt,
        "operator.less_equal": operator.le,
        "operator.greater": operator.gt,
        "operator.greater_equal": operator.ge,
    }
    if name in {"operator.negative", "operator.absolute"}:
        op = operator_map[name]
        return _pair_tasks(lambda: op(sd_a), lambda: op(np_a))
    if name == "operator.matmul":
        side = _linalg_side(case.size)
        left_np = _np_values((side, side), dtype)
        right_np = _np_values((side, side), dtype, 2)
        left_sd = _sd_array(left_np, dtype)
        right_sd = _sd_array(right_np, dtype)
        return _pair_tasks(
            lambda: left_sd @ right_sd,
            lambda: left_np @ right_np,
        )
    if name in operator_map:
        op = operator_map[name]
        right_np = (
            np.full(shape, 2, dtype=np_dtype)
            if name == "operator.power"
            else np_b
        )
        right_sd = _sd_array(right_np, dtype)
        np_operator = (
            np.floor_divide
            if name == "operator.truediv" and dtype == "int64"
            else op
        )
        return _pair_tasks(
            lambda: op(sd_a, right_sd),
            lambda: np_operator(np_a, right_np),
        )

    raise KeyError(f"no benchmark task builder for {name}")
