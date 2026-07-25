"""Shape, dtype, and profile axes for the benchmark matrix."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

DTYPES = ("bool", "int64", "float64", "complex128")
SIZE_ELEMENTS = {
    "small": 64,
    "medium": 4_096,
    "large": 262_144,
}
NDIMS = (1, 2, 3, 6)
PROFILES = ("smoke", "standard", "full")


@dataclass(frozen=True)
class MatrixPoint:
    size: str
    ndim: int
    dtype: str
    shape: tuple[int, ...]


def shape_for(size: str, ndim: int) -> tuple[int, ...]:
    """Build a balanced positive shape near the requested element budget."""
    if size not in SIZE_ELEMENTS:
        raise ValueError(f"unknown size: {size}")
    if ndim not in NDIMS:
        raise ValueError(f"unsupported ndim: {ndim}")

    target = SIZE_ELEMENTS[size]
    side = max(1, int(round(target ** (1.0 / ndim))))
    shape = [side] * ndim
    tail = prod(shape[1:]) if ndim > 1 else 1
    shape[0] = max(1, target // tail)
    return tuple(shape)


def profile_axes(profile: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if profile == "smoke":
        return ("small",), (2,)
    if profile == "standard":
        return ("small", "medium", "large"), NDIMS
    if profile == "full":
        return tuple(SIZE_ELEMENTS), NDIMS
    raise ValueError(f"unknown profile: {profile}")


def points(
    profile: str,
    *,
    dtypes: tuple[str, ...] = DTYPES,
) -> list[MatrixPoint]:
    sizes, ndims = profile_axes(profile)
    if profile == "standard":
        combinations = [
            ("small", 2),
            ("medium", 1),
            ("medium", 2),
            ("medium", 3),
            ("medium", 6),
            ("large", 2),
        ]
    else:
        combinations = [(size, ndim) for size in sizes for ndim in ndims]
    return [
        MatrixPoint(size, ndim, dtype, shape_for(size, ndim))
        for size, ndim in combinations
        for dtype in dtypes
    ]
