"""M2: the n-d iterator's plan (broadcast, drop, reorder, coalesce)."""

import numpy as np
import pytest

import stardust_numpy as sdnp

plan = sdnp._core._iter_plan


def sd(np_array):
    return sdnp.asarray(np_array)  # zero-copy, keeps numpy's strides


def test_contiguous_array_becomes_one_run():
    assert plan([sdnp.zeros((2, 3, 4))]) == {"shape": (24,), "strides": [(8,)], "empty": False}


def test_transposed_array_is_walked_in_memory_order():
    t = sd(np.zeros((2, 3, 4)).T)  # shape (4, 3, 2), strides (8, 32, 96)
    assert plan([t], "naive")["strides"] == [(8, 32, 96)]
    assert plan([t], "reorder") == {"shape": (2, 3, 4), "strides": [(96, 32, 8)], "empty": False}
    assert plan([t]) == {"shape": (24,), "strides": [(8,)], "empty": False}


def test_uniform_slices_coalesce_across_rows():
    # Every second element of a (4, 6) array is one evenly spaced run of 12.
    assert plan([sd(np.zeros((4, 6))[:, ::2])])["shape"] == (12,)
    # Every second row is not: two runs of 6, 96 bytes apart.
    assert plan([sd(np.zeros((4, 6))[::2])]) == {"shape": (2, 6), "strides": [(96, 8)], "empty": False}


def test_extent_one_axes_are_dropped():
    assert plan([sdnp.zeros((1, 5, 1))]) == {"shape": (5,), "strides": [(8,)], "empty": False}
    assert plan([sdnp.zeros((1, 1))]) == {"shape": (1,), "strides": [(0,)], "empty": False}


def test_broadcast_operands_get_stride_zero():
    p = plan([sdnp.zeros((3, 1)), sdnp.zeros(4)])
    assert p == {"shape": (3, 4), "strides": [(8, 0), (0, 8)], "empty": False}


def test_coalescing_requires_every_operand_to_agree():
    c = sdnp.zeros((3, 4))
    t = sd(np.zeros((4, 3)).T)
    p = plan([t, c])  # input transposed, output C-order
    assert p["shape"] == (3, 4)  # cannot merge: contiguous for one, not the other
    assert p["strides"][1] == (32, 8)  # the output (last operand) decides the order


def test_empty_iteration():
    assert plan([sdnp.zeros((3, 0, 2))])["empty"] is True


@pytest.mark.parametrize("shapes", [[(2, 3), (3, 2)], [(4,), (5,)], [(2, 1, 3), (4, 3, 1), (2,)]])
def test_incompatible_broadcast_raises(shapes):
    with pytest.raises(ValueError):
        plan([sdnp.zeros(s) for s in shapes])


@pytest.mark.parametrize(
    "shapes", [[(5, 1, 4), (3, 1)], [(1,), (2, 3)], [(2, 1, 3), (4, 1), (1, 3)], [(0, 3), (1, 3)]]
)
def test_broadcast_shape_matches_numpy(shapes):
    arrays = [sdnp.zeros(s) for s in shapes]
    want = np.broadcast_shapes(*shapes)
    got = plan(arrays, "naive")["shape"]
    assert got == want
