"""M0: the extension builds, imports, and was compiled for the intended target."""

import stardust_numpy as sdnp


def test_import():
    assert sdnp._core.__name__ == "stardust_numpy._core"


def test_build_targets_this_mac():
    info = sdnp._core._build_info()
    assert info["cplusplus"] >= 202002  # C++20
    assert info["neon"]
    assert info["fcma"]  # complex multiply-accumulate, used in M7


def test_accelerate_is_linked_with_ilp64():
    info = sdnp._core._build_info()
    assert info["blas_ilp64"]
    assert info["accelerate_ddot"] == 32.0
