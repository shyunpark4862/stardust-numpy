"""Type stubs for the top-level ``sdnp`` package.

``sdnp`` is a thin Python wrapper that re-exports everything from the
compiled extension module ``sdnp.sdnp`` (see ``sdnp/sdnp.pyi`` for the full,
NumPy-style documented surface: the ``Array`` class, iterator types, and
every module-level free function).

This file only forwards the wildcard re-export so that ``import sdnp as np``
gives static type checkers (mypy, pyright/basedpyright) and IDEs the same
completions and docstrings as the underlying extension module.
"""

from .sdnp import *  # noqa: F401,F403
from .sdnp import __all__ as __all__
