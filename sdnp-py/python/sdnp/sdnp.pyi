"""Static typing facade for the compiled ``sdnp.sdnp`` extension.

The implementation is a single PyO3 extension module.  Its typing declarations
are split across private ``_*.pyi`` files by feature area for maintainability;
this facade explicitly re-exports the unchanged public API.  Users should
continue to write ``import sdnp`` rather than importing private stub modules.
"""

from typing import Literal

from ._array import (
    Array as Array,
    flatiter as flatiter,
    axis0iter as axis0iter,
)

from ._creation import (
    array as array,
    zeros as zeros,
    ones as ones,
    full as full,
    arange as arange,
    linspace as linspace,
    logspace as logspace,
    geomspace as geomspace,
    meshgrid as meshgrid,
    eye as eye,
    eye_with as eye_with,
    tri as tri,
    tri_with as tri_with,
    tril as tril,
    triu as triu,
    diag as diag,
)

from ._ufunc import (
    add as add,
    subtract as subtract,
    multiply as multiply,
    divide as divide,
    trunc_divide as trunc_divide,
    remainder as remainder,
    power as power,
    negative as negative,
    absolute as absolute,
    equal as equal,
    not_equal as not_equal,
    less as less,
    less_equal as less_equal,
    greater as greater,
    greater_equal as greater_equal,
    logical_and as logical_and,
    logical_or as logical_or,
    logical_not as logical_not,
    isnan as isnan,
    isinf as isinf,
    isfinite as isfinite,
    conj as conj,
    real as real,
    imag as imag,
)

from ._reduction import (
    sum as sum,
    prod as prod,
    min as min,
    max as max,
    mean as mean,
    var as var,
    std as std,
    any as any,
    all as all,
    argmin as argmin,
    argmax as argmax,
    cumsum as cumsum,
    cumprod as cumprod,
)

from ._manipulation import (
    concatenate as concatenate,
    stack as stack,
    vstack as vstack,
    hstack as hstack,
)

from ._selection import (
    where as where,
    nonzero as nonzero,
    clip as clip,
)

from ._sorting import (
    sort as sort,
    argsort as argsort,
    unique as unique,
)

from ._linalg import (
    dot as dot,
    matmul as matmul,
    vdot as vdot,
    outer as outer,
    diagonal as diagonal,
    trace as trace,
)

from ._iteration import (
    ndindex as ndindex,
    ndenumerate as ndenumerate,
    nditer as nditer,
)

__optimized__: bool
__build_profile__: Literal["debug", "release"]
__all__: list[str]
