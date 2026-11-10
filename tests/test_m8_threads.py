"""M8: threads. Every result must be bit-for-bit the same for any thread count
(DESIGN.md §2.4), and the pool must survive several Python threads at once."""

import math
import threading

import numpy as np
import pytest

import stardust_numpy as sdnp

COUNTS = [1, 2, 3, 7, 10]


@pytest.fixture(autouse=True)
def restore_threads():
    before = sdnp.get_num_threads()
    yield
    sdnp.set_num_threads(before)


def bits(r):
    """Comparable bytes of any result (array, scalar or tuple of them)."""
    if isinstance(r, tuple):
        return tuple(bits(x) for x in r)
    if isinstance(r, sdnp.ndarray):
        a = np.asarray(r)
        return (a.shape, a.dtype.str, np.ascontiguousarray(a).tobytes())
    return repr(r)


def battery():
    """Large enough to cross the parallel threshold, in several layouts."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal(600_000)
    f[::997] = np.nan
    m = rng.standard_normal((700, 900))
    t3 = rng.standard_normal((60, 70, 80)).transpose(2, 0, 1)
    i = rng.integers(-1000, 1000, (500, 700))
    c = rng.standard_normal(300_000) + 1j * rng.standard_normal(300_000)
    b = rng.random((800, 600)) > 0.5
    idx = rng.integers(0, 600_000, 400_000)
    S = sdnp.asarray
    F, M, T3, I, C, B, IDX = S(f), S(m), S(t3), S(i), S(c), S(b), S(idx)
    clean = S(np.nan_to_num(f))
    P = S(1.0 + np.nan_to_num(f) * 1e-4)
    ops = {
        "f + m-ish": lambda: F * 2.0 + 1.0,
        "exp": lambda: sdnp.exp(M),
        "transposed copy": lambda: T3.copy(),
        "int + float (buffered cast)": lambda: I + M[:500, :700],
        "where": lambda: sdnp.where(S(b[:700, :600]), S(m[:700, :600]), 0.0),
        "clip": lambda: sdnp.clip(M, min=-0.5, max=0.5),
        "sum full": lambda: sdnp.sum(clean),
        "sum full nan": lambda: sdnp.sum(F),
        "sum axis 0": lambda: sdnp.sum(M, axis=0),
        "sum axis 1": lambda: sdnp.sum(M, axis=1),
        "sum 3d axes": lambda: sdnp.sum(T3, axis=(0, 2)),
        "sum complex": lambda: sdnp.sum(S(c)),
        "sum bool": lambda: sdnp.sum(B),
        "sum int": lambda: sdnp.sum(I),
        "prod int": lambda: sdnp.prod(I % 3 + 1),
        "mean/var/std": lambda: (sdnp.mean(M), sdnp.var(M), sdnp.std(M, axis=0)),
        "max/min nan": lambda: (sdnp.max(F), sdnp.min(clean), sdnp.max(M, axis=1)),
        "argmax/argmin": lambda: (sdnp.argmax(F), sdnp.argmin(clean), sdnp.argmax(M, axis=0), sdnp.argmin(M, axis=1)),
        "any/all": lambda: (sdnp.any(B), sdnp.all(B), sdnp.any(B, axis=0)),
        "cumulative_sum": lambda: sdnp.cumulative_sum(M, axis=1),
        "sort rows": lambda: (sdnp.sort(M), sdnp.argsort(M, axis=0)),
        "unique": lambda: sdnp.unique_values(I),
        "matmul int": lambda: I[:300, :400] @ I[:400, :350],
        "matmul float": lambda: M[:300] @ M[:300].T,
        "gather": lambda: F[IDX],
        "astype": lambda: sdnp.astype(M, sdnp.int64),
        "sum complex big": lambda: sdnp.sum(C),
        # M10+: blocked over a reduced axis / multi-accumulator / blocked
        # scans. Not numpy's bits any more, but still the same for every
        # thread count.
        "reduced-axis blocks": lambda: (sdnp.mean(M, axis=0), sdnp.var(M, axis=0), sdnp.prod(M, axis=0),
                                        sdnp.max(M, axis=0), sdnp.sum(B, axis=0), sdnp.sum(I, axis=0)),
        "prod float/complex": lambda: (sdnp.prod(P), sdnp.prod(P[::3]), sdnp.prod(S(c[:200_000] / abs(c[:200_000]))),
                                       sdnp.prod(M[:, :400] * 0.001 + 1.0, axis=1)),
        "long scans": lambda: (sdnp.cumulative_sum(clean), sdnp.cumulative_prod(P), sdnp.cumulative_sum(C),
                               sdnp.cumulative_sum(F[::-2]), sdnp.cumulative_sum(I.reshape(-1)[:100_003]),
                               sdnp.cumulative_sum(S(m.reshape(-1)[:90_000].reshape(30_000, 3)), axis=0),
                               sdnp.cumulative_prod(P, include_initial=True)),
        # M10+ tier 2: take along one axis, masks in chunks.
        "take with blocks": lambda: (M[:, S(idx[:500] % 900)], T3[:, :, S(idx[:40] % 70)], M[S(idx[:300] % 700)]),
        "mask gather": lambda: (F[S(f > 0.3)], F[S(f > 2.5)], M[S(m > 0)], M[:, S(m[0] > 0)]),
        "mask scatter": lambda: masked_assign(F, S(f > 0.3)),
    }
    return ops


def masked_assign(x, mask):
    a = x.copy()
    a[mask] = -1.0
    b = x.copy()
    b[mask] = sdnp.arange(float(sdnp.sum(mask)))
    return a, b


def test_results_do_not_depend_on_thread_count():
    ops = battery()
    reference = {}
    for n in COUNTS:
        sdnp.set_num_threads(n)
        for name, fn in ops.items():
            got = bits(fn())
            if n == COUNTS[0]:
                reference[name] = got
            else:
                assert got == reference[name], f"{name}: differs with {n} threads"


def test_parallel_sum_still_equals_numpy():
    # The parallel pairwise sum walks numpy's recursion tree: same bits.
    x = np.random.default_rng(1).standard_normal(3_000_001)
    for n in (1, 10):
        sdnp.set_num_threads(n)
        assert sdnp.sum(sdnp.asarray(x)) == np.sum(x)
        c = x[:2_000_000] + 1j * x[1_000_001:]
        assert sdnp.sum(sdnp.asarray(c)) == np.sum(c)


def test_set_and_get_num_threads():
    sdnp.set_num_threads(3)
    assert sdnp.get_num_threads() == 3
    sdnp.set_num_threads(1)
    assert sdnp.get_num_threads() == 1
    for bad, exc in [(0, ValueError), (-1, ValueError), (True, TypeError), (2.0, TypeError), (2000, ValueError)]:
        with pytest.raises(exc):
            sdnp.set_num_threads(bad)
    with pytest.raises(TypeError):
        sdnp.set_num_threads(n=2)  # positional-only


def test_default_is_the_performance_cores():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", "import stardust_numpy as s; print(s.get_num_threads())"],
                         capture_output=True, text=True, check=True).stdout
    perf = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout)
    assert int(out) == perf


def test_several_python_threads_at_once():
    # While one thread's job holds the pool (with the GIL released), another
    # thread's job runs on its own thread: both must still be right.
    sdnp.set_num_threads(10)
    rng = np.random.default_rng(2)
    data = [rng.standard_normal(400_000) for _ in range(6)]
    sdnp.set_num_threads(1)  # the reference: computed alone, on one thread
    want = [(sdnp.sum(sdnp.asarray(d)), sdnp.max(sdnp.asarray(d)), sdnp.exp(sdnp.asarray(d)).tolist()[:1000])
            for d in data]
    sdnp.set_num_threads(10)
    errors = []

    def work(k):
        try:
            x = sdnp.asarray(data[k])
            for _ in range(20):
                got = (sdnp.sum(x), sdnp.max(x), sdnp.exp(x).tolist()[:1000])
                if got[0] != want[k][0] or got[1] != want[k][1] or got[2] != want[k][2]:
                    errors.append(k)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(repr(e))

    threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_resize_while_others_compute():
    stop = False
    errors = []
    x = sdnp.asarray(np.ones(500_000))

    def work():
        while not stop:
            if sdnp.sum(x) != 500_000.0:
                errors.append("wrong")

    t = threading.Thread(target=work)
    t.start()
    for n in [1, 4, 10, 2, 10]:
        sdnp.set_num_threads(n)
    stop = True
    t.join()
    assert errors == []
    assert math.isfinite(sdnp.sum(x))
