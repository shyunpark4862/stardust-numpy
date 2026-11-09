// Python entry point of the native core (`stardust_numpy._core`).

#include "bind/ndarray_bind.hpp"
#include "bind/pyutil.hpp"
#include "ops/copy.hpp"
#include "ops/creation.hpp"
#include "ops/shape.hpp"
#include "ops/ufunc.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <Accelerate/Accelerate.h>

namespace nb = nanobind;

namespace {

// Calls into Accelerate once so a broken link or a wrong ILP64 setup fails
// loudly at import-test time rather than in M6.
double accelerate_smoke_test() {
    const double x[] = {1.0, 2.0, 3.0};
    const double y[] = {4.0, 5.0, 6.0};
    return cblas_ddot(3, x, 1, y, 1);  // 1*4 + 2*5 + 3*6 = 32
}

nb::dict build_info() {
    nb::dict info;
    info["compiler"] = std::string(__VERSION__);
    info["cplusplus"] = static_cast<long>(__cplusplus);
#if defined(__ARM_NEON)
    info["neon"] = true;
#else
    info["neon"] = false;
#endif
#if defined(__ARM_FEATURE_COMPLEX)
    info["fcma"] = true;
#else
    info["fcma"] = false;
#endif
    info["blas_ilp64"] = sizeof(__LAPACK_int) == 8;
    info["accelerate_ddot"] = accelerate_smoke_test();
    return info;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "stardust-numpy native core";
    // nanobind reports objects still alive when its module is torn down. A
    // user's global dict holding sdnp.float64 at interpreter exit (finalization
    // order is not ours to control) triggers it without any real leak, so the
    // check is off; tests/test_m1_ndarray.py measures refcounts directly instead.
    nb::set_leak_warnings(false);
    sd::register_dtypes(m);  // first: other bindings use the dtype objects
    sd::bind_ndarray(m);
    sd::bind_creation(m);
    sd::bind_copy(m);
    sd::bind_shape(m);
    sd::bind_ufuncs(m);
    m.def("_build_info", &build_info, "Toolchain facts captured at compile time.");
}
