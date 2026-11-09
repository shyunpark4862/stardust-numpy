// Python entry point of the native core (`stardust_numpy._core`).
//
// M0 only proves that the toolchain works end to end: nanobind binding,
// the target CPU features, and linking against Accelerate.

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
    m.def("_build_info", &build_info, "Toolchain facts captured at compile time.");
}
