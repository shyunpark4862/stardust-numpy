// The meaning of every elementwise operation, one element at a time.
//
// Each operation is a struct with
//   template <typename T> static constexpr bool ok;   which storage types it has a loop for
//   static R f(T x) / f(T a, T b);                     the per-element function
// The loop templates (kernels/elementwise.hpp) turn these into 1-D strided
// inner loops; the dispatcher picks the loop dtype from `ok` (ops/ufunc.cpp).
//
// Semantics are DESIGN.md §2.4: integers wrap and never fail (D1), floats and
// complex follow IEEE 754 and, where numpy spells a formula out (complex
// multiply/divide, floor_divide on floats, ...), the same formula in the same
// order so results are bit-for-bit numpy's. -ffp-contract=off keeps the
// compiler from fusing a*b+c into one FMA behind our back.
#pragma once

#include "core/dtype.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace sd::ops {

using i64 = std::int64_t;
using u64 = std::uint64_t;
using c128 = complex128;

template <typename T> inline constexpr bool is_bool = std::is_same_v<T, bool>;
template <typename T> inline constexpr bool is_int = std::is_same_v<T, i64>;
template <typename T> inline constexpr bool is_float = std::is_same_v<T, double>;
template <typename T> inline constexpr bool is_cplx = std::is_same_v<T, c128>;

// Wrapping integer arithmetic: signed overflow is undefined behaviour in C++,
// unsigned overflow is defined modulo 2^64 -- which is exactly two's
// complement wrap once converted back.
inline i64 wrap_add(i64 a, i64 b) { return static_cast<i64>(static_cast<u64>(a) + static_cast<u64>(b)); }
inline i64 wrap_sub(i64 a, i64 b) { return static_cast<i64>(static_cast<u64>(a) - static_cast<u64>(b)); }
inline i64 wrap_mul(i64 a, i64 b) { return static_cast<i64>(static_cast<u64>(a) * static_cast<u64>(b)); }

// numpy's complex multiply and divide (Smith's algorithm), written out:
// std::complex's operators add C99 Annex G inf/nan recovery that numpy lacks.
//
// numpy is compiled with the compiler's default FMA contraction, which turns
// a*b - c*d into fma(a, b, -(c*d)): the first product is never rounded. To be
// bit-for-bit equal we write those fmas explicitly (DESIGN.md §2.4: FMA where a
// kernel needs it, spelled out in the code; -ffp-contract=off everywhere else).
// Measured: 4000 random products, all equal to numpy with this pattern.
inline c128 cmul(c128 a, c128 b) {
    const double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    return {std::fma(ar, br, -(ai * bi)), std::fma(ar, bi, ai * br)};
}

inline c128 cdiv(c128 a, c128 b) {
    const double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    const double abs_br = std::fabs(br), abs_bi = std::fabs(bi);
    if (abs_br >= abs_bi) {
        if (abs_br == 0 && abs_bi == 0) return {ar / abs_br, ai / abs_bi};  // division by zero: inf/nan
        const double rat = bi / br, scl = 1.0 / std::fma(bi, rat, br);
        return {std::fma(ai, rat, ar) * scl, std::fma(-ar, rat, ai) * scl};
    }
    const double rat = br / bi, scl = 1.0 / std::fma(br, rat, bi);
    return {std::fma(ar, rat, ai) * scl, std::fma(ai, rat, -ar) * scl};
}

// ---- C99 complex functions (libm) -----------------------------------------
// libc++'s std::exp(std::complex) etc. are its own implementations; numpy
// calls the C library's cexp etc. These are declared here because C++ has no
// <complex.h>; `__extension__` marks the C99 _Complex type as intentional.
__extension__ typedef double _Complex c99_t;
extern "C" {
c99_t cexp(c99_t); c99_t clog(c99_t); c99_t csqrt(c99_t); c99_t cpow(c99_t, c99_t);
c99_t csin(c99_t); c99_t ccos(c99_t); c99_t ctan(c99_t);
c99_t casin(c99_t); c99_t cacos(c99_t); c99_t catan(c99_t);
c99_t csinh(c99_t); c99_t ccosh(c99_t); c99_t ctanh(c99_t);
c99_t casinh(c99_t); c99_t cacosh(c99_t); c99_t catanh(c99_t);
}

// std::complex<double> and double _Complex are both laid out as double[2].
inline c99_t to_c99(c128 z) { c99_t r; std::memcpy(&r, &z, sizeof r); return r; }
inline c128 from_c99(c99_t z) { c128 r; std::memcpy(&r, &z, sizeof r); return r; }
inline c128 call(c99_t (*f)(c99_t), c128 z) { return from_c99(f(to_c99(z))); }

// ---- comparison helpers ---------------------------------------------------

// IEEE 754-2019 maximum/minimum: NaN propagates and -0 < +0 (what numpy
// returns, and what the NEON FMAX/FMIN instructions do).
inline double fmax_nan(double a, double b) {
    if (a != a) return a;
    if (b != b) return b;
    if (a == b) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}
inline double fmin_nan(double a, double b) {
    if (a != a) return a;
    if (b != b) return b;
    if (a == b) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}

// ============================================================ binary: arithmetic
// C13: no bool loops, so bool inputs are computed in int64.

struct Add {
    static constexpr const char* name = "add";
    template <typename T> static constexpr bool ok = !is_bool<T>;
    static i64 f(i64 a, i64 b) { return wrap_add(a, b); }
    static double f(double a, double b) { return a + b; }
    static c128 f(c128 a, c128 b) { return {a.real() + b.real(), a.imag() + b.imag()}; }
};

struct Subtract {
    static constexpr const char* name = "subtract";
    template <typename T> static constexpr bool ok = !is_bool<T>;
    static i64 f(i64 a, i64 b) { return wrap_sub(a, b); }
    static double f(double a, double b) { return a - b; }
    static c128 f(c128 a, c128 b) { return {a.real() - b.real(), a.imag() - b.imag()}; }
};

struct Multiply {
    static constexpr const char* name = "multiply";
    template <typename T> static constexpr bool ok = !is_bool<T>;
    static i64 f(i64 a, i64 b) { return wrap_mul(a, b); }
    static double f(double a, double b) { return a * b; }
    static c128 f(c128 a, c128 b) { return cmul(a, b); }
};

// No integer loop: integers divide in float64.
struct Divide {
    static constexpr const char* name = "divide";
    template <typename T> static constexpr bool ok = is_float<T> || is_cplx<T>;
    static double f(double a, double b) { return a / b; }
    static c128 f(c128 a, c128 b) { return cdiv(a, b); }
};

// numpy's npy_divmod: fmod, then fix the sign and snap the quotient.
inline double float_divmod(double a, double b, double* mod_out) {
    double mod = std::fmod(a, b);
    if (!b) {
        *mod_out = mod;
        return a / b;
    }
    double div = (a - mod) / b;
    if (mod) {
        if ((b < 0) != (mod < 0)) {
            mod += b;
            div -= 1.0;
        }
    } else {
        mod = std::copysign(0.0, b);
    }
    double floordiv;
    if (div) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, a / b);
    }
    *mod_out = mod;
    return floordiv;
}

// D1: a // 0 == 0 and a % 0 == a, so a == (a // b) * b + a % b always holds.
struct FloorDivide {
    static constexpr const char* name = "floor_divide";
    template <typename T> static constexpr bool ok = is_int<T> || is_float<T>;
    static i64 f(i64 a, i64 b) {
        if (b == 0) return 0;
        if (b == -1) return wrap_sub(0, a);  // INT_MIN // -1 wraps to INT_MIN
        i64 q = a / b;                         // C++ truncates toward zero ...
        if ((a % b != 0) && ((a < 0) != (b < 0))) --q;  // ... floor rounds down
        return q;
    }
    static double f(double a, double b) {
        if (!b) return a / b;
        double mod;
        return float_divmod(a, b, &mod);
    }
};

struct Remainder {
    static constexpr const char* name = "remainder";
    template <typename T> static constexpr bool ok = is_int<T> || is_float<T>;
    static i64 f(i64 a, i64 b) {
        if (b == 0) return a;
        if (b == -1) return 0;
        i64 r = a % b;                          // sign of a ...
        if (r != 0 && ((r < 0) != (b < 0))) r += b;  // ... Python: sign of b
        return r;
    }
    static double f(double a, double b) {
        if (!b) return std::fmod(a, b);
        double mod;
        float_divmod(a, b, &mod);
        return mod;
    }
};

// Integer power by repeated squaring (wraps). Negative exponents keep an
// integer result (D1): only 1 and -1 have integer reciprocals; 0**-n is 0,
// like division by zero.
inline i64 int_pow(i64 base, i64 exp) {
    if (exp < 0) {
        if (base == 1) return 1;
        if (base == -1) return (exp & 1) ? -1 : 1;
        return 0;
    }
    i64 result = 1;
    while (exp) {
        if (exp & 1) result = wrap_mul(result, base);
        exp >>= 1;
        if (exp) base = wrap_mul(base, base);
    }
    return result;
}

// numpy's nc_pow: small integer exponents by repeated multiplication, the
// rest through the C library's cpow.
inline c128 complex_pow(c128 a, c128 b) {
    const double br = b.real(), bi = b.imag();
    if (br == 0. && bi == 0.) return {1., 0.};
    if (a.real() == 0. && a.imag() == 0.) {
        if (br > 0) return {0., 0.};
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
    if (bi == 0 && br >= -100 && br <= 100 && static_cast<double>(static_cast<i64>(br)) == br) {
        i64 n = static_cast<i64>(br);
        if (n == 1) return a;
        if (n == 2) return cmul(a, a);
        if (n == 3) return cmul(a, cmul(a, a));  // operand order matters with fused products
        if (n > -100 && n < 100) {
            c128 acc{1., 0.}, p = a;
            i64 mask = 1;
            if (n < 0) n = -n;
            for (;;) {
                if (n & mask) acc = cmul(acc, p);
                mask <<= 1;
                if (n < mask || mask <= 0) break;
                p = cmul(p, p);
            }
            if (br < 0) acc = cdiv({1., 0.}, acc);
            return acc;
        }
    }
    return from_c99(cpow(to_c99(a), to_c99(b)));
}

struct Pow {
    static constexpr const char* name = "pow";
    template <typename T> static constexpr bool ok = !is_bool<T>;
    static i64 f(i64 a, i64 b) { return int_pow(a, b); }
    static double f(double a, double b) { return std::pow(a, b); }
    static c128 f(c128 a, c128 b) { return complex_pow(a, b); }
};

struct Atan2 {
    static constexpr const char* name = "atan2";
    template <typename T> static constexpr bool ok = is_float<T>;
    static double f(double a, double b) { return std::atan2(a, b); }
};

// Ordering is undefined for complex (S5): no complex loop.
struct Maximum {
    static constexpr const char* name = "maximum";
    template <typename T> static constexpr bool ok = !is_cplx<T>;
    static bool f(bool a, bool b) { return a || b; }
    static i64 f(i64 a, i64 b) { return a > b ? a : b; }
    static double f(double a, double b) { return fmax_nan(a, b); }
};

struct Minimum {
    static constexpr const char* name = "minimum";
    template <typename T> static constexpr bool ok = !is_cplx<T>;
    static bool f(bool a, bool b) { return a && b; }
    static i64 f(i64 a, i64 b) { return a < b ? a : b; }
    static double f(double a, double b) { return fmin_nan(a, b); }
};

// ============================================================ binary: comparison
// Every dtype keeps its own loop (bool included: comparing is not arithmetic).

struct Equal {
    static constexpr const char* name = "equal";
    template <typename T> static constexpr bool ok = true;
    template <typename T> static bool f(T a, T b) { return a == b; }
};
struct NotEqual {
    static constexpr const char* name = "not_equal";
    template <typename T> static constexpr bool ok = true;
    template <typename T> static bool f(T a, T b) { return a != b; }
};
struct Less {
    static constexpr const char* name = "less";
    template <typename T> static constexpr bool ok = !is_cplx<T>;
    template <typename T> static bool f(T a, T b) { return a < b; }
};
struct LessEqual {
    static constexpr const char* name = "less_equal";
    template <typename T> static constexpr bool ok = !is_cplx<T>;
    template <typename T> static bool f(T a, T b) { return a <= b; }
};
struct Greater {
    static constexpr const char* name = "greater";
    template <typename T> static constexpr bool ok = !is_cplx<T>;
    template <typename T> static bool f(T a, T b) { return a > b; }
};
struct GreaterEqual {
    static constexpr const char* name = "greater_equal";
    template <typename T> static constexpr bool ok = !is_cplx<T>;
    template <typename T> static bool f(T a, T b) { return a >= b; }
};

// ============================================================ binary: logical and bitwise
// Logical operations take bool only (M4 decision, Array API): the dispatcher
// finds no loop for int64 and raises TypeError.

struct LogicalAnd {
    static constexpr const char* name = "logical_and";
    template <typename T> static constexpr bool ok = is_bool<T>;
    static bool f(bool a, bool b) { return a && b; }
};
struct LogicalOr {
    static constexpr const char* name = "logical_or";
    template <typename T> static constexpr bool ok = is_bool<T>;
    static bool f(bool a, bool b) { return a || b; }
};
struct LogicalXor {
    static constexpr const char* name = "logical_xor";
    template <typename T> static constexpr bool ok = is_bool<T>;
    static bool f(bool a, bool b) { return a != b; }
};

struct BitwiseAnd {
    static constexpr const char* name = "bitwise_and";
    template <typename T> static constexpr bool ok = is_bool<T> || is_int<T>;
    template <typename T> static T f(T a, T b) { return a & b; }
};
struct BitwiseOr {
    static constexpr const char* name = "bitwise_or";
    template <typename T> static constexpr bool ok = is_bool<T> || is_int<T>;
    template <typename T> static T f(T a, T b) { return a | b; }
};
struct BitwiseXor {
    static constexpr const char* name = "bitwise_xor";
    template <typename T> static constexpr bool ok = is_bool<T> || is_int<T>;
    template <typename T> static T f(T a, T b) { return a ^ b; }
};

// The shift count is read as unsigned: 64 or more (negative counts included)
// shifts every bit out. In C++ shifting by >= 64 is undefined, hence the test.
struct LeftShift {
    static constexpr const char* name = "bitwise_left_shift";
    static constexpr bool no_bool_inputs = true;
    template <typename T> static constexpr bool ok = is_int<T>;
    static i64 f(i64 a, i64 b) { return static_cast<u64>(b) >= 64 ? 0 : static_cast<i64>(static_cast<u64>(a) << b); }
};
struct RightShift {
    static constexpr const char* name = "bitwise_right_shift";
    static constexpr bool no_bool_inputs = true;
    template <typename T> static constexpr bool ok = is_int<T>;
    static i64 f(i64 a, i64 b) { return static_cast<u64>(b) >= 64 ? (a < 0 ? -1 : 0) : (a >> b); }  // arithmetic
};

// ============================================================ unary: arithmetic

struct Negative {
    static constexpr const char* name = "negative";
    template <typename T> static constexpr bool ok = !is_bool<T>;
    static i64 f(i64 a) { return wrap_sub(0, a); }
    static double f(double a) { return -a; }
    static c128 f(c128 a) { return {-a.real(), -a.imag()}; }
};

struct Positive {
    static constexpr const char* name = "positive";
    template <typename T> static constexpr bool ok = !is_bool<T>;
    template <typename T> static T f(T a) { return a; }
};

// numpy's |z| (its SIMD cabs): larger * sqrt(1 + (smaller/larger)^2), with
// an inf part winning over a nan part. Not libm's hypot, which differs by up
// to 1 ULP.
inline double complex_abs(c128 z) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    double re = std::fabs(z.real()), im = std::fabs(z.imag());
    if (re == inf || im == inf) return inf;
    if (re != re || im != im) return nan;
    const double larger = std::max(re, im), smaller = std::min(re, im);
    const double ratio = larger == 0 ? 0.0 : smaller / larger;
    return std::sqrt(std::fma(ratio, ratio, 1.0)) * larger;
}

struct Abs {
    static constexpr const char* name = "abs";
    template <typename T> static constexpr bool ok = !is_bool<T>;
    static i64 f(i64 a) { return a < 0 ? wrap_sub(0, a) : a; }  // abs(INT_MIN) == INT_MIN
    static double f(double a) { return std::fabs(a); }
    static double f(c128 a) { return complex_abs(a); }  // complex -> float64
};

struct Square {
    static constexpr const char* name = "square";
    template <typename T> static constexpr bool ok = !is_bool<T>;
    static i64 f(i64 a) { return wrap_mul(a, a); }
    static double f(double a) { return a * a; }
    static c128 f(c128 a) {  // contracted like cmul
        const double r = a.real(), i = a.imag();
        return {std::fma(r, r, -(i * i)), std::fma(r, i, i * r)};
    }
};

struct Sign {
    static constexpr const char* name = "sign";
    template <typename T> static constexpr bool ok = !is_bool<T>;
    static i64 f(i64 a) { return (a > 0) - (a < 0); }
    static double f(double a) { return a > 0 ? 1. : (a < 0 ? -1. : (a == 0 ? 0. : a)); }  // nan stays nan
    // z / |z|, with numpy's handling of zero, infinite and nan inputs.
    static c128 f(c128 z) {
        const double re = z.real(), im = z.imag(), abs = std::hypot(re, im);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(abs)) return {nan, nan};
        if (std::isinf(abs)) {
            if (std::isinf(re)) return std::isinf(im) ? c128{nan, nan} : c128{re > 0 ? 1. : -1., 0.};
            return {0., im > 0 ? 1. : -1.};
        }
        if (abs == 0) return {0., 0.};
        return {re / abs, im / abs};
    }
};

// ============================================================ unary: floating-point math
// No integer loops: integers are computed in float64.

#define SD_FLOAT_FN(Name, pyname, real_expr, cplx_expr)                     \
    struct Name {                                                           \
        static constexpr const char* name = pyname;                         \
        template <typename T> static constexpr bool ok = is_float<T> || is_cplx<T>; \
        static double f(double x) { return real_expr; }                     \
        static c128 f(c128 z) { return cplx_expr; }                         \
    };

// numpy's formulas for the complex functions C99 does not have.
inline c128 complex_expm1(c128 z) {
    const double a = std::sin(z.imag() / 2);
    return {std::fma(std::expm1(z.real()), std::cos(z.imag()), -(2 * a * a)), std::exp(z.real()) * std::sin(z.imag())};
}
inline c128 complex_log1p(c128 z) {
    const double l = std::hypot(z.real() + 1, z.imag());
    return {std::log(l), std::atan2(z.imag(), z.real() + 1)};
}
inline c128 complex_log_scaled(c128 z, double scale) {
    const c128 l = call(clog, z);
    return {l.real() * scale, l.imag() * scale};
}
inline constexpr double kLog2E = 1.442695040888963407359924681001892137;
inline constexpr double kLog10E = 0.434294481903251827651128918916605082;

SD_FLOAT_FN(Sqrt, "sqrt", std::sqrt(x), call(csqrt, z))
SD_FLOAT_FN(Exp, "exp", std::exp(x), call(cexp, z))
SD_FLOAT_FN(Expm1, "expm1", std::expm1(x), complex_expm1(z))
SD_FLOAT_FN(Log, "log", std::log(x), call(clog, z))
SD_FLOAT_FN(Log1p, "log1p", std::log1p(x), complex_log1p(z))
SD_FLOAT_FN(Log2, "log2", std::log2(x), complex_log_scaled(z, kLog2E))
SD_FLOAT_FN(Log10, "log10", std::log10(x), complex_log_scaled(z, kLog10E))
SD_FLOAT_FN(Sin, "sin", std::sin(x), call(csin, z))
SD_FLOAT_FN(Cos, "cos", std::cos(x), call(ccos, z))
SD_FLOAT_FN(Tan, "tan", std::tan(x), call(ctan, z))
SD_FLOAT_FN(Asin, "asin", std::asin(x), call(casin, z))
SD_FLOAT_FN(Acos, "acos", std::acos(x), call(cacos, z))
SD_FLOAT_FN(Atan, "atan", std::atan(x), call(catan, z))
SD_FLOAT_FN(Sinh, "sinh", std::sinh(x), call(csinh, z))
SD_FLOAT_FN(Cosh, "cosh", std::cosh(x), call(ccosh, z))
SD_FLOAT_FN(Tanh, "tanh", std::tanh(x), call(ctanh, z))
SD_FLOAT_FN(Asinh, "asinh", std::asinh(x), call(casinh, z))
SD_FLOAT_FN(Acosh, "acosh", std::acosh(x), call(cacosh, z))
SD_FLOAT_FN(Atanh, "atanh", std::atanh(x), call(catanh, z))

#undef SD_FLOAT_FN

// ============================================================ unary: rounding
// Integers are already whole: their loops are the identity, and the dtype is
// kept. No complex loops, except round (each part separately).

#define SD_ROUND_FN(Name, pyname, expr)                                      \
    struct Name {                                                           \
        static constexpr const char* name = pyname;                         \
        template <typename T> static constexpr bool ok = !is_cplx<T>;      \
        static double f(double x) { return expr; }                          \
        template <typename T> static T f(T x) { return x; }                 \
    };

SD_ROUND_FN(Floor, "floor", std::floor(x))
SD_ROUND_FN(Ceil, "ceil", std::ceil(x))
SD_ROUND_FN(Trunc, "trunc", std::trunc(x))

#undef SD_ROUND_FN

// Half to even: rint rounds in the current mode, which is round-to-nearest-even.
struct Round {
    static constexpr const char* name = "round";
    template <typename T> static constexpr bool ok = true;
    static double f(double x) { return std::rint(x); }
    static c128 f(c128 z) { return {std::rint(z.real()), std::rint(z.imag())}; }
    template <typename T> static T f(T x) { return x; }
};

// ============================================================ unary: predicates and the rest

struct IsNan {
    static constexpr const char* name = "isnan";
    template <typename T> static constexpr bool ok = true;
    static bool f(double x) { return std::isnan(x); }
    static bool f(c128 z) { return std::isnan(z.real()) || std::isnan(z.imag()); }
    template <typename T> static bool f(T) { return false; }
};
struct IsInf {
    static constexpr const char* name = "isinf";
    template <typename T> static constexpr bool ok = true;
    static bool f(double x) { return std::isinf(x); }
    static bool f(c128 z) { return std::isinf(z.real()) || std::isinf(z.imag()); }
    template <typename T> static bool f(T) { return false; }
};
struct IsFinite {
    static constexpr const char* name = "isfinite";
    template <typename T> static constexpr bool ok = true;
    static bool f(double x) { return std::isfinite(x); }
    static bool f(c128 z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }
    template <typename T> static bool f(T) { return true; }
};

struct LogicalNot {
    static constexpr const char* name = "logical_not";
    template <typename T> static constexpr bool ok = is_bool<T>;
    static bool f(bool x) { return !x; }
};

// On bool, ~ is logical negation (not the integer ~1 == -2).
struct BitwiseInvert {
    static constexpr const char* name = "bitwise_invert";
    template <typename T> static constexpr bool ok = is_bool<T> || is_int<T>;
    static bool f(bool x) { return !x; }
    static i64 f(i64 x) { return ~x; }
};

struct Conj {
    static constexpr const char* name = "conj";
    template <typename T> static constexpr bool ok = true;
    static c128 f(c128 z) { return {z.real(), -z.imag()}; }
    template <typename T> static T f(T x) { return x; }
};

}  // namespace sd::ops
