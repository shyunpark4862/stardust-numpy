//! Scalar element types, promotion rules, and casting traits.
//!
//! This module defines which Rust types may appear as [`Array`](crate::Array)
//! elements and how they combine in binary operations. Promotion follows a
//! fixed NumPy-like ladder: `bool < i64 < f64 < Complex<f64>`. Wider
//! conversions are explicit via [`CastTo`] and [`ArrayCast`].
//!
//! **Promotion vs cast:** [`Promote`] and [`CastTo`] apply at compile time
//! when ufuncs widen operands to a common dtype. [`ArrayCast`] covers every
//! supported conversion when the user calls [`Array::astype`], including
//! narrowing and complex-to-real extraction.

mod cast;
mod promotion;
mod scalar;

pub use cast::ArrayCast;
pub use promotion::{CastTo, Promote};
pub use scalar::{AsBool, Complex64, Scalar};
