//! Reference optimization problems, independent of execution backends.

mod foundations;
mod sphere;

pub use foundations::{BenchmarkError, Rastrigin, Zdt1};
pub use sphere::{Sphere, SphereError};
