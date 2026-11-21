//! Composable neuroevolution building blocks.
//!
//! Reusable algorithms and tasks share the Generate -> Evaluate -> Select
//! protocol. Chapter 1 evolves a small neural controller; Chapter 2 adds
//! foundational search methods, Pareto operators, and scalar neural primitives.
//! Benchmarks and tasks remain independent of concrete search algorithms.
//!
//! ```
//! use nevo::benchmarks::Sphere;
//!
//! let sphere = Sphere::new(2)?;
//! assert_eq!(sphere.value(&[3.0, 4.0])?, 25.0);
//! # Ok::<(), nevo::benchmarks::SphereError>(())
//! ```

pub mod algorithms;
pub mod benchmarks;
pub mod multiobjective;
pub mod nn;
pub mod operators;
pub mod tasks;

pub use nevo_core::{Algorithm, Evaluated, Evaluator, Job, JobId, StepError, step};

/// Common protocol types for examples and downstream algorithms.
pub mod prelude {
    pub use crate::{Algorithm, Evaluated, Evaluator, Job, JobId, step};
}
