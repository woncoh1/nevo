//! Composable neuroevolution building blocks.
//!
//! Reusable algorithms and tasks share the Generate -> Evaluate -> Select
//! protocol. The Chapter 1 companion evolves a small neural controller;
//! the Sphere problem remains a separate numerical reference benchmark.
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
pub mod tasks;

pub use nevo_core::{Algorithm, Evaluated, Evaluator, Job, JobId, StepError, step};

/// Common protocol types for examples and downstream algorithms.
pub mod prelude {
    pub use crate::{Algorithm, Evaluated, Evaluator, Job, JobId, step};
}
