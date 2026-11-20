use std::error::Error;
use std::fmt;

use crate::Evaluator;

/// The minimization problem `f(x) = sum(x[i] * x[i])`.
///
/// The mathematical optimum is zero at the all-zero vector. The evaluator
/// accepts any finite point of the configured dimension, provided the result
/// is representable as a finite `f64`. Initialization bounds are a search
/// configuration, not a restriction on this objective's domain.
///
/// Computation is scalar, left-to-right, with `f64` accumulation. Extremely
/// small squares can underflow; cross-backend bitwise identity is not promised.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sphere {
    dimensions: usize,
}

impl Sphere {
    /// Construct a Sphere problem with at least one coordinate.
    ///
    /// # Errors
    /// Returns [`SphereError::ZeroDimensions`] for a zero-dimensional problem.
    pub fn new(dimensions: usize) -> Result<Self, SphereError> {
        if dimensions == 0 {
            return Err(SphereError::ZeroDimensions);
        }
        Ok(Self { dimensions })
    }

    /// Number of coordinates required by this problem.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Compute the raw objective value: smaller is better.
    ///
    /// # Errors
    /// Rejects the wrong dimension, nonfinite coordinates, and overflow.
    /// It neither clips coordinates nor silently replaces invalid scores.
    pub fn value(&self, point: &[f64]) -> Result<f64, SphereError> {
        if point.len() != self.dimensions {
            return Err(SphereError::DimensionMismatch {
                expected: self.dimensions,
                actual: point.len(),
            });
        }

        let mut total = 0.0;
        for (index, &value) in point.iter().enumerate() {
            if !value.is_finite() {
                return Err(SphereError::NonFiniteCoordinate { index });
            }
            total += value * value;
            if !total.is_finite() {
                return Err(SphereError::Overflow);
            }
        }
        Ok(total)
    }
}

impl<T: AsRef<[f64]> + ?Sized> Evaluator<T> for Sphere {
    type Output = f64;
    type Error = SphereError;

    fn evaluate(&mut self, input: &T, _seed: u64) -> Result<f64, Self::Error> {
        // Sphere is deterministic: its value does not depend on the seed.
        self.value(input.as_ref())
    }
}

/// Invalid configuration or input for the Sphere reference evaluator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SphereError {
    /// At least one coordinate is required.
    ZeroDimensions,
    /// The input length differs from the configured dimension.
    DimensionMismatch {
        /// Configured number of coordinates.
        expected: usize,
        /// Supplied number of coordinates.
        actual: usize,
    },
    /// A coordinate is NaN or infinite.
    NonFiniteCoordinate {
        /// Zero-based coordinate index.
        index: usize,
    },
    /// A square or accumulated sum cannot be represented as a finite `f64`.
    Overflow,
}

impl fmt::Display for SphereError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions => write!(f, "Sphere requires at least one dimension"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "expected {expected} coordinates, received {actual}")
            }
            Self::NonFiniteCoordinate { index } => {
                write!(f, "coordinate {index} is not finite")
            }
            Self::Overflow => write!(f, "Sphere objective overflowed f64"),
        }
    }
}

impl Error for SphereError {}
