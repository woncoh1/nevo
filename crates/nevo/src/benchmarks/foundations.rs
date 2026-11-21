use crate::Evaluator;
use std::{error::Error, fmt};

/// Invalid input or arithmetic for the Chapter 2 reference benchmarks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BenchmarkError {
    /// The benchmark's minimum dimension was not met.
    InvalidDimensions,
    /// An input has the wrong number of coordinates.
    DimensionMismatch {
        /// Configured input dimension.
        expected: usize,
        /// Supplied input dimension.
        actual: usize,
    },
    /// A coordinate is nonfinite or outside the benchmark's domain.
    InvalidCoordinate {
        /// Zero-based coordinate index.
        index: usize,
    },
    /// The computed objective is not finite.
    Overflow,
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "benchmark minimum dimension not met"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "expected {expected} coordinates, received {actual}")
            }
            Self::InvalidCoordinate { index } => {
                write!(f, "coordinate {index} is invalid for this benchmark")
            }
            Self::Overflow => write!(f, "benchmark arithmetic overflowed f64"),
        }
    }
}
impl Error for BenchmarkError {}

fn check_dimension(point: &[f64], expected: usize) -> Result<(), BenchmarkError> {
    if point.len() != expected {
        return Err(BenchmarkError::DimensionMismatch {
            expected,
            actual: point.len(),
        });
    }
    Ok(())
}

/// The unrotated Rastrigin minimization problem, `sum(x² + 10(1 − cos(2πx)))`.
///
/// The optimum is zero at the origin. This evaluator accepts any finite input
/// for which its scalar `f64` arithmetic remains finite; it does not impose the
/// customary initialization interval. It is a multimodal contrast to Sphere.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rastrigin {
    dimensions: usize,
}

impl Rastrigin {
    /// Construct a nonempty Rastrigin problem.
    pub fn new(dimensions: usize) -> Result<Self, BenchmarkError> {
        if dimensions == 0 {
            return Err(BenchmarkError::InvalidDimensions);
        }
        Ok(Self { dimensions })
    }
    /// Number of required coordinates.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    /// Compute the raw minimization objective with checked input and arithmetic.
    pub fn value(&self, point: &[f64]) -> Result<f64, BenchmarkError> {
        check_dimension(point, self.dimensions)?;
        let mut total = 0.0;
        for (index, &x) in point.iter().enumerate() {
            if !x.is_finite() {
                return Err(BenchmarkError::InvalidCoordinate { index });
            }
            total += x * x + 10.0 * (1.0 - (std::f64::consts::TAU * x).cos());
            if !total.is_finite() {
                return Err(BenchmarkError::Overflow);
            }
        }
        Ok(total)
    }
}
impl<T: AsRef<[f64]> + ?Sized> Evaluator<T> for Rastrigin {
    type Output = f64;
    type Error = BenchmarkError;
    fn evaluate(&mut self, point: &T, _seed: u64) -> Result<f64, Self::Error> {
        self.value(point.as_ref())
    }
}

/// ZDT1 with a configurable dimension of at least two and domain `[0,1]^d`.
///
/// Both objectives are minimized: `f1=x[0]`, `g=1+9*mean(x[1..])`,
/// `f2=g*(1-sqrt(f1/g))`. The Pareto front is `f2=1-sqrt(f1)` when all
/// remaining genes are zero. The original published instance uses 30 genes;
/// small dimensional demonstrations must state their changed dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Zdt1 {
    dimensions: usize,
}

impl Zdt1 {
    /// Construct ZDT1 with at least two coordinates.
    pub fn new(dimensions: usize) -> Result<Self, BenchmarkError> {
        if dimensions < 2 {
            return Err(BenchmarkError::InvalidDimensions);
        }
        Ok(Self { dimensions })
    }
    /// Number of required coordinates.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    /// Compute both minimization objectives, rejecting any coordinate outside `[0,1]`.
    pub fn value(&self, point: &[f64]) -> Result<Vec<f64>, BenchmarkError> {
        check_dimension(point, self.dimensions)?;
        for (index, &x) in point.iter().enumerate() {
            if !x.is_finite() || !(0.0..=1.0).contains(&x) {
                return Err(BenchmarkError::InvalidCoordinate { index });
            }
        }
        let g = 1.0 + 9.0 * point[1..].iter().sum::<f64>() / (self.dimensions - 1) as f64;
        let output = vec![point[0], g * (1.0 - (point[0] / g).sqrt())];
        if output.iter().any(|v| !v.is_finite()) {
            return Err(BenchmarkError::Overflow);
        }
        Ok(output)
    }
}
impl<T: AsRef<[f64]> + ?Sized> Evaluator<T> for Zdt1 {
    type Output = Vec<f64>;
    type Error = BenchmarkError;
    fn evaluate(&mut self, point: &T, _seed: u64) -> Result<Vec<f64>, Self::Error> {
        self.value(point.as_ref())
    }
}
