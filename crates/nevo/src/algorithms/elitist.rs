//! An original mutation-only teaching search, not a canonical GA or Gaussian ES.

use std::error::Error;
use std::fmt;

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};

use crate::{Algorithm, Evaluated, Job, JobId};

/// Preference for an unmodified scalar objective value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Objective {
    /// Retain smaller values.
    Minimize,
    /// Retain larger values.
    Maximize,
}

impl Objective {
    fn prefers(self, candidate: f64, incumbent: f64) -> bool {
        match self {
            Self::Minimize => candidate < incumbent,
            Self::Maximize => candidate > incumbent,
        }
    }
}

/// Configuration for [`ElitistSearch`].
///
/// Initialization samples each coordinate in `[-initial_bound, initial_bound)`.
/// Subsequent batches perturb the retained point independently in each
/// coordinate by a value in `[-mutation_scale, mutation_scale)`. Search is
/// unbounded after initialization. Both ranges must be positive and finite.
#[derive(Clone, Debug, PartialEq)]
pub struct ElitistConfig {
    /// Number of coordinates in every candidate; must be positive.
    pub dimensions: usize,
    /// Number of evaluated candidates per batch; must be positive.
    pub offspring: usize,
    /// Half-width of the initial sampling interval.
    pub initial_bound: f64,
    /// Half-width of the uniform mutation interval.
    pub mutation_scale: f64,
    /// Whether smaller or larger objective values are preferred.
    pub objective: Objective,
    /// Root seed for independent candidate and evaluation-seed streams.
    pub seed: u64,
}

impl ElitistConfig {
    /// Use 32 offspring, initial bound 1, mutation scale 0.2, and minimization.
    pub fn new(dimensions: usize, seed: u64) -> Self {
        Self {
            dimensions,
            offspring: 32,
            initial_bound: 1.0,
            mutation_scale: 0.2,
            objective: Objective::Minimize,
            seed,
        }
    }

    fn validate(&self) -> Result<(), ElitistError> {
        if self.dimensions == 0 {
            return Err(ElitistError::InvalidConfig("dimensions must be positive"));
        }
        if self.offspring == 0 {
            return Err(ElitistError::InvalidConfig("offspring must be positive"));
        }
        for (value, message) in [
            (
                self.initial_bound,
                "initial_bound and twice its value must be positive and finite",
            ),
            (
                self.mutation_scale,
                "mutation_scale and twice its value must be positive and finite",
            ),
        ] {
            if value <= 0.0 || !value.is_finite() || !(2.0 * value).is_finite() {
                return Err(ElitistError::InvalidConfig(message));
            }
        }
        self.dimensions
            .checked_mul(self.offspring)
            .ok_or(ElitistError::InvalidConfig(
                "candidate count overflows usize",
            ))?;
        Ok(())
    }
}

/// Retain the best point while sampling uniformly perturbed descendants.
///
/// The first batch is random initialization. Every later batch contains
/// `offspring` mutations of the retained point. Selection compares all new
/// evaluations with the incumbent; exact ties keep the incumbent, or the
/// earliest issued point during initialization. The incumbent is not evaluated
/// again. This cached-score policy is intended for deterministic evaluators;
/// it is not a noise-aware optimizer for stochastic fitness.
///
/// This teaching variant has `(1 + lambda)` elitism after initialization, but
/// no crossover, adaptive step size, or Gaussian mutation. It is not presented
/// as the book's canonical genetic algorithm or evolution strategy.
///
/// Randomness uses the pinned `rand_chacha` implementation of `ChaCha8Rng`.
/// Both RNGs use `seed_from_u64(config.seed)`: stream 0 supplies candidate
/// coordinates and stream 1 supplies evaluation seeds. A uniform coordinate
/// uses the high 53 bits of one `next_u64()` divided by `2^53`.
#[derive(Debug)]
pub struct ElitistSearch {
    config: ElitistConfig,
    candidates_rng: ChaCha8Rng,
    seeds_rng: ChaCha8Rng,
    pending: Option<Vec<Job<Vec<f64>>>>,
    best: Option<Evaluated<Vec<f64>, f64>>,
    next_id: Option<u64>,
    batches: usize,
    evaluations: usize,
}

impl ElitistSearch {
    /// Validate a configuration and initialize the two random streams.
    ///
    /// # Errors
    /// Rejects zero dimensions/populations and invalid numeric ranges.
    pub fn new(config: ElitistConfig) -> Result<Self, ElitistError> {
        config.validate()?;
        let mut candidates_rng = ChaCha8Rng::seed_from_u64(config.seed);
        candidates_rng.set_stream(0);
        let mut seeds_rng = ChaCha8Rng::seed_from_u64(config.seed);
        seeds_rng.set_stream(1);
        Ok(Self {
            config,
            candidates_rng,
            seeds_rng,
            pending: None,
            best: None,
            next_id: Some(0),
            batches: 0,
            evaluations: 0,
        })
    }

    /// Configuration used for the entire run.
    pub fn config(&self) -> &ElitistConfig {
        &self.config
    }

    /// Best accepted evaluation, absent before the first successful selection.
    pub fn best(&self) -> Option<&Evaluated<Vec<f64>, f64>> {
        self.best.as_ref()
    }

    /// Number of complete batches incorporated by successful selection.
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// Number of evaluations incorporated by successful selection.
    ///
    /// Calls made by an evaluator during a failed batch are not observable
    /// here. On successful runs this equals the actual objective-call count.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }
}

impl Algorithm for ElitistSearch {
    type Input = Vec<f64>;
    type Output = f64;
    type Error = ElitistError;

    fn generate(&mut self) -> Result<Vec<Job<Vec<f64>>>, Self::Error> {
        if self.pending.is_some() {
            return Err(ElitistError::PendingBatch);
        }
        let first_id = self.next_id.ok_or(ElitistError::IdExhausted)?;
        let count = u64::try_from(self.config.offspring).map_err(|_| ElitistError::IdExhausted)?;
        let last_id = first_id
            .checked_add(count - 1)
            .ok_or(ElitistError::IdExhausted)?;
        let mut jobs = Vec::with_capacity(self.config.offspring);
        for offset in 0..count {
            let mut input = Vec::with_capacity(self.config.dimensions);
            for coordinate in 0..self.config.dimensions {
                let value = if let Some(best) = &self.best {
                    best.job.input[coordinate]
                        + symmetric(&mut self.candidates_rng, self.config.mutation_scale)
                } else {
                    symmetric(&mut self.candidates_rng, self.config.initial_bound)
                };
                if !value.is_finite() {
                    return Err(ElitistError::NonFiniteCandidate { coordinate });
                }
                input.push(value);
            }
            jobs.push(Job {
                id: JobId(first_id + offset),
                input,
                seed: self.seeds_rng.next_u64(),
            });
        }
        // Retain issued inputs to detect accidental input/metadata changes.
        // This concrete teaching algorithm can clone Vec<f64>; the core
        // protocol imposes no Clone requirement on other implementations.
        self.pending = Some(jobs.clone());
        self.next_id = last_id.checked_add(1);
        Ok(jobs)
    }

    fn select(&mut self, results: Vec<Evaluated<Vec<f64>, f64>>) -> Result<(), Self::Error> {
        let pending = self.pending.as_ref().ok_or(ElitistError::NoPendingBatch)?;
        if results.len() != pending.len() {
            return Err(ElitistError::BatchSize {
                expected: pending.len(),
                actual: results.len(),
            });
        }
        // Validate everything before changing the incumbent or accounting.
        for (index, (issued, result)) in pending.iter().zip(&results).enumerate() {
            if issued.id != result.job.id {
                return Err(ElitistError::UnexpectedId {
                    index,
                    expected: issued.id,
                    actual: result.job.id,
                });
            }
            if issued.seed != result.job.seed
                || issued.input.len() != result.job.input.len()
                || !issued
                    .input
                    .iter()
                    .zip(&result.job.input)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Err(ElitistError::ChangedJob(issued.id));
            }
            if !result.output.is_finite() {
                return Err(ElitistError::NonFiniteOutput(issued.id));
            }
        }
        let batches = self
            .batches
            .checked_add(1)
            .ok_or(ElitistError::CountOverflow)?;
        let evaluations = self
            .evaluations
            .checked_add(results.len())
            .ok_or(ElitistError::CountOverflow)?;
        for result in results {
            if self
                .best
                .as_ref()
                .is_none_or(|best| self.config.objective.prefers(result.output, best.output))
            {
                self.best = Some(result);
            }
        }
        self.pending = None;
        self.batches = batches;
        self.evaluations = evaluations;
        Ok(())
    }
}

fn symmetric(rng: &mut ChaCha8Rng, half_width: f64) -> f64 {
    let unit = (rng.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
    let value = (2.0 * unit - 1.0) * half_width;
    // With subnormal widths, rounding can reach the exclusive upper bound.
    // The width is strictly positive, so decrementing its bits finds the
    // preceding nonnegative f64 without requiring a newer Rust toolchain.
    if value >= half_width {
        f64::from_bits(half_width.to_bits() - 1)
    } else {
        value
    }
}

/// Invalid configuration, protocol violation, or arithmetic exhaustion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ElitistError {
    /// An invalid configuration with a short diagnostic.
    InvalidConfig(&'static str),
    /// Generate was called while the previous batch awaits selection.
    PendingBatch,
    /// Select was called without a generated batch.
    NoPendingBatch,
    /// No unique ID range remains for the requested batch.
    IdExhausted,
    /// An accounting counter cannot be represented by `usize`.
    CountOverflow,
    /// Mutation overflowed a candidate coordinate.
    NonFiniteCandidate {
        /// Zero-based coordinate index.
        coordinate: usize,
    },
    /// The result batch has missing or extra evaluations.
    BatchSize {
        /// Number of issued jobs.
        expected: usize,
        /// Number of supplied evaluations.
        actual: usize,
    },
    /// An ID is duplicated, unknown, or out of original submission order.
    UnexpectedId {
        /// Zero-based result position.
        index: usize,
        /// ID issued at this position.
        expected: JobId,
        /// ID supplied at this position.
        actual: JobId,
    },
    /// An evaluated job's input or seed differs from the issued job.
    ChangedJob(JobId),
    /// The supplied objective is NaN or infinite.
    NonFiniteOutput(JobId),
}

impl fmt::Display for ElitistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(f, "invalid elitist search configuration: {message}")
            }
            Self::PendingBatch => write!(f, "select the pending batch before generating again"),
            Self::NoPendingBatch => write!(f, "no generated batch is awaiting selection"),
            Self::IdExhausted => write!(f, "evaluation job IDs exhausted"),
            Self::CountOverflow => write!(f, "search accounting counter overflowed"),
            Self::NonFiniteCandidate { coordinate } => {
                write!(f, "mutation overflowed coordinate {coordinate}")
            }
            Self::BatchSize { expected, actual } => {
                write!(f, "expected {expected} evaluations, received {actual}")
            }
            Self::UnexpectedId {
                index,
                expected,
                actual,
            } => write!(f, "result {index} has job ID {actual}; expected {expected}"),
            Self::ChangedJob(id) => write!(f, "job {id} input or seed changed during evaluation"),
            Self::NonFiniteOutput(id) => write!(f, "job {id} objective must be finite"),
        }
    }
}

impl Error for ElitistError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_id_is_issued_once_then_exhaustion_is_reported() {
        let mut config = ElitistConfig::new(1, 7);
        config.offspring = 1;
        let mut algorithm = ElitistSearch::new(config).unwrap();
        algorithm.next_id = Some(u64::MAX);
        let jobs = algorithm.generate().unwrap();
        assert_eq!(jobs[0].id, JobId(u64::MAX));
        algorithm
            .select(
                jobs.into_iter()
                    .map(|job| Evaluated { job, output: 0.0 })
                    .collect(),
            )
            .unwrap();
        assert_eq!(algorithm.generate(), Err(ElitistError::IdExhausted));
    }

    #[test]
    fn insufficient_id_range_is_rejected_before_issuing_any_jobs() {
        let mut algorithm = ElitistSearch::new(ElitistConfig::new(1, 7)).unwrap();
        algorithm.next_id = Some(u64::MAX);
        assert_eq!(algorithm.generate(), Err(ElitistError::IdExhausted));
        assert!(algorithm.pending.is_none());
    }

    #[test]
    fn rounded_subnormal_samples_keep_the_upper_bound_exclusive() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let width = f64::from_bits(1);
        for _ in 0..64 {
            let sample = symmetric(&mut rng, width);
            assert!((-width..width).contains(&sample));
        }
    }
}
