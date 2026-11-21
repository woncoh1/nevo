//! Internal numerical helpers and the shared complete-batch protocol.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};

use super::Objective;
use crate::{Evaluated, Job, JobId};

/// A configuration, protocol, or numerical failure in a foundational search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SearchError {
    /// A configuration parameter is invalid.
    InvalidConfig(&'static str),
    /// A previously generated batch is still outstanding.
    PendingBatch,
    /// Selection was called without an issued batch.
    NoPendingBatch,
    /// The complete evaluated batch failed identity, shape, or value checks.
    InvalidResults(&'static str),
    /// Arithmetic became invalid or unrepresentable.
    Numerical(&'static str),
    /// Unique evaluation identifiers have been exhausted.
    IdExhausted,
    /// The batch or evaluation counter would overflow.
    CountOverflow,
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid search configuration: {message}"),
            Self::PendingBatch => write!(f, "select the pending batch before generating again"),
            Self::NoPendingBatch => write!(f, "no generated batch is awaiting selection"),
            Self::InvalidResults(message) => write!(f, "invalid evaluation batch: {message}"),
            Self::Numerical(message) => write!(f, "search numerical failure: {message}"),
            Self::IdExhausted => write!(f, "evaluation job IDs exhausted"),
            Self::CountOverflow => write!(f, "search counter overflowed"),
        }
    }
}

impl Error for SearchError {}

#[derive(Debug)]
pub(crate) struct BatchState {
    next_id: Option<u64>,
    pending: Option<Vec<Job<Vec<f64>>>>,
    batches: usize,
    evaluations: usize,
}

impl Default for BatchState {
    fn default() -> Self {
        Self {
            next_id: Some(0),
            pending: None,
            batches: 0,
            evaluations: 0,
        }
    }
}

impl BatchState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn ensure_ready(&self) -> Result<(), SearchError> {
        if self.pending.is_some() {
            Err(SearchError::PendingBatch)
        } else {
            Ok(())
        }
    }

    pub(crate) fn issue(
        &mut self,
        inputs: Vec<Vec<f64>>,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<Job<Vec<f64>>>, SearchError> {
        self.ensure_ready()?;
        let seeds = (0..inputs.len()).map(|_| rng.next_u64()).collect();
        self.issue_with_seeds(inputs, seeds)
    }

    pub(crate) fn issue_with_seeds(
        &mut self,
        inputs: Vec<Vec<f64>>,
        seeds: Vec<u64>,
    ) -> Result<Vec<Job<Vec<f64>>>, SearchError> {
        self.ensure_ready()?;
        if inputs.is_empty() || inputs.len() != seeds.len() {
            return Err(SearchError::InvalidResults(
                "nonempty inputs and matching seeds required",
            ));
        }
        if inputs
            .iter()
            .any(|v| v.is_empty() || v.iter().any(|x| !x.is_finite()))
        {
            return Err(SearchError::Numerical(
                "generated candidates must be finite and nonempty",
            ));
        }
        let first = self.next_id.ok_or(SearchError::IdExhausted)?;
        let count = u64::try_from(inputs.len()).map_err(|_| SearchError::IdExhausted)?;
        let last = first
            .checked_add(count - 1)
            .ok_or(SearchError::IdExhausted)?;
        let jobs: Vec<_> = inputs
            .into_iter()
            .zip(seeds)
            .enumerate()
            .map(|(index, (input, seed))| Job {
                id: JobId(first + index as u64),
                input,
                seed,
            })
            .collect();
        // Concrete numerical algorithms validate returned genes as well as IDs.
        // This small reference implementation deliberately keeps an issued copy.
        self.pending = Some(jobs.clone());
        self.next_id = last.checked_add(1);
        Ok(jobs)
    }

    pub(crate) fn validate<T>(
        &self,
        results: &[Evaluated<Vec<f64>, T>],
    ) -> Result<(), SearchError> {
        let issued = self.pending.as_ref().ok_or(SearchError::NoPendingBatch)?;
        if issued.len() != results.len() {
            return Err(SearchError::InvalidResults("missing or extra results"));
        }
        for (job, result) in issued.iter().zip(results) {
            if job.id != result.job.id {
                return Err(SearchError::InvalidResults(
                    "duplicate, unknown, or reordered job ID",
                ));
            }
            if job.seed != result.job.seed
                || job.input.len() != result.job.input.len()
                || !job
                    .input
                    .iter()
                    .zip(&result.job.input)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Err(SearchError::InvalidResults(
                    "input or evaluation seed changed",
                ));
            }
        }
        self.batches
            .checked_add(1)
            .ok_or(SearchError::CountOverflow)?;
        self.evaluations
            .checked_add(results.len())
            .ok_or(SearchError::CountOverflow)?;
        Ok(())
    }

    pub(crate) fn validate_scalar(
        &self,
        results: &[Evaluated<Vec<f64>, f64>],
    ) -> Result<(), SearchError> {
        self.validate(results)?;
        if results.iter().any(|r| !r.output.is_finite()) {
            return Err(SearchError::InvalidResults(
                "objective values must be finite",
            ));
        }
        Ok(())
    }

    pub(crate) fn commit(&mut self) {
        let count = self
            .pending
            .take()
            .expect("selection must validate its pending batch")
            .len();
        self.batches += 1;
        self.evaluations += count;
    }

    pub(crate) fn batches(&self) -> usize {
        self.batches
    }
    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations
    }
}

pub(crate) fn seeded_rng(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(stream);
    rng
}

pub(crate) fn uniform01(rng: &mut impl RngCore) -> f64 {
    (rng.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
}

pub(crate) fn symmetric(rng: &mut impl RngCore, bound: f64) -> f64 {
    let value = (2.0 * uniform01(rng) - 1.0) * bound;
    if value >= bound {
        f64::from_bits(bound.to_bits() - 1)
    } else {
        value
    }
}

pub(crate) fn normal(rng: &mut impl RngCore) -> f64 {
    // Uncached Box-Muller: u is in (0,1], v in [0,1). Always two u64 draws.
    let u = ((rng.next_u64() >> 11) as f64 + 1.0) / 9_007_199_254_740_992.0;
    let v = uniform01(rng);
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

pub(crate) fn uniform_index(upper: usize, rng: &mut impl RngCore) -> usize {
    assert!(upper > 0, "sampling requires a nonempty range");
    let n = upper as u64;
    let threshold = n.wrapping_neg() % n;
    loop {
        let value = rng.next_u64();
        if value >= threshold {
            return (value % n) as usize;
        }
    }
}

pub(crate) fn score_cmp(a: f64, b: f64, objective: Objective) -> Ordering {
    let order = a
        .partial_cmp(&b)
        .expect("objectives must be validated as finite");
    match objective {
        Objective::Minimize => order,
        Objective::Maximize => order.reverse(),
    }
}

pub(crate) fn update_best(
    best: &mut Option<Evaluated<Vec<f64>, f64>>,
    results: &[Evaluated<Vec<f64>, f64>],
    objective: Objective,
) {
    for result in results {
        if best
            .as_ref()
            .is_none_or(|incumbent| score_cmp(result.output, incumbent.output, objective).is_lt())
        {
            *best = Some(result.clone());
        }
    }
}
