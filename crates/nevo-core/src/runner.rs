use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use crate::{Algorithm, Evaluated, Evaluator, JobId};

/// A failure in one serial Generate -> Evaluate -> Select step.
#[derive(Debug)]
pub enum StepError<A, E> {
    /// The algorithm could not generate a batch.
    Generate(A),
    /// Empty batches are not termination signals in this protocol.
    EmptyBatch,
    /// Two jobs in the generated batch have the same identity.
    DuplicateJobId(JobId),
    /// A specific job could not be evaluated.
    Evaluate {
        /// Identity of the failed evaluation.
        job_id: JobId,
        /// The evaluator's original error.
        source: E,
    },
    /// The algorithm rejected results or failed to select its next state.
    Select(A),
}

impl<A: fmt::Display, E: fmt::Display> fmt::Display for StepError<A, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generate(source) => write!(f, "generation failed: {source}"),
            Self::EmptyBatch => write!(f, "algorithm generated an empty batch"),
            Self::DuplicateJobId(id) => write!(f, "duplicate job ID in batch: {id}"),
            Self::Evaluate { job_id, source } => {
                write!(f, "evaluation of job {job_id} failed: {source}")
            }
            Self::Select(source) => write!(f, "algorithm selection failed: {source}"),
        }
    }
}

impl<A, E> Error for StepError<A, E>
where
    A: Error + 'static,
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Generate(source) | Self::Select(source) => Some(source),
            Self::Evaluate { source, .. } => Some(source),
            Self::EmptyBatch | Self::DuplicateJobId(_) => None,
        }
    }
}

/// Execute one complete batch serially and return its evaluation count.
///
/// The caller owns stopping criteria and decides how often to call `step`.
/// Results are delivered in generation order, not sorted by ID or fitness.
/// IDs need not be consecutive. This helper rejects duplicate IDs *within*
/// a batch; algorithms remain responsible for uniqueness across batches.
///
/// # Errors
///
/// Empty or duplicate-ID batches fail before evaluation starts. The first
/// evaluation failure aborts the step: no partial batch reaches `select`.
/// Earlier evaluator side effects are not undone, and `generate` may already
/// have advanced algorithm state. Do not retry `step` on a failed run without
/// restoring or reconstructing it. Panics are not caught.
pub fn step<A, E>(
    algorithm: &mut A,
    evaluator: &mut E,
) -> Result<usize, StepError<A::Error, E::Error>>
where
    A: Algorithm,
    E: Evaluator<A::Input, Output = A::Output>,
{
    let jobs = algorithm.generate().map_err(StepError::Generate)?;
    if jobs.is_empty() {
        return Err(StepError::EmptyBatch);
    }

    let mut seen = BTreeSet::new();
    for job in &jobs {
        if !seen.insert(job.id) {
            return Err(StepError::DuplicateJobId(job.id));
        }
    }

    let count = jobs.len();
    let mut results = Vec::with_capacity(count);
    for job in jobs {
        let output =
            evaluator
                .evaluate(&job.input, job.seed)
                .map_err(|source| StepError::Evaluate {
                    job_id: job.id,
                    source,
                })?;
        results.push(Evaluated { job, output });
    }

    algorithm.select(results).map_err(StepError::Select)?;
    Ok(count)
}
