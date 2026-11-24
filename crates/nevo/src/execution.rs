//! Optional parallel evaluation while preserving the Generate → Evaluate → Select contract.
//!
//! Enable the `parallel` Cargo feature. Algorithms still generate and select on
//! the calling thread. Rayon distributes independent evaluations over its current
//! thread pool; no performance or bitwise cross-platform guarantee is implied.

use std::collections::BTreeSet;

use rayon::prelude::*;

use crate::{Algorithm, Evaluated, Evaluator, StepError};

/// Evaluate one complete generated batch in parallel and select once on success.
///
/// `factory` constructs a **fresh evaluator for each job**, avoiding accidental
/// state reuse between candidates. It may capture immutable configuration or
/// thread-safe shared data. Each evaluator receives the original input and
/// evaluation seed; results reach `select` in submission order, regardless of
/// completion order. Only transported inputs, outputs, and evaluator errors need
/// `Send`; the algorithm and locally constructed evaluator do not.
///
/// The caller controls the pool, for example with
/// `rayon::ThreadPoolBuilder::new().num_threads(4).build()?.install(|| ...)`.
/// Reproducibility requires evaluators whose results depend on their explicit
/// inputs/seeds, not scheduling or mutable shared state.
///
/// # Errors
/// Empty or duplicate-ID batches fail before any evaluator is constructed.
/// Evaluations are collected without short-circuiting: other jobs may execute
/// even if one fails. If several fail, return the error of the lowest submission
/// index, not the smallest job ID or first completion. No partial results reach
/// `select`, and evaluator side effects are not undone. As with [`crate::step`],
/// any error terminates the run; restoring/reconstructing it is the caller's job.
/// Factory/evaluator panics are not caught and propagate through Rayon.
pub fn step_parallel<A, E, F>(
    algorithm: &mut A,
    factory: F,
) -> Result<usize, StepError<A::Error, E::Error>>
where
    A: Algorithm,
    A::Input: Send,
    A::Output: Send,
    E: Evaluator<A::Input, Output = A::Output>,
    E::Error: Send,
    F: Fn() -> E + Send + Sync,
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
    // Collect Vec<Result<...>>, not Result<Vec<...>>: preserve submission order
    // and choose a deterministic error after all normal job returns are available.
    let outcomes: Vec<_> = jobs
        .into_par_iter()
        .map(|job| {
            let mut evaluator = factory();
            match evaluator.evaluate(&job.input, job.seed) {
                Ok(output) => Ok(Evaluated { job, output }),
                Err(source) => Err((job.id, source)),
            }
        })
        .collect();
    let mut results = Vec::with_capacity(count);
    for outcome in outcomes {
        results.push(outcome.map_err(|(job_id, source)| StepError::Evaluate { job_id, source })?);
    }
    algorithm.select(results).map_err(StepError::Select)?;
    Ok(count)
}
