//! Backend-independent Generate -> Evaluate -> Select interfaces.
//!
//! This crate defines a protocol, not a particular evolutionary algorithm.
//! Inputs may be genomes, agent-environment pairs, or other evaluation tasks.
//! Outputs may be scalars, objective vectors, descriptors, or custom records.
//!
//! The initial protocol is synchronous: one complete, nonempty batch at a time.
//! See [`Algorithm`] and [`step`] for lifecycle and failure semantics.

mod runner;

pub use runner::{StepError, step};

/// Identity of one evaluation job within a run.
///
/// Algorithms assign IDs and must not reuse them during a run. Repeated trials
/// of the same genome receive distinct IDs, even when their seeds are equal.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct JobId(pub u64);

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An owned input plus explicit evaluation identity and randomness.
///
/// `seed` belongs to this evaluation, not to the algorithm's mutation RNG.
/// Evaluators may deliberately share seeds between jobs for matched trials.
/// No `Clone`, `Send`, `Sync`, or serialization bound is required by the core.
///
/// ```
/// use nevo_core::{Job, JobId};
///
/// let job = Job {
///     id: JobId(0),
///     input: vec![1.0, -2.0],
///     seed: 42,
/// };
/// assert_eq!(job.id, JobId(0));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Job<T> {
    /// Run-local identity assigned before execution.
    pub id: JobId,
    /// Data or handles needed for evaluation.
    pub input: T,
    /// Explicit seed for this evaluation.
    pub seed: u64,
}

/// A job returned with its typed evaluation output.
///
/// Returning ownership lets an algorithm retain selected inputs without
/// requiring the executor to clone genomes or simulation configurations.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluated<T, O> {
    /// The original job, unchanged by the serial executor.
    pub job: Job<T>,
    /// An objective value or a user-defined evaluation record.
    pub output: O,
}

/// A stateful producer and consumer of evaluation work.
///
/// # Protocol
///
/// 1. `generate` issues one nonempty batch with run-unique IDs.
/// 2. The caller evaluates every job without altering its input or metadata.
/// 3. `select` receives exactly that batch in its original submission order.
/// 4. Only after successful selection may the caller generate another batch.
///
/// `select` is the broad state-transition stage: survivor selection, archive
/// insertion, distribution adaptation, and agent transfer all belong here when
/// appropriate. It is not restricted to choosing a subset of individuals.
///
/// Algorithms own their state and their generation RNG. Selection, mutation,
/// archive insertion, distribution updates, and transfer belong inside the
/// relevant algorithm or its composed operators, not in the executor.
///
/// # Failure
///
/// On any error, this minimal protocol stops the run. It does not promise
/// rollback, partial-result updates, retries, or asynchronous completion.
/// Restore or reconstruct the run before restarting. Implementations should
/// validate a complete result batch before committing an update.
pub trait Algorithm {
    /// Evaluation input; not necessarily a single genome.
    type Input;
    /// The result schema required by this algorithm.
    type Output;
    /// Configuration, protocol, or algorithm-specific failure.
    type Error: std::error::Error + 'static;

    /// Produce the next batch of evaluation jobs.
    ///
    /// # Errors
    /// Returns an implementation-specific error if no batch can be issued.
    fn generate(&mut self) -> Result<Vec<Job<Self::Input>>, Self::Error>;

    /// Select the next search state from one complete evaluated batch.
    ///
    /// This includes any algorithm-specific updates, not only survival.
    ///
    /// # Errors
    /// Returns an implementation-specific error for invalid results or a
    /// failed update. Callers must treat an error as terminal for this run.
    fn select(
        &mut self,
        results: Vec<Evaluated<Self::Input, Self::Output>>,
    ) -> Result<(), Self::Error>;
}

/// Evaluate an input in a problem-specific context.
///
/// The mutable receiver permits reusable scratch buffers and simulation state.
/// For reproducibility, implementations must reset task-dependent state or
/// explicitly include persistent state in their evaluation protocol.
/// `T: ?Sized` permits borrowed slices as well as owned input types.
///
/// A future parallel executor will impose its own thread-safety bounds;
/// this trait deliberately does not require them.
pub trait Evaluator<T: ?Sized> {
    /// Scalar or structured information produced by evaluation.
    type Output;
    /// Invalid-input or evaluation-specific failure.
    type Error: std::error::Error + 'static;

    /// Evaluate `input` using an explicit evaluation seed.
    ///
    /// Deterministic evaluators may ignore the seed. Stochastic evaluators
    /// should use it instead of hidden thread-local or global randomness.
    ///
    /// # Errors
    /// Returns an implementation-specific error rather than silently
    /// converting invalid evaluations into favorable or unfavorable scores.
    fn evaluate(&mut self, input: &T, seed: u64) -> Result<Self::Output, Self::Error>;
}
