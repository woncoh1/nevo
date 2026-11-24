//! Complete-batch bookkeeping for structured genomes.

use super::SearchError;
use crate::{Evaluated, Job, JobId};
use rand_chacha::{ChaCha8Rng, rand_core::RngCore};

#[derive(Debug)]
pub(crate) struct Batch<T> {
    pending: Option<Vec<Job<T>>>,
    next_id: Option<u64>,
    batches: usize,
    evaluations: usize,
}

impl<T: Clone + PartialEq> Batch<T> {
    pub(crate) fn new() -> Self {
        Self {
            pending: None,
            next_id: Some(0),
            batches: 0,
            evaluations: 0,
        }
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
        inputs: Vec<T>,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<Job<T>>, SearchError> {
        self.ensure_ready()?;
        if inputs.is_empty() {
            return Err(SearchError::InvalidResults("a batch must be nonempty"));
        }
        let first = self.next_id.ok_or(SearchError::IdExhausted)?;
        let count = u64::try_from(inputs.len()).map_err(|_| SearchError::IdExhausted)?;
        let last = first
            .checked_add(count - 1)
            .ok_or(SearchError::IdExhausted)?;
        let jobs: Vec<_> = inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| Job {
                id: JobId(first + index as u64),
                input,
                seed: rng.next_u64(),
            })
            .collect();
        self.pending = Some(jobs.clone());
        self.next_id = last.checked_add(1);
        Ok(jobs)
    }

    pub(crate) fn validate_scalar(&self, results: &[Evaluated<T, f64>]) -> Result<(), SearchError> {
        let jobs = self.pending.as_ref().ok_or(SearchError::NoPendingBatch)?;
        if jobs.len() != results.len() {
            return Err(SearchError::InvalidResults("missing or extra results"));
        }
        for (job, result) in jobs.iter().zip(results) {
            if job.id != result.job.id
                || job.seed != result.job.seed
                || job.input != result.job.input
            {
                return Err(SearchError::InvalidResults(
                    "job identity, input, seed, or submission order changed",
                ));
            }
            if !result.output.is_finite() {
                return Err(SearchError::InvalidResults(
                    "objective value must be finite",
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

    pub(crate) fn commit(&mut self) {
        let size = self
            .pending
            .take()
            .expect("selection validates before commit")
            .len();
        self.batches += 1;
        self.evaluations += size;
    }
    pub(crate) fn batches(&self) -> usize {
        self.batches
    }
    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations
    }
}
