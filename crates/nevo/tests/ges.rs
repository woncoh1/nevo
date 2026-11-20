//! An integration fixture, not an evolutionary optimizer.

use std::error::Error;
use std::fmt;

use nevo::StepError;
use nevo::benchmarks::{Sphere, SphereError};
use nevo::prelude::*;

#[derive(Debug)]
struct FixtureError(&'static str);

impl fmt::Display for FixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl Error for FixtureError {}

struct ProbeAlgorithm {
    jobs: Option<Vec<Job<Vec<f64>>>>,
    pending_ids: Option<Vec<JobId>>,
    results: Vec<Evaluated<Vec<f64>, f64>>,
}

impl ProbeAlgorithm {
    fn new(points: Vec<Vec<f64>>) -> Self {
        Self {
            jobs: Some(
                points
                    .into_iter()
                    .enumerate()
                    .map(|(id, input)| Job {
                        id: JobId(u64::try_from(id).expect("small fixture")),
                        input,
                        seed: 42,
                    })
                    .collect(),
            ),
            pending_ids: None,
            results: Vec::new(),
        }
    }
}

impl Algorithm for ProbeAlgorithm {
    type Input = Vec<f64>;
    type Output = f64;
    type Error = FixtureError;

    fn generate(&mut self) -> Result<Vec<Job<Vec<f64>>>, FixtureError> {
        if self.pending_ids.is_some() {
            return Err(FixtureError("a batch is still pending"));
        }
        let jobs = self.jobs.take().ok_or(FixtureError("fixture exhausted"))?;
        self.pending_ids = Some(jobs.iter().map(|job| job.id).collect());
        Ok(jobs)
    }

    fn select(&mut self, results: Vec<Evaluated<Vec<f64>, f64>>) -> Result<(), FixtureError> {
        let expected = self
            .pending_ids
            .as_ref()
            .ok_or(FixtureError("no pending batch"))?;
        let received: Vec<JobId> = results.iter().map(|result| result.job.id).collect();
        if &received != expected || results.iter().any(|result| !result.output.is_finite()) {
            return Err(FixtureError("invalid result batch"));
        }
        self.results = results;
        self.pending_ids = None;
        Ok(())
    }
}

#[test]
fn sphere_runs_through_the_entire_ges_protocol() {
    let mut algorithm = ProbeAlgorithm::new(vec![vec![1.0; 8], vec![-5.0; 8], vec![0.0; 8]]);
    let mut evaluator = Sphere::new(8).unwrap();
    assert_eq!(step(&mut algorithm, &mut evaluator).unwrap(), 3);
    let values: Vec<f64> = algorithm
        .results
        .iter()
        .map(|result| result.output)
        .collect();
    assert_eq!(values, vec![8.0, 200.0, 0.0]);
}

#[test]
fn invalid_sphere_input_does_not_select_in_the_algorithm() {
    let mut algorithm = ProbeAlgorithm::new(vec![vec![1.0; 8], vec![1.0; 7]]);
    let mut evaluator = Sphere::new(8).unwrap();
    let error = step(&mut algorithm, &mut evaluator).unwrap_err();
    assert!(matches!(
        error,
        StepError::Evaluate {
            job_id: JobId(1),
            source: SphereError::DimensionMismatch {
                expected: 8,
                actual: 7,
            },
        }
    ));
    assert!(algorithm.results.is_empty());
    assert!(algorithm.pending_ids.is_some());
}

#[test]
fn fixture_rejects_a_second_generate_while_results_are_pending() {
    let mut algorithm = ProbeAlgorithm::new(vec![vec![1.0; 8]]);
    algorithm.generate().unwrap();
    assert!(algorithm.generate().is_err());
}

#[test]
fn fixture_validates_before_committing_selection() {
    let mut algorithm = ProbeAlgorithm::new(vec![vec![1.0; 8]]);
    let jobs = algorithm.generate().unwrap();
    let job = jobs.into_iter().next().unwrap();
    let invalid = vec![Evaluated {
        job,
        output: f64::NAN,
    }];
    assert!(algorithm.select(invalid).is_err());
    assert!(algorithm.results.is_empty());
    assert!(algorithm.pending_ids.is_some());
}
