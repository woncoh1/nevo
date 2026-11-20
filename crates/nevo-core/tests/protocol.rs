//! Tests for the synchronous GES contract and serial executor.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

use nevo_core::{Algorithm, Evaluated, Evaluator, Job, JobId, StepError, step};

#[derive(Debug)]
struct ProbeError(&'static str);

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl Error for ProbeError {}

#[derive(Default)]
struct ProbeAlgorithm {
    pending: Option<Vec<Job<Vec<f64>>>>,
    received: Vec<Evaluated<Vec<f64>, f64>>,
    select_calls: usize,
    fail_generate: bool,
    fail_select: bool,
}

impl Algorithm for ProbeAlgorithm {
    type Input = Vec<f64>;
    type Output = f64;
    type Error = ProbeError;

    fn generate(&mut self) -> Result<Vec<Job<Self::Input>>, Self::Error> {
        if self.fail_generate {
            return Err(ProbeError("generate"));
        }
        self.pending.take().ok_or(ProbeError("no batch"))
    }

    fn select(
        &mut self,
        results: Vec<Evaluated<Self::Input, Self::Output>>,
    ) -> Result<(), Self::Error> {
        self.select_calls += 1;
        if self.fail_select {
            return Err(ProbeError("select"));
        }
        self.received = results;
        Ok(())
    }
}

#[derive(Default)]
struct ProbeEvaluator {
    calls: Vec<(Vec<f64>, u64)>,
    fail_on_call: Option<usize>,
}

impl Evaluator<Vec<f64>> for ProbeEvaluator {
    type Output = f64;
    type Error = ProbeError;

    fn evaluate(&mut self, input: &Vec<f64>, seed: u64) -> Result<f64, Self::Error> {
        self.calls.push((input.clone(), seed));
        if self.fail_on_call == Some(self.calls.len()) {
            return Err(ProbeError("evaluate"));
        }
        Ok(input.iter().sum())
    }
}

fn job(id: u64, input: Vec<f64>, seed: u64) -> Job<Vec<f64>> {
    Job {
        id: JobId(id),
        input,
        seed,
    }
}

fn algorithm(jobs: Vec<Job<Vec<f64>>>) -> ProbeAlgorithm {
    ProbeAlgorithm {
        pending: Some(jobs),
        ..Default::default()
    }
}

#[test]
fn step_preserves_inputs_ids_seeds_and_submission_order() {
    // Deliberately use nonconsecutive, unsorted IDs.
    let jobs = vec![job(80, vec![1.0, 2.0], 42), job(3, vec![-4.0], 99)];
    let expected = jobs.clone();
    let mut algorithm = algorithm(jobs);
    let mut evaluator = ProbeEvaluator::default();

    assert_eq!(step(&mut algorithm, &mut evaluator).unwrap(), 2);
    assert_eq!(algorithm.select_calls, 1);
    assert_eq!(algorithm.received[0].job, expected[0]);
    assert_eq!(algorithm.received[1].job, expected[1]);
    assert_eq!(algorithm.received[0].output, 3.0);
    assert_eq!(algorithm.received[1].output, -4.0);
    assert_eq!(
        evaluator.calls,
        vec![(vec![1.0, 2.0], 42), (vec![-4.0], 99)]
    );
}

#[test]
fn duplicate_ids_fail_before_any_evaluation() {
    let mut algorithm = algorithm(vec![job(7, vec![1.0], 1), job(7, vec![2.0], 2)]);
    let mut evaluator = ProbeEvaluator::default();
    let error = step(&mut algorithm, &mut evaluator).unwrap_err();

    assert!(matches!(error, StepError::DuplicateJobId(JobId(7))));
    assert!(evaluator.calls.is_empty());
    assert_eq!(algorithm.select_calls, 0);
}

#[test]
fn empty_batch_is_an_error_not_termination() {
    let mut algorithm = algorithm(Vec::new());
    let mut evaluator = ProbeEvaluator::default();
    assert!(matches!(
        step(&mut algorithm, &mut evaluator),
        Err(StepError::EmptyBatch)
    ));
    assert!(evaluator.calls.is_empty());
    assert_eq!(algorithm.select_calls, 0);
}

#[test]
fn generate_error_skips_evaluation_and_select() {
    let mut algorithm = ProbeAlgorithm {
        fail_generate: true,
        ..Default::default()
    };
    let mut evaluator = ProbeEvaluator::default();
    let error = step(&mut algorithm, &mut evaluator).unwrap_err();

    assert!(matches!(&error, StepError::Generate(_)));
    assert_eq!(error.source().unwrap().to_string(), "generate");
    assert!(evaluator.calls.is_empty());
    assert_eq!(algorithm.select_calls, 0);
}

#[test]
fn evaluation_error_never_delivers_a_partial_selection() {
    let mut algorithm = algorithm(vec![
        job(10, vec![1.0], 1),
        job(11, vec![2.0], 2),
        job(12, vec![3.0], 3),
    ]);
    let mut evaluator = ProbeEvaluator {
        fail_on_call: Some(2),
        ..Default::default()
    };
    let error = step(&mut algorithm, &mut evaluator).unwrap_err();

    assert!(matches!(
        &error,
        StepError::Evaluate {
            job_id: JobId(11),
            ..
        }
    ));
    assert_eq!(error.source().unwrap().to_string(), "evaluate");
    assert_eq!(evaluator.calls.len(), 2);
    assert_eq!(algorithm.select_calls, 0);
    assert!(algorithm.received.is_empty());
}

#[test]
fn select_error_preserves_its_source() {
    let mut algorithm = algorithm(vec![job(0, vec![1.0], 42)]);
    algorithm.fail_select = true;
    let mut evaluator = ProbeEvaluator::default();
    let error = step(&mut algorithm, &mut evaluator).unwrap_err();

    assert!(matches!(&error, StepError::Select(_)));
    assert_eq!(error.source().unwrap().to_string(), "select");
    assert_eq!(algorithm.select_calls, 1);
}

#[test]
fn distinct_trials_can_share_the_same_seed() {
    let mut algorithm = algorithm(vec![job(0, vec![1.0], 42), job(1, vec![2.0], 42)]);
    let mut evaluator = ProbeEvaluator::default();
    assert_eq!(step(&mut algorithm, &mut evaluator).unwrap(), 2);
    assert_eq!(evaluator.calls[0].1, evaluator.calls[1].1);
}

#[test]
fn local_non_clone_inputs_and_structured_outputs_are_supported() {
    // Rc makes the input non-Send. Neither input nor output derives Clone.
    struct LocalInput(Rc<i32>);
    struct Report {
        reward: i32,
        descriptor: [usize; 2],
    }
    struct LocalAlgorithm {
        pending: Option<Job<LocalInput>>,
        result: Option<Evaluated<LocalInput, Report>>,
    }
    impl Algorithm for LocalAlgorithm {
        type Input = LocalInput;
        type Output = Report;
        type Error = Infallible;

        fn generate(&mut self) -> Result<Vec<Job<LocalInput>>, Infallible> {
            Ok(vec![self.pending.take().expect("one batch only")])
        }

        fn select(
            &mut self,
            results: Vec<Evaluated<LocalInput, Report>>,
        ) -> Result<(), Infallible> {
            self.result = results.into_iter().next();
            Ok(())
        }
    }
    struct LocalEvaluator;
    impl Evaluator<LocalInput> for LocalEvaluator {
        type Output = Report;
        type Error = Infallible;

        fn evaluate(&mut self, input: &LocalInput, _seed: u64) -> Result<Report, Infallible> {
            Ok(Report {
                reward: *input.0,
                descriptor: [1, 2],
            })
        }
    }

    let mut algorithm = LocalAlgorithm {
        pending: Some(Job {
            id: JobId(0),
            input: LocalInput(Rc::new(7)),
            seed: 42,
        }),
        result: None,
    };
    assert_eq!(step(&mut algorithm, &mut LocalEvaluator).unwrap(), 1);
    let result = algorithm.result.unwrap();
    assert_eq!(result.output.reward, 7);
    assert_eq!(result.output.descriptor, [1, 2]);
}

#[test]
fn job_ids_have_readable_error_messages() {
    assert_eq!(JobId(42).to_string(), "42");
    let error: StepError<ProbeError, ProbeError> = StepError::DuplicateJobId(JobId(9));
    assert_eq!(error.to_string(), "duplicate job ID in batch: 9");
    assert!(error.source().is_none());
}
