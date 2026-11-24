//! Stable parallel evaluation, failure semantics, and algorithm replay.
#![cfg(feature = "parallel")]

use std::cell::Cell;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nevo::algorithms::{DeepGa, DeepGaConfig, SearchError};
use nevo::benchmarks::Sphere;
use nevo::encoding::{SeedGenome, SeededEncoding};
use nevo::execution::step_parallel;
use nevo::{Algorithm, Evaluated, Evaluator, Job, JobId, StepError, step};

// A job input is Send, but deliberately neither Clone nor Sync.
#[derive(Debug)]
struct Payload(Cell<u64>);

// Algorithm errors remain on the calling thread and need not implement Send.
#[derive(Debug)]
struct LocalError {
    message: &'static str,
    _not_send: Rc<()>,
}

impl LocalError {
    fn new(message: &'static str) -> Self {
        Self {
            message,
            _not_send: Rc::new(()),
        }
    }
}

impl fmt::Display for LocalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}
impl Error for LocalError {}

#[derive(Debug)]
struct EvaluationError(u64);
impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rejected {}", self.0)
    }
}
impl Error for EvaluationError {}

struct RecordingAlgorithm {
    jobs: Option<Vec<Job<Payload>>>,
    selected: Vec<Evaluated<Payload, u64>>,
    selections: usize,
    fail_generate: bool,
    fail_select: bool,
    _not_send: Rc<()>,
}

impl RecordingAlgorithm {
    fn new(ids: &[u64]) -> Self {
        Self {
            jobs: Some(
                ids.iter()
                    .enumerate()
                    .map(|(index, &id)| Job {
                        id: JobId(id),
                        input: Payload(Cell::new(index as u64 + 10)),
                        seed: 9000 + index as u64,
                    })
                    .collect(),
            ),
            selected: vec![],
            selections: 0,
            fail_generate: false,
            fail_select: false,
            _not_send: Rc::new(()),
        }
    }

    fn records(&self) -> Vec<(u64, u64, u64, u64)> {
        self.selected
            .iter()
            .map(|r| (r.job.id.0, r.job.input.0.get(), r.job.seed, r.output))
            .collect()
    }
}

impl Algorithm for RecordingAlgorithm {
    type Input = Payload;
    type Output = u64;
    type Error = LocalError;

    fn generate(&mut self) -> Result<Vec<Job<Self::Input>>, Self::Error> {
        if self.fail_generate {
            return Err(LocalError::new("generate"));
        }
        self.jobs
            .take()
            .ok_or_else(|| LocalError::new("already generated"))
    }

    fn select(
        &mut self,
        results: Vec<Evaluated<Self::Input, Self::Output>>,
    ) -> Result<(), Self::Error> {
        self.selections += 1;
        if self.fail_select {
            return Err(LocalError::new("select"));
        }
        self.selected = results;
        Ok(())
    }
}

// Construction happens within each worker; this evaluator intentionally is !Send.
struct IdentityEvaluator {
    _local: Rc<()>,
}
impl IdentityEvaluator {
    fn new() -> Self {
        Self {
            _local: Rc::new(()),
        }
    }
}
impl Evaluator<Payload> for IdentityEvaluator {
    type Output = u64;
    type Error = EvaluationError;
    fn evaluate(&mut self, input: &Payload, seed: u64) -> Result<u64, Self::Error> {
        Ok(input.0.get().rotate_left(7) ^ seed)
    }
}

#[test]
fn parallel_results_preserve_serial_job_identity_input_seed_and_issue_order() {
    let ids = [90, 2, 17, 4, 100, 1];
    let mut serial = RecordingAlgorithm::new(&ids);
    assert_eq!(
        step(&mut serial, &mut IdentityEvaluator::new()).unwrap(),
        ids.len()
    );
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        // Construct the deliberately !Send algorithm inside the install closure.
        let records = pool.install(|| {
            let mut parallel = RecordingAlgorithm::new(&ids);
            assert_eq!(
                step_parallel(&mut parallel, IdentityEvaluator::new).unwrap(),
                ids.len()
            );
            assert_eq!(parallel.selections, 1);
            parallel.records()
        });
        assert_eq!(records, serial.records());
    }
}

#[test]
fn every_job_gets_a_fresh_evaluator_even_on_a_single_worker() {
    struct StatefulEvaluator {
        calls: u64,
    }
    impl Evaluator<Payload> for StatefulEvaluator {
        type Output = u64;
        type Error = EvaluationError;
        fn evaluate(&mut self, _: &Payload, _: u64) -> Result<u64, Self::Error> {
            self.calls += 1;
            Ok(self.calls)
        }
    }
    let constructions = AtomicUsize::new(0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let mut search = RecordingAlgorithm::new(&[8, 6, 4, 2]);
        step_parallel(&mut search, || {
            constructions.fetch_add(1, Ordering::Relaxed);
            StatefulEvaluator { calls: 0 }
        })
        .unwrap();
        assert!(search.selected.iter().all(|r| r.output == 1));
    });
    assert_eq!(constructions.load(Ordering::Relaxed), 4);
}

#[test]
fn evaluation_failure_reports_lowest_submission_index_and_never_selects() {
    struct FailingEvaluator<'a>(&'a AtomicUsize);
    impl Evaluator<Payload> for FailingEvaluator<'_> {
        type Output = u64;
        type Error = EvaluationError;
        fn evaluate(&mut self, input: &Payload, _: u64) -> Result<u64, Self::Error> {
            self.0.fetch_add(1, Ordering::Relaxed);
            let value = input.0.get();
            if value == 10 || value == 12 {
                Err(EvaluationError(value))
            } else {
                Ok(value)
            }
        }
    }
    for threads in [1, 4] {
        let calls = AtomicUsize::new(0);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut search = RecordingAlgorithm::new(&[90, 7, 1, 2]);
            match step_parallel(&mut search, || FailingEvaluator(&calls)) {
                Err(StepError::Evaluate { job_id, source }) => {
                    assert_eq!(job_id, JobId(90));
                    assert_eq!(source.0, 10);
                }
                result => panic!("unexpected result: {result:?}"),
            }
            assert_eq!(search.selections, 0);
            assert!(search.selected.is_empty());
        });
        assert_eq!(calls.load(Ordering::Relaxed), 4);
    }
}

#[test]
fn empty_and_duplicate_batches_fail_before_constructing_evaluators() {
    for ids in [vec![], vec![7, 3, 7]] {
        let constructions = AtomicUsize::new(0);
        let mut search = RecordingAlgorithm::new(&ids);
        let result = step_parallel(&mut search, || {
            constructions.fetch_add(1, Ordering::Relaxed);
            IdentityEvaluator::new()
        });
        if ids.is_empty() {
            assert!(matches!(result, Err(StepError::EmptyBatch)));
        } else {
            assert!(matches!(result, Err(StepError::DuplicateJobId(JobId(7)))));
        }
        assert_eq!(constructions.load(Ordering::Relaxed), 0);
        assert_eq!(search.selections, 0);
    }
}

#[test]
fn generation_and_selection_errors_keep_their_correct_phase() {
    let mut search = RecordingAlgorithm::new(&[4]);
    search.fail_generate = true;
    assert!(matches!(
        step_parallel(&mut search, IdentityEvaluator::new),
        Err(StepError::Generate(_))
    ));
    assert_eq!(search.selections, 0);

    let mut search = RecordingAlgorithm::new(&[4]);
    search.fail_select = true;
    assert!(matches!(
        step_parallel(&mut search, IdentityEvaluator::new),
        Err(StepError::Select(_))
    ));
    assert_eq!(search.selections, 1);
}

struct EncodedSphere {
    encoding: SeededEncoding,
    sphere: Sphere,
}
impl EncodedSphere {
    fn new(encoding: SeededEncoding) -> Self {
        Self {
            sphere: Sphere::new(encoding.dimensions()).unwrap(),
            encoding,
        }
    }
}
impl Evaluator<SeedGenome> for EncodedSphere {
    type Output = f64;
    type Error = SearchError;
    fn evaluate(&mut self, input: &SeedGenome, _: u64) -> Result<f64, Self::Error> {
        self.sphere
            .value(&self.encoding.decode(input)?)
            .map_err(|_| SearchError::Numerical("Sphere failed"))
    }
}

#[test]
fn deep_ga_replays_identically_serially_and_with_one_or_four_workers() {
    let mut config = DeepGaConfig::new(5, 42);
    config.population = 32;
    config.parents = 8;
    let mut serial = DeepGa::new(config.clone()).unwrap();
    let mut evaluator = EncodedSphere::new(*serial.encoding());
    for _ in 0..30 {
        step(&mut serial, &mut evaluator).unwrap();
    }

    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut parallel = DeepGa::new(config.clone()).unwrap();
        let encoding = *parallel.encoding();
        for _ in 0..30 {
            pool.install(|| step_parallel(&mut parallel, || EncodedSphere::new(encoding)))
                .unwrap();
        }
        assert_eq!(parallel.parents(), serial.parents());
        assert_eq!(parallel.best(), serial.best());
        assert_eq!(parallel.batches(), 30);
        assert_eq!(parallel.evaluations(), 960);
        let serial_next = serial
            .encoding()
            .decode(&serial.best().unwrap().job.input)
            .unwrap();
        let parallel_next = parallel
            .encoding()
            .decode(&parallel.best().unwrap().job.input)
            .unwrap();
        assert_eq!(parallel_next, serial_next);
    }
}
