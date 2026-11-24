//! Seed-lineage reconstruction and mutation-only truncation search contracts.

use nevo::algorithms::{DeepGa, DeepGaConfig, Objective, SearchError};
use nevo::benchmarks::Sphere;
use nevo::encoding::{SeedGenome, SeededEncoding};
use nevo::{Algorithm, Evaluated, Evaluator, Job, JobId, step};

fn scored(jobs: Vec<Job<SeedGenome>>, scores: &[f64]) -> Vec<Evaluated<SeedGenome, f64>> {
    assert_eq!(jobs.len(), scores.len());
    jobs.into_iter()
        .zip(scores)
        .map(|(job, &output)| Evaluated { job, output })
        .collect()
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

    fn evaluate(&mut self, genome: &SeedGenome, _seed: u64) -> Result<f64, Self::Error> {
        self.sphere
            .value(&self.encoding.decode(genome)?)
            .map_err(|_| SearchError::Numerical("Sphere evaluation failed"))
    }
}

#[test]
fn encoding_matches_an_independent_three_coordinate_reference() {
    // Reference values calculated from ChaCha8 words and the declared uniform /
    // Box-Muller equations, without calling the encoding or shared math helpers.
    let encoding = SeededEncoding::new(3, 2.0, 0.125).unwrap();
    let mut genome = SeedGenome {
        initial_seed: 42,
        mutation_seeds: vec![],
    };
    let cases = [
        [0.7275847692266857, 1.801101630689936, -0.2899343885739212],
        [0.8459575624475333, 1.785859700694736, -0.36991628223365436],
        [0.9972935662926963, 1.6250804802206238, -0.21973301091600278],
    ];
    for (index, expected) in cases.into_iter().enumerate() {
        let actual = encoding.decode(&genome).unwrap();
        for (&got, want) in actual.iter().zip(expected) {
            assert!((got - want).abs() < 2e-14, "{got} != {want}");
        }
        if index < 2 {
            genome.mutation_seeds.push([7, 11][index]);
        }
    }
    let before = genome.clone();
    let first = encoding.decode(&genome).unwrap();
    let second = encoding.decode(&genome).unwrap();
    assert_eq!(first, second);
    assert_eq!(genome, before);
    assert_eq!(encoding.dimensions(), 3);
}

#[test]
fn zero_scale_replays_initialization_and_invalid_settings_or_overflow_fail() {
    let encoding = SeededEncoding::new(16, 0.25, 0.0).unwrap();
    let initial = SeedGenome {
        initial_seed: 42,
        mutation_seeds: vec![],
    };
    let mutated = SeedGenome {
        mutation_seeds: vec![0, 7, 11, u64::MAX],
        ..initial.clone()
    };
    let parameters = encoding.decode(&initial).unwrap();
    assert!(parameters.iter().all(|&x| (-0.25..0.25).contains(&x)));
    assert_eq!(parameters, encoding.decode(&mutated).unwrap());

    for (dimensions, bound, sigma) in [
        (0, 1.0, 0.1),
        (usize::MAX, 1.0, 0.1),
        (2, 0.0, 0.1),
        (2, -1.0, 0.1),
        (2, f64::NAN, 0.1),
        (2, f64::INFINITY, 0.1),
        (2, f64::MAX, 0.1),
        (2, 1.0, -0.1),
        (2, 1.0, f64::NAN),
        (2, 1.0, f64::INFINITY),
    ] {
        assert!(matches!(
            SeededEncoding::new(dimensions, bound, sigma),
            Err(SearchError::InvalidConfig(_))
        ));
    }
    let huge = SeededEncoding::new(3, 1.0, f64::MAX).unwrap();
    assert!(matches!(
        huge.decode(&SeedGenome {
            initial_seed: 42,
            mutation_seeds: vec![11],
        }),
        Err(SearchError::Numerical(_))
    ));
}

#[test]
fn truncation_keeps_the_best_parents_for_both_objective_directions() {
    for (objective, expected_ids) in [(Objective::Minimize, [3, 1]), (Objective::Maximize, [2, 0])]
    {
        let mut config = DeepGaConfig::new(3, 42);
        config.population = 6;
        config.parents = 2;
        config.objective = objective;
        let mut search = DeepGa::new(config.clone()).unwrap();
        assert_eq!(search.config(), &config);
        assert!(search.parents().is_empty());
        assert!(search.best().is_none());
        let jobs = search.generate().unwrap();
        assert!(jobs.iter().all(|j| j.input.mutation_seeds.is_empty()));
        search
            .select(scored(jobs, &[4.0, 1.0, 5.0, 0.0, 3.0, 2.0]))
            .unwrap();
        assert_eq!(
            search
                .parents()
                .iter()
                .map(|r| r.job.id.0)
                .collect::<Vec<_>>(),
            expected_ids
        );
        let parents = search.parents().to_vec();
        let next = search.generate().unwrap();
        assert_eq!(next[0].input, parents[0].job.input);
        assert_eq!(next[0].id, JobId(6));
        for (index, child) in next.iter().enumerate() {
            assert_eq!(child.id, JobId(6 + index as u64));
        }
        for child in &next[1..] {
            assert_eq!(child.input.mutation_seeds.len(), 1);
            assert!(
                parents
                    .iter()
                    .any(|p| p.job.input.initial_seed == child.input.initial_seed)
            );
        }
        assert_eq!(search.batches(), 1);
        assert_eq!(search.evaluations(), 6);
    }
}

#[test]
fn reevaluated_elites_do_not_bypass_current_scores_or_truncation() {
    let mut config = DeepGaConfig::new(2, 13);
    config.population = 4;
    config.parents = 1;
    config.mutation_sigma = 0.0;
    let mut search = DeepGa::new(config).unwrap();
    let initial = search.generate().unwrap();
    search
        .select(scored(initial, &[3.0, 1.0, 2.0, 0.0]))
        .unwrap();
    let historical = search.best().unwrap().clone();
    let next = search.generate().unwrap();
    let unchanged = search.encoding().decode(&historical.job.input).unwrap();
    for job in &next {
        assert_eq!(search.encoding().decode(&job.input).unwrap(), unchanged);
    }
    let expected_parent = next[1].input.clone();
    search
        .select(scored(next, &[103.0, 100.0, 101.0, 102.0]))
        .unwrap();
    assert_eq!(search.best(), Some(&historical));
    assert_eq!(search.parents()[0].job.input, expected_parent);
    assert_eq!(search.parents()[0].output, 100.0);
    assert_eq!(search.evaluations(), 8);
    let third = search.generate().unwrap();
    assert_eq!(third[0].input, expected_parent);
    assert_eq!(third[0].input.mutation_seeds.len(), 1);
    assert!(third[1..].iter().all(|j| j.input.mutation_seeds.len() == 2));
}

#[test]
fn exact_ties_keep_issue_order_and_the_earliest_best_record() {
    let mut config = DeepGaConfig::new(2, 55);
    config.population = 4;
    config.parents = 2;
    let mut search = DeepGa::new(config).unwrap();
    let jobs = search.generate().unwrap();
    search
        .select(scored(jobs, &[0.0, -0.0, 0.0, -0.0]))
        .unwrap();
    assert_eq!(search.best().unwrap().job.id, JobId(0));
    assert_eq!(search.parents()[0].job.id, JobId(0));
    assert_eq!(search.parents()[1].job.id, JobId(1));
    let jobs = search.generate().unwrap();
    search
        .select(scored(jobs, &[-0.0, 0.0, -0.0, 0.0]))
        .unwrap();
    assert_eq!(search.best().unwrap().job.id, JobId(0));
    assert_eq!(search.parents()[0].job.id, JobId(4));
}

#[test]
fn replay_preserves_every_job_history_seed_and_selected_state() {
    let config = DeepGaConfig::new(4, 1234);
    let mut a = DeepGa::new(config.clone()).unwrap();
    let mut b = DeepGa::new(config).unwrap();
    let mut evaluator = EncodedSphere::new(*a.encoding());
    for _ in 0..12 {
        let jobs_a = a.generate().unwrap();
        let jobs_b = b.generate().unwrap();
        assert_eq!(jobs_a, jobs_b);
        let results: Vec<_> = jobs_a
            .into_iter()
            .map(|job| {
                let output = evaluator.evaluate(&job.input, job.seed).unwrap();
                Evaluated { job, output }
            })
            .collect();
        a.select(results.clone()).unwrap();
        b.select(results).unwrap();
        assert_eq!(a.parents(), b.parents());
        assert_eq!(a.best(), b.best());
    }
    assert_eq!(a.batches(), 12);
    assert_eq!(a.evaluations(), 12 * 64);
}

#[test]
fn malformed_complete_batches_cannot_commit_or_release_pending_work() {
    for fault in 0..9 {
        let mut config = DeepGaConfig::new(2, 42);
        config.population = 4;
        config.parents = 2;
        let mut search = DeepGa::new(config).unwrap();
        assert_eq!(search.select(vec![]), Err(SearchError::NoPendingBatch));
        let jobs = search.generate().unwrap();
        assert_eq!(search.generate(), Err(SearchError::PendingBatch));
        let mut results = scored(jobs, &[0.0, 1.0, 2.0, 3.0]);
        match fault {
            0 => {
                results.pop();
            }
            1 => results.push(results[0].clone()),
            2 => results.swap(0, 1),
            3 => results[3].job.id = results[0].job.id,
            4 => results[3].job.seed ^= 1,
            5 => results[3].job.input.initial_seed ^= 1,
            6 => results[3].job.input.mutation_seeds.push(7),
            7 => results[3].output = f64::NAN,
            8 => results[3].output = f64::INFINITY,
            _ => unreachable!(),
        }
        assert!(matches!(
            search.select(results),
            Err(SearchError::InvalidResults(_))
        ));
        assert_eq!(search.batches(), 0);
        assert_eq!(search.evaluations(), 0);
        assert!(search.best().is_none());
        assert!(search.parents().is_empty());
        assert_eq!(search.generate(), Err(SearchError::PendingBatch));
    }
}

#[test]
fn a_late_invalid_score_leaves_existing_parents_and_best_unchanged() {
    let mut search = DeepGa::new(DeepGaConfig::new(3, 42)).unwrap();
    let mut evaluator = EncodedSphere::new(*search.encoding());
    step(&mut search, &mut evaluator).unwrap();
    let parents = search.parents().to_vec();
    let best = search.best().cloned();
    let jobs = search.generate().unwrap();
    let mut scores = vec![-100.0; 64];
    scores[63] = f64::NAN;
    assert!(search.select(scored(jobs, &scores)).is_err());
    assert_eq!(search.parents(), parents);
    assert_eq!(search.best(), best.as_ref());
    assert_eq!(search.batches(), 1);
    assert_eq!(search.evaluations(), 64);
}

#[test]
fn invalid_parent_elite_and_dimension_configurations_are_rejected() {
    let defaults = DeepGaConfig::new(4, 42);
    let mut invalid = vec![];
    for (population, parents, elite_count) in
        [(0, 0, 0), (4, 0, 0), (4, 5, 1), (4, 2, 3), (4, 4, 4)]
    {
        invalid.push(DeepGaConfig {
            population,
            parents,
            elite_count,
            ..defaults.clone()
        });
    }
    invalid.push(DeepGaConfig {
        dimensions: 0,
        ..defaults.clone()
    });
    invalid.push(DeepGaConfig {
        dimensions: usize::MAX,
        ..defaults.clone()
    });
    invalid.push(DeepGaConfig {
        initial_bound: f64::NAN,
        ..defaults.clone()
    });
    invalid.push(DeepGaConfig {
        mutation_sigma: f64::INFINITY,
        ..defaults
    });
    for config in invalid {
        assert!(matches!(
            DeepGa::new(config),
            Err(SearchError::InvalidConfig(_))
        ));
    }
}

#[test]
fn several_seeded_sphere_runs_improve_with_equal_evaluation_budgets() {
    for seed in [7, 19, 42, 91] {
        let mut config = DeepGaConfig::new(4, seed);
        config.population = 32;
        config.parents = 8;
        let mut search = DeepGa::new(config).unwrap();
        let mut evaluator = EncodedSphere::new(*search.encoding());
        step(&mut search, &mut evaluator).unwrap();
        let initial = search.best().unwrap().output;
        for _ in 1..40 {
            assert_eq!(step(&mut search, &mut evaluator).unwrap(), 32);
        }
        assert!(search.best().unwrap().output < initial, "seed {seed}");
        assert_eq!(search.batches(), 40);
        assert_eq!(search.evaluations(), 1280);
    }
}
