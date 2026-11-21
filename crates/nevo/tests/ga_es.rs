//! Operator, selection, protocol, and reproducibility checks for GA and ES.

use std::collections::VecDeque;

use nevo::algorithms::{Es, EsConfig, Ga, GaConfig, Objective, SearchError};
use nevo::benchmarks::Sphere;
use nevo::operators::{gaussian_mutation, tournament_selection, uniform_crossover};
use nevo::{Algorithm, Evaluated, Job, JobId, step};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};

struct ScriptedRng(VecDeque<u64>);

impl ScriptedRng {
    fn new(values: &[u64]) -> Self {
        Self(values.iter().copied().collect())
    }
}

impl RngCore for ScriptedRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn next_u64(&mut self) -> u64 {
        self.0.pop_front().expect("unexpected RNG draw")
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}

fn score(jobs: Vec<Job<Vec<f64>>>, values: &[f64]) -> Vec<Evaluated<Vec<f64>, f64>> {
    assert_eq!(jobs.len(), values.len());
    jobs.into_iter()
        .zip(values)
        .map(|(job, &output)| Evaluated { job, output })
        .collect()
}

fn ga_config(objective: Objective) -> GaConfig {
    let mut config = GaConfig::new(2, 42);
    config.population = 4;
    config.objective = objective;
    config
}

fn es_config(objective: Objective) -> EsConfig {
    let mut config = EsConfig::new(2, 42);
    config.population = 4;
    config.parents = 2;
    config.objective = objective;
    config
}

#[test]
fn tournament_returns_an_original_index_and_obeys_both_directions() {
    let population: Vec<_> = [10.0, -2.0, 1.0, 20.0]
        .into_iter()
        .enumerate()
        .map(|(i, output)| Evaluated {
            job: Job {
                id: JobId(i as u64),
                input: vec![i as f64],
                seed: 0,
            },
            output,
        })
        .collect();
    // Population length four has no rejection region: draws select 2, 1, 3.
    assert_eq!(
        tournament_selection(
            &population,
            3,
            Objective::Minimize,
            &mut ScriptedRng::new(&[2, 1, 3])
        )
        .unwrap(),
        1
    );
    assert_eq!(
        tournament_selection(
            &population,
            3,
            Objective::Maximize,
            &mut ScriptedRng::new(&[2, 1, 3])
        )
        .unwrap(),
        3
    );
    assert_eq!(population[0].job.id, JobId(0));
}

#[test]
fn uniform_crossover_inherits_the_selected_parent_coordinate_without_averaging() {
    let child = uniform_crossover(
        &[1.0, 2.0, 3.0, 4.0],
        &[10.0, 20.0, 30.0, 40.0],
        &mut ScriptedRng::new(&[0, u64::MAX, 0, u64::MAX]),
    )
    .unwrap();
    assert_eq!(child, [1.0, 20.0, 3.0, 40.0]);
}

#[test]
fn gaussian_mutation_uses_a_normal_deviate_and_can_be_disabled_without_rng_draws() {
    let mut point = [1.0, -1.0];
    // For u = 2^-53 and angle zero, Box-Muller gives sqrt(106*ln(2)).
    let expected_shift = 0.5 * (106.0 * std::f64::consts::LN_2).sqrt();
    gaussian_mutation(&mut point, 0.5, 1.0, &mut ScriptedRng::new(&[0; 6])).unwrap();
    assert!((point[0] - (1.0 + expected_shift)).abs() < 1e-14);
    assert!((point[1] - (-1.0 + expected_shift)).abs() < 1e-14);
    let unchanged = point;
    gaussian_mutation(&mut point, 0.0, 1.0, &mut ScriptedRng::new(&[])).unwrap();
    gaussian_mutation(&mut point, 1.0, 0.0, &mut ScriptedRng::new(&[])).unwrap();
    assert_eq!(point, unchanged);
}

#[test]
fn operators_reject_invalid_inputs_and_mutation_failure_keeps_the_input() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    assert!(tournament_selection(&[], 2, Objective::Minimize, &mut rng).is_err());
    assert!(uniform_crossover(&[1.0], &[1.0, 2.0], &mut rng).is_err());
    assert!(uniform_crossover(&[f64::NAN], &[1.0], &mut rng).is_err());
    let mut point = [1.0, 2.0];
    assert!(gaussian_mutation(&mut point, -1.0, 0.5, &mut rng).is_err());
    assert!(gaussian_mutation(&mut point, 1.0, f64::NAN, &mut rng).is_err());
    assert_eq!(point, [1.0, 2.0]);
    let mut large = [f64::MAX];
    assert!(matches!(
        gaussian_mutation(&mut large, f64::MAX, 1.0, &mut ScriptedRng::new(&[0; 3])),
        Err(SearchError::Numerical(_))
    ));
    assert_eq!(large, [f64::MAX]);
}

#[test]
fn same_seed_ga_and_es_have_identical_initial_jobs() {
    let mut ga = Ga::new(GaConfig::new(8, 17)).unwrap();
    let mut es = Es::new(EsConfig::new(8, 17)).unwrap();
    assert_eq!(ga.generate().unwrap(), es.generate().unwrap());
}

#[test]
fn ga_selects_the_expected_current_population_and_copies_elites_with_fresh_ids() {
    for (objective, winner) in [(Objective::Minimize, 1), (Objective::Maximize, 2)] {
        let mut ga = Ga::new(ga_config(objective)).unwrap();
        let jobs = ga.generate().unwrap();
        let expected = jobs[winner].clone();
        ga.select(score(jobs, &[3.0, -2.0, 8.0, 0.0])).unwrap();
        assert_eq!(ga.parents().len(), 4);
        assert_eq!(ga.parents()[0].job, expected);
        assert_eq!(ga.best().unwrap().job, expected);
        let next = ga.generate().unwrap();
        assert_eq!(next.len(), 4);
        assert_eq!(next[0].input, expected.input);
        assert_eq!(next[0].id, JobId(4));
        ga.select(score(next, &[10.0, 11.0, 12.0, 13.0])).unwrap();
        assert_eq!(ga.evaluations(), 8); // The copied elite was evaluated too.
        assert_eq!(ga.batches(), 2);
    }
}

#[test]
fn ga_allows_operator_ablation_and_keeps_signed_zero_ties_in_issue_order() {
    let mut config = ga_config(Objective::Minimize);
    config.elite_count = 0;
    config.crossover_probability = 0.0;
    config.mutation_sigma = 0.0;
    let mut ga = Ga::new(config).unwrap();
    let jobs = ga.generate().unwrap();
    let old = jobs.clone();
    ga.select(score(jobs, &[0.0, -0.0, 0.0, -0.0])).unwrap();
    assert_eq!(
        ga.parents().iter().map(|p| p.job.id).collect::<Vec<_>>(),
        [JobId(0), JobId(1), JobId(2), JobId(3)]
    );
    for child in ga.generate().unwrap() {
        assert!(old.iter().any(|parent| parent.input == child.input));
    }
}

#[test]
fn es_comma_replacement_forgets_even_better_old_parents_but_keeps_best_reporting() {
    for objective in [Objective::Minimize, Objective::Maximize] {
        let mut es = Es::new(es_config(objective)).unwrap();
        let direction = if objective == Objective::Minimize {
            1.0
        } else {
            -1.0
        };
        let initial = es.generate().unwrap();
        es.select(score(
            initial,
            &[0.0, direction, 2.0 * direction, 3.0 * direction],
        ))
        .unwrap();
        assert_eq!(
            es.parents()
                .iter()
                .map(|p| p.evaluation.job.id)
                .collect::<Vec<_>>(),
            [JobId(0), JobId(1)]
        );
        let offspring = es.generate().unwrap();
        es.select(score(
            offspring,
            &[
                10.0 * direction,
                11.0 * direction,
                12.0 * direction,
                13.0 * direction,
            ],
        ))
        .unwrap();
        assert_eq!(
            es.parents()
                .iter()
                .map(|p| p.evaluation.job.id)
                .collect::<Vec<_>>(),
            [JobId(4), JobId(5)]
        );
        assert_eq!(es.best().unwrap().job.id, JobId(0));
        assert_eq!(es.evaluations(), 8);
        assert_eq!(es.batches(), 2);
    }
}

#[test]
fn es_respects_explicit_sigma_bounds_and_never_drops_strategy_state() {
    let mut config = es_config(Objective::Minimize);
    config.initial_sigma = 0.2;
    config.sigma_min = 0.2;
    config.sigma_max = 0.2;
    config.tau = 1000.0;
    let mut es = Es::new(config).unwrap();
    let mut sphere = Sphere::new(2).unwrap();
    for _ in 0..5 {
        step(&mut es, &mut sphere).unwrap();
        assert_eq!(es.parents().len(), 2);
        assert!(es.parents().iter().all(|p| p.sigma == 0.2));
    }
}

fn assert_replay<A>(first: &mut A, second: &mut A)
where
    A: Algorithm<Input = Vec<f64>, Output = f64, Error = SearchError>,
{
    let sphere = Sphere::new(2).unwrap();
    for _ in 0..6 {
        let jobs = first.generate().unwrap();
        assert_eq!(jobs, second.generate().unwrap());
        let results: Vec<_> = jobs
            .into_iter()
            .map(|job| {
                let output = sphere.value(&job.input).unwrap();
                Evaluated { job, output }
            })
            .collect();
        first.select(results.clone()).unwrap();
        second.select(results).unwrap();
    }
}

#[test]
fn ga_and_es_replay_their_complete_seeded_runs() {
    let mut ga1 = Ga::new(ga_config(Objective::Minimize)).unwrap();
    let mut ga2 = Ga::new(ga_config(Objective::Minimize)).unwrap();
    assert_replay(&mut ga1, &mut ga2);
    assert_eq!(ga1.parents(), ga2.parents());
    assert_eq!(ga1.best(), ga2.best());
    let mut es1 = Es::new(es_config(Objective::Minimize)).unwrap();
    let mut es2 = Es::new(es_config(Objective::Minimize)).unwrap();
    assert_replay(&mut es1, &mut es2);
    assert_eq!(es1.parents(), es2.parents());
    assert_eq!(es1.best(), es2.best());
}

fn invalid_batches_preserve_pending<A>(mut algorithm: A)
where
    A: Algorithm<Input = Vec<f64>, Output = f64, Error = SearchError>,
{
    assert_eq!(
        algorithm.select(Vec::new()),
        Err(SearchError::NoPendingBatch)
    );
    let jobs = algorithm.generate().unwrap();
    assert_eq!(algorithm.generate(), Err(SearchError::PendingBatch));
    let values = vec![1.0; jobs.len()];
    let original = score(jobs, &values);
    for mode in 0..8 {
        let mut results = original.clone();
        let last = results.len() - 1;
        match mode {
            0 => {
                results.pop();
            }
            1 => results.push(results[0].clone()),
            2 => results[last].job.id = results[0].job.id,
            3 => results[last].job.id = JobId(999),
            4 => results.swap(0, 1),
            5 => results[last].job.seed ^= 1,
            6 => results[last].job.input.push(0.0),
            _ => results[last].output = f64::NAN,
        }
        assert!(matches!(
            algorithm.select(results),
            Err(SearchError::InvalidResults(_))
        ));
        assert_eq!(algorithm.generate(), Err(SearchError::PendingBatch));
    }
    // Inspection of one object after errors tests validation atomicity; normal
    // application code follows the core fail-stop protocol and aborts the run.
}

#[test]
fn ga_and_es_reject_complete_batch_protocol_violations() {
    invalid_batches_preserve_pending(Ga::new(ga_config(Objective::Minimize)).unwrap());
    invalid_batches_preserve_pending(Es::new(es_config(Objective::Minimize)).unwrap());
}

#[test]
fn late_invalid_results_do_not_change_ga_or_es_selected_state() {
    let mut ga = Ga::new(ga_config(Objective::Minimize)).unwrap();
    let mut es = Es::new(es_config(Objective::Minimize)).unwrap();
    let mut sphere = Sphere::new(2).unwrap();
    step(&mut ga, &mut sphere).unwrap();
    step(&mut es, &mut sphere).unwrap();
    let ga_parents = ga.parents().to_vec();
    let es_parents = es.parents().to_vec();
    let ga_best = ga.best().cloned();
    let es_best = es.best().cloned();
    let ga_results = score(
        ga.generate().unwrap(),
        &[-100.0, -90.0, -80.0, f64::INFINITY],
    );
    let es_results = score(
        es.generate().unwrap(),
        &[-100.0, -90.0, -80.0, f64::NEG_INFINITY],
    );
    assert!(ga.select(ga_results).is_err());
    assert!(es.select(es_results).is_err());
    assert_eq!(ga.parents(), ga_parents);
    assert_eq!(es.parents(), es_parents);
    assert_eq!(ga.best(), ga_best.as_ref());
    assert_eq!(es.best(), es_best.as_ref());
    assert_eq!((ga.evaluations(), es.evaluations()), (4, 4));
    assert_eq!((ga.batches(), es.batches()), (1, 1));
}

#[test]
fn invalid_ga_and_es_configurations_are_rejected_before_a_run() {
    for dimensions in [0, usize::MAX] {
        assert!(Ga::new(GaConfig::new(dimensions, 0)).is_err());
        assert!(Es::new(EsConfig::new(dimensions, 0)).is_err());
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let mut config = ga_config(Objective::Minimize);
        config.mutation_sigma = invalid;
        assert!(Ga::new(config).is_err());
        let mut config = es_config(Objective::Minimize);
        config.initial_sigma = invalid;
        assert!(Es::new(config).is_err());
    }
    let mut config = ga_config(Objective::Minimize);
    config.elite_count = config.population;
    assert!(Ga::new(config).is_err());
    let mut config = ga_config(Objective::Minimize);
    config.mutation_probability = 1.1;
    assert!(Ga::new(config).is_err());
    let mut config = ga_config(Objective::Minimize);
    config.initial_bound = f64::MAX;
    assert!(Ga::new(config).is_err());
    let mut config = es_config(Objective::Minimize);
    config.parents = config.population;
    assert!(Es::new(config).is_err());
    let mut config = es_config(Objective::Minimize);
    config.sigma_min = config.initial_sigma * 2.0;
    assert!(Es::new(config).is_err());
    let mut config = es_config(Objective::Minimize);
    config.tau = 0.0;
    assert!(Es::new(config).is_err());
}

#[test]
fn ga_and_es_improve_sphere_over_multiple_seeds_at_the_same_evaluation_budget() {
    for seed in [7, 19, 42, 91] {
        let mut ga = Ga::new(GaConfig::new(8, seed)).unwrap();
        let mut es = Es::new(EsConfig::new(8, seed)).unwrap();
        let mut sphere = Sphere::new(8).unwrap();
        step(&mut ga, &mut sphere).unwrap();
        step(&mut es, &mut sphere).unwrap();
        let initial = ga.best().unwrap().output;
        assert_eq!(initial, es.best().unwrap().output);
        for _ in 1..40 {
            step(&mut ga, &mut sphere).unwrap();
            step(&mut es, &mut sphere).unwrap();
        }
        assert!(ga.best().unwrap().output < initial);
        assert!(es.best().unwrap().output < initial);
        assert_eq!((ga.evaluations(), es.evaluations()), (2560, 2560));
    }
}
