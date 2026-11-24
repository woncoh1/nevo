//! Differential mutation, binomial crossover, target survival and protocol checks.

use nevo::algorithms::{De, DeConfig, Objective, SearchError};
use nevo::benchmarks::Sphere;
use nevo::{Algorithm, Evaluated, Job, JobId, step};

fn config() -> DeConfig {
    let mut config = DeConfig::new(4, 42);
    config.population = 4;
    config
}

fn score(jobs: Vec<Job<Vec<f64>>>, outputs: &[f64]) -> Vec<Evaluated<Vec<f64>, f64>> {
    assert_eq!(jobs.len(), outputs.len());
    jobs.into_iter()
        .zip(outputs)
        .map(|(job, &output)| Evaluated { job, output })
        .collect()
}

fn donor_permutations(target: usize) -> Vec<[usize; 3]> {
    let others: Vec<_> = (0..4).filter(|&index| index != target).collect();
    let [a, b, c] = [others[0], others[1], others[2]];
    vec![
        [a, b, c],
        [a, c, b],
        [b, a, c],
        [b, c, a],
        [c, a, b],
        [c, b, a],
    ]
}

fn mutant(parents: &[Job<Vec<f64>>], indices: [usize; 3], weight: f64) -> Vec<f64> {
    let [first, second, third] = indices;
    parents[first]
        .input
        .iter()
        .zip(&parents[second].input)
        .zip(&parents[third].input)
        .map(|((&a, &b), &c)| a + weight * (b - c))
        .collect()
}

#[test]
fn full_crossover_uses_one_frozen_distinct_non_target_donor_triple() {
    let mut config = config();
    config.crossover_probability = 1.0;
    config.differential_weight = 1.9;
    let mut search = De::new(config.clone()).unwrap();
    let initial = search.generate().unwrap();
    search
        .select(score(initial.clone(), &[4.0, 3.0, 2.0, 1.0]))
        .unwrap();
    let trials = search.generate().unwrap();
    for (target, trial) in trials.iter().enumerate() {
        // With four candidates, the donor triple must be a permutation of
        // exactly the three other parents. One shared triple must explain ALL
        // coordinates; arbitrary per-gene donors cannot pass this assertion.
        assert!(donor_permutations(target).into_iter().any(|indices| {
            mutant(&initial, indices, config.differential_weight) == trial.input
        }));
        assert_eq!(trial.id, JobId(4 + target as u64));
    }
    assert!(
        trials
            .iter()
            .flat_map(|job| &job.input)
            .any(|value| value.abs() > config.initial_bound)
    );
    assert_eq!(
        search.parents().iter().map(|p| &p.job).collect::<Vec<_>>(),
        initial.iter().collect::<Vec<_>>()
    );
}

#[test]
fn zero_crossover_still_forces_one_mutant_coordinate() {
    let mut config = config();
    config.crossover_probability = 0.0;
    let weight = config.differential_weight;
    let mut search = De::new(config).unwrap();
    let initial = search.generate().unwrap();
    search.select(score(initial.clone(), &[0.0; 4])).unwrap();
    for (target, trial) in search.generate().unwrap().iter().enumerate() {
        let changed: Vec<_> = initial[target]
            .input
            .iter()
            .zip(&trial.input)
            .enumerate()
            .filter_map(|(coordinate, (&a, &b))| (a.to_bits() != b.to_bits()).then_some(coordinate))
            .collect();
        assert_eq!(changed.len(), 1);
        let coordinate = changed[0];
        assert!(donor_permutations(target).into_iter().any(|indices| {
            mutant(&initial, indices, weight)[coordinate] == trial.input[coordinate]
        }));
    }
}

#[test]
fn one_dimensional_forced_crossover_does_not_depend_on_probability() {
    let mut low = config();
    low.dimensions = 1;
    low.crossover_probability = 0.0;
    let mut high = low.clone();
    high.crossover_probability = 1.0;
    let mut first = De::new(low).unwrap();
    let mut second = De::new(high).unwrap();
    let jobs = first.generate().unwrap();
    assert_eq!(jobs, second.generate().unwrap());
    first.select(score(jobs.clone(), &[0.0; 4])).unwrap();
    second.select(score(jobs, &[0.0; 4])).unwrap();
    assert_eq!(first.generate().unwrap(), second.generate().unwrap());
}

#[test]
fn trial_survival_is_one_to_one_and_strict_with_target_ties() {
    for objective in [Objective::Minimize, Objective::Maximize] {
        let sign = if objective == Objective::Minimize {
            1.0
        } else {
            -1.0
        };
        let mut config = config();
        config.objective = objective;
        let mut search = De::new(config).unwrap();
        let initial = search.generate().unwrap();
        search
            .select(score(
                initial,
                &[10.0 * sign, 20.0 * sign, 30.0 * sign, 40.0 * sign],
            ))
            .unwrap();
        let trials = search.generate().unwrap();
        search
            .select(score(
                trials,
                &[9.0 * sign, 20.0 * sign, 100.0 * sign, -sign],
            ))
            .unwrap();
        // Global truncation would choose old target 0 rather than target 2.
        // DE must instead preserve each target slot's individual competition.
        assert_eq!(
            search
                .parents()
                .iter()
                .map(|p| p.job.id)
                .collect::<Vec<_>>(),
            [JobId(4), JobId(1), JobId(2), JobId(7)]
        );
        assert_eq!(search.best().unwrap().job.id, JobId(7));
        assert_eq!(search.batches(), 2);
        assert_eq!(search.evaluations(), 8);
    }
}

#[test]
fn repeated_selection_requires_a_new_complete_batch() {
    let mut search = De::new(config()).unwrap();
    assert_eq!(search.select(Vec::new()), Err(SearchError::NoPendingBatch));
    let jobs = search.generate().unwrap();
    assert_eq!(search.generate(), Err(SearchError::PendingBatch));
    let results = score(jobs, &[0.0; 4]);
    search.select(results.clone()).unwrap();
    assert_eq!(search.select(results), Err(SearchError::NoPendingBatch));
    assert_eq!(search.evaluations(), 4);
}

#[test]
fn malformed_late_results_never_partially_replace_targets() {
    for mode in 0..10 {
        let mut search = De::new(config()).unwrap();
        let initial = search.generate().unwrap();
        search
            .select(score(initial, &[10.0, 20.0, 30.0, 40.0]))
            .unwrap();
        let before = search.parents().to_vec();
        let best = search.best().cloned();
        let mut bad = score(search.generate().unwrap(), &[-10.0, -20.0, -30.0, -40.0]);
        match mode {
            0 => {
                bad.pop();
            }
            1 => bad.push(bad[0].clone()),
            2 => bad[3].job.id = bad[0].job.id,
            3 => bad[3].job.id = JobId(999),
            4 => bad.swap(2, 3),
            5 => bad[3].job.seed ^= 1,
            6 => bad[3].job.input[0] += 0.25,
            7 => bad[3].job.input.push(0.0),
            8 => bad[3].output = f64::NAN,
            _ => bad[3].output = f64::INFINITY,
        }
        assert!(matches!(
            search.select(bad),
            Err(SearchError::InvalidResults(_))
        ));
        assert_eq!(search.parents(), before);
        assert_eq!(search.best(), best.as_ref());
        assert_eq!(search.evaluations(), 4);
        assert_eq!(search.batches(), 1);
        assert_eq!(search.generate(), Err(SearchError::PendingBatch));
    }
}

#[test]
fn invalid_configurations_are_rejected_before_sampling() {
    for dimensions in [0, usize::MAX] {
        assert!(De::new(DeConfig::new(dimensions, 0)).is_err());
    }
    for population in [0, 1, 2, 3, usize::MAX] {
        let mut config = config();
        config.population = population;
        assert!(De::new(config).is_err());
    }
    for weight in [0.0, -1.0, 2.0, f64::INFINITY, f64::NAN] {
        let mut config = config();
        config.differential_weight = weight;
        assert!(De::new(config).is_err());
    }
    for probability in [-0.01, 1.01, f64::INFINITY, f64::NAN] {
        let mut config = config();
        config.crossover_probability = probability;
        assert!(De::new(config).is_err());
    }
    for bound in [0.0, -1.0, f64::MAX, f64::INFINITY, f64::NAN] {
        let mut config = config();
        config.initial_bound = bound;
        assert!(De::new(config).is_err());
    }
}

#[test]
fn mutation_overflow_is_reported_and_does_not_replace_parents() {
    let mut config = config();
    config.dimensions = 64;
    config.initial_bound = f64::MAX * 0.4;
    config.differential_weight = 1.99;
    config.crossover_probability = 1.0;
    let mut search = De::new(config).unwrap();
    let initial = search.generate().unwrap();
    search.select(score(initial, &[0.0; 4])).unwrap();
    let before = search.parents().to_vec();
    assert!(matches!(search.generate(), Err(SearchError::Numerical(_))));
    assert_eq!(search.parents(), before);
    assert_eq!(search.evaluations(), 4);
}

#[test]
fn seeded_runs_replay_and_make_progress_on_sphere() {
    for seed in [7, 42, 91] {
        let config = DeConfig::new(4, seed);
        let mut first = De::new(config.clone()).unwrap();
        let mut second = De::new(config).unwrap();
        let mut sphere = Sphere::new(4).unwrap();
        step(&mut first, &mut sphere).unwrap();
        step(&mut second, &mut sphere).unwrap();
        let initial = first.best().unwrap().output;
        for _ in 1..60 {
            let jobs = first.generate().unwrap();
            assert_eq!(jobs, second.generate().unwrap());
            let results: Vec<_> = jobs
                .into_iter()
                .map(|job| Evaluated {
                    output: sphere.value(&job.input).unwrap(),
                    job,
                })
                .collect();
            first.select(results.clone()).unwrap();
            second.select(results).unwrap();
        }
        assert_eq!(first.parents(), second.parents());
        assert_eq!(first.best(), second.best());
        assert!(first.best().unwrap().output < initial * 0.1);
        assert_eq!(first.evaluations(), 3840);
        assert_eq!(first.batches(), 60);
    }
}
