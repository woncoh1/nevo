//! Correctness and reproducibility of the uniform-mutation teaching search.

use nevo::algorithms::{ElitistConfig, ElitistError, ElitistSearch, Objective};
use nevo::benchmarks::Sphere;
use nevo::{Algorithm, Evaluated, Evaluator, Job, JobId, step};

fn small_search(objective: Objective) -> ElitistSearch {
    let mut config = ElitistConfig::new(3, 42);
    config.offspring = 4;
    config.objective = objective;
    ElitistSearch::new(config).unwrap()
}

fn scored(jobs: Vec<Job<Vec<f64>>>, scores: &[f64]) -> Vec<Evaluated<Vec<f64>, f64>> {
    assert_eq!(jobs.len(), scores.len());
    jobs.into_iter()
        .zip(scores)
        .map(|(job, &output)| Evaluated { job, output })
        .collect()
}

#[test]
fn invalid_population_and_numeric_ranges_are_rejected() {
    for (dimensions, offspring) in [(0, 4), (3, 0)] {
        let mut config = ElitistConfig::new(dimensions, 42);
        config.offspring = offspring;
        assert!(matches!(
            ElitistSearch::new(config),
            Err(ElitistError::InvalidConfig(_))
        ));
    }
    for invalid in [
        0.0,
        -0.1,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
    ] {
        let mut config = ElitistConfig::new(3, 42);
        config.initial_bound = invalid;
        assert!(matches!(
            ElitistSearch::new(config),
            Err(ElitistError::InvalidConfig(_))
        ));
        let mut config = ElitistConfig::new(3, 42);
        config.mutation_scale = invalid;
        assert!(matches!(
            ElitistSearch::new(config),
            Err(ElitistError::InvalidConfig(_))
        ));
    }
}

#[test]
fn minimization_and_maximization_retain_the_correct_unchanged_candidate() {
    for (objective, winner) in [(Objective::Minimize, 1), (Objective::Maximize, 2)] {
        let mut search = small_search(objective);
        let jobs = search.generate().unwrap();
        let expected = jobs[winner].clone();
        search.select(scored(jobs, &[3.0, -2.0, 8.0, 0.0])).unwrap();
        assert_eq!(search.best().unwrap().job, expected);
        assert_eq!(search.batches(), 1);
        assert_eq!(search.evaluations(), 4);
    }
}

#[test]
fn equal_scores_keep_earliest_initial_candidate_then_the_incumbent() {
    let mut search = small_search(Objective::Minimize);
    let jobs = search.generate().unwrap();
    let expected = jobs[0].clone();
    search.select(scored(jobs, &[1.0; 4])).unwrap();
    assert_eq!(search.best().unwrap().job, expected);
    let jobs = search.generate().unwrap();
    search.select(scored(jobs, &[1.0; 4])).unwrap();
    assert_eq!(search.best().unwrap().job, expected);
    assert_eq!(search.evaluations(), 8);
}

#[test]
fn offspring_mutate_the_selected_parent_with_fresh_ids_and_bounded_offsets() {
    let mut search = small_search(Objective::Minimize);
    let jobs = search.generate().unwrap();
    for job in &jobs {
        assert_eq!(job.input.len(), 3);
        assert!(job.input.iter().all(|&x| (-1.0..1.0).contains(&x)));
    }
    let parent = jobs[2].input.clone();
    search.select(scored(jobs, &[3.0, 2.0, 1.0, 4.0])).unwrap();
    let offspring = search.generate().unwrap();
    for (index, job) in offspring.iter().enumerate() {
        assert_eq!(job.id, JobId(4 + index as u64));
        for (coordinate, original) in job.input.iter().zip(&parent) {
            // A tolerance permits the rounding from adding the perturbation.
            assert!((coordinate - original).abs() <= 0.2 + f64::EPSILON);
        }
    }
    assert_eq!(search.batches(), 1);
    assert_eq!(search.evaluations(), 4);
}

#[test]
fn a_complete_seeded_run_replays_candidates_seeds_results_and_accounting() {
    let mut first = small_search(Objective::Minimize);
    let mut second = small_search(Objective::Minimize);
    let mut evaluator = Sphere::new(3).unwrap();
    for batch in 0..5 {
        let jobs = first.generate().unwrap();
        assert_eq!(jobs, second.generate().unwrap());
        let results: Vec<_> = jobs
            .into_iter()
            .map(|job| {
                let output = evaluator.evaluate(&job.input, job.seed).unwrap();
                Evaluated { job, output }
            })
            .collect();
        first.select(results.clone()).unwrap();
        second.select(results).unwrap();
        assert_eq!(first.best(), second.best());
        assert_eq!(first.evaluations(), 4 * (batch + 1));
    }
}

#[test]
fn evaluation_seed_stream_is_independent_of_candidate_dimension() {
    let mut first = ElitistSearch::new(ElitistConfig::new(1, 19)).unwrap();
    let mut second = ElitistSearch::new(ElitistConfig::new(8, 19)).unwrap();
    let first = first.generate().unwrap();
    let second = second.generate().unwrap();
    assert_eq!(
        first.iter().map(|job| job.seed).collect::<Vec<_>>(),
        second.iter().map(|job| job.seed).collect::<Vec<_>>()
    );
}

#[test]
fn protocol_rejects_select_before_generate_and_overlapping_batches() {
    let mut search = small_search(Objective::Minimize);
    assert_eq!(search.select(vec![]), Err(ElitistError::NoPendingBatch));
    let jobs = search.generate().unwrap();
    assert_eq!(search.generate(), Err(ElitistError::PendingBatch));
    let results = scored(jobs, &[1.0; 4]);
    search.select(results.clone()).unwrap();
    assert_eq!(search.select(results), Err(ElitistError::NoPendingBatch));
}

#[test]
fn missing_and_extra_results_are_rejected_without_a_commit() {
    for extra in [false, true] {
        let mut search = small_search(Objective::Minimize);
        let mut results = scored(search.generate().unwrap(), &[1.0; 4]);
        if extra {
            results.push(results[0].clone());
        } else {
            results.pop();
        }
        assert!(matches!(
            search.select(results),
            Err(ElitistError::BatchSize { .. })
        ));
        assert!(search.best().is_none());
        assert_eq!(search.evaluations(), 0);
        assert_eq!(search.batches(), 0);
    }
}

#[test]
fn duplicate_unknown_and_reordered_ids_are_rejected_without_a_commit() {
    for mode in 0..3 {
        let mut search = small_search(Objective::Minimize);
        let mut results = scored(search.generate().unwrap(), &[1.0; 4]);
        match mode {
            0 => results[3].job.id = results[0].job.id,
            1 => results[3].job.id = JobId(999),
            _ => results.swap(1, 2),
        }
        assert!(matches!(
            search.select(results),
            Err(ElitistError::UnexpectedId { .. })
        ));
        assert!(search.best().is_none());
        assert_eq!(search.evaluations(), 0);
    }
}

#[test]
fn changed_seed_or_input_is_rejected_even_when_it_has_the_best_score() {
    for change_seed in [false, true] {
        let mut search = small_search(Objective::Minimize);
        let jobs = search.generate().unwrap();
        search.select(scored(jobs, &[2.0; 4])).unwrap();
        let incumbent = search.best().cloned();
        let mut results = scored(search.generate().unwrap(), &[1.0, 0.5, 0.1, -1.0]);
        if change_seed {
            results[3].job.seed ^= 1;
        } else {
            results[3].job.input[0] += 1.0;
        }
        assert!(matches!(
            search.select(results),
            Err(ElitistError::ChangedJob(_))
        ));
        assert_eq!(search.best(), incumbent.as_ref());
        assert_eq!(search.evaluations(), 4);
        assert_eq!(search.batches(), 1);
    }
}

#[test]
fn nonfinite_scores_do_not_replace_the_incumbent_or_advance_accounting() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut search = small_search(Objective::Minimize);
        let jobs = search.generate().unwrap();
        search.select(scored(jobs, &[2.0; 4])).unwrap();
        let incumbent = search.best().cloned();
        let jobs = search.generate().unwrap();
        assert!(matches!(
            search.select(scored(jobs, &[1.0, 0.0, -1.0, invalid])),
            Err(ElitistError::NonFiniteOutput(_))
        ));
        assert_eq!(search.best(), incumbent.as_ref());
        assert_eq!(search.evaluations(), 4);
    }
}

#[test]
fn sphere_search_improves_across_fixed_seeds_without_claiming_a_convergence_rate() {
    for seed in [7, 19, 42, 91] {
        let mut config = ElitistConfig::new(8, seed);
        config.initial_bound = 5.0;
        let mut search = ElitistSearch::new(config).unwrap();
        let mut evaluator = Sphere::new(8).unwrap();
        step(&mut search, &mut evaluator).unwrap();
        let initial = search.best().unwrap().output;
        let mut previous = initial;
        for _ in 1..30 {
            assert_eq!(step(&mut search, &mut evaluator).unwrap(), 32);
            let current = search.best().unwrap().output;
            assert!(current <= previous);
            previous = current;
        }
        assert!(previous < initial);
        assert_eq!(search.evaluations(), 960);
        assert_eq!(search.batches(), 30);
    }
}
