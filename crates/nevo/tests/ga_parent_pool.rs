//! Truncation of the mating pool is independent of elite copying and scoring.

use nevo::algorithms::{Ga, GaConfig, Objective, SearchError};
use nevo::benchmarks::Sphere;
use nevo::{Algorithm, Evaluated, JobId};

#[test]
fn parent_pool_size_must_be_positive_and_no_larger_than_population() {
    for size in [0, 9, usize::MAX] {
        let mut config = GaConfig::new(4, 42);
        config.population = 8;
        config.parent_pool_size = Some(size);
        assert!(matches!(
            Ga::new(config),
            Err(SearchError::InvalidConfig(_))
        ));
    }
    for size in [None, Some(1), Some(8)] {
        let mut config = GaConfig::new(4, 42);
        config.population = 8;
        config.parent_pool_size = size;
        assert!(Ga::new(config).is_ok());
    }
}

#[test]
fn offspring_inherit_only_from_the_best_pool_in_both_objective_directions() {
    for (objective, eligible) in [(Objective::Minimize, [3, 6]), (Objective::Maximize, [2, 5])] {
        for crossover_probability in [0.0, 1.0] {
            let mut config = GaConfig::new(4, 42);
            config.population = 8;
            config.parent_pool_size = Some(2);
            config.tournament_size = 1;
            config.elite_count = 2;
            config.mutation_sigma = 0.0;
            config.crossover_probability = crossover_probability;
            config.objective = objective;
            let mut ga = Ga::new(config).unwrap();
            let initial = ga.generate().unwrap();
            let first = initial[eligible[0]].input.clone();
            let second = initial[eligible[1]].input.clone();
            ga.select(
                initial
                    .into_iter()
                    .zip([5.0, 3.0, 7.0, 0.0, 2.0, 6.0, 1.0, 4.0])
                    .map(|(job, output)| Evaluated { job, output })
                    .collect(),
            )
            .unwrap();
            assert_eq!(ga.parents().len(), 8);
            let offspring = ga.generate().unwrap();
            assert_eq!(offspring[0].input, first);
            assert_eq!(offspring[1].input, second);
            for child in offspring.iter().skip(2) {
                for coordinate in 0..4 {
                    assert!(
                        child.input[coordinate] == first[coordinate]
                            || child.input[coordinate] == second[coordinate]
                    );
                }
                if crossover_probability == 0.0 {
                    assert!(child.input == first || child.input == second);
                }
            }
        }
    }
}

#[test]
fn a_one_candidate_pool_can_reproduce_without_copying_any_elites() {
    let mut config = GaConfig::new(3, 42);
    config.population = 8;
    config.parent_pool_size = Some(1);
    config.elite_count = 0;
    config.mutation_sigma = 0.0;
    config.crossover_probability = 1.0;
    let mut ga = Ga::new(config).unwrap();
    let initial = ga.generate().unwrap();
    let winner = initial[7].input.clone();
    ga.select(
        initial
            .into_iter()
            .enumerate()
            .map(|(index, job)| Evaluated {
                job,
                output: -(index as f64),
            })
            .collect(),
    )
    .unwrap();
    assert!(ga.generate().unwrap().iter().all(|job| job.input == winner));
}

#[test]
fn explicit_full_pool_preserves_default_jobs_random_streams_and_selected_state() {
    let config = GaConfig::new(4, 42);
    assert_eq!(config.parent_pool_size, None);
    let mut full_config = config.clone();
    full_config.parent_pool_size = Some(config.population);
    let mut default = Ga::new(config).unwrap();
    let mut full = Ga::new(full_config).unwrap();
    let sphere = Sphere::new(4).unwrap();
    for _ in 0..8 {
        let jobs = default.generate().unwrap();
        assert_eq!(jobs, full.generate().unwrap());
        let results: Vec<_> = jobs
            .into_iter()
            .map(|job| Evaluated {
                output: sphere.value(&job.input).unwrap(),
                job,
            })
            .collect();
        default.select(results.clone()).unwrap();
        full.select(results).unwrap();
        assert_eq!(default.parents(), full.parents());
        assert_eq!(default.best(), full.best());
    }
}

#[test]
fn twenty_retained_elites_are_new_jobs_whose_scores_are_reevaluated() {
    let mut config = GaConfig::new(3, 42);
    config.population = 100;
    config.parent_pool_size = Some(20);
    config.elite_count = 20;
    config.tournament_size = 1;
    config.crossover_probability = 1.0;
    let mut ga = Ga::new(config).unwrap();
    let initial = ga.generate().unwrap();
    let elite_genomes: Vec<_> = initial
        .iter()
        .take(20)
        .map(|job| job.input.clone())
        .collect();
    ga.select(
        initial
            .into_iter()
            .enumerate()
            .map(|(index, job)| Evaluated {
                job,
                output: index as f64,
            })
            .collect(),
    )
    .unwrap();
    let next = ga.generate().unwrap();
    assert_eq!(next.len(), 100);
    for (index, genome) in elite_genomes.iter().enumerate() {
        assert_eq!(&next[index].input, genome);
        assert_eq!(next[index].id, JobId(100 + index as u64));
    }
    ga.select(
        next.into_iter()
            .map(|job| Evaluated {
                job,
                output: 1000.0,
            })
            .collect(),
    )
    .unwrap();
    assert!(ga.parents().iter().all(|parent| parent.output == 1000.0));
    assert_eq!(ga.best().unwrap().output, 0.0);
    assert_eq!(ga.evaluations(), 200);
}
