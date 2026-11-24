//! Explicit fixed-scale and parent-plus-offspring ES behavior.

use nevo::algorithms::{Es, EsConfig, EsSelection, EsStepSize, Objective, SearchError};
use nevo::benchmarks::Sphere;
use nevo::{Algorithm, Evaluated, Job, JobId, step};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};

fn config(selection: EsSelection, step_size: EsStepSize) -> EsConfig {
    let mut config = EsConfig::new(2, 42);
    config.population = 4;
    config.parents = 1;
    config.selection = selection;
    config.step_size = step_size;
    config
}

fn score(jobs: Vec<Job<Vec<f64>>>, outputs: &[f64]) -> Vec<Evaluated<Vec<f64>, f64>> {
    assert_eq!(jobs.len(), outputs.len());
    jobs.into_iter()
        .zip(outputs)
        .map(|(job, &output)| Evaluated { job, output })
        .collect()
}

#[test]
fn original_defaults_remain_self_adaptive_comma() {
    let config = EsConfig::new(8, 42);
    assert_eq!(config.selection, EsSelection::Comma);
    assert_eq!(config.step_size, EsStepSize::SelfAdaptive);
    assert_eq!(EsSelection::default(), EsSelection::Comma);
    assert_eq!(EsStepSize::default(), EsStepSize::SelfAdaptive);
}

#[test]
fn fixed_scale_draws_only_parent_index_and_coordinate_gaussians() {
    let config = config(EsSelection::Comma, EsStepSize::Fixed);
    let mut search = Es::new(config.clone()).unwrap();
    let initial = search.generate().unwrap();
    let parent = initial[0].input.clone();
    search
        .select(score(initial, &[0.0, 1.0, 2.0, 3.0]))
        .unwrap();
    let offspring = search.generate().unwrap();

    // Independently account for random draws: 4*2 initial uniform coordinates,
    // one parent-index draw per offspring, then two draws per Box-Muller value.
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..8 {
        rng.next_u64();
    }
    for child in &offspring {
        rng.next_u64(); // There is one parent, so the index is necessarily zero.
        for (&center, &actual) in parent.iter().zip(&child.input) {
            let u = ((rng.next_u64() >> 11) as f64 + 1.0) / 9_007_199_254_740_992.0;
            let v = (rng.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
            let normal = (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos();
            let expected = center + config.initial_sigma * normal;
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    search
        .select(score(offspring, &[0.0, 1.0, 2.0, 3.0]))
        .unwrap();
    assert_eq!(search.parents()[0].sigma, config.initial_sigma);
}

#[test]
fn fixed_mode_ignores_unused_adaptation_parameters() {
    let mut config = config(EsSelection::Plus, EsStepSize::Fixed);
    config.tau = f64::NAN;
    config.sigma_min = -1.0;
    config.sigma_max = 0.0;
    let mut search = Es::new(config).unwrap();
    let mut sphere = Sphere::new(2).unwrap();
    for _ in 0..4 {
        step(&mut search, &mut sphere).unwrap();
        assert!(search.parents().iter().all(|parent| parent.sigma == 0.5));
    }
}

#[test]
fn plus_retains_better_parents_and_parent_ties_in_both_objective_directions() {
    for objective in [Objective::Minimize, Objective::Maximize] {
        let direction = if objective == Objective::Minimize {
            1.0
        } else {
            -1.0
        };
        let mut config = config(EsSelection::Plus, EsStepSize::SelfAdaptive);
        config.parents = 2;
        config.objective = objective;
        let mut search = Es::new(config).unwrap();
        let initial = search.generate().unwrap();
        search
            .select(score(
                initial,
                &[0.0, direction, 2.0 * direction, 3.0 * direction],
            ))
            .unwrap();
        let original = search.parents().to_vec();
        let children = search.generate().unwrap();
        search
            .select(score(
                children,
                &[direction, 5.0 * direction, 6.0 * direction, 7.0 * direction],
            ))
            .unwrap();
        assert_eq!(search.parents(), original);
        assert_eq!(search.evaluations(), 8); // Cached parents incur no extra calls.
        assert_eq!(search.batches(), 2);

        let children = search.generate().unwrap();
        search
            .select(score(
                children,
                &[
                    -direction,
                    8.0 * direction,
                    9.0 * direction,
                    10.0 * direction,
                ],
            ))
            .unwrap();
        assert_eq!(search.parents()[0].evaluation.job.id, JobId(8));
        assert_eq!(search.parents()[1], original[0]);
        assert_eq!(search.best().unwrap().job.id, JobId(8));
    }
}

#[test]
fn plus_keeps_each_surviving_parents_adapted_scale() {
    let mut config = config(EsSelection::Plus, EsStepSize::SelfAdaptive);
    config.parents = 2;
    let mut search = Es::new(config).unwrap();
    let initial = search.generate().unwrap();
    search
        .select(score(initial, &[3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    let children = search.generate().unwrap();
    search
        .select(score(children, &[0.0, 1.0, 7.0, 8.0]))
        .unwrap();
    let adapted = search.parents().to_vec();
    assert!(adapted.iter().all(|parent| parent.sigma != 0.5));
    let children = search.generate().unwrap();
    search
        .select(score(children, &[10.0, 11.0, 12.0, 13.0]))
        .unwrap();
    assert_eq!(search.parents(), adapted);
}

#[test]
fn one_plus_one_fixed_es_keeps_incumbent_until_strict_improvement() {
    let mut config = config(EsSelection::Plus, EsStepSize::Fixed);
    config.population = 1;
    config.parents = 1;
    let mut search = Es::new(config).unwrap();
    let initial = search.generate().unwrap();
    search.select(score(initial, &[5.0])).unwrap();
    let incumbent = search.parents()[0].clone();
    for worse_or_equal in [6.0, 5.0] {
        let child = search.generate().unwrap();
        assert_eq!(child.len(), 1);
        search.select(score(child, &[worse_or_equal])).unwrap();
        assert_eq!(search.parents(), [incumbent.clone()]);
    }
    let child = search.generate().unwrap();
    let improved = child[0].clone();
    search.select(score(child, &[4.0])).unwrap();
    assert_eq!(search.parents()[0].evaluation.job, improved);
    assert_eq!(search.parents()[0].sigma, 0.5);
    assert_eq!(search.best().unwrap().job, improved);
    assert_eq!(search.batches(), 4);
    assert_eq!(search.evaluations(), 4);
}

#[test]
fn all_four_modes_replay_and_improve_sphere() {
    for selection in [EsSelection::Comma, EsSelection::Plus] {
        for step_size in [EsStepSize::SelfAdaptive, EsStepSize::Fixed] {
            let mut config = EsConfig::new(4, 42);
            config.selection = selection;
            config.step_size = step_size;
            let mut first = Es::new(config.clone()).unwrap();
            let mut second = Es::new(config).unwrap();
            let sphere = Sphere::new(4).unwrap();
            let mut initial = 0.0;
            for batch in 0..20 {
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
                if batch == 0 {
                    initial = first.best().unwrap().output;
                }
            }
            assert_eq!(first.parents(), second.parents());
            assert!(first.best().unwrap().output < initial);
            assert_eq!(first.evaluations(), 1280);
            assert_eq!(first.batches(), 20);
        }
    }
}

#[test]
fn plus_rejects_bad_pending_results_without_changing_selected_state() {
    let mut search = Es::new(config(EsSelection::Plus, EsStepSize::Fixed)).unwrap();
    let mut sphere = Sphere::new(2).unwrap();
    step(&mut search, &mut sphere).unwrap();
    let parents = search.parents().to_vec();
    let best = search.best().cloned();
    let mut invalid = score(search.generate().unwrap(), &[-100.0, -99.0, -98.0, -97.0]);
    invalid[3].job.seed ^= 1;
    assert!(matches!(
        search.select(invalid),
        Err(SearchError::InvalidResults(_))
    ));
    assert_eq!(search.parents(), parents);
    assert_eq!(search.best(), best.as_ref());
    assert_eq!(search.evaluations(), 4);
    assert_eq!(search.generate(), Err(SearchError::PendingBatch));
}

#[test]
fn both_modes_reject_invalid_required_scale_and_population() {
    for step_size in [EsStepSize::Fixed, EsStepSize::SelfAdaptive] {
        for sigma in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let mut config = config(EsSelection::Plus, step_size);
            config.initial_sigma = sigma;
            assert!(Es::new(config).is_err());
        }
        let mut config = config(EsSelection::Plus, step_size);
        config.population = usize::MAX;
        assert!(Es::new(config).is_err());
    }
}
