//! Pareto reference cases and NSGA-II survival, protocol, and search checks.

use nevo::algorithms::{Nsga2, Nsga2Config, Objective, SearchError};
use nevo::benchmarks::Zdt1;
use nevo::multiobjective::{crowding_distances, dominates, nondominated_fronts};
use nevo::{Algorithm, Evaluated, Job};

fn scored(
    jobs: Vec<Job<Vec<f64>>>,
    fitness: impl Fn(&[f64]) -> Vec<f64>,
) -> Vec<Evaluated<Vec<f64>, Vec<f64>>> {
    jobs.into_iter()
        .map(|job| {
            let output = fitness(&job.input);
            Evaluated { job, output }
        })
        .collect()
}

fn assigned(jobs: Vec<Job<Vec<f64>>>, values: &[[f64; 2]]) -> Vec<Evaluated<Vec<f64>, Vec<f64>>> {
    assert_eq!(jobs.len(), values.len());
    jobs.into_iter()
        .zip(values)
        .map(|(job, output)| Evaluated {
            job,
            output: output.to_vec(),
        })
        .collect()
}

#[test]
fn dominance_requires_one_strict_improvement_and_supports_mixed_directions() {
    let directions = [
        Objective::Minimize,
        Objective::Maximize,
        Objective::Minimize,
    ];
    assert!(dominates(&[1.0, 9.0, 3.0], &[1.0, 7.0, 3.0], &directions).unwrap());
    assert!(!dominates(&[1.0, 9.0, 4.0], &[1.0, 7.0, 3.0], &directions).unwrap());
    assert!(!dominates(&[1.0, 7.0, 3.0], &[1.0, 9.0, 3.0], &directions).unwrap());
    assert!(!dominates(&[1.0, 9.0, 3.0], &[1.0, 9.0, 3.0], &directions).unwrap());
    assert!(!dominates(&[-0.0], &[0.0], &[Objective::Minimize]).unwrap());
    assert!(!dominates(&[0.0], &[-0.0], &[Objective::Maximize]).unwrap());
}

#[test]
fn nondominated_sort_matches_four_hand_calculated_fronts_with_duplicates() {
    let values = vec![
        vec![1.0, 5.0],
        vec![2.0, 3.0],
        vec![4.0, 1.0],
        vec![2.0, 5.0],
        vec![3.0, 4.0],
        vec![4.0, 3.0],
        vec![4.0, 5.0],
        vec![5.0, 6.0],
        vec![2.0, 3.0],
    ];
    let expected = vec![vec![0, 1, 2, 8], vec![3, 4, 5], vec![6], vec![7]];
    assert_eq!(
        nondominated_fronts(&values, &[Objective::Minimize; 2]).unwrap(),
        expected
    );
    let mixed: Vec<_> = values.iter().map(|v| vec![v[0], -v[1]]).collect();
    assert_eq!(
        nondominated_fronts(&mixed, &[Objective::Minimize, Objective::Maximize]).unwrap(),
        expected
    );
    assert!(
        nondominated_fronts(&[], &[Objective::Minimize])
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        nondominated_fronts(&[vec![0.0], vec![-0.0]], &[Objective::Minimize]).unwrap(),
        [vec![0, 1]]
    );
}

#[test]
fn crowding_matches_normalized_neighbor_gaps_and_ignores_constant_objectives() {
    let values = vec![
        vec![0.0, 8.0, 42.0],
        vec![1.0, 7.0, 42.0],
        vec![4.0, 3.0, 42.0],
        vec![8.0, 0.0, 42.0],
    ];
    assert_eq!(
        crowding_distances(&values).unwrap(),
        [f64::INFINITY, 1.125, 1.75, f64::INFINITY]
    );
    assert_eq!(
        crowding_distances(&[vec![2.0, -0.0], vec![2.0, 0.0], vec![2.0, 0.0]]).unwrap(),
        [0.0; 3]
    );
    assert_eq!(crowding_distances(&[vec![1.0, 2.0]]).unwrap(), [0.0]);
    assert!(crowding_distances(&[]).unwrap().is_empty());
    assert_eq!(
        crowding_distances(&[vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap(),
        [f64::INFINITY; 2]
    );
    let duplicates = vec![
        vec![0.0, 3.0],
        vec![1.0, 2.0],
        vec![1.0, 2.0],
        vec![3.0, 0.0],
    ];
    assert_eq!(
        crowding_distances(&duplicates).unwrap(),
        [f64::INFINITY, 1.0, 1.0, f64::INFINITY]
    );
}

#[test]
fn crowding_handles_extreme_and_subnormal_objective_ranges() {
    let huge = f64::MAX;
    assert_eq!(
        crowding_distances(&[vec![-huge], vec![0.0], vec![huge]]).unwrap(),
        [f64::INFINITY, 1.0, f64::INFINITY]
    );
    let tiny = f64::from_bits(1);
    assert_eq!(
        crowding_distances(&[vec![-tiny], vec![0.0], vec![tiny]]).unwrap(),
        [f64::INFINITY, 1.0, f64::INFINITY]
    );
    let adjacent = vec![
        vec![f64::from_bits(huge.to_bits() - 2)],
        vec![f64::from_bits(huge.to_bits() - 1)],
        vec![huge],
    ];
    assert_eq!(
        crowding_distances(&adjacent).unwrap(),
        [f64::INFINITY, 1.0, f64::INFINITY]
    );
}

#[test]
fn pareto_primitives_reject_invalid_shapes_and_values() {
    assert!(dominates(&[], &[], &[]).is_err());
    assert!(dominates(&[0.0], &[0.0, 1.0], &[Objective::Minimize]).is_err());
    assert!(nondominated_fronts(&[], &[]).is_err());
    assert!(crowding_distances(&[vec![]]).is_err());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(dominates(&[bad], &[0.0], &[Objective::Minimize]).is_err());
        assert!(nondominated_fronts(&[vec![bad]], &[Objective::Minimize]).is_err());
        assert!(crowding_distances(&[vec![0.0], vec![bad]]).is_err());
    }
    assert!(nondominated_fronts(&[vec![1.0], vec![1.0, 2.0]], &[Objective::Minimize]).is_err());
    assert!(crowding_distances(&[vec![1.0], vec![1.0, 2.0]]).is_err());
}

#[test]
fn nsga2_keeps_best_combined_fronts_with_mixed_objective_directions() {
    for directions in [
        [Objective::Minimize; 2],
        [Objective::Minimize, Objective::Maximize],
    ] {
        let mut config = Nsga2Config::new(2, directions.to_vec(), 7);
        config.population = 5;
        let mut search = Nsga2::new(config).unwrap();
        for values in [
            [
                [0.0, 10.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [11.0, 11.0],
                [12.0, 12.0],
            ],
            [[0.0, 0.0], [1.0, 4.0], [4.0, 1.0], [6.0, 6.0], [20.0, 20.0]],
        ] {
            let mut results = assigned(search.generate().unwrap(), &values);
            if directions[1] == Objective::Maximize {
                for result in &mut results {
                    result.output[1] = -result.output[1];
                }
            }
            search.select(results).unwrap();
        }
        let ids: Vec<_> = search.parents().iter().map(|p| p.job.id.0).collect();
        // New (0,0) is rank 0; old extremes and the new tradeoff points are rank 1.
        assert_eq!(ids, [5, 0, 1, 6, 7]);
        assert_eq!(search.pareto_front().len(), 1);
        assert_eq!(search.pareto_front()[0].job.id.0, 5);
        assert_eq!(search.evaluations(), 10);
    }
}

#[test]
fn nsga2_issues_only_fresh_offspring_and_caches_retained_parent_scores() {
    let mut config = Nsga2Config::new(2, vec![Objective::Minimize; 2], 42);
    config.population = 4;
    let mut search = Nsga2::new(config).unwrap();
    let first = scored(search.generate().unwrap(), |_| vec![0.0; 2]);
    search.select(first.clone()).unwrap();
    for generation in 1..4 {
        let jobs = search.generate().unwrap();
        assert_eq!(jobs.len(), 4);
        assert_eq!(
            jobs.iter().map(|j| j.id.0).collect::<Vec<_>>(),
            ((generation * 4) as u64..((generation + 1) * 4) as u64).collect::<Vec<_>>()
        );
        let score = if generation == 1 { 1.0 } else { 0.0 };
        search.select(scored(jobs, |_| vec![score; 2])).unwrap();
        // Inferior offspring lose; completely tied, zero-crowding offspring
        // lose the stable survivor tie break to the existing parents.
        assert_eq!(search.parents(), first);
        assert_eq!(search.batches(), generation + 1);
        assert_eq!(search.evaluations(), 4 * (generation + 1));
    }
}

#[test]
fn nsga2_rejects_malformed_batches_without_mutating_state_or_consuming_pending_work() {
    let mut config = Nsga2Config::new(2, vec![Objective::Minimize; 2], 23);
    config.population = 4;
    let mut search = Nsga2::new(config.clone()).unwrap();
    let mut control = Nsga2::new(config).unwrap();
    assert_eq!(search.select(Vec::new()), Err(SearchError::NoPendingBatch));
    let fitness = |x: &[f64]| vec![x.iter().map(|v| v * v).sum(), x.iter().sum()];
    let first = scored(search.generate().unwrap(), fitness);
    assert_eq!(first, scored(control.generate().unwrap(), fitness));
    search.select(first.clone()).unwrap();
    control.select(first).unwrap();
    let valid = scored(search.generate().unwrap(), fitness);
    assert_eq!(valid, scored(control.generate().unwrap(), fitness));
    let before = search.parents().to_vec();
    assert_eq!(search.generate(), Err(SearchError::PendingBatch));
    for corruption in 0..10 {
        let mut bad = valid.clone();
        match corruption {
            0 => {
                bad.pop();
            }
            1 => {
                bad.push(bad[0].clone());
            }
            2 => bad.swap(0, 1),
            3 => bad[1].job.id = bad[0].job.id,
            4 => bad[0].job.seed ^= 1,
            5 => bad[0].job.input[0] += 1.0,
            6 => {
                bad[0].job.input.pop();
            }
            7 => {
                bad[0].output.pop();
            }
            8 => bad[0].output[0] = f64::NAN,
            9 => bad[0].output[0] = f64::INFINITY,
            _ => unreachable!(),
        }
        assert!(matches!(
            search.select(bad),
            Err(SearchError::InvalidResults(_))
        ));
        assert_eq!(search.parents(), before);
        assert_eq!(search.batches(), 1);
        assert_eq!(search.evaluations(), 4);
        assert_eq!(search.generate(), Err(SearchError::PendingBatch));
    }
    search.select(valid.clone()).unwrap();
    control.select(valid.clone()).unwrap();
    assert_eq!(search.select(valid), Err(SearchError::NoPendingBatch));
    assert_eq!(search.parents(), control.parents());
    assert_eq!(search.generate().unwrap(), control.generate().unwrap());
}

#[test]
fn nsga2_seeded_replay_and_clipping_preserve_configured_bounds() {
    let mut config = Nsga2Config::new(3, vec![Objective::Minimize; 2], 17);
    config.population = 12;
    config.lower_bound = -2.0;
    config.upper_bound = 0.25;
    config.mutation_sigma = 10.0;
    config.mutation_probability = 1.0;
    config.crossover_probability = 1.0;
    let mut first = Nsga2::new(config.clone()).unwrap();
    let mut replay = Nsga2::new(config).unwrap();
    let mut saw_clipped = false;
    for _ in 0..10 {
        let jobs = first.generate().unwrap();
        assert_eq!(jobs, replay.generate().unwrap());
        for gene in jobs.iter().flat_map(|job| &job.input) {
            assert!(gene.is_finite() && (-2.0..=0.25).contains(gene));
            saw_clipped |= *gene == -2.0 || *gene == 0.25;
        }
        let results = scored(jobs, |x| {
            vec![
                x.iter().map(|v| (v + 2.0).powi(2)).sum(),
                x.iter().map(|v| (v - 0.25).powi(2)).sum(),
            ]
        });
        first.select(results.clone()).unwrap();
        replay.select(results).unwrap();
        assert_eq!(first.parents(), replay.parents());
    }
    assert!(saw_clipped);
}

#[test]
fn nsga2_rejects_invalid_configuration() {
    let valid = Nsga2Config::new(2, vec![Objective::Minimize; 2], 7);
    for corruption in 0..12 {
        let mut bad = valid.clone();
        match corruption {
            0 => bad.dimensions = 0,
            1 => bad.population = 1,
            2 => {
                bad.objectives.pop();
            }
            3 => bad.lower_bound = bad.upper_bound,
            4 => bad.lower_bound = f64::NEG_INFINITY,
            5 => {
                bad.lower_bound = -f64::MAX;
                bad.upper_bound = f64::MAX;
            }
            6 => bad.mutation_sigma = -1.0,
            7 => bad.mutation_sigma = f64::NAN,
            8 => bad.mutation_probability = -0.01,
            9 => bad.mutation_probability = f64::NAN,
            10 => bad.crossover_probability = 1.01,
            11 => bad.crossover_probability = f64::INFINITY,
            _ => unreachable!(),
        }
        assert!(matches!(
            Nsga2::new(bad),
            Err(SearchError::InvalidConfig(_))
        ));
    }
}

#[test]
fn nsga2_approaches_zdt1_tradeoff_front_across_fixed_seeds() {
    // A small six-gene integration problem, not the published 30-gene setting.
    let problem = Zdt1::new(6).unwrap();
    let mean_gap = |search: &Nsga2| {
        let front = search.pareto_front();
        front
            .iter()
            .map(|p| p.output[1] - (1.0 - p.output[0].sqrt()))
            .sum::<f64>()
            / front.len() as f64
    };
    for seed in [1, 7, 42] {
        let mut config = Nsga2Config::new(6, vec![Objective::Minimize; 2], seed);
        config.population = 40;
        let mut search = Nsga2::new(config).unwrap();
        let results = scored(search.generate().unwrap(), |x| problem.value(x).unwrap());
        search.select(results).unwrap();
        let initial_gap = mean_gap(&search);
        for _ in 1..100 {
            let results = scored(search.generate().unwrap(), |x| problem.value(x).unwrap());
            search.select(results).unwrap();
        }
        let final_gap = mean_gap(&search);
        assert!(
            final_gap < initial_gap * 0.25,
            "seed {seed}: {initial_gap} -> {final_gap}"
        );
        assert!(final_gap < 0.25, "seed {seed}: {final_gap}");
        let front = search.pareto_front();
        let low = front
            .iter()
            .map(|p| p.output[0])
            .fold(f64::INFINITY, f64::min);
        let high = front
            .iter()
            .map(|p| p.output[0])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            high - low > 0.8,
            "seed {seed}: narrow front [{low}, {high}]"
        );
        for (i, a) in front.iter().enumerate() {
            for b in &front[(i + 1)..] {
                assert!(!dominates(&a.output, &b.output, &[Objective::Minimize; 2]).unwrap());
                assert!(!dominates(&b.output, &a.output, &[Objective::Minimize; 2]).unwrap());
            }
        }
        assert_eq!(search.batches(), 100);
        assert_eq!(search.evaluations(), 4_000);
    }
}
