//! Bounded property tests for numerical, graph, and Generate–Evaluate–Select contracts.

use nevo::algorithms::{Ga, GaConfig, Objective, SearchError};
use nevo::benchmarks::Sphere;
use nevo::multiobjective::nondominated_fronts;
use nevo::neat::{
    InnovationTracker, MutationConfig, add_connection, allocate_offspring, crossover, mutate,
};
use nevo::nn::softmax;
use nevo::{Algorithm, Evaluated, step};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};

fn property_config() -> ProptestConfig {
    let mut config = ProptestConfig::default();
    if std::env::var_os("PROPTEST_CASES").is_none() {
        config.cases = 64;
    }
    // Integration tests are outside src/, so persist beside this source file.
    config.failure_persistence = Some(Box::new(FileFailurePersistence::WithSource(
        "proptest-regressions",
    )));
    config
}

proptest! {
    #![proptest_config(property_config())]

    #[test]
    fn softmax_is_normalized_and_unchanged_by_a_common_shift(
        logits in prop::collection::vec(-1000i16..=1000, 1..33),
        shift in -10_000i32..=10_000,
    ) {
        // Integers and eighths are exact here, isolating the softmax invariant.
        let logits: Vec<_> = logits.into_iter().map(|x| f64::from(x) / 8.0).collect();
        let shifted: Vec<_> = logits.iter().map(|x| x + f64::from(shift)).collect();
        let probabilities = softmax(&logits).unwrap();
        let shifted_probabilities = softmax(&shifted).unwrap();
        prop_assert_eq!(probabilities.len(), logits.len());
        prop_assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for (probability, shifted) in probabilities.iter().zip(shifted_probabilities) {
            prop_assert!(probability.is_finite() && (0.0..=1.0).contains(probability));
            prop_assert!((probability - shifted).abs() < 1e-12);
        }
    }

    #[test]
    fn pareto_fronts_partition_candidates_and_assign_their_dominance_ranks(
        points in prop::collection::vec(prop::array::uniform2(-8i16..=8), 1..25),
        maximize in any::<[bool; 2]>(),
    ) {
        let objectives = maximize.map(|yes| if yes { Objective::Maximize } else { Objective::Minimize });
        let values: Vec<_> = points.iter().map(|point| point.map(f64::from).to_vec()).collect();
        let fronts = nondominated_fronts(&values, &objectives).unwrap();
        let mut ranks = vec![None; points.len()];
        for (rank, front) in fronts.iter().enumerate() {
            prop_assert!(!front.is_empty());
            for &index in front {
                prop_assert!(index < points.len());
                prop_assert!(ranks[index].replace(rank).is_none());
            }
        }
        prop_assert!(ranks.iter().all(Option::is_some));
        // Independent integer comparisons also exercise mixed directions and ties.
        for (index, point) in points.iter().enumerate() {
            let dominators = points.iter().enumerate().filter(|(_, other)| {
                let better = |axis: usize| if maximize[axis] {
                    other[axis] > point[axis]
                } else {
                    other[axis] < point[axis]
                };
                (0..2).all(|axis| other[axis] == point[axis] || better(axis))
                    && (0..2).any(better)
            });
            let expected = dominators.map(|(other, _)| ranks[other].unwrap() + 1).max().unwrap_or(0);
            prop_assert_eq!(ranks[index], Some(expected));
        }
    }

    #[test]
    fn feedforward_mutation_and_crossover_preserve_valid_graphs(
        seed in any::<u64>(),
        generations in 1usize..7,
        weights in prop::array::uniform3(-16i16..=16),
        input in prop::array::uniform2(-1.0f64..1.0),
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut tracker = InnovationTracker::new(2, 1).unwrap();
        let mut left = tracker.initial_genome(weights.map(|x| f64::from(x) / 8.0).to_vec()).unwrap();
        let mut right = left.clone();
        let config = MutationConfig {
            add_node_probability: 1.0,
            add_connection_probability: 1.0,
            ..MutationConfig::default()
        };
        for _ in 0..generations {
            tracker.begin_generation();
            mutate(&mut left, &mut tracker, &config, &mut rng).unwrap();
            mutate(&mut right, &mut tracker, &config, &mut rng).unwrap();
            let child = crossover(&left, 1.0, &right, 1.0, 0.75, &mut rng).unwrap();
            for genome in [&left, &right, &child] {
                tracker.validate_genome(genome).unwrap();
                let output = genome.compile().unwrap().forward(&input).unwrap();
                prop_assert_eq!(output.len(), 1);
                prop_assert!(output[0].is_finite() && (0.0..=1.0).contains(&output[0]));
            }
            left = child;
        }
    }

    #[test]
    fn recurrent_reset_replays_a_sequence_from_fresh_state(
        seed in any::<u64>(),
        sequence in prop::collection::vec(-1.0f64..1.0, 1..25),
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut tracker = InnovationTracker::new_recurrent(1, 1).unwrap();
        let mut genome = tracker.initial_genome(vec![1.0, -0.5]).unwrap();
        add_connection(&mut genome, &mut tracker, 2, 2, 1.0).unwrap();
        let config = MutationConfig {
            add_node_probability: 1.0,
            add_connection_probability: 1.0,
            ..MutationConfig::default()
        };
        for _ in 0..3 {
            tracker.begin_generation();
            mutate(&mut genome, &mut tracker, &config, &mut rng).unwrap();
        }
        let mut network = genome.compile_recurrent().unwrap();
        let expected: Vec<_> = sequence.iter().map(|&input| network.step(&[input]).unwrap()).collect();
        network.reset();
        for (&input, expected) in sequence.iter().zip(expected) {
            prop_assert_eq!(network.step(&[input]).unwrap(), expected);
        }
    }

    #[test]
    fn offspring_quotas_preserve_the_requested_population(
        weights in prop::collection::vec(0u16..=100, 1..33),
        population in 1usize..1025,
    ) {
        let weights: Vec<_> = weights.into_iter().map(f64::from).collect();
        let quotas = allocate_offspring(&weights, population).unwrap();
        prop_assert_eq!(quotas.len(), weights.len());
        prop_assert_eq!(quotas.iter().sum::<usize>(), population);
        if weights.iter().any(|&weight| weight > 0.0) {
            for (&weight, &quota) in weights.iter().zip(&quotas) {
                if weight == 0.0 {
                    prop_assert_eq!(quota, 0);
                }
            }
        }
    }

    #[test]
    fn malformed_results_do_not_commit_selection(
        seed in any::<u64>(),
        dimensions in 1usize..9,
        population in 2usize..17,
        corruption in 0u8..6,
        index in any::<usize>(),
    ) {
        let mut config = GaConfig::new(dimensions, seed);
        config.population = population;
        let mut search = Ga::new(config).unwrap();
        let mut sphere = Sphere::new(dimensions).unwrap();
        step(&mut search, &mut sphere).unwrap();
        let before = search.parents().to_vec();
        let best = search.best().cloned();
        let mut results: Vec<_> = search.generate().unwrap().into_iter().map(|job| {
            let output = sphere.value(&job.input).unwrap();
            Evaluated { job, output }
        }).collect();
        let index = index % population;
        match corruption {
            0 => results[index].job.id.0 ^= 1,
            1 => results[index].job.seed ^= 1,
            2 => results[index].job.input[0] += 1.0,
            3 => results[index].output = f64::NAN,
            4 => { results.remove(index); }
            _ => results.rotate_left(1),
        }
        prop_assert!(matches!(search.select(results), Err(SearchError::InvalidResults(_))));
        prop_assert_eq!(search.parents(), before);
        prop_assert_eq!(search.best(), best.as_ref());
        prop_assert_eq!(search.batches(), 1);
        prop_assert_eq!(search.evaluations(), population);
        // This checks pending-work preservation; it does not establish a retry API.
        prop_assert_eq!(search.generate(), Err(SearchError::PendingBatch));
    }
}
