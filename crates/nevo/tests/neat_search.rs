//! NEAT species sharing, exact work budgets, GES integrity, and a fixed XOR smoke run.

use nevo::algorithms::{Neat, NeatConfig, SearchError};
use nevo::neat::{Genome, NeatError};
use nevo::{Algorithm, Evaluated, Job};

fn scored(jobs: Vec<Job<Genome>>, fitness: impl Fn(&Genome) -> f64) -> Vec<Evaluated<Genome, f64>> {
    jobs.into_iter()
        .map(|job| {
            let output = fitness(&job.input);
            Evaluated { job, output }
        })
        .collect()
}

fn fixed_species_config(population: usize) -> NeatConfig {
    let mut config = NeatConfig::new(2, 1, 42);
    config.population = population;
    config.compatibility_threshold = 1e-12;
    config.champion_min_offspring = 1;
    config.crossover_probability = 0.0;
    config.mutation.weight_probability = 0.0;
    config.mutation.add_node_probability = 0.0;
    config.mutation.add_connection_probability = 0.0;
    config
}

#[test]
fn initialization_is_minimal_and_fresh_work_includes_species_champions() {
    let population = 8;
    let mut search = Neat::new(fixed_species_config(population)).unwrap();
    assert!(search.best().is_none());
    assert!(search.parents().is_empty());
    assert!(search.species().is_empty());
    let mut original = Vec::new();
    for batch in 0..3 {
        let jobs = search.generate().unwrap();
        assert_eq!(jobs.len(), population);
        for (index, job) in jobs.iter().enumerate() {
            assert_eq!(job.id.0, (batch * population + index) as u64);
            assert_eq!(job.input.hidden_node_count(), 0);
            assert_eq!(job.input.enabled_connection_count(), 3);
            if batch > 0 {
                assert_eq!(job.input, original[index]);
            }
        }
        if batch == 0 {
            original = jobs.iter().map(|job| job.input.clone()).collect();
        }
        search.select(scored(jobs, |_| 0.0)).unwrap();
        assert_eq!(search.batches(), batch + 1);
        assert_eq!(search.evaluations(), population * (batch + 1));
        assert_eq!(search.species().len(), population);
        assert!(
            search
                .species()
                .iter()
                .all(|species| species.offspring == 1)
        );
    }
    assert_eq!(search.best().unwrap().job.id.0, 0);
}

#[test]
fn sharing_uses_species_means_before_culling_not_species_fitness_sums() {
    let mut config = NeatConfig::new(2, 1, 13);
    config.population = 18;
    config.compatibility.weight = 1.0;
    config.compatibility_threshold = 0.4;
    config.survival_fraction = 0.2;
    let mut search = Neat::new(config).unwrap();
    let jobs = search.generate().unwrap();
    // For these equal-length initial genomes there are no disjoint/excess
    // genes, so distance is simply the mean absolute weight difference.
    let mut representatives: Vec<&Genome> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    for job in &jobs {
        let found = representatives.iter().position(|representative| {
            job.input
                .connections()
                .iter()
                .zip(representative.connections())
                .map(|(a, b)| (a.weight - b.weight).abs())
                .sum::<f64>()
                / 3.0
                < 0.4
        });
        if let Some(index) = found {
            sizes[index] += 1;
        } else {
            representatives.push(&job.input);
            sizes.push(1);
        }
    }
    assert!(sizes.len() > 1);
    assert!(sizes.windows(2).any(|pair| pair[0] != pair[1]));
    search.select(scored(jobs, |_| 4.0)).unwrap();
    let species = search.species();
    assert_eq!(species.iter().map(|s| s.size).collect::<Vec<_>>(), sizes);
    for (index, group) in species.iter().enumerate() {
        assert_eq!(group.shared_fitness, 4.0);
        assert_eq!(
            group.offspring,
            18 / species.len() + usize::from(index < 18 % species.len())
        );
    }
    assert_eq!(species.iter().map(|s| s.offspring).sum::<usize>(), 18);
}

#[test]
fn species_ids_persist_and_stagnation_protects_the_current_champions_species() {
    let mut config = fixed_species_config(6);
    config.stagnation_generations = 2;
    let mut search = Neat::new(config).unwrap();
    for generation in 0..3 {
        let jobs = search.generate().unwrap();
        search.select(scored(jobs, |_| 1.0)).unwrap();
        assert_eq!(
            search.species().iter().map(|s| s.id).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5]
        );
        assert!(
            search
                .species()
                .iter()
                .all(|s| s.stagnant_generations == generation)
        );
    }
    assert!(search.species()[0].protected);
    assert_eq!(search.species()[0].offspring, 6);
    assert!(
        search.species()[1..]
            .iter()
            .all(|s| !s.protected && s.offspring == 0)
    );
    let jobs = search.generate().unwrap();
    search.select(scored(jobs, |_| 1.0)).unwrap();
    assert_eq!(search.species().len(), 1);
    assert_eq!(search.species()[0].id, 0);
    assert_eq!(search.species()[0].offspring, 6);
    assert_eq!(search.evaluations(), 24);
}

#[test]
fn proportional_species_quotas_use_raw_nonnegative_fitness() {
    let mut search = Neat::new(fixed_species_config(6)).unwrap();
    let results = search
        .generate()
        .unwrap()
        .into_iter()
        .zip([1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
        .map(|(job, output)| Evaluated { job, output })
        .collect();
    search.select(results).unwrap();
    assert_eq!(
        search
            .species()
            .iter()
            .map(|s| s.offspring)
            .collect::<Vec<_>>(),
        [1, 1, 2, 2, 0, 0]
    );
    assert_eq!(search.best().unwrap().output, 4.0);
    assert_eq!(search.generate().unwrap().len(), 6);
}

#[test]
fn malformed_selection_keeps_all_state_and_pending_work_unchanged() {
    let mut config = NeatConfig::new(2, 1, 7);
    config.population = 12;
    let mut search = Neat::new(config.clone()).unwrap();
    let mut control = Neat::new(config).unwrap();
    assert_eq!(
        search.select(vec![]),
        Err(NeatError::Protocol(SearchError::NoPendingBatch))
    );
    let fitness = |genome: &Genome| 1.0 + genome.connections()[0].weight.abs();
    let first = scored(search.generate().unwrap(), fitness);
    assert_eq!(first, scored(control.generate().unwrap(), fitness));
    search.select(first.clone()).unwrap();
    control.select(first).unwrap();
    let valid = scored(search.generate().unwrap(), fitness);
    assert_eq!(valid, scored(control.generate().unwrap(), fitness));
    let parents = search.parents().to_vec();
    let species = search.species().to_vec();
    let best = search.best().cloned();
    for corruption in 0..9 {
        let mut bad = valid.clone();
        match corruption {
            0 => {
                bad.pop();
            }
            1 => {
                bad.push(bad[0].clone());
            }
            2 => bad.swap(0, 1),
            3 => bad[0].job.seed ^= 1,
            4 => bad[1].job.id = bad[0].job.id,
            5 => {
                let genome = &bad[0].job.input;
                let mut links = genome.connections().to_vec();
                links[0].weight += 1.0;
                bad[0].job.input = Genome::new(genome.nodes().to_vec(), links).unwrap();
            }
            6 => bad[0].output = f64::NAN,
            7 => bad[0].output = f64::INFINITY,
            8 => bad[0].output = -1.0,
            _ => unreachable!(),
        }
        assert!(matches!(
            search.select(bad),
            Err(NeatError::Protocol(SearchError::InvalidResults(_)))
        ));
        assert_eq!(search.parents(), parents);
        assert_eq!(search.species(), species);
        assert_eq!(search.best(), best.as_ref());
        assert_eq!(search.batches(), 1);
        assert_eq!(search.evaluations(), 12);
        assert_eq!(
            search.generate(),
            Err(NeatError::Protocol(SearchError::PendingBatch))
        );
    }
    search.select(valid.clone()).unwrap();
    control.select(valid.clone()).unwrap();
    assert_eq!(
        search.select(valid),
        Err(NeatError::Protocol(SearchError::NoPendingBatch))
    );
    assert_eq!(search.parents(), control.parents());
    assert_eq!(search.species(), control.species());
    assert_eq!(search.generate().unwrap(), control.generate().unwrap());
}

#[test]
fn seeded_replay_preserves_structure_species_and_evaluation_counts() {
    let mut config = NeatConfig::new(2, 1, 21);
    config.population = 16;
    config.mutation.add_node_probability = 0.5;
    config.mutation.add_connection_probability = 0.5;
    let mut first = Neat::new(config.clone()).unwrap();
    let mut second = Neat::new(config).unwrap();
    let mut grew = false;
    for batch in 1..=12 {
        let jobs = first.generate().unwrap();
        assert_eq!(jobs, second.generate().unwrap());
        grew |= jobs.iter().any(|job| job.input.hidden_node_count() > 0);
        let results = scored(jobs, |genome| {
            genome.validate().unwrap();
            1.0 + genome.compile().unwrap().forward(&[0.0, 1.0]).unwrap()[0]
        });
        first.select(results.clone()).unwrap();
        second.select(results).unwrap();
        assert_eq!(first.species(), second.species());
        assert_eq!(first.best(), second.best());
        assert_eq!(
            first.species().iter().map(|s| s.offspring).sum::<usize>(),
            16
        );
        assert_eq!(first.evaluations(), 16 * batch);
    }
    assert!(grew);
}

#[test]
fn finite_extreme_fitness_does_not_overflow_sharing() {
    let mut search = Neat::new(fixed_species_config(6)).unwrap();
    let results = search
        .generate()
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, job)| Evaluated {
            job,
            output: if i % 2 == 0 { f64::MAX } else { f64::MAX / 2.0 },
        })
        .collect();
    search.select(results).unwrap();
    assert_eq!(
        search.species().iter().map(|s| s.offspring).sum::<usize>(),
        6
    );
    assert!(
        search
            .species()
            .iter()
            .all(|s| s.shared_fitness.is_finite())
    );
    assert_eq!(search.generate().unwrap().len(), 6);
}

#[test]
fn invalid_neat_configuration_is_rejected_before_search() {
    for invalid in 0..12 {
        let mut config = NeatConfig::new(2, 1, 1);
        match invalid {
            0 => config.inputs = 0,
            1 => config.outputs = 0,
            2 => config.population = 1,
            3 => config.compatibility_threshold = 0.0,
            4 => config.compatibility.weight = f64::NAN,
            5 => config.mutation.weight_bound = f64::INFINITY,
            6 => config.mutation.add_node_probability = 1.1,
            7 => config.crossover_probability = -0.1,
            8 => config.disable_probability = f64::NAN,
            9 => config.survival_fraction = 0.0,
            10 => config.stagnation_generations = 0,
            11 => config.champion_min_offspring = 0,
            _ => unreachable!(),
        }
        assert!(matches!(
            Neat::new(config),
            Err(NeatError::InvalidConfig(_))
        ));
    }
}

fn xor_outputs(genome: &Genome) -> [f64; 4] {
    let network = genome.compile().unwrap();
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        .map(|input| network.forward(&input).unwrap()[0])
}

#[test]
fn declared_seed_and_budget_evolve_hidden_structure_for_xor() {
    // This is a deterministic integration smoke case, not an estimated success
    // rate. The recorded companion experiment retains all ten seeds, including
    // failures; this test uses its declared seed 1 with an unchanged 300-batch cap.
    let mut search = Neat::new(NeatConfig::new(2, 1, 1)).unwrap();
    let targets = [0.0, 1.0, 1.0, 0.0];
    let mut solved = false;
    for _ in 0..300 {
        let results = scored(search.generate().unwrap(), |genome| {
            let error: f64 = xor_outputs(genome)
                .iter()
                .zip(targets)
                .map(|(&prediction, target)| (prediction - target).abs())
                .sum();
            (4.0 - error).powi(2)
        });
        search.select(results).unwrap();
        if xor_outputs(&search.best().unwrap().job.input)
            .iter()
            .zip(targets)
            .all(|(&prediction, target)| (prediction - target).abs() <= 0.1)
        {
            solved = true;
            break;
        }
    }
    assert!(solved, "seed 1 did not solve XOR within 300 batches");
    assert!(search.best().unwrap().job.input.hidden_node_count() > 0);
    assert_eq!(search.evaluations(), search.batches() * 150);
    assert!(search.evaluations() <= 45_000);
}
