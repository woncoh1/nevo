//! Synchronous recurrence, state isolation, structural history, and GES identity.

use nevo::algorithms::{Neat, NeatConfig, SearchError};
use nevo::neat::{
    CompatibilityConfig, ConnectionGene, Genome, InnovationTracker, MutationConfig, NeatError,
    NetworkMode, NodeGene, NodeKind, add_connection, compatibility_distance, crossover, mutate,
    split_connection,
};
use nevo::{Algorithm, Evaluated};
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{RngCore, SeedableRng},
};

fn node(id: u64, kind: NodeKind) -> NodeGene {
    NodeGene { id, kind }
}

fn link(innovation: u64, from: u64, to: u64, weight: f64) -> ConnectionGene {
    ConnectionGene {
        innovation,
        from,
        to,
        weight,
        enabled: true,
    }
}

fn base_nodes() -> Vec<NodeGene> {
    vec![
        node(0, NodeKind::Input),
        node(1, NodeKind::Bias),
        node(2, NodeKind::Output),
    ]
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-4.9 * value).exp())
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

struct ZeroDraws;

impl RngCore for ZeroDraws {
    fn next_u32(&mut self) -> u32 {
        0
    }
    fn next_u64(&mut self) -> u64 {
        0
    }
    fn fill_bytes(&mut self, destination: &mut [u8]) {
        destination.fill(0);
    }
}

#[test]
fn output_self_loop_retains_a_pulse_until_explicit_episode_reset() {
    let genome = Genome::new_recurrent(
        base_nodes(),
        vec![link(0, 0, 2, 2.0), link(1, 1, 2, -1.0), link(2, 2, 2, 2.5)],
    )
    .unwrap();
    let mut network = genome.compile_recurrent().unwrap();
    assert_eq!(network.input_count(), 1);
    assert_eq!(network.output_count(), 1);
    let first = network.step(&[1.0]).unwrap()[0];
    close(first, sigmoid(1.0));
    let second = network.step(&[0.0]).unwrap()[0];
    close(second, sigmoid(-1.0 + 2.5 * first));
    assert!(second > 0.99);
    let mut cloned_state = network.clone();
    assert_eq!(network.step(&[0.0]), cloned_state.step(&[0.0]));

    network.reset();
    let fresh = network.step(&[0.0]).unwrap()[0];
    close(fresh, sigmoid(-1.0));
    assert!(fresh < 0.01);
    assert_eq!(
        fresh,
        genome.compile_recurrent().unwrap().step(&[0.0]).unwrap()[0]
    );
    // Running a phenotype never places episode state inside its genotype.
    assert_eq!(genome.mode(), NetworkMode::Recurrent);
    assert_eq!(genome.connections()[2].weight, 2.5);
}

#[test]
fn external_inputs_are_current_but_every_neuron_source_is_delayed_one_tick() {
    let mut nodes = base_nodes();
    nodes.push(node(3, NodeKind::Hidden));
    let genome =
        Genome::new_recurrent(nodes, vec![link(0, 0, 3, 1.0), link(1, 3, 2, 2.0)]).unwrap();
    let mut network = genome.compile_recurrent().unwrap();
    assert_eq!(network.step(&[1.0]).unwrap(), [0.5]);
    close(
        network.step(&[0.0]).unwrap()[0],
        sigmoid(2.0 * sigmoid(1.0)),
    );
    close(network.step(&[-1.0]).unwrap()[0], sigmoid(2.0 * 0.5));
    network.reset();
    assert_eq!(network.step(&[-1.0]).unwrap(), [0.5]);
}

#[test]
fn synchronous_values_do_not_depend_on_neuron_ids_or_input_gene_storage_order() {
    let mut nodes = base_nodes();
    nodes.extend([node(3, NodeKind::Hidden), node(4, NodeKind::Hidden)]);
    let links = vec![
        link(0, 0, 3, 1.0),
        link(1, 1, 3, -0.1),
        link(2, 3, 4, 0.7),
        link(3, 4, 2, -0.5),
        link(4, 2, 3, 0.8),
        link(5, 1, 2, 0.2),
        link(6, 4, 4, 0.9),
    ];
    let mut original = Genome::new_recurrent(nodes.clone(), links.clone())
        .unwrap()
        .compile_recurrent()
        .unwrap();
    // Original output ID 2 is updated first. Renamed output ID 4 is last.
    let rename = |id| match id {
        2 => 4,
        3 => 2,
        4 => 3,
        other => other,
    };
    let renamed_nodes = nodes
        .into_iter()
        .rev()
        .map(|gene| node(rename(gene.id), gene.kind))
        .collect();
    let renamed_links = links
        .into_iter()
        .rev()
        .map(|gene| {
            link(
                gene.innovation,
                rename(gene.from),
                rename(gene.to),
                gene.weight,
            )
        })
        .collect();
    let mut renamed = Genome::new_recurrent(renamed_nodes, renamed_links)
        .unwrap()
        .compile_recurrent()
        .unwrap();
    for input in [1.0, 0.0, -1.0, 0.2, 0.7, 0.0] {
        assert_eq!(original.step(&[input]), renamed.step(&[input]));
    }
}

#[test]
fn recurrence_allows_cycles_and_output_feedback_but_never_input_or_bias_destinations() {
    let mut nodes = base_nodes();
    nodes.push(node(3, NodeKind::Hidden));
    let links = vec![link(0, 2, 3, 0.5), link(1, 3, 2, 0.5), link(2, 3, 3, 0.5)];
    assert!(Genome::new_recurrent(nodes.clone(), links.clone()).is_ok());
    assert!(Genome::new(nodes.clone(), links).is_err());
    for destination in [0, 1] {
        for enabled in [true, false] {
            let mut invalid = link(0, 2, destination, 0.5);
            invalid.enabled = enabled;
            assert!(Genome::new_recurrent(nodes.clone(), vec![invalid]).is_err());
        }
    }
    assert!(Genome::new_recurrent(nodes.clone(), vec![link(0, 2, 99, 0.5)]).is_err());
    assert!(Genome::new_recurrent(nodes.clone(), vec![link(0, 2, 3, f64::NAN)]).is_err());
    assert!(Genome::new_recurrent(nodes, vec![link(0, 2, 3, 0.5), link(1, 2, 3, 0.6)]).is_err());
}

#[test]
fn phenotype_compilation_and_historical_operators_require_the_explicit_matching_mode() {
    let feedforward = InnovationTracker::new(1, 1).unwrap();
    let recurrent = InnovationTracker::new_recurrent(1, 1).unwrap();
    assert_eq!(feedforward.mode(), NetworkMode::FeedForward);
    assert_eq!(recurrent.mode(), NetworkMode::Recurrent);
    let direct = feedforward.initial_genome(vec![0.5, -0.2]).unwrap();
    let temporal = recurrent.initial_genome(vec![0.5, -0.2]).unwrap();
    assert_eq!(direct.nodes(), temporal.nodes());
    assert_eq!(direct.connections(), temporal.connections());
    assert_ne!(direct, temporal);
    assert!(direct.compile_recurrent().is_err());
    assert!(temporal.compile().is_err());
    assert!(feedforward.validate_genome(&temporal).is_err());
    assert!(recurrent.validate_genome(&direct).is_err());
    assert!(compatibility_distance(&direct, &temporal, &CompatibilityConfig::default()).is_err());
    assert!(crossover(&direct, 1.0, &temporal, 1.0, 0.75, &mut ZeroDraws).is_err());
    let direct = direct.compile().unwrap();
    let mut temporal = temporal.compile_recurrent().unwrap();
    for input in [1.0, 0.0, -1.0, 0.25] {
        assert_eq!(direct.forward(&[input]), temporal.step(&[input]));
    }
}

#[test]
fn failed_tick_does_not_commit_earlier_neuron_updates_or_corrupt_the_next_tick() {
    let genome = Genome::new_recurrent(
        vec![
            node(0, NodeKind::Input),
            node(1, NodeKind::Input),
            node(2, NodeKind::Bias),
            node(3, NodeKind::Hidden),
            node(10, NodeKind::Output),
        ],
        vec![
            link(0, 0, 3, 0.5),
            link(1, 10, 3, 0.9),
            link(2, 2, 3, -0.1),
            link(3, 3, 10, 0.3),
            link(4, 1, 10, f64::MAX),
        ],
    )
    .unwrap();
    let mut network = genome.compile_recurrent().unwrap();
    network.step(&[0.5, 0.0]).unwrap();
    let mut control = network.clone();
    // Hidden node 3 is computed before output 10 overflows on the second input.
    assert!(matches!(
        network.step(&[10.0, 2.0]),
        Err(NeatError::Numerical(_))
    ));
    assert!(network.step(&[0.0]).is_err());
    assert!(network.step(&[f64::NAN, 0.0]).is_err());
    assert!(network.step(&[0.0, f64::INFINITY]).is_err());
    for input in [[0.25, 0.0], [-0.5, 0.0], [0.0, 1.0], [0.0, 0.0]] {
        assert_eq!(network.step(&input), control.step(&input));
    }
    network.reset();
    assert_eq!(
        network.step(&[0.5, 0.0]),
        genome.compile_recurrent().unwrap().step(&[0.5, 0.0])
    );
}

#[test]
fn feedback_addition_and_splitting_preserve_mode_and_generation_local_innovations() {
    let mut tracker = InnovationTracker::new_recurrent(1, 1).unwrap();
    let base = tracker.initial_genome(vec![1.0, 0.0]).unwrap();
    let mut left = base.clone();
    let mut right = base.clone();
    for genome in [&mut left, &mut right] {
        assert!(add_connection(genome, &mut tracker, 2, 2, 0.7).unwrap());
        assert!(!add_connection(genome, &mut tracker, 2, 0, 0.7).unwrap());
        assert!(!add_connection(genome, &mut tracker, 2, 1, 0.7).unwrap());
        assert!(!add_connection(genome, &mut tracker, 2, 2, 0.7).unwrap());
        assert!(split_connection(genome, &mut tracker, 2).unwrap());
        assert_eq!(genome.mode(), NetworkMode::Recurrent);
        assert_eq!(genome.connections()[2].innovation, 2);
        assert!(!genome.connections()[2].enabled);
        assert_eq!(genome.connections()[3], link(3, 2, 3, 1.0));
        assert_eq!(genome.connections()[4], link(4, 3, 2, 0.7));
        tracker.validate_genome(genome).unwrap();
        genome.compile_recurrent().unwrap().step(&[1.0]).unwrap();
    }
    assert_eq!(left, right);
    assert!(add_connection(&mut left, &mut tracker, 3, 3, 0.2).unwrap());
    tracker.begin_generation();
    let mut later = base;
    assert!(add_connection(&mut later, &mut tracker, 2, 2, 0.7).unwrap());
    assert!(later.connections()[2].innovation > left.connections()[5].innovation);
    tracker.validate_genome(&left).unwrap();
    tracker.validate_genome(&later).unwrap();
}

#[test]
fn recurrent_crossover_can_reenable_inherited_feedback_without_discarding_cycles() {
    let mut tracker = InnovationTracker::new_recurrent(1, 1).unwrap();
    let mut split = tracker.initial_genome(vec![1.0, 0.0]).unwrap();
    add_connection(&mut split, &mut tracker, 2, 2, 0.7).unwrap();
    let unsplit = split.clone();
    split_connection(&mut split, &mut tracker, 2).unwrap();
    let child = crossover(&split, 2.0, &unsplit, 1.0, 0.0, &mut ZeroDraws).unwrap();
    assert_eq!(child.mode(), NetworkMode::Recurrent);
    assert_eq!(child.nodes(), split.nodes());
    assert_eq!(child.connections().len(), 5);
    assert!(child.connections().iter().all(|gene| gene.enabled));
    assert_eq!(child.connections()[2], link(2, 2, 2, 0.7));
    assert_eq!(child.connections()[3], link(3, 2, 3, 1.0));
    assert_eq!(child.connections()[4], link(4, 3, 2, 0.7));
    tracker.validate_genome(&child).unwrap();
    child.compile_recurrent().unwrap().step(&[1.0]).unwrap();
    let disabled = crossover(&split, 2.0, &unsplit, 1.0, 1.0, &mut ZeroDraws).unwrap();
    assert!(!disabled.connections()[2].enabled);
}

#[test]
fn seeded_recurrent_mutation_replays_and_minimal_graph_can_gain_an_output_self_loop() {
    let mut first_tracker = InnovationTracker::new_recurrent(1, 1).unwrap();
    let mut second_tracker = first_tracker.clone();
    let mut first = first_tracker.initial_genome(vec![1.0, 0.0]).unwrap();
    let mut second = first.clone();
    let mut first_rng = ChaCha8Rng::seed_from_u64(27);
    let mut second_rng = first_rng.clone();
    let mut config = MutationConfig {
        weight_probability: 0.0,
        add_node_probability: 0.0,
        add_connection_probability: 1.0,
        ..MutationConfig::default()
    };
    for generation in 0..12 {
        first_tracker.begin_generation();
        second_tracker.begin_generation();
        mutate(&mut first, &mut first_tracker, &config, &mut first_rng).unwrap();
        mutate(&mut second, &mut second_tracker, &config, &mut second_rng).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.mode(), NetworkMode::Recurrent);
        first_tracker.validate_genome(&first).unwrap();
        first.compile_recurrent().unwrap().step(&[0.5]).unwrap();
        if generation == 0 {
            assert_eq!(
                (first.connections()[2].from, first.connections()[2].to),
                (2, 2)
            );
        }
        if generation == 1 {
            // No missing endpoint pairs remain before hidden nodes are added.
            assert_eq!(first.connections().len(), 3);
            config.add_node_probability = 0.7;
            config.weight_probability = 0.8;
        }
    }
    assert!(first.hidden_node_count() > 0);
    assert_eq!(first_rng.next_u64(), second_rng.next_u64());
}

#[test]
fn recurrent_search_replays_and_select_rejects_mode_substitution_without_losing_work() {
    let mut config = NeatConfig::new(1, 1, 99);
    assert_eq!(config.mode, NetworkMode::FeedForward);
    config.mode = NetworkMode::Recurrent;
    config.population = 16;
    config.mutation.add_node_probability = 0.3;
    config.mutation.add_connection_probability = 1.0;
    let mut first = Neat::new(config.clone()).unwrap();
    let mut second = Neat::new(config).unwrap();
    for generation in 0..6 {
        let jobs = first.generate().unwrap();
        assert_eq!(jobs, second.generate().unwrap());
        assert!(
            jobs.iter()
                .all(|job| job.input.mode() == NetworkMode::Recurrent)
        );
        let valid: Vec<_> = jobs
            .into_iter()
            .map(|job| {
                let output = 1.0 + job.input.enabled_connection_count() as f64;
                Evaluated { job, output }
            })
            .collect();
        if generation == 0 {
            let mut changed = valid.clone();
            let genome = &changed[0].job.input;
            changed[0].job.input =
                Genome::new(genome.nodes().to_vec(), genome.connections().to_vec()).unwrap();
            assert!(matches!(
                first.select(changed),
                Err(NeatError::Protocol(SearchError::InvalidResults(_)))
            ));
            assert_eq!(first.batches(), 0);
            assert!(first.parents().is_empty());
            assert!(first.best().is_none());
            assert_eq!(
                first.generate(),
                Err(NeatError::Protocol(SearchError::PendingBatch))
            );
        }
        first.select(valid.clone()).unwrap();
        second.select(valid).unwrap();
        assert_eq!(first.parents(), second.parents());
        assert_eq!(first.species(), second.species());
        assert_eq!(first.best(), second.best());
    }
    assert_eq!(first.evaluations(), 96);
    assert!(first.parents().iter().any(|parent| {
        parent
            .job
            .input
            .connections()
            .iter()
            .any(|gene| gene.enabled && gene.from == gene.to)
    }));
}
