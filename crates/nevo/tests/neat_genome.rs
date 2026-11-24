//! Historical identity, graph execution, and independent NEAT operator cases.

use nevo::neat::{
    CompatibilityConfig, ConnectionGene, Genome, InnovationTracker, MutationConfig, NeatError,
    NodeGene, NodeKind, add_connection, allocate_offspring, compatibility_distance, crossover,
    mutate, mutate_weights, split_connection,
};
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{RngCore, SeedableRng},
};

fn node(id: u64, kind: NodeKind) -> NodeGene {
    NodeGene { id, kind }
}
fn link(innovation: u64, from: u64, to: u64, weight: f64, enabled: bool) -> ConnectionGene {
    ConnectionGene {
        innovation,
        from,
        to,
        weight,
        enabled,
    }
}
fn base_nodes() -> Vec<NodeGene> {
    vec![
        node(0, NodeKind::Input),
        node(1, NodeKind::Bias),
        node(2, NodeKind::Output),
    ]
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-4.9 * x).exp())
}
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

// Crossover only uses uniform floating draws. Zero selects the first matching
// parent and includes each equal-fitness unmatched gene, making exact examples
// independent of a particular pseudo-random sample sequence.
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
fn minimal_graph_uses_shared_ids_bias_and_checked_stateless_execution() {
    let tracker = InnovationTracker::new(2, 1).unwrap();
    assert_eq!(tracker.initial_connection_count(), 3);
    let genome = tracker.initial_genome(vec![1.0, 2.0, -0.5]).unwrap();
    assert_eq!(
        genome.nodes(),
        [
            node(0, NodeKind::Input),
            node(1, NodeKind::Input),
            node(2, NodeKind::Bias),
            node(3, NodeKind::Output)
        ]
    );
    assert_eq!(genome.hidden_node_count(), 0);
    assert_eq!(genome.enabled_connection_count(), 3);
    let network = genome.compile().unwrap();
    assert_eq!(network.input_count(), 2);
    assert_eq!(network.output_count(), 1);
    assert_eq!(network.forward(&[0.5, 0.0]).unwrap(), [0.5]);
    close(network.forward(&[0.0, 0.0]).unwrap()[0], sigmoid(-0.5));
    assert_eq!(network.forward(&[0.5, 0.0]).unwrap(), [0.5]);
    assert!(network.forward(&[0.0]).is_err());
    assert!(network.forward(&[0.0, f64::NAN]).is_err());
    assert!(tracker.initial_genome(vec![0.0; 2]).is_err());
    assert!(
        tracker
            .initial_genome(vec![0.0, 1.0, f64::INFINITY])
            .is_err()
    );
}

#[test]
fn node_split_preserves_historical_gene_and_uses_one_then_original_weight() {
    let mut tracker = InnovationTracker::new(2, 1).unwrap();
    let mut genome = tracker.initial_genome(vec![2.5, 0.0, 0.0]).unwrap();
    assert!(split_connection(&mut genome, &mut tracker, 0).unwrap());
    assert_eq!(genome.hidden_node_count(), 1);
    assert_eq!(genome.connections().len(), 5);
    assert_eq!(genome.enabled_connection_count(), 4);
    assert_eq!(genome.connections()[0], link(0, 0, 3, 2.5, false));
    assert_eq!(genome.connections()[3], link(3, 0, 4, 1.0, true));
    assert_eq!(genome.connections()[4], link(4, 4, 3, 2.5, true));
    // New hidden ID 4 is greater than output ID 3, so evaluating by node ID
    // instead of topological order would give the wrong answer.
    close(
        genome.compile().unwrap().forward(&[1.0, 0.0]).unwrap()[0],
        sigmoid(2.5 * sigmoid(1.0)),
    );
    let before = genome.clone();
    assert!(!split_connection(&mut genome, &mut tracker, 0).unwrap());
    assert_eq!(genome, before);
}

#[test]
fn innovations_deduplicate_within_a_generation_and_remain_global_across_generations() {
    let mut tracker = InnovationTracker::new(2, 1).unwrap();
    let mut left = tracker.initial_genome(vec![2.0, 0.0, 0.0]).unwrap();
    let mut right = tracker.initial_genome(vec![-3.0, 0.0, 0.0]).unwrap();
    split_connection(&mut left, &mut tracker, 0).unwrap();
    split_connection(&mut right, &mut tracker, 0).unwrap();
    assert_eq!(left.nodes(), right.nodes());
    assert_eq!(
        left.connections()[3].innovation,
        right.connections()[3].innovation
    );
    assert_eq!(
        left.connections()[4].innovation,
        right.connections()[4].innovation
    );
    assert_eq!(left.connections()[4].weight, 2.0);
    assert_eq!(right.connections()[4].weight, -3.0);
    let before_link = left.clone();
    assert!(add_connection(&mut left, &mut tracker, 1, 4, 0.5).unwrap());
    assert!(add_connection(&mut right, &mut tracker, 1, 4, -0.5).unwrap());
    assert_eq!(
        left.connections().last().unwrap().innovation,
        right.connections().last().unwrap().innovation
    );
    let old_innovation = left.connections().last().unwrap().innovation;
    tracker.begin_generation();
    let mut later = before_link;
    assert!(add_connection(&mut later, &mut tracker, 1, 4, 1.0).unwrap());
    assert!(later.connections().last().unwrap().innovation > old_innovation);
    let mut later_split = tracker.initial_genome(vec![1.0; 3]).unwrap();
    split_connection(&mut later_split, &mut tracker, 0).unwrap();
    assert!(later_split.nodes().last().unwrap().id > 4);
    assert!(
        later_split.connections()[3].innovation > later.connections().last().unwrap().innovation
    );
    tracker.validate_genome(&left).unwrap();
    tracker.validate_genome(&later_split).unwrap();
}

#[test]
fn add_connection_respects_roles_duplicates_cycles_and_transactional_validation() {
    let mut tracker = InnovationTracker::new(2, 1).unwrap();
    let mut genome = tracker.initial_genome(vec![1.0; 3]).unwrap();
    split_connection(&mut genome, &mut tracker, 0).unwrap();
    split_connection(&mut genome, &mut tracker, 1).unwrap();
    // The two independent hidden nodes are IDs 4 and 5.
    assert!(add_connection(&mut genome, &mut tracker, 4, 5, 0.25).unwrap());
    let before = genome.clone();
    for (from, to) in [(5, 4), (4, 4), (3, 4), (4, 0), (0, 3), (4, 5)] {
        assert!(!add_connection(&mut genome, &mut tracker, from, to, 1.0).unwrap());
        assert_eq!(genome, before);
    }
    assert!(add_connection(&mut genome, &mut tracker, 1, 4, f64::NAN).is_err());
    assert!(add_connection(&mut genome, &mut tracker, 999, 4, 0.0).is_err());
    assert!(split_connection(&mut genome, &mut tracker, u64::MAX).is_err());
    assert_eq!(genome, before);
    tracker.validate_genome(&genome).unwrap();
}

#[test]
fn validation_rejects_bad_identity_endpoints_weights_and_enabled_cycles() {
    let nodes = base_nodes();
    let valid = vec![link(0, 0, 2, 1.0, true)];
    assert!(
        Genome::new(
            vec![
                nodes[0].clone(),
                nodes[0].clone(),
                nodes[1].clone(),
                nodes[2].clone()
            ],
            valid.clone()
        )
        .is_err()
    );
    assert!(Genome::new(nodes[..2].to_vec(), vec![]).is_err());
    assert!(
        Genome::new(
            nodes.clone(),
            vec![link(0, 0, 2, 0.0, true), link(0, 1, 2, 1.0, true)]
        )
        .is_err()
    );
    assert!(
        Genome::new(
            nodes.clone(),
            vec![link(0, 0, 2, 0.0, true), link(1, 0, 2, 1.0, false)]
        )
        .is_err()
    );
    for (from, to, weight) in [
        (0, 99, 1.0),
        (0, 2, f64::NAN),
        (0, 2, f64::INFINITY),
        (2, 0, 1.0),
        (1, 0, 1.0),
        (0, 1, 1.0),
    ] {
        assert!(Genome::new(nodes.clone(), vec![link(0, from, to, weight, true)]).is_err());
    }
    let mut nodes = nodes;
    nodes.extend([node(3, NodeKind::Hidden), node(4, NodeKind::Hidden)]);
    let mut links = vec![
        link(0, 0, 3, 1.0, true),
        link(1, 3, 4, 1.0, true),
        link(2, 4, 3, 1.0, true),
    ];
    assert!(Genome::new(nodes.clone(), links.clone()).is_err());
    links[2].enabled = false;
    assert!(Genome::new(nodes, links).is_ok());
}

#[test]
fn equality_checks_weight_bits_and_network_arithmetic_fails_explicitly() {
    let tracker = InnovationTracker::new(1, 1).unwrap();
    let positive_zero = tracker.initial_genome(vec![0.0, 0.0]).unwrap();
    let negative_zero = tracker.initial_genome(vec![-0.0, 0.0]).unwrap();
    assert_ne!(positive_zero, negative_zero);
    let huge = tracker
        .initial_genome(vec![f64::MAX, 0.0])
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(huge.forward(&[1.0]).unwrap(), [1.0]);
    assert!(matches!(huge.forward(&[2.0]), Err(NeatError::Numerical(_))));
}

#[test]
fn compatibility_matches_hand_calculated_excess_disjoint_and_disabled_weight_terms() {
    let mut nodes = base_nodes();
    nodes.push(node(3, NodeKind::Hidden));
    let left = Genome::new(
        nodes.clone(),
        vec![
            link(1, 0, 3, 1.0, false),
            link(2, 1, 3, 2.0, true),
            link(4, 3, 2, 3.0, true),
        ],
    )
    .unwrap();
    let right = Genome::new(
        nodes,
        vec![
            link(1, 0, 3, 3.0, true),
            link(2, 1, 3, 4.0, true),
            link(3, 0, 2, 0.0, true),
            link(5, 1, 2, 0.0, true),
        ],
    )
    .unwrap();
    // E=1 (5), D=2 (3,4), mean matching weight difference=2, N=1.
    let config = CompatibilityConfig::default();
    close(compatibility_distance(&left, &right, &config).unwrap(), 3.8);
    close(compatibility_distance(&right, &left, &config).unwrap(), 3.8);
    assert_eq!(compatibility_distance(&left, &left, &config).unwrap(), 0.0);
    let normalized = CompatibilityConfig {
        small_genome_threshold: 0,
        ..config
    };
    close(
        compatibility_distance(&left, &right, &normalized).unwrap(),
        1.55,
    );
    let empty = Genome::new(base_nodes(), vec![]).unwrap();
    let single = Genome::new(base_nodes(), vec![link(10, 0, 2, 1.0, true)]).unwrap();
    close(
        compatibility_distance(&empty, &single, &CompatibilityConfig::default()).unwrap(),
        1.0,
    );
}

#[test]
fn crossover_uses_homology_fitter_unmatched_genes_and_disabled_inheritance() {
    let left = Genome::new(
        base_nodes(),
        vec![link(0, 0, 2, 1.0, false), link(1, 1, 2, 3.0, true)],
    )
    .unwrap();
    let mut right_nodes = base_nodes();
    right_nodes.push(node(3, NodeKind::Hidden));
    let right = Genome::new(
        right_nodes,
        vec![
            link(0, 0, 2, 2.0, true),
            link(2, 0, 3, 4.0, true),
            link(3, 3, 2, 5.0, true),
        ],
    )
    .unwrap();
    let child = crossover(&left, 2.0, &right, 1.0, 1.0, &mut ZeroDraws).unwrap();
    assert_eq!(child.connections(), left.connections());
    assert_eq!(child.hidden_node_count(), 0);
    let enabled = crossover(&left, 2.0, &right, 1.0, 0.0, &mut ZeroDraws).unwrap();
    assert!(enabled.connections()[0].enabled);
    assert_eq!(enabled.connections()[0].weight, 1.0);
    let other_fitter = crossover(&left, 1.0, &right, 2.0, 1.0, &mut ZeroDraws).unwrap();
    assert_eq!(
        other_fitter
            .connections()
            .iter()
            .map(|gene| gene.innovation)
            .collect::<Vec<_>>(),
        [0, 2, 3]
    );
    assert_eq!(other_fitter.hidden_node_count(), 1);
    let equal = crossover(&left, 1.0, &right, 1.0, 1.0, &mut ZeroDraws).unwrap();
    assert_eq!(
        equal
            .connections()
            .iter()
            .map(|gene| gene.innovation)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
}

#[test]
fn crossover_guards_cycles_created_by_dag_union_or_reenabling() {
    let mut nodes = base_nodes();
    nodes.extend([node(3, NodeKind::Hidden), node(4, NodeKind::Hidden)]);
    let common = vec![
        link(1, 0, 3, 1.0, true),
        link(2, 0, 4, 1.0, true),
        link(3, 3, 2, 1.0, true),
        link(4, 4, 2, 1.0, true),
    ];
    let mut left_links = common.clone();
    left_links.push(link(5, 3, 4, 1.0, true));
    let mut right_links = common;
    right_links.push(link(6, 4, 3, 1.0, true));
    let left = Genome::new(nodes.clone(), left_links.clone()).unwrap();
    let right = Genome::new(nodes.clone(), right_links.clone()).unwrap();
    let child = crossover(&left, 1.0, &right, 1.0, 0.0, &mut ZeroDraws).unwrap();
    assert!(
        child
            .connections()
            .iter()
            .find(|gene| gene.innovation == 5)
            .unwrap()
            .enabled
    );
    assert!(
        !child
            .connections()
            .iter()
            .find(|gene| gene.innovation == 6)
            .unwrap()
            .enabled
    );
    child.compile().unwrap();
    left_links.push(link(6, 4, 3, 1.0, false));
    right_links.push(link(5, 3, 4, 1.0, false));
    let left = Genome::new(nodes.clone(), left_links).unwrap();
    let right = Genome::new(nodes, right_links).unwrap();
    let child = crossover(&left, 2.0, &right, 1.0, 0.0, &mut ZeroDraws).unwrap();
    assert_eq!(child.enabled_connection_count(), 5);
    assert!(!child.connections().last().unwrap().enabled);
}

#[test]
fn crossover_preserves_first_historical_id_for_duplicate_endpoint_pairs() {
    let left = Genome::new(base_nodes(), vec![link(3, 0, 2, 1.0, true)]).unwrap();
    let right = Genome::new(base_nodes(), vec![link(7, 0, 2, 2.0, true)]).unwrap();
    let child = crossover(&left, 1.0, &right, 1.0, 0.75, &mut ZeroDraws).unwrap();
    assert_eq!(child.connections(), left.connections());
    let conflicting = Genome::new(base_nodes(), vec![link(3, 1, 2, 2.0, true)]).unwrap();
    assert!(matches!(
        crossover(&left, 1.0, &conflicting, 1.0, 0.75, &mut ZeroDraws),
        Err(NeatError::IncompatibleHistory(_))
    ));
    assert!(compatibility_distance(&left, &conflicting, &CompatibilityConfig::default()).is_err());
}

#[test]
fn mutation_is_replayable_and_preserves_valid_histories() {
    let mut first_tracker = InnovationTracker::new(2, 1).unwrap();
    let mut second_tracker = first_tracker.clone();
    let mut first = first_tracker.initial_genome(vec![0.0; 3]).unwrap();
    let mut second = first.clone();
    let mut first_rng = ChaCha8Rng::seed_from_u64(42);
    let mut second_rng = first_rng.clone();
    let config = MutationConfig {
        add_node_probability: 1.0,
        add_connection_probability: 1.0,
        weight_probability: 1.0,
        ..MutationConfig::default()
    };
    for _ in 0..6 {
        first_tracker.begin_generation();
        second_tracker.begin_generation();
        mutate(&mut first, &mut first_tracker, &config, &mut first_rng).unwrap();
        mutate(&mut second, &mut second_tracker, &config, &mut second_rng).unwrap();
        assert_eq!(first, second);
        first_tracker.validate_genome(&first).unwrap();
        first.compile().unwrap().forward(&[0.0, 1.0]).unwrap();
    }
    assert_eq!(first.hidden_node_count(), 6);
    let before = first.clone();
    let invalid = MutationConfig {
        perturb_probability: f64::NAN,
        ..config
    };
    assert!(mutate_weights(&mut first, &invalid, &mut first_rng).is_err());
    assert_eq!(first, before);
}

#[test]
fn largest_remainders_preserve_budget_sharing_and_all_zero_fallback() {
    assert_eq!(allocate_offspring(&[1.0, 1.0, 1.0], 8).unwrap(), [3, 3, 2]);
    assert_eq!(allocate_offspring(&[0.0, 1.0, 3.0], 7).unwrap(), [0, 2, 5]);
    assert_eq!(allocate_offspring(&[0.0, 0.0, 0.0], 8).unwrap(), [3, 3, 2]);
    assert_eq!(
        allocate_offspring(&[f64::MAX, f64::MAX / 2.0], 9).unwrap(),
        [6, 3]
    );
    for (weights, population) in [
        (&[][..], 10),
        (&[1.0][..], 0),
        (&[-1.0][..], 10),
        (&[f64::NAN][..], 10),
        (&[f64::INFINITY][..], 10),
    ] {
        assert!(allocate_offspring(weights, population).is_err());
    }
}
