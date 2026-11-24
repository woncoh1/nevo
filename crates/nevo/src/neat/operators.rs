use std::collections::{BTreeMap, BTreeSet};

use rand_chacha::rand_core::RngCore;

use super::genome::creates_cycle;
use super::{ConnectionGene, Genome, InnovationTracker, NeatError, NodeGene, NodeKind};
use crate::algorithms::common::{symmetric, uniform_index, uniform01};

/// Coefficients for excess/disjoint counts and mean matching-weight difference.
#[derive(Clone, Debug, PartialEq)]
pub struct CompatibilityConfig {
    /// Coefficient on the number of excess genes.
    pub excess: f64,
    /// Coefficient on the number of disjoint genes.
    pub disjoint: f64,
    /// Coefficient on the mean absolute weight difference of matching genes.
    pub weight: f64,
    /// Use normalization 1 when both genomes have fewer than this many genes.
    pub small_genome_threshold: usize,
}

impl Default for CompatibilityConfig {
    fn default() -> Self {
        Self {
            excess: 1.0,
            disjoint: 1.0,
            weight: 0.4,
            small_genome_threshold: 20,
        }
    }
}

impl CompatibilityConfig {
    pub(crate) fn validate(&self) -> Result<(), NeatError> {
        if [self.excess, self.disjoint, self.weight]
            .iter()
            .any(|c| !c.is_finite() || *c < 0.0)
        {
            return Err(NeatError::InvalidConfig(
                "compatibility coefficients must be finite and nonnegative",
            ));
        }
        Ok(())
    }
}

/// Explicit mutation choices for this scalar feedforward NEAT variant.
#[derive(Clone, Debug, PartialEq)]
pub struct MutationConfig {
    /// Probability of changing the genome's connection weights.
    pub weight_probability: f64,
    /// Given weight mutation, probability of perturbing each weight rather than resetting it.
    pub perturb_probability: f64,
    /// Half-width of a uniform additive weight perturbation; zero is allowed.
    pub perturb_bound: f64,
    /// Half-width for initial, reset, and newly added connection weights.
    pub weight_bound: f64,
    /// Independent per-child probability of splitting an enabled connection.
    pub add_node_probability: f64,
    /// Independent per-child probability of adding one admissible connection.
    pub add_connection_probability: f64,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            weight_probability: 0.8,
            perturb_probability: 0.9,
            perturb_bound: 0.5,
            weight_bound: 1.0,
            add_node_probability: 0.03,
            add_connection_probability: 0.05,
        }
    }
}

impl MutationConfig {
    pub(crate) fn validate(&self) -> Result<(), NeatError> {
        if [
            self.weight_probability,
            self.perturb_probability,
            self.add_node_probability,
            self.add_connection_probability,
        ]
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return Err(NeatError::InvalidConfig(
                "mutation probabilities must lie in [0,1]",
            ));
        }
        if !self.weight_bound.is_finite()
            || self.weight_bound <= 0.0
            || !(2.0 * self.weight_bound).is_finite()
            || !self.perturb_bound.is_finite()
            || self.perturb_bound < 0.0
            || !(2.0 * self.perturb_bound).is_finite()
        {
            return Err(NeatError::InvalidConfig(
                "weight bound must be positive, perturbation bound nonnegative, and twice each bound finite",
            ));
        }
        Ok(())
    }
}

fn compatible_history(left: &Genome, right: &Genome) -> Result<(), NeatError> {
    left.validate()?;
    right.validate()?;
    let left_base: Vec<_> = left
        .nodes
        .iter()
        .filter(|node| node.kind != NodeKind::Hidden)
        .collect();
    let right_base: Vec<_> = right
        .nodes
        .iter()
        .filter(|node| node.kind != NodeKind::Hidden)
        .collect();
    if left_base != right_base {
        return Err(NeatError::IncompatibleHistory(
            "parents have different input, bias, or output nodes",
        ));
    }
    let nodes: BTreeMap<_, _> = left.nodes.iter().map(|node| (node.id, node.kind)).collect();
    for node in &right.nodes {
        if nodes.get(&node.id).is_some_and(|&kind| kind != node.kind) {
            return Err(NeatError::IncompatibleHistory(
                "shared node IDs have different roles",
            ));
        }
    }
    let links: BTreeMap<_, _> = left
        .connections
        .iter()
        .map(|link| (link.innovation, (link.from, link.to)))
        .collect();
    for link in &right.connections {
        if links
            .get(&link.innovation)
            .is_some_and(|&endpoints| endpoints != (link.from, link.to))
        {
            return Err(NeatError::IncompatibleHistory(
                "matching innovations have different endpoints",
            ));
        }
    }
    Ok(())
}

/// Historical NEAT distance `cE*E/N + cD*D/N + cW*mean(abs(weight differences))`.
///
/// Disabled genes are included. `N` is the larger gene count, except that small
/// genomes use 1. With no matching genes the weight-difference term is zero.
pub fn compatibility_distance(
    left: &Genome,
    right: &Genome,
    config: &CompatibilityConfig,
) -> Result<f64, NeatError> {
    config.validate()?;
    compatible_history(left, right)?;
    let mut i = 0;
    let mut j = 0;
    let mut disjoint = 0;
    let mut differences = Vec::new();
    while i < left.connections.len() && j < right.connections.len() {
        let a = &left.connections[i];
        let b = &right.connections[j];
        match a.innovation.cmp(&b.innovation) {
            std::cmp::Ordering::Less => {
                disjoint += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                disjoint += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                let difference = (a.weight - b.weight).abs();
                if !difference.is_finite() {
                    return Err(NeatError::Numerical(
                        "matching-weight difference overflowed",
                    ));
                }
                differences.push(difference);
                i += 1;
                j += 1;
            }
        }
    }
    let excess = left.connections.len() - i + right.connections.len() - j;
    let size = left.connections.len().max(right.connections.len());
    let normalization = if size < config.small_genome_threshold {
        1
    } else {
        size.max(1)
    } as f64;
    let mean = if differences.is_empty() {
        0.0
    } else {
        let scale = differences.iter().copied().fold(0.0_f64, f64::max);
        if scale == 0.0 {
            0.0
        } else {
            (differences.iter().map(|d| d / scale).sum::<f64>() / differences.len() as f64).min(1.0)
                * scale
        }
    };
    let distance = config.excess * excess as f64 / normalization
        + config.disjoint * disjoint as f64 / normalization
        + config.weight * mean;
    if !distance.is_finite() {
        return Err(NeatError::Numerical("compatibility distance overflowed"));
    }
    Ok(distance)
}

/// Add a new endpoint pair without changing existing historical markings.
///
/// Returns `false` when the pair already exists (even if disabled), targets an
/// input/bias, leaves an output, is a self-link, or would create a cycle.
/// Unknown endpoints and invalid weights are errors. On error the genome and
/// tracker remain unchanged.
pub fn add_connection(
    genome: &mut Genome,
    tracker: &mut InnovationTracker,
    from: u64,
    to: u64,
    weight: f64,
) -> Result<bool, NeatError> {
    tracker.validate_genome(genome)?;
    if !weight.is_finite() {
        return Err(NeatError::InvalidGenome(
            "new connection weight must be finite",
        ));
    }
    let from_node = genome
        .nodes
        .iter()
        .find(|node| node.id == from)
        .ok_or(NeatError::InvalidGenome("unknown connection source"))?;
    let to_node = genome
        .nodes
        .iter()
        .find(|node| node.id == to)
        .ok_or(NeatError::InvalidGenome("unknown connection destination"))?;
    if from_node.kind == NodeKind::Output
        || matches!(to_node.kind, NodeKind::Input | NodeKind::Bias)
        || from == to
        || genome
            .connections
            .iter()
            .any(|link| (link.from, link.to) == (from, to))
        || creates_cycle(&genome.connections, from, to)
    {
        return Ok(false);
    }
    let mut next_tracker = tracker.clone();
    let innovation = next_tracker.connection(from, to)?;
    let mut links = genome.connections.clone();
    links.push(ConnectionGene {
        innovation,
        from,
        to,
        weight,
        enabled: true,
    });
    let next = Genome::new(genome.nodes.clone(), links)?;
    next_tracker.validate_genome(&next)?;
    *genome = next;
    *tracker = next_tracker;
    Ok(true)
}

/// Split an enabled connection: disable it, then insert weights `1` and `old_weight`.
///
/// Returns `false` if the specified gene is already disabled. A repeated split
/// in another genome during this generation reuses the new node and two
/// innovations. The new sigmoid changes the function; this is not an exact
/// function-preserving mutation.
pub fn split_connection(
    genome: &mut Genome,
    tracker: &mut InnovationTracker,
    innovation: u64,
) -> Result<bool, NeatError> {
    tracker.validate_genome(genome)?;
    let index = genome
        .connections
        .iter()
        .position(|link| link.innovation == innovation)
        .ok_or(NeatError::InvalidGenome(
            "cannot split a missing connection",
        ))?;
    let old = &genome.connections[index];
    if !old.enabled {
        return Ok(false);
    }
    let mut next_tracker = tracker.clone();
    let split = next_tracker.split(innovation)?;
    if genome.nodes.iter().any(|node| node.id == split.node) {
        return Ok(false);
    }
    let mut nodes = genome.nodes.clone();
    nodes.push(NodeGene {
        id: split.node,
        kind: NodeKind::Hidden,
    });
    let mut links = genome.connections.clone();
    links[index].enabled = false;
    links.push(ConnectionGene {
        innovation: split.incoming,
        from: old.from,
        to: split.node,
        weight: 1.0,
        enabled: true,
    });
    links.push(ConnectionGene {
        innovation: split.outgoing,
        from: split.node,
        to: old.to,
        weight: old.weight,
        enabled: true,
    });
    let next = Genome::new(nodes, links)?;
    next_tracker.validate_genome(&next)?;
    *genome = next;
    *tracker = next_tracker;
    Ok(true)
}

/// Mutate finite weights, including disabled historical genes, without changing topology.
///
/// One genome-level probability gates mutation. If selected, each weight is
/// either uniformly perturbed or reset. Successful weights are not clipped;
/// `weight_bound` controls initialization/reset, not the entire search domain.
pub fn mutate_weights(
    genome: &mut Genome,
    config: &MutationConfig,
    rng: &mut impl RngCore,
) -> Result<(), NeatError> {
    config.validate()?;
    genome.validate()?;
    if uniform01(rng) >= config.weight_probability {
        return Ok(());
    }
    let mut next = genome.clone();
    for link in &mut next.connections {
        link.weight = if uniform01(rng) < config.perturb_probability {
            link.weight
                + if config.perturb_bound == 0.0 {
                    0.0
                } else {
                    symmetric(rng, config.perturb_bound)
                }
        } else {
            symmetric(rng, config.weight_bound)
        };
        if !link.weight.is_finite() {
            return Err(NeatError::Numerical("weight mutation overflowed"));
        }
    }
    *genome = next;
    Ok(())
}

/// Compose weight mutation, optional node splitting, and optional edge addition.
///
/// Structural probabilities are independent and applied in that order. A node
/// mutation samples uniformly from enabled genes; edge addition samples
/// uniformly from all admissible missing endpoint pairs. A saturated graph
/// makes the corresponding mutation a no-op instead of failing the run.
pub fn mutate(
    genome: &mut Genome,
    tracker: &mut InnovationTracker,
    config: &MutationConfig,
    rng: &mut impl RngCore,
) -> Result<(), NeatError> {
    config.validate()?;
    tracker.validate_genome(genome)?;
    let mut next = genome.clone();
    // The historical registry can grow much larger than one genome. Stage a
    // copy only when a structural event actually needs to modify it.
    let mut next_tracker = None;
    mutate_weights(&mut next, config, rng)?;
    if uniform01(rng) < config.add_node_probability {
        let enabled: Vec<_> = next
            .connections
            .iter()
            .filter(|link| link.enabled)
            .map(|link| link.innovation)
            .collect();
        if !enabled.is_empty() {
            split_connection(
                &mut next,
                next_tracker.get_or_insert_with(|| tracker.clone()),
                enabled[uniform_index(enabled.len(), rng)],
            )?;
        }
    }
    if uniform01(rng) < config.add_connection_probability {
        let mut candidates = Vec::new();
        for from in next
            .nodes
            .iter()
            .filter(|node| node.kind != NodeKind::Output)
        {
            for to in next
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::Hidden | NodeKind::Output))
            {
                if from.id != to.id
                    && !next
                        .connections
                        .iter()
                        .any(|link| link.from == from.id && link.to == to.id)
                    && !creates_cycle(&next.connections, from.id, to.id)
                {
                    candidates.push((from.id, to.id));
                }
            }
        }
        if !candidates.is_empty() {
            let (from, to) = candidates[uniform_index(candidates.len(), rng)];
            add_connection(
                &mut next,
                next_tracker.get_or_insert_with(|| tracker.clone()),
                from,
                to,
                symmetric(rng, config.weight_bound),
            )?;
        }
    }
    next_tracker
        .as_ref()
        .unwrap_or(tracker)
        .validate_genome(&next)?;
    *genome = next;
    if let Some(next_tracker) = next_tracker {
        *tracker = next_tracker;
    }
    Ok(())
}

/// Align connection innovations and inherit homologous genes from either parent.
///
/// Nonmatching genes come from the fitter parent. At equal fitness, each
/// nonmatching gene is included with probability one half. If a matching gene
/// is disabled in either parent, its child enable flag is independently chosen
/// using `disable_probability` (normally 0.75).
///
/// Feedforward variant rules: consider genes in innovation order; keep the
/// earliest innovation if endpoint pairs repeat, and retain an otherwise
/// inherited cycle-closing gene as disabled. These guards are necessary even
/// when both parents are DAGs. Historical IDs are never rewritten.
pub fn crossover(
    left: &Genome,
    left_fitness: f64,
    right: &Genome,
    right_fitness: f64,
    disable_probability: f64,
    rng: &mut impl RngCore,
) -> Result<Genome, NeatError> {
    compatible_history(left, right)?;
    if !left_fitness.is_finite()
        || left_fitness < 0.0
        || !right_fitness.is_finite()
        || right_fitness < 0.0
        || !disable_probability.is_finite()
        || !(0.0..=1.0).contains(&disable_probability)
    {
        return Err(NeatError::InvalidConfig(
            "crossover needs nonnegative finite fitness and a valid disable probability",
        ));
    }
    let left_genes: BTreeMap<_, _> = left
        .connections
        .iter()
        .map(|link| (link.innovation, link))
        .collect();
    let right_genes: BTreeMap<_, _> = right
        .connections
        .iter()
        .map(|link| (link.innovation, link))
        .collect();
    let innovations: BTreeSet<_> = left_genes
        .keys()
        .chain(right_genes.keys())
        .copied()
        .collect();
    let mut links = Vec::new();
    let mut endpoints = BTreeSet::new();
    for innovation in innovations {
        let candidate = match (left_genes.get(&innovation), right_genes.get(&innovation)) {
            (Some(&a), Some(&b)) => {
                let mut link = if uniform01(rng) < 0.5 {
                    a.clone()
                } else {
                    b.clone()
                };
                if !a.enabled || !b.enabled {
                    link.enabled = uniform01(rng) >= disable_probability;
                }
                Some(link)
            }
            (Some(&a), None)
                if left_fitness > right_fitness
                    || (left_fitness == right_fitness && uniform01(rng) < 0.5) =>
            {
                Some(a.clone())
            }
            (None, Some(&b))
                if right_fitness > left_fitness
                    || (left_fitness == right_fitness && uniform01(rng) < 0.5) =>
            {
                Some(b.clone())
            }
            _ => None,
        };
        if let Some(mut link) = candidate {
            if !endpoints.insert((link.from, link.to)) {
                continue;
            }
            if link.enabled && creates_cycle(&links, link.from, link.to) {
                link.enabled = false;
            }
            links.push(link);
        }
    }
    let referenced: BTreeSet<_> = links.iter().flat_map(|link| [link.from, link.to]).collect();
    let nodes: BTreeMap<_, _> = left
        .nodes
        .iter()
        .chain(&right.nodes)
        .filter(|node| node.kind != NodeKind::Hidden || referenced.contains(&node.id))
        .map(|node| (node.id, node.clone()))
        .collect();
    Genome::new(nodes.into_values().collect(), links)
}
