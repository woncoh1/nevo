use std::collections::{BTreeMap, BTreeSet};

use super::NeatError;

/// Execution and structural rules carried by a NEAT genotype.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NetworkMode {
    /// Acyclic, stateless evaluation in topological order; the original default.
    #[default]
    FeedForward,
    /// Synchronous stateful evaluation, with one-tick delay on neuron-source edges.
    ///
    /// External inputs and bias use their current values. Hidden/output sources
    /// use their values from the previous tick, including on otherwise acyclic
    /// edges. Self-links and output feedback are allowed. This is an explicit
    /// temporal convention, not the original NEAT software's activation loop.
    Recurrent,
}

/// The role of a node in a NEAT graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// An external input, ordered by ascending node ID at evaluation.
    Input,
    /// The single constant input whose value is always one.
    Bias,
    /// An evolved internal neuron with a steepened logistic activation.
    Hidden,
    /// An output neuron, ordered by ascending node ID at evaluation.
    Output,
}

/// A historically identified neuron; node IDs are not topological positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeGene {
    /// Stable identifier allocated by the run's shared innovation tracker.
    pub id: u64,
    /// Input, bias, hidden, or output role.
    pub kind: NodeKind,
}

/// One historical connection, including disabled genes retained for alignment.
#[derive(Clone, Debug)]
pub struct ConnectionGene {
    /// Stable historical marking, distinct from an evaluation job ID.
    pub innovation: u64,
    /// Source node identifier.
    pub from: u64,
    /// Destination node identifier.
    pub to: u64,
    /// Finite connection weight.
    pub weight: f64,
    /// Whether this connection is expressed in the phenotype.
    pub enabled: bool,
}

impl PartialEq for ConnectionGene {
    fn eq(&self, other: &Self) -> bool {
        self.innovation == other.innovation
            && self.from == other.from
            && self.to == other.to
            && self.weight.to_bits() == other.weight.to_bits()
            && self.enabled == other.enabled
    }
}
impl Eq for ConnectionGene {}

/// A validated, canonically ordered NEAT genotype with explicit execution mode.
///
/// Node genes are sorted by ID; connections are sorted by innovation. The
/// graph contains inputs, exactly one bias, and outputs. Inputs/bias can never
/// be destinations. Feedforward mode additionally requires output sinks, no
/// self-links, and an enabled DAG. Recurrent mode permits cycles and output
/// feedback. Every endpoint pair and innovation is unique. Disabled genes
/// remain in historical alignment/compatibility but are not executed.
///
/// Equality compares execution mode and weight bits, preserving the GES
/// input-identity contract even for a change between `0.0` and `-0.0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Genome {
    pub(super) nodes: Vec<NodeGene>,
    pub(super) connections: Vec<ConnectionGene>,
    pub(super) mode: NetworkMode,
}

impl Genome {
    /// Validate feedforward genes and canonicalize their storage order.
    pub fn new(nodes: Vec<NodeGene>, connections: Vec<ConnectionGene>) -> Result<Self, NeatError> {
        Self::with_mode(nodes, connections, NetworkMode::FeedForward)
    }

    /// Validate genes for synchronous recurrence and canonicalize their order.
    ///
    /// The genotype contains no episode state; compiling it creates a fresh
    /// phenotype. Use [`Self::compile_recurrent`] and reset between trials.
    pub fn new_recurrent(
        nodes: Vec<NodeGene>,
        connections: Vec<ConnectionGene>,
    ) -> Result<Self, NeatError> {
        Self::with_mode(nodes, connections, NetworkMode::Recurrent)
    }

    pub(super) fn with_mode(
        mut nodes: Vec<NodeGene>,
        mut connections: Vec<ConnectionGene>,
        mode: NetworkMode,
    ) -> Result<Self, NeatError> {
        nodes.sort_by_key(|node| node.id);
        connections.sort_by_key(|connection| connection.innovation);
        let genome = Self {
            nodes,
            connections,
            mode,
        };
        genome.validate()?;
        Ok(genome)
    }

    /// Structural and temporal interpretation shared by every gene in this genome.
    pub fn mode(&self) -> NetworkMode {
        self.mode
    }

    /// All node genes, including any currently disconnected hidden nodes.
    pub fn nodes(&self) -> &[NodeGene] {
        &self.nodes
    }
    /// All connection genes, including disabled historical genes.
    pub fn connections(&self) -> &[ConnectionGene] {
        &self.connections
    }
    /// Number of evolved hidden node genes.
    pub fn hidden_node_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Hidden)
            .count()
    }
    /// Number of connections currently expressed in the phenotype.
    pub fn enabled_connection_count(&self) -> usize {
        self.connections
            .iter()
            .filter(|connection| connection.enabled)
            .count()
    }
    /// Compile feedforward edges into a deterministic, stateless evaluator.
    ///
    /// A recurrent genome is rejected, even if its current edges form a DAG.
    pub fn compile(&self) -> Result<FeedForwardNetwork, NeatError> {
        FeedForwardNetwork::from_genome(self)
    }

    /// Compile a recurrent genome with zero initial hidden/output activations.
    ///
    /// A feedforward genome is rejected, so its instantaneous propagation is
    /// never silently changed to synchronous recurrence.
    pub fn compile_recurrent(&self) -> Result<RecurrentNetwork, NeatError> {
        RecurrentNetwork::from_genome(self)
    }

    /// Check identifiers, roles, endpoints, weights, and mode-specific structure.
    pub fn validate(&self) -> Result<(), NeatError> {
        if !self.nodes.iter().any(|node| node.kind == NodeKind::Input)
            || !self.nodes.iter().any(|node| node.kind == NodeKind::Output)
            || self
                .nodes
                .iter()
                .filter(|node| node.kind == NodeKind::Bias)
                .count()
                != 1
        {
            return Err(NeatError::InvalidGenome(
                "inputs, outputs, and exactly one bias are required",
            ));
        }
        if self.nodes.windows(2).any(|pair| pair[0].id >= pair[1].id)
            || self
                .connections
                .windows(2)
                .any(|pair| pair[0].innovation >= pair[1].innovation)
        {
            return Err(NeatError::InvalidGenome(
                "node IDs and innovations must be unique and sorted",
            ));
        }
        let roles: BTreeMap<_, _> = self.nodes.iter().map(|node| (node.id, node.kind)).collect();
        let mut endpoints = BTreeSet::new();
        for connection in &self.connections {
            let Some(from) = roles.get(&connection.from) else {
                return Err(NeatError::InvalidGenome("connection source is missing"));
            };
            let Some(to) = roles.get(&connection.to) else {
                return Err(NeatError::InvalidGenome(
                    "connection destination is missing",
                ));
            };
            if matches!(to, NodeKind::Input | NodeKind::Bias) {
                return Err(NeatError::InvalidGenome(
                    "inputs and bias cannot be connection destinations",
                ));
            }
            if self.mode == NetworkMode::FeedForward
                && (*from == NodeKind::Output || connection.from == connection.to)
            {
                return Err(NeatError::InvalidGenome(
                    "feedforward outputs must be sinks and self-links are forbidden",
                ));
            }
            if !connection.weight.is_finite() {
                return Err(NeatError::InvalidGenome(
                    "connection weights must be finite",
                ));
            }
            if !endpoints.insert((connection.from, connection.to)) {
                return Err(NeatError::InvalidGenome("duplicate connection endpoints"));
            }
        }
        if self.mode == NetworkMode::FeedForward {
            self.topological_order()?;
        }
        Ok(())
    }

    pub(super) fn topological_order(&self) -> Result<Vec<u64>, NeatError> {
        let mut degree: BTreeMap<_, usize> = self.nodes.iter().map(|node| (node.id, 0)).collect();
        let mut outgoing: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for connection in self
            .connections
            .iter()
            .filter(|connection| connection.enabled)
        {
            *degree
                .get_mut(&connection.to)
                .ok_or(NeatError::InvalidGenome("missing endpoint"))? += 1;
            outgoing
                .entry(connection.from)
                .or_default()
                .push(connection.to);
        }
        let mut ready: BTreeSet<_> = degree
            .iter()
            .filter_map(|(&id, &n)| (n == 0).then_some(id))
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = ready.pop_first() {
            order.push(id);
            if let Some(destinations) = outgoing.get(&id) {
                for destination in destinations {
                    let count = degree.get_mut(destination).expect("validated endpoint");
                    *count -= 1;
                    if *count == 0 {
                        ready.insert(*destination);
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(NeatError::InvalidGenome(
                "enabled connections contain a cycle",
            ));
        }
        Ok(order)
    }
}

pub(super) fn creates_cycle(connections: &[ConnectionGene], from: u64, to: u64) -> bool {
    let mut visited = BTreeSet::new();
    let mut pending = vec![to];
    while let Some(node) = pending.pop() {
        if node == from {
            return true;
        }
        if visited.insert(node) {
            pending.extend(
                connections
                    .iter()
                    .filter(|connection| connection.enabled && connection.from == node)
                    .map(|connection| connection.to),
            );
        }
    }
    false
}

/// A compiled, stateless feedforward phenotype with `sigmoid(4.9 * sum)` nodes.
///
/// Input/output ordering follows ascending node IDs. Incoming terms are summed
/// in innovation order. The evaluator uses the DAG's topological order rather
/// than treating increasing historical IDs as a valid execution order.
#[derive(Clone, Debug)]
pub struct FeedForwardNetwork {
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    bias: usize,
    order: Vec<usize>,
    incoming: Vec<Vec<(usize, f64)>>,
}

impl FeedForwardNetwork {
    /// Validate a genome and compile its currently enabled connections.
    pub fn from_genome(genome: &Genome) -> Result<Self, NeatError> {
        genome.validate()?;
        if genome.mode != NetworkMode::FeedForward {
            return Err(NeatError::InvalidGenome(
                "recurrent genomes require a recurrent phenotype",
            ));
        }
        let indices: BTreeMap<_, _> = genome
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id, i))
            .collect();
        let inputs = genome
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| (node.kind == NodeKind::Input).then_some(i))
            .collect();
        let outputs = genome
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| (node.kind == NodeKind::Output).then_some(i))
            .collect();
        let bias = genome
            .nodes
            .iter()
            .position(|node| node.kind == NodeKind::Bias)
            .expect("validated bias");
        let order = genome
            .topological_order()?
            .into_iter()
            .map(|id| indices[&id])
            .filter(|&i| matches!(genome.nodes[i].kind, NodeKind::Hidden | NodeKind::Output))
            .collect();
        let mut incoming = vec![Vec::new(); genome.nodes.len()];
        for connection in genome
            .connections
            .iter()
            .filter(|connection| connection.enabled)
        {
            incoming[indices[&connection.to]].push((indices[&connection.from], connection.weight));
        }
        Ok(Self {
            inputs,
            outputs,
            bias,
            order,
            incoming,
        })
    }

    /// Number of external input values required by [`Self::forward`].
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }
    /// Number of output values returned by [`Self::forward`].
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Evaluate one input with fresh state and checked finite affine arithmetic.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>, NeatError> {
        if input.len() != self.inputs.len() || input.iter().any(|value| !value.is_finite()) {
            return Err(NeatError::InvalidGenome(
                "input must have the expected length and finite values",
            ));
        }
        let mut values = vec![0.0; self.incoming.len()];
        values[self.bias] = 1.0;
        for (&index, &value) in self.inputs.iter().zip(input) {
            values[index] = value;
        }
        for &index in &self.order {
            let mut sum = 0.0;
            for &(source, weight) in &self.incoming[index] {
                sum += values[source] * weight;
                if !sum.is_finite() {
                    return Err(NeatError::Numerical("network affine sum overflowed"));
                }
            }
            // Saturation remains well-defined even if 4.9*sum reaches infinity;
            // exponentials only receive nonpositive arguments in either branch.
            values[index] = if sum >= 0.0 {
                1.0 / (1.0 + (-4.9 * sum).exp())
            } else {
                let e = (4.9 * sum).exp();
                e / (1.0 + e)
            };
        }
        Ok(self.outputs.iter().map(|&index| values[index]).collect())
    }
}

/// A synchronous recurrent phenotype with explicit episode state.
///
/// At tick `t`, inputs and the bias use their current values. Every connection
/// whose source is a hidden or output neuron reads that neuron's value from
/// tick `t-1`. All hidden/output activations are then committed simultaneously,
/// each using `sigmoid(4.9 * sum)`. Thus a hidden-to-output edge has a one-tick
/// delay even when it is not part of a directed cycle. This convention permits
/// self-links and output feedback without relying on node or traversal order.
///
/// New phenotypes and [`Self::reset`] start with all neuron activations zero,
/// not `sigmoid(0)`. Reset between independent episodes. Cloning a compiled
/// phenotype copies its current state; compiling the same genome creates a
/// fresh zero-state phenotype. The genotype never contains episode state.
///
/// This is an explicit synchronous NEAT variant, not a reproduction of the
/// original software's activation-until-output loop or pole-balancing protocol.
/// The previous-neuron/current-input convention is also used by
/// [NEAT-Python's recurrent phenotype](https://neat-python.readthedocs.io/en/latest/_modules/nn/recurrent.html).
#[derive(Clone, Debug)]
pub struct RecurrentNetwork {
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    bias: usize,
    neurons: Vec<usize>,
    external: Vec<bool>,
    incoming: Vec<Vec<(usize, f64)>>,
    values: Vec<f64>,
    scratch: Vec<f64>,
}

impl RecurrentNetwork {
    /// Validate and compile a recurrent genome, initializing fresh episode state.
    ///
    /// Feedforward mode is rejected to prevent an implicit change in temporal
    /// semantics. Incoming terms retain historical innovation order.
    pub fn from_genome(genome: &Genome) -> Result<Self, NeatError> {
        genome.validate()?;
        if genome.mode != NetworkMode::Recurrent {
            return Err(NeatError::InvalidGenome(
                "feedforward genomes require a feedforward phenotype",
            ));
        }
        let indices: BTreeMap<_, _> = genome
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id, index))
            .collect();
        let inputs = genome
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| (node.kind == NodeKind::Input).then_some(index))
            .collect();
        let outputs = genome
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| (node.kind == NodeKind::Output).then_some(index))
            .collect();
        let bias = genome
            .nodes
            .iter()
            .position(|node| node.kind == NodeKind::Bias)
            .expect("validated bias");
        let external: Vec<_> = genome
            .nodes
            .iter()
            .map(|node| matches!(node.kind, NodeKind::Input | NodeKind::Bias))
            .collect();
        let neurons = external
            .iter()
            .enumerate()
            .filter_map(|(index, &is_external)| (!is_external).then_some(index))
            .collect();
        let mut incoming = vec![Vec::new(); genome.nodes.len()];
        for link in genome.connections.iter().filter(|link| link.enabled) {
            incoming[indices[&link.to]].push((indices[&link.from], link.weight));
        }
        let mut values = vec![0.0; genome.nodes.len()];
        values[bias] = 1.0;
        Ok(Self {
            inputs,
            outputs,
            bias,
            neurons,
            external,
            incoming,
            scratch: values.clone(),
            values,
        })
    }

    /// Number of external input coordinates required at each tick.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Number of output coordinates returned at each tick.
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Clear episode state, setting every hidden/output activation to zero.
    pub fn reset(&mut self) {
        self.values.fill(0.0);
        self.scratch.fill(0.0);
        self.values[self.bias] = 1.0;
        self.scratch[self.bias] = 1.0;
    }

    /// Advance exactly one synchronous tick and return the new output values.
    ///
    /// Input/output order follows ascending node IDs. Incorrect input shape,
    /// nonfinite inputs, or overflowing affine sums return an error without
    /// committing any new neuron state. Saturated finite affine sums use a
    /// stable sigmoid. No convergence iteration or hidden warm-up is performed.
    pub fn step(&mut self, input: &[f64]) -> Result<Vec<f64>, NeatError> {
        if input.len() != self.inputs.len() || input.iter().any(|value| !value.is_finite()) {
            return Err(NeatError::InvalidGenome(
                "input must have the expected length and finite values",
            ));
        }
        self.scratch[self.bias] = 1.0;
        for (&index, &value) in self.inputs.iter().zip(input) {
            self.scratch[index] = value;
        }
        for &index in &self.neurons {
            let mut sum = 0.0;
            for &(source, weight) in &self.incoming[index] {
                let value = if self.external[source] {
                    self.scratch[source]
                } else {
                    self.values[source]
                };
                sum += value * weight;
                if !sum.is_finite() {
                    return Err(NeatError::Numerical("network affine sum overflowed"));
                }
            }
            self.scratch[index] = if sum >= 0.0 {
                1.0 / (1.0 + (-4.9 * sum).exp())
            } else {
                let exponential = (4.9 * sum).exp();
                exponential / (1.0 + exponential)
            };
        }
        std::mem::swap(&mut self.values, &mut self.scratch);
        Ok(self
            .outputs
            .iter()
            .map(|&index| self.values[index])
            .collect())
    }
}
