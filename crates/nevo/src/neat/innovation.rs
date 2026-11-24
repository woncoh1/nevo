use std::collections::BTreeMap;

use super::{ConnectionGene, Genome, NeatError, NodeGene, NodeKind};

#[derive(Clone, Copy, Debug)]
pub(super) struct Split {
    pub node: u64,
    pub incoming: u64,
    pub outgoing: u64,
}

/// Shared historical identity for every genome in one evolutionary run.
///
/// New node and connection IDs increase globally. Structural mutations are
/// deduplicated **within a generation**: adding the same endpoint pair or
/// splitting the same historical connection reuses that generation's IDs.
/// [`Self::begin_generation`] clears those lookup caches without resetting ID
/// counters. A later independent mutation receives new historical markings.
/// The permanent ID-to-structure maps validate history; they are not lifetime
/// endpoint-to-ID caches.
#[derive(Clone, Debug)]
pub struct InnovationTracker {
    base_nodes: Vec<NodeGene>,
    initial_connections: Vec<(u64, u64, u64)>,
    nodes: BTreeMap<u64, NodeKind>,
    connections: BTreeMap<u64, (u64, u64)>,
    next_node: Option<u64>,
    next_innovation: Option<u64>,
    connections_this_generation: BTreeMap<(u64, u64), u64>,
    splits_this_generation: BTreeMap<u64, Split>,
}

impl InnovationTracker {
    /// Allocate common inputs, one bias, outputs, and direct initial connections.
    ///
    /// Initial node IDs are inputs, then bias, then outputs. Initial connection
    /// weights are supplied in output-major order, with input weights then bias.
    pub fn new(inputs: usize, outputs: usize) -> Result<Self, NeatError> {
        if inputs == 0 || outputs == 0 {
            return Err(NeatError::InvalidConfig(
                "input and output counts must be positive",
            ));
        }
        let sources = inputs.checked_add(1).ok_or(NeatError::IdExhausted)?;
        let count = sources.checked_add(outputs).ok_or(NeatError::IdExhausted)?;
        let links = sources.checked_mul(outputs).ok_or(NeatError::IdExhausted)?;
        let next_node = Some(u64::try_from(count).map_err(|_| NeatError::IdExhausted)?);
        let next_innovation = Some(u64::try_from(links).map_err(|_| NeatError::IdExhausted)?);
        let mut base_nodes = Vec::with_capacity(count);
        for id in 0..inputs {
            base_nodes.push(NodeGene {
                id: id as u64,
                kind: NodeKind::Input,
            });
        }
        base_nodes.push(NodeGene {
            id: inputs as u64,
            kind: NodeKind::Bias,
        });
        for index in 0..outputs {
            base_nodes.push(NodeGene {
                id: (sources + index) as u64,
                kind: NodeKind::Output,
            });
        }
        let mut initial_connections = Vec::with_capacity(links);
        for output in 0..outputs {
            for source in 0..sources {
                initial_connections.push((
                    initial_connections.len() as u64,
                    source as u64,
                    (sources + output) as u64,
                ));
            }
        }
        Ok(Self {
            nodes: base_nodes.iter().map(|node| (node.id, node.kind)).collect(),
            connections: initial_connections
                .iter()
                .map(|&(id, from, to)| (id, (from, to)))
                .collect(),
            base_nodes,
            initial_connections,
            next_node,
            next_innovation,
            connections_this_generation: BTreeMap::new(),
            splits_this_generation: BTreeMap::new(),
        })
    }

    /// Number of weights required for the minimal direct-connectivity genome.
    pub fn initial_connection_count(&self) -> usize {
        self.initial_connections.len()
    }

    /// Build a minimal genome sharing the run's initial historical markings.
    pub fn initial_genome(&self, weights: Vec<f64>) -> Result<Genome, NeatError> {
        if weights.len() != self.initial_connections.len() {
            return Err(NeatError::InvalidGenome(
                "wrong number of initial connection weights",
            ));
        }
        let connections = self
            .initial_connections
            .iter()
            .zip(weights)
            .map(|(&(innovation, from, to), weight)| ConnectionGene {
                innovation,
                from,
                to,
                weight,
                enabled: true,
            })
            .collect();
        Genome::new(self.base_nodes.clone(), connections)
    }

    /// Start a reproduction generation; retain all IDs and historical mappings.
    pub fn begin_generation(&mut self) {
        self.connections_this_generation.clear();
        self.splits_this_generation.clear();
    }

    /// Check that every gene agrees with this run's recorded historical identity.
    pub fn validate_genome(&self, genome: &Genome) -> Result<(), NeatError> {
        genome.validate()?;
        for node in genome.nodes() {
            if self.nodes.get(&node.id) != Some(&node.kind) {
                return Err(NeatError::IncompatibleHistory(
                    "unknown node or changed node role",
                ));
            }
        }
        for connection in genome.connections() {
            if self.connections.get(&connection.innovation)
                != Some(&(connection.from, connection.to))
            {
                return Err(NeatError::IncompatibleHistory(
                    "innovation does not identify these endpoints",
                ));
            }
        }
        let base: Vec<_> = genome
            .nodes()
            .iter()
            .filter(|node| node.kind != NodeKind::Hidden)
            .collect();
        if base != self.base_nodes.iter().collect::<Vec<_>>() {
            return Err(NeatError::IncompatibleHistory(
                "input, bias, or output nodes differ from the run",
            ));
        }
        Ok(())
    }

    pub(super) fn connection(&mut self, from: u64, to: u64) -> Result<u64, NeatError> {
        if let Some(&id) = self.connections_this_generation.get(&(from, to)) {
            return Ok(id);
        }
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&to) {
            return Err(NeatError::IncompatibleHistory(
                "new connection has an unknown endpoint",
            ));
        }
        let id = self.next_innovation.ok_or(NeatError::IdExhausted)?;
        self.next_innovation = id.checked_add(1);
        self.connections.insert(id, (from, to));
        self.connections_this_generation.insert((from, to), id);
        Ok(id)
    }

    pub(super) fn split(&mut self, innovation: u64) -> Result<Split, NeatError> {
        if let Some(&split) = self.splits_this_generation.get(&innovation) {
            return Ok(split);
        }
        let &(from, to) =
            self.connections
                .get(&innovation)
                .ok_or(NeatError::IncompatibleHistory(
                    "cannot split an unknown innovation",
                ))?;
        let node = self.next_node.ok_or(NeatError::IdExhausted)?;
        // Both connection IDs must be available before changing this tracker.
        self.next_innovation
            .ok_or(NeatError::IdExhausted)?
            .checked_add(1)
            .ok_or(NeatError::IdExhausted)?;
        self.next_node = node.checked_add(1);
        self.nodes.insert(node, NodeKind::Hidden);
        let split = Split {
            node,
            incoming: self.connection(from, node)?,
            outgoing: self.connection(node, to)?,
        };
        self.splits_this_generation.insert(innovation, split);
        Ok(split)
    }
}
