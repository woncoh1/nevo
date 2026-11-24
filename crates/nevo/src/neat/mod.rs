//! Historical graph genes and composable feedforward NEAT building blocks.
//!
//! This implementation starts with direct input/bias-to-output connections,
//! aligns genes by innovation number, and grows structure through connection
//! and node mutations. Inputs and the single bias are sources; outputs are
//! sinks. Enabled connections form a DAG. Recurrent NEAT is a separate future
//! variant, not an implicit update rule for cyclic graphs.
//!
//! An innovation number is meaningful within its shared [`InnovationTracker`].
//! Genomes from independent evolutionary histories must not be mixed merely
//! because their integer identifiers happen to agree.
//!
//! Reference: Stanley and Miikkulainen (2002),
//! [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).

mod genome;
mod innovation;
mod operators;
mod species;

pub use genome::{ConnectionGene, FeedForwardNetwork, Genome, NodeGene, NodeKind};
pub use innovation::InnovationTracker;
pub use operators::{
    CompatibilityConfig, MutationConfig, add_connection, compatibility_distance, crossover, mutate,
    mutate_weights, split_connection,
};
pub use species::{SpeciesStats, allocate_offspring};

use std::{error::Error, fmt};

use crate::algorithms::SearchError;

/// An invalid NEAT configuration, graph, history, evaluation, or numerical step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NeatError {
    /// A probability, bound, dimension, or population setting is invalid.
    InvalidConfig(&'static str),
    /// A graph violates its shape, identity, or feedforward invariants.
    InvalidGenome(&'static str),
    /// Parents or a genome and its tracker disagree about historical identity.
    IncompatibleHistory(&'static str),
    /// A finite input led to an unrepresentable numerical result.
    Numerical(&'static str),
    /// A node, innovation, or species identifier cannot be allocated uniquely.
    IdExhausted,
    /// The complete-batch Generate–Evaluate–Select contract was violated.
    Protocol(SearchError),
}

impl fmt::Display for NeatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid NEAT configuration: {message}"),
            Self::InvalidGenome(message) => write!(f, "invalid NEAT genome: {message}"),
            Self::IncompatibleHistory(message) => write!(f, "incompatible NEAT history: {message}"),
            Self::Numerical(message) => write!(f, "NEAT numerical failure: {message}"),
            Self::IdExhausted => write!(f, "NEAT node, innovation, or species IDs exhausted"),
            Self::Protocol(error) => error.fmt(f),
        }
    }
}

impl Error for NeatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Protocol(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SearchError> for NeatError {
    fn from(error: SearchError) -> Self {
        Self::Protocol(error)
    }
}
