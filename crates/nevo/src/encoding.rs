//! Parameter vectors represented by initialization and mutation seeds.
//!
//! A lineage stores `8 * (1 + mutation_count)` bytes of seed payload rather than
//! one `f64` per parameter. This excludes vector metadata and allocation costs;
//! histories grow with generations, so this representation is not always smaller.
//! Decoding costs `O(dimensions * (1 + mutation_count))` and allocates a vector.

use crate::algorithms::SearchError;
use crate::algorithms::common::{normal, seeded_rng, symmetric};

/// A reproducible initialization followed by an ordered mutation history.
///
/// The encoding settings must be saved alongside this value: dimensions,
/// initialization bound, and mutation scale are not stored in the genome.
/// Evaluation seeds and evaluation job IDs are independent of these seeds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SeedGenome {
    /// Seed for the initial vector.
    pub initial_seed: u64,
    /// Seeds for Gaussian mutations, applied in this exact order.
    pub mutation_seeds: Vec<u64>,
}

/// Decode seed histories into flat real-valued parameters.
///
/// Every seed initializes a fresh ChaCha8 stream 0. Initial coordinates use
/// high-53-bit uniforms on `[-initial_bound, initial_bound)`. Each later seed
/// supplies a dense Gaussian perturbation using uncached Box–Muller sampling,
/// with two `u64` draws per coordinate. Additions are applied in lineage order.
///
/// Replay is exact for the same settings and numerical implementation. It is
/// not a promise of bitwise identity across platforms or future math/RNG changes.
/// This reference encoding is inspired by Such et al. (2017),
/// <https://arxiv.org/abs/1712.06567>, with an explicitly uniform initialization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeededEncoding {
    dimensions: usize,
    initial_bound: f64,
    mutation_sigma: f64,
}

impl SeededEncoding {
    /// Validate the number of coordinates, initialization range, and mutation scale.
    ///
    /// A zero mutation scale is allowed and makes every mutation an identity.
    ///
    /// # Errors
    /// Rejects zero or overflowing dimensions, nonpositive/nonfinite bounds,
    /// an overflowing doubled bound, and negative/nonfinite mutation scales.
    pub fn new(
        dimensions: usize,
        initial_bound: f64,
        mutation_sigma: f64,
    ) -> Result<Self, SearchError> {
        if dimensions == 0
            || dimensions
                .checked_mul(std::mem::size_of::<f64>())
                .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(SearchError::InvalidConfig(
                "encoding dimensions must be positive and fit a parameter vector",
            ));
        }
        if !initial_bound.is_finite() || initial_bound <= 0.0 || !(2.0 * initial_bound).is_finite()
        {
            return Err(SearchError::InvalidConfig(
                "initial_bound and twice its value must be positive and finite",
            ));
        }
        if !mutation_sigma.is_finite() || mutation_sigma < 0.0 {
            return Err(SearchError::InvalidConfig(
                "mutation_sigma must be finite and nonnegative",
            ));
        }
        Ok(Self {
            dimensions,
            initial_bound,
            mutation_sigma,
        })
    }

    /// Number of decoded parameters.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Reconstruct a finite parameter vector from its ordered seed history.
    ///
    /// # Errors
    /// Returns a numerical error if a Gaussian perturbation or accumulated
    /// coordinate overflows. The genome is never changed.
    pub fn decode(&self, genome: &SeedGenome) -> Result<Vec<f64>, SearchError> {
        let mut rng = seeded_rng(genome.initial_seed, 0);
        let mut parameters: Vec<_> = (0..self.dimensions)
            .map(|_| symmetric(&mut rng, self.initial_bound))
            .collect();
        if self.mutation_sigma == 0.0 {
            return Ok(parameters);
        }
        for &seed in &genome.mutation_seeds {
            let mut rng = seeded_rng(seed, 0);
            for coordinate in &mut parameters {
                *coordinate += self.mutation_sigma * normal(&mut rng);
                if !coordinate.is_finite() {
                    return Err(SearchError::Numerical(
                        "seeded parameter reconstruction overflowed",
                    ));
                }
            }
        }
        Ok(parameters)
    }
}
