//! A mutation-only genetic algorithm with truncation selection and seed lineages.

use rand_chacha::{ChaCha8Rng, rand_core::RngCore};

use super::batch::Batch;
use super::common::{score_cmp, seeded_rng, uniform_index};
use super::{Objective, SearchError};
use crate::encoding::{SeedGenome, SeededEncoding};
use crate::{Algorithm, Evaluated, Job};

/// Settings for [`DeepGa`].
#[derive(Clone, Debug, PartialEq)]
pub struct DeepGaConfig {
    /// Number of real-valued parameters decoded from each seed lineage.
    pub dimensions: usize,
    /// Number of newly identified candidates evaluated in each batch.
    pub population: usize,
    /// Number of top current candidates retained as potential parents.
    pub parents: usize,
    /// Top parents copied unchanged into each subsequent evaluated batch.
    pub elite_count: usize,
    /// Uniform initialization half-width; later parameter vectors are unbounded.
    pub initial_bound: f64,
    /// Fixed standard deviation of dense Gaussian mutations; zero is allowed.
    pub mutation_sigma: f64,
    /// Direction of the raw scalar objective.
    pub objective: Objective,
    /// Root seed: stream 0 chooses lineages, stream 1 supplies evaluation seeds.
    pub seed: u64,
}

impl DeepGaConfig {
    /// Use 64 candidates, 16 parents, one elite, bound 1, sigma 0.1, and minimization.
    pub fn new(dimensions: usize, seed: u64) -> Self {
        Self {
            dimensions,
            population: 64,
            parents: 16,
            elite_count: 1,
            initial_bound: 1.0,
            mutation_sigma: 0.1,
            objective: Objective::Minimize,
            seed,
        }
    }

    fn validate(&self) -> Result<SeededEncoding, SearchError> {
        if self.population == 0 || self.parents == 0 || self.parents > self.population {
            return Err(SearchError::InvalidConfig(
                "population must be positive and 0 < parents <= population",
            ));
        }
        if self.elite_count > self.parents || self.elite_count >= self.population {
            return Err(SearchError::InvalidConfig(
                "elite_count must not exceed parents and must be smaller than population",
            ));
        }
        self.dimensions
            .checked_mul(self.population)
            .ok_or(SearchError::InvalidConfig(
                "population dimensions overflow usize",
            ))?;
        SeededEncoding::new(self.dimensions, self.initial_bound, self.mutation_sigma)
    }
}

/// Truncation selection, uniform parent sampling, dense Gaussian mutation, and elitism.
///
/// The first batch contains independent initialization seeds. Subsequent batches
/// copy `elite_count` selected lineages and produce the remaining children by
/// uniformly choosing a retained parent and appending one mutation seed. There
/// is no crossover. [`Self::encoding`] reconstructs the parameters inside the
/// user's evaluator; the optimizer itself handles compact seed histories.
///
/// Every candidate, including a copied elite, gets a fresh job ID and is evaluated
/// again. Selection retains the best `parents` candidates from this batch, with
/// submission order breaking ties. Historical best is a separate reporting record.
///
/// This teaching variant follows the mutation/truncation pattern in Such et al.
/// (2017), <https://arxiv.org/abs/1712.06567>. It uses uniform initialization and
/// does not implement the paper's additional elite-contender reevaluations,
/// distributed noise table, or large Atari/humanoid experiments.
#[derive(Debug)]
pub struct DeepGa {
    config: DeepGaConfig,
    encoding: SeededEncoding,
    candidates_rng: ChaCha8Rng,
    seeds_rng: ChaCha8Rng,
    batch: Batch<SeedGenome>,
    parents: Vec<Evaluated<SeedGenome, f64>>,
    best: Option<Evaluated<SeedGenome, f64>>,
}

impl DeepGa {
    /// Construct a run with validated population and encoding settings.
    ///
    /// # Errors
    /// Rejects invalid dimensions, parent/elite counts, initialization bounds,
    /// or mutation scales.
    pub fn new(config: DeepGaConfig) -> Result<Self, SearchError> {
        let encoding = config.validate()?;
        Ok(Self {
            candidates_rng: seeded_rng(config.seed, 0),
            seeds_rng: seeded_rng(config.seed, 1),
            config,
            encoding,
            batch: Batch::new(),
            parents: Vec::new(),
            best: None,
        })
    }

    /// Immutable settings used for this run.
    pub fn config(&self) -> &DeepGaConfig {
        &self.config
    }

    /// Decoder for this run's seed genomes.
    ///
    /// Copy this small settings value into an evaluator before borrowing the
    /// search mutably for a step.
    pub fn encoding(&self) -> &SeededEncoding {
        &self.encoding
    }

    /// Current retained parents, best first; empty before the initial selection.
    pub fn parents(&self) -> &[Evaluated<SeedGenome, f64>] {
        &self.parents
    }

    /// Best observed evaluation; equal scores retain the earlier record.
    ///
    /// This record does not independently enter reproduction. Under noisy
    /// evaluation its score is a historical observation, not a quality estimate.
    pub fn best(&self) -> Option<&Evaluated<SeedGenome, f64>> {
        self.best.as_ref()
    }

    /// Number of successfully selected batches, including initialization.
    pub fn batches(&self) -> usize {
        self.batch.batches()
    }

    /// Number of evaluations incorporated by successful selection.
    pub fn evaluations(&self) -> usize {
        self.batch.evaluations()
    }
}

impl Algorithm for DeepGa {
    type Input = SeedGenome;
    type Output = f64;
    type Error = SearchError;

    fn generate(&mut self) -> Result<Vec<Job<SeedGenome>>, Self::Error> {
        self.batch.ensure_ready()?;
        let mut inputs = Vec::with_capacity(self.config.population);
        if self.parents.is_empty() {
            for _ in 0..self.config.population {
                inputs.push(SeedGenome {
                    initial_seed: self.candidates_rng.next_u64(),
                    mutation_seeds: Vec::new(),
                });
            }
        } else {
            inputs.extend(
                self.parents
                    .iter()
                    .take(self.config.elite_count)
                    .map(|parent| parent.job.input.clone()),
            );
            while inputs.len() < self.config.population {
                let index = uniform_index(self.parents.len(), &mut self.candidates_rng);
                let mut child = self.parents[index].job.input.clone();
                child
                    .mutation_seeds
                    .len()
                    .checked_add(1)
                    .ok_or(SearchError::CountOverflow)?;
                child.mutation_seeds.push(self.candidates_rng.next_u64());
                inputs.push(child);
            }
        }
        self.batch.issue(inputs, &mut self.seeds_rng)
    }

    fn select(&mut self, mut results: Vec<Evaluated<SeedGenome, f64>>) -> Result<(), Self::Error> {
        self.batch.validate_scalar(&results)?;
        for result in &results {
            if self.best.as_ref().is_none_or(|best| {
                score_cmp(result.output, best.output, self.config.objective).is_lt()
            }) {
                self.best = Some(result.clone());
            }
        }
        results.sort_by(|a, b| score_cmp(a.output, b.output, self.config.objective));
        results.truncate(self.config.parents);
        self.parents = results;
        self.batch.commit();
        Ok(())
    }
}
