//! A generational real-valued genetic algorithm.

use rand_chacha::ChaCha8Rng;

use super::common::{BatchState, score_cmp, seeded_rng, symmetric, uniform01, update_best};
use super::{Objective, SearchError};
use crate::operators::{gaussian_mutation, tournament_selection, uniform_crossover};
use crate::{Algorithm, Evaluated, Job};

/// Settings for [`Ga`].
#[derive(Clone, Debug, PartialEq)]
pub struct GaConfig {
    /// Number of real-valued genes; must be positive.
    pub dimensions: usize,
    /// Number of candidates evaluated in every batch.
    pub population: usize,
    /// Initialization half-width; search is unbounded after initialization.
    pub initial_bound: f64,
    /// Standard deviation of each Gaussian mutation; zero disables mutation.
    pub mutation_sigma: f64,
    /// Independent mutation probability for each coordinate.
    pub mutation_probability: f64,
    /// Number of contestants sampled with replacement per parent selection.
    pub tournament_size: usize,
    /// Best previous candidates copied unchanged into the next evaluated batch.
    pub elite_count: usize,
    /// Probability of uniform crossover for each nonelite offspring.
    pub crossover_probability: f64,
    /// Direction of the scalar objective.
    pub objective: Objective,
    /// Root seed for candidate stream 0 and evaluation-seed stream 1.
    pub seed: u64,
}

impl GaConfig {
    /// Set population 64, bound 5, mutation sigma 0.1, and minimization.
    ///
    /// The remaining defaults are per-gene mutation probability `1/d`,
    /// tournament size 3, one copied elite, and crossover probability 0.9.
    pub fn new(dimensions: usize, seed: u64) -> Self {
        Self {
            dimensions,
            population: 64,
            initial_bound: 5.0,
            mutation_sigma: 0.1,
            mutation_probability: 1.0 / dimensions as f64,
            tournament_size: 3,
            elite_count: 1,
            crossover_probability: 0.9,
            objective: Objective::Minimize,
            seed,
        }
    }

    fn validate(&self) -> Result<(), SearchError> {
        if self.dimensions == 0 || self.population == 0 {
            return Err(SearchError::InvalidConfig(
                "dimensions and population must be positive",
            ));
        }
        self.dimensions
            .checked_mul(self.population)
            .ok_or(SearchError::InvalidConfig(
                "population dimensions overflow usize",
            ))?;
        if !self.initial_bound.is_finite()
            || self.initial_bound <= 0.0
            || !(2.0 * self.initial_bound).is_finite()
        {
            return Err(SearchError::InvalidConfig(
                "initial_bound and twice its value must be positive and finite",
            ));
        }
        if !self.mutation_sigma.is_finite() || self.mutation_sigma < 0.0 {
            return Err(SearchError::InvalidConfig(
                "mutation_sigma must be finite and nonnegative",
            ));
        }
        if [self.mutation_probability, self.crossover_probability]
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return Err(SearchError::InvalidConfig(
                "probabilities must be finite and in [0, 1]",
            ));
        }
        if self.tournament_size == 0 || self.elite_count >= self.population {
            return Err(SearchError::InvalidConfig(
                "tournament_size must be positive and elite_count smaller than population",
            ));
        }
        Ok(())
    }
}

/// Tournament selection, uniform crossover, Gaussian mutation, and elitism.
///
/// Every batch contains `population` newly identified jobs, including copied
/// elites. Copied candidates are reevaluated rather than reusing their prior
/// scores; tournaments use the current evaluated parent scores. After selection, the current population is sorted by objective,
/// with issue order breaking ties. Best-ever reporting is kept separately.
///
/// This explicitly specified real-valued GA is an independent implementation
/// of standard operators, not a line-by-line port of the book's exercise.
#[derive(Debug)]
pub struct Ga {
    config: GaConfig,
    candidates_rng: ChaCha8Rng,
    seeds_rng: ChaCha8Rng,
    state: BatchState,
    parents: Vec<Evaluated<Vec<f64>, f64>>,
    best: Option<Evaluated<Vec<f64>, f64>>,
}

impl Ga {
    /// Construct a run after validating all settings.
    ///
    /// # Errors
    /// Rejects invalid dimensions, population, bounds, probabilities, or operators.
    pub fn new(config: GaConfig) -> Result<Self, SearchError> {
        config.validate()?;
        Ok(Self {
            candidates_rng: seeded_rng(config.seed, 0),
            seeds_rng: seeded_rng(config.seed, 1),
            config,
            state: BatchState::new(),
            parents: Vec::new(),
            best: None,
        })
    }

    /// Immutable settings used for this run.
    pub fn config(&self) -> &GaConfig {
        &self.config
    }

    /// Current evaluated population, best first; empty before initial selection.
    pub fn parents(&self) -> &[Evaluated<Vec<f64>, f64>] {
        &self.parents
    }

    /// Best evaluation observed during the run; ties keep the earlier record.
    pub fn best(&self) -> Option<&Evaluated<Vec<f64>, f64>> {
        self.best.as_ref()
    }

    /// Number of successfully selected batches.
    pub fn batches(&self) -> usize {
        self.state.batches()
    }

    /// Number of evaluations in successfully selected batches.
    pub fn evaluations(&self) -> usize {
        self.state.evaluations()
    }
}

impl Algorithm for Ga {
    type Input = Vec<f64>;
    type Output = f64;
    type Error = SearchError;

    fn generate(&mut self) -> Result<Vec<Job<Vec<f64>>>, Self::Error> {
        self.state.ensure_ready()?;
        let mut inputs = Vec::with_capacity(self.config.population);
        if self.parents.is_empty() {
            for _ in 0..self.config.population {
                inputs.push(
                    (0..self.config.dimensions)
                        .map(|_| symmetric(&mut self.candidates_rng, self.config.initial_bound))
                        .collect(),
                );
            }
        } else {
            for elite in self.parents.iter().take(self.config.elite_count) {
                inputs.push(elite.job.input.clone());
            }
            while inputs.len() < self.config.population {
                let first = tournament_selection(
                    &self.parents,
                    self.config.tournament_size,
                    self.config.objective,
                    &mut self.candidates_rng,
                )?;
                let mut child =
                    if uniform01(&mut self.candidates_rng) < self.config.crossover_probability {
                        let second = tournament_selection(
                            &self.parents,
                            self.config.tournament_size,
                            self.config.objective,
                            &mut self.candidates_rng,
                        )?;
                        uniform_crossover(
                            &self.parents[first].job.input,
                            &self.parents[second].job.input,
                            &mut self.candidates_rng,
                        )?
                    } else {
                        self.parents[first].job.input.clone()
                    };
                gaussian_mutation(
                    &mut child,
                    self.config.mutation_sigma,
                    self.config.mutation_probability,
                    &mut self.candidates_rng,
                )?;
                inputs.push(child);
            }
        }
        self.state.issue(inputs, &mut self.seeds_rng)
    }

    fn select(&mut self, mut results: Vec<Evaluated<Vec<f64>, f64>>) -> Result<(), Self::Error> {
        self.state.validate_scalar(&results)?;
        update_best(&mut self.best, &results, self.config.objective);
        // Stable sorting preserves original issue order for equal scores.
        results.sort_by(|a, b| score_cmp(a.output, b.output, self.config.objective));
        self.parents = results;
        self.state.commit();
        Ok(())
    }
}
