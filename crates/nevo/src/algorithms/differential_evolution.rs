//! Synchronous differential evolution with the `DE/rand/1/bin` variation rule.

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::RngCore;

use super::common::{
    BatchState, score_cmp, seeded_rng, symmetric, uniform_index, uniform01, update_best,
};
use super::{Objective, SearchError};
use crate::{Algorithm, Evaluated, Job};

/// Settings for [`De`]. Defaults are local teaching choices, not book prescriptions.
#[derive(Clone, Debug, PartialEq)]
pub struct DeConfig {
    /// Number of real-valued coordinates; must be positive.
    pub dimensions: usize,
    /// Population and trial-batch size; at least four are required.
    pub population: usize,
    /// Initial coordinate half-width; subsequent search is unbounded.
    pub initial_bound: f64,
    /// Differential weight `F`, strictly between zero and two.
    pub differential_weight: f64,
    /// Independent probability of inheriting a mutant coordinate, in `[0,1]`.
    /// One uniformly chosen coordinate is always taken from the mutant.
    pub crossover_probability: f64,
    /// Direction of the scalar objective.
    pub objective: Objective,
    /// Root seed for candidate stream 0 and evaluation-seed stream 1.
    pub seed: u64,
}

impl DeConfig {
    /// Choose population 64, initial bound 5, `F=0.8`, `CR=0.9`, and minimization.
    pub fn new(dimensions: usize, seed: u64) -> Self {
        Self {
            dimensions,
            population: 64,
            initial_bound: 5.0,
            differential_weight: 0.8,
            crossover_probability: 0.9,
            objective: Objective::Minimize,
            seed,
        }
    }

    fn validate(&self) -> Result<(), SearchError> {
        if self.dimensions == 0 || self.population < 4 {
            return Err(SearchError::InvalidConfig(
                "DE needs positive dimensions and at least four candidates",
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
        if !self.differential_weight.is_finite()
            || self.differential_weight <= 0.0
            || self.differential_weight >= 2.0
            || !self.crossover_probability.is_finite()
            || !(0.0..=1.0).contains(&self.crossover_probability)
        {
            return Err(SearchError::InvalidConfig(
                "DE requires finite 0 < F < 2 and 0 <= CR <= 1",
            ));
        }
        Ok(())
    }
}

/// Differential mutation, binomial crossover, and one-to-one greedy survival.
///
/// For each target `x_i`, choose three distinct donors, also distinct from `i`,
/// from the frozen current population. Form `v = x_r1 + F * (x_r2 - x_r3)`.
/// A trial independently inherits each coordinate from `v` with probability
/// `CR`, otherwise from `x_i`; one random coordinate is forced to use `v`.
/// This guarantees mutant inheritance, not necessarily a numerical change.
///
/// [`Algorithm::select`] compares each evaluated trial only with its own target.
/// Strictly better trials replace targets; equal scores retain targets. The
/// population is updated synchronously after the complete batch is validated.
/// Parent scores are cached, so use a stationary deterministic evaluator or
/// explicitly accept comparisons with earlier noisy observations. No objective
/// calls, score transformations, clipping, or restarts are hidden here.
///
/// The first batch evaluates a uniform initial population. Later batches contain
/// one fresh evaluation per target. `parents()` preserves target-slot order,
/// rather than fitness order, and `best()` is a separate evaluated record.
///
/// This explicit `DE/rand/1/bin` choice implements the differential-evolution
/// mechanism described in *Neuroevolution*, Chapter 2, §2.2.6, pp.34–35. Donor
/// exclusion, forced-coordinate binomial crossover, initialization, and tie
/// behavior are declared implementation choices.
#[derive(Debug)]
pub struct De {
    config: DeConfig,
    candidates_rng: ChaCha8Rng,
    seeds_rng: ChaCha8Rng,
    state: BatchState,
    parents: Vec<Evaluated<Vec<f64>, f64>>,
    best: Option<Evaluated<Vec<f64>, f64>>,
}

impl De {
    /// Validate settings and create a new, unevaluated search.
    pub fn new(config: DeConfig) -> Result<Self, SearchError> {
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
    pub fn config(&self) -> &DeConfig {
        &self.config
    }

    /// Current evaluated population in target-slot order, empty before selection.
    pub fn parents(&self) -> &[Evaluated<Vec<f64>, f64>] {
        &self.parents
    }

    /// Best evaluated point observed so far; equal scores keep the earlier record.
    pub fn best(&self) -> Option<&Evaluated<Vec<f64>, f64>> {
        self.best.as_ref()
    }

    /// Number of successfully selected batches, including initialization.
    pub fn batches(&self) -> usize {
        self.state.batches()
    }

    /// Evaluations in successfully selected batches; cached parents do not count.
    pub fn evaluations(&self) -> usize {
        self.state.evaluations()
    }
}

impl Algorithm for De {
    type Input = Vec<f64>;
    type Output = f64;
    type Error = SearchError;

    fn generate(&mut self) -> Result<Vec<Job<Vec<f64>>>, SearchError> {
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
            for target in 0..self.parents.len() {
                let [first, second, third] =
                    donor_indices(target, self.parents.len(), &mut self.candidates_rng);
                let forced = uniform_index(self.config.dimensions, &mut self.candidates_rng);
                let mut trial = Vec::with_capacity(self.config.dimensions);
                for coordinate in 0..self.config.dimensions {
                    let take_mutant = uniform01(&mut self.candidates_rng)
                        < self.config.crossover_probability
                        || coordinate == forced;
                    let value = if take_mutant {
                        let difference = self.parents[second].job.input[coordinate]
                            - self.parents[third].job.input[coordinate];
                        self.parents[first].job.input[coordinate]
                            + self.config.differential_weight * difference
                    } else {
                        self.parents[target].job.input[coordinate]
                    };
                    if !value.is_finite() {
                        return Err(SearchError::Numerical("differential mutation overflowed"));
                    }
                    trial.push(value);
                }
                inputs.push(trial);
            }
        }
        self.state.issue(inputs, &mut self.seeds_rng)
    }

    fn select(&mut self, results: Vec<Evaluated<Vec<f64>, f64>>) -> Result<(), SearchError> {
        self.state.validate_scalar(&results)?;
        update_best(&mut self.best, &results, self.config.objective);
        if self.parents.is_empty() {
            self.parents = results;
        } else {
            for (parent, trial) in self.parents.iter_mut().zip(results) {
                if score_cmp(trial.output, parent.output, self.config.objective).is_lt() {
                    *parent = trial;
                }
            }
        }
        self.state.commit();
        Ok(())
    }
}

fn donor_indices(target: usize, population: usize, rng: &mut impl RngCore) -> [usize; 3] {
    let mut donors = [0; 3];
    for position in 0..donors.len() {
        loop {
            let index = uniform_index(population, rng);
            if index != target && !donors[..position].contains(&index) {
                donors[position] = index;
                break;
            }
        }
    }
    donors
}
