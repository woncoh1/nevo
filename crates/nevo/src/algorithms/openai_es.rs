//! Mirrored Gaussian evolution strategies with an explicit SGD mean update.
//!
//! Inspired by Salimans et al. (2017), <https://arxiv.org/abs/1703.03864>.
//! This local variant uses the paper's `1 / sigma` estimator scaling. It does
//! not reproduce the original distributed system, Adam, L2 penalty, observation
//! normalization, or shared noise table.

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::RngCore;

use super::Objective;
use super::common::{BatchState, SearchError, normal, score_cmp, seeded_rng, update_best};
use crate::{Algorithm, Evaluated, Job};

/// Transform scalar evaluations before estimating a search direction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreTransform {
    /// Use raw rewards: maximize the score, or negate it for minimization.
    Raw,
    /// Map ranks to `[-0.5, 0.5]`, with the best score receiving +0.5.
    ///
    /// Tied values receive their average rank. All-equal scores therefore
    /// produce zero utilities and no artificial mean movement.
    CenteredRanks,
}

/// Configuration for the mirrored Gaussian [`OpenAiEs`] variant.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiEsConfig {
    /// Initial distribution mean; nonempty and finite.
    pub initial_mean: Vec<f64>,
    /// Gaussian directions per batch; each direction produces two evaluations.
    pub directions: usize,
    /// Fixed positive Gaussian perturbation scale.
    pub sigma: f64,
    /// Positive SGD learning rate applied to the estimated reward gradient.
    pub learning_rate: f64,
    /// Raw rewards or centered rank utilities.
    pub score_transform: ScoreTransform,
    /// Whether the original objective should be minimized or maximized.
    pub objective: Objective,
    /// Root seed; stream 0 samples noise and stream 1 supplies pair seeds.
    pub seed: u64,
}

impl OpenAiEsConfig {
    /// Use 32 mirrored pairs, sigma 0.5, learning rate 0.1, centered ranks,
    /// and minimization.
    pub fn new(initial_mean: Vec<f64>, seed: u64) -> Self {
        Self {
            initial_mean,
            directions: 32,
            sigma: 0.5,
            learning_rate: 0.1,
            score_transform: ScoreTransform::CenteredRanks,
            objective: Objective::Minimize,
            seed,
        }
    }

    fn validate(&self) -> Result<(), SearchError> {
        if self.initial_mean.is_empty() || self.initial_mean.iter().any(|x| !x.is_finite()) {
            return Err(SearchError::InvalidConfig(
                "OpenAI-style ES needs a nonempty finite mean",
            ));
        }
        if self.directions == 0 {
            return Err(SearchError::InvalidConfig(
                "OpenAI-style ES needs at least one direction",
            ));
        }
        for value in [self.sigma, self.learning_rate] {
            if !value.is_finite() || value <= 0.0 {
                return Err(SearchError::InvalidConfig(
                    "ES sigma and learning rate must be positive and finite",
                ));
            }
        }
        self.directions
            .checked_mul(2)
            .and_then(|count| count.checked_mul(self.initial_mean.len()))
            .ok_or(SearchError::InvalidConfig(
                "ES population size is too large",
            ))?;
        Ok(())
    }
}

/// Gaussian-smoothed reward search using mirrored sampling and SGD.
///
/// For `K` directions, each batch evaluates `mean + sigma * epsilon_i` and
/// `mean - sigma * epsilon_i`. The update is
/// `mean += learning_rate * sum((u_plus-u_minus)*epsilon)/(2*K*sigma)`.
/// `u` denotes raw reward or centered rank utility. Perturbation sigma stays
/// fixed. For stochastic evaluators, the two members of each pair deliberately
/// receive the same evaluation seed; different pairs consume fresh seeds.
///
/// The best evaluated sample is tracked separately. The new mean is not
/// automatically evaluated and is never assigned a sample's score.
#[derive(Debug)]
pub struct OpenAiEs {
    config: OpenAiEsConfig,
    mean: Vec<f64>,
    pending_noise: Vec<Vec<f64>>,
    candidates_rng: ChaCha8Rng,
    seeds_rng: ChaCha8Rng,
    batch: BatchState,
    best: Option<Evaluated<Vec<f64>, f64>>,
}

impl OpenAiEs {
    /// Initialize a fixed-scale mirrored Gaussian search.
    ///
    /// # Errors
    /// Rejects nonfinite/empty means and invalid scale, rate, or direction count.
    pub fn new(config: OpenAiEsConfig) -> Result<Self, SearchError> {
        config.validate()?;
        Ok(Self {
            mean: config.initial_mean.clone(),
            pending_noise: Vec::new(),
            candidates_rng: seeded_rng(config.seed, 0),
            seeds_rng: seeded_rng(config.seed, 1),
            batch: BatchState::new(),
            best: None,
            config,
        })
    }

    /// Initial configuration; the evolving mean is exposed separately.
    pub fn config(&self) -> &OpenAiEsConfig {
        &self.config
    }

    /// Current search mean, generally not itself evaluated.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Best evaluated sample observed so far.
    pub fn best(&self) -> Option<&Evaluated<Vec<f64>, f64>> {
        self.best.as_ref()
    }

    /// Number of successfully selected batches.
    pub fn batches(&self) -> usize {
        self.batch.batches()
    }

    /// Evaluations incorporated by successful selection, excluding failed batches.
    pub fn evaluations(&self) -> usize {
        self.batch.evaluations()
    }
}

impl Algorithm for OpenAiEs {
    type Input = Vec<f64>;
    type Output = f64;
    type Error = SearchError;

    fn generate(&mut self) -> Result<Vec<Job<Vec<f64>>>, SearchError> {
        self.batch.ensure_ready()?;
        let count = 2 * self.config.directions;
        let mut inputs = Vec::with_capacity(count);
        let mut seeds = Vec::with_capacity(count);
        let mut noise = Vec::with_capacity(self.config.directions);
        for _ in 0..self.config.directions {
            let epsilon: Vec<_> = (0..self.mean.len())
                .map(|_| normal(&mut self.candidates_rng))
                .collect();
            for sign in [1.0, -1.0] {
                let input: Vec<_> = self
                    .mean
                    .iter()
                    .zip(&epsilon)
                    .map(|(&mean, &e)| mean + sign * self.config.sigma * e)
                    .collect();
                if input.iter().any(|x| !x.is_finite()) {
                    return Err(SearchError::Numerical("mirrored ES sampling overflowed"));
                }
                inputs.push(input);
            }
            let pair_seed = self.seeds_rng.next_u64();
            seeds.extend([pair_seed, pair_seed]);
            noise.push(epsilon);
        }
        let jobs = self.batch.issue_with_seeds(inputs, seeds)?;
        self.pending_noise = noise;
        Ok(jobs)
    }

    fn select(&mut self, results: Vec<Evaluated<Vec<f64>, f64>>) -> Result<(), SearchError> {
        self.batch.validate_scalar(&results)?;
        let utilities = utilities(&results, self.config.score_transform, self.config.objective);
        let mut gradient = vec![0.0; self.mean.len()];
        for (pair, epsilon) in self.pending_noise.iter().enumerate() {
            let difference = utilities[2 * pair] - utilities[2 * pair + 1];
            let coefficient = difference / results.len() as f64 / self.config.sigma;
            for (component, &noise) in gradient.iter_mut().zip(epsilon) {
                *component += coefficient * noise;
            }
        }
        let mean: Vec<_> = self
            .mean
            .iter()
            .zip(&gradient)
            .map(|(&mean, &gradient)| mean + self.config.learning_rate * gradient)
            .collect();
        if gradient.iter().chain(mean.iter()).any(|x| !x.is_finite()) {
            return Err(SearchError::Numerical("mirrored ES update overflowed"));
        }
        update_best(&mut self.best, &results, self.config.objective);
        self.mean = mean;
        self.pending_noise.clear();
        self.batch.commit();
        Ok(())
    }
}

fn utilities(
    results: &[Evaluated<Vec<f64>, f64>],
    transform: ScoreTransform,
    objective: Objective,
) -> Vec<f64> {
    if transform == ScoreTransform::Raw {
        return results
            .iter()
            .map(|result| match objective {
                Objective::Minimize => -result.output,
                Objective::Maximize => result.output,
            })
            .collect();
    }
    let mut order: Vec<_> = (0..results.len()).collect();
    order.sort_by(|&a, &b| score_cmp(results[a].output, results[b].output, objective));
    let mut utilities = vec![0.0; results.len()];
    let mut first = 0;
    while first < order.len() {
        let mut end = first + 1;
        while end < order.len() && results[order[first]].output == results[order[end]].output {
            end += 1;
        }
        let average_rank = (first as f64 + (end - 1) as f64) / 2.0;
        let utility = 0.5 - average_rank / (results.len() - 1) as f64;
        for &index in &order[first..end] {
            utilities[index] = utility;
        }
        first = end;
    }
    utilities
}
