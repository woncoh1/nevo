//! A self-adaptive, isotropic `(mu, lambda)` evolution strategy.

use rand_chacha::ChaCha8Rng;

use super::common::{
    BatchState, normal, score_cmp, seeded_rng, symmetric, uniform_index, update_best,
};
use super::{Objective, SearchError};
use crate::{Algorithm, Evaluated, Job};

/// Settings for [`Es`], with one inherited step size per individual.
#[derive(Clone, Debug, PartialEq)]
pub struct EsConfig {
    /// Number of real-valued genes; must be positive.
    pub dimensions: usize,
    /// Offspring count lambda, including the initial evaluated batch.
    pub population: usize,
    /// Parent count mu; must satisfy `0 < parents < population`.
    pub parents: usize,
    /// Initial coordinate half-width; there is no later clipping of coordinates.
    pub initial_bound: f64,
    /// Step size assigned to every member of the initial population.
    pub initial_sigma: f64,
    /// Learning rate for the scalar log-normal step-size mutation.
    pub tau: f64,
    /// Explicit lower bound on inherited step sizes.
    pub sigma_min: f64,
    /// Explicit upper bound on inherited step sizes.
    pub sigma_max: f64,
    /// Direction of the raw scalar objective.
    pub objective: Objective,
    /// Root seed for candidate stream 0 and evaluation-seed stream 1.
    pub seed: u64,
}

impl EsConfig {
    /// Set lambda 64, mu 16, bound 5, sigma 0.5, and minimization.
    ///
    /// The default learning rate is `1 / sqrt(2*d)` and sigma bounds are
    /// `[1e-12, 1e3]`. These are declared defaults, not universal optima.
    pub fn new(dimensions: usize, seed: u64) -> Self {
        Self {
            dimensions,
            population: 64,
            parents: 16,
            initial_bound: 5.0,
            initial_sigma: 0.5,
            tau: 1.0 / (2.0 * dimensions as f64).sqrt(),
            sigma_min: 1e-12,
            sigma_max: 1e3,
            objective: Objective::Minimize,
            seed,
        }
    }

    fn validate(&self) -> Result<(), SearchError> {
        if self.dimensions == 0 || self.parents == 0 || self.parents >= self.population {
            return Err(SearchError::InvalidConfig(
                "dimensions must be positive and 0 < parents < population",
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
        if [self.initial_sigma, self.sigma_min, self.sigma_max, self.tau]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
            || self.sigma_min > self.initial_sigma
            || self.initial_sigma > self.sigma_max
        {
            return Err(SearchError::InvalidConfig(
                "sigma values and tau must be finite and positive, with sigma_min <= initial_sigma <= sigma_max",
            ));
        }
        Ok(())
    }
}

/// One current ES parent, coupling its evaluated genes with its inherited scale.
#[derive(Clone, Debug, PartialEq)]
pub struct EsParent {
    /// Objective evaluation of the parent's object variables.
    pub evaluation: Evaluated<Vec<f64>, f64>,
    /// Step size inherited together with those object variables.
    pub sigma: f64,
}

/// Isotropic Gaussian ES with self-adaptation and comma-only replacement.
///
/// Each child samples one of the current mu parents uniformly, mutates its
/// step size by `sigma' = sigma * exp(tau * N(0,1))`, then perturbs every gene
/// using `sigma' * N(0,1)`. The next mu parents come from the new lambda
/// offspring only. There is no recombination or best-ever reinsertion.
///
/// The initial lambda random points are evaluated and truncated to mu parents.
/// Step sizes are clipped in log space to the declared sigma bounds before
/// exponentiation. This numerical safeguard is an explicit implementation
/// choice. All object variables remain unbounded after initialization.
///
/// See Beyer and Schwefel (2002), *Evolution strategies: A comprehensive
/// introduction*, Figure 1 and Section 4.2, DOI `10.1023/A:1015059928466`.
#[derive(Debug)]
pub struct Es {
    config: EsConfig,
    candidates_rng: ChaCha8Rng,
    seeds_rng: ChaCha8Rng,
    state: BatchState,
    parents: Vec<EsParent>,
    pending_sigmas: Vec<f64>,
    best: Option<Evaluated<Vec<f64>, f64>>,
}

impl Es {
    /// Construct a run after validating population and adaptation settings.
    ///
    /// # Errors
    /// Rejects invalid dimensions, population sizes, sampling ranges, or scales.
    pub fn new(config: EsConfig) -> Result<Self, SearchError> {
        config.validate()?;
        Ok(Self {
            candidates_rng: seeded_rng(config.seed, 0),
            seeds_rng: seeded_rng(config.seed, 1),
            config,
            state: BatchState::new(),
            parents: Vec::new(),
            pending_sigmas: Vec::new(),
            best: None,
        })
    }

    /// Immutable settings used for this run.
    pub fn config(&self) -> &EsConfig {
        &self.config
    }

    /// Current mu parents and their coupled step sizes, sorted best first.
    pub fn parents(&self) -> &[EsParent] {
        &self.parents
    }

    /// Best evaluation ever observed, even if it is no longer a parent.
    ///
    /// This reporting record never participates in reproduction by itself.
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

impl Algorithm for Es {
    type Input = Vec<f64>;
    type Output = f64;
    type Error = SearchError;

    fn generate(&mut self) -> Result<Vec<Job<Vec<f64>>>, Self::Error> {
        self.state.ensure_ready()?;
        let mut inputs = Vec::with_capacity(self.config.population);
        let mut sigmas = Vec::with_capacity(self.config.population);
        if self.parents.is_empty() {
            for _ in 0..self.config.population {
                inputs.push(
                    (0..self.config.dimensions)
                        .map(|_| symmetric(&mut self.candidates_rng, self.config.initial_bound))
                        .collect(),
                );
                sigmas.push(self.config.initial_sigma);
            }
        } else {
            for _ in 0..self.config.population {
                let parent =
                    &self.parents[uniform_index(self.parents.len(), &mut self.candidates_rng)];
                let proposed_log_sigma =
                    parent.sigma.ln() + self.config.tau * normal(&mut self.candidates_rng);
                if !proposed_log_sigma.is_finite() {
                    return Err(SearchError::Numerical("step-size adaptation overflowed"));
                }
                let sigma = proposed_log_sigma
                    .clamp(self.config.sigma_min.ln(), self.config.sigma_max.ln())
                    .exp()
                    .clamp(self.config.sigma_min, self.config.sigma_max);
                if !sigma.is_finite() || sigma <= 0.0 {
                    return Err(SearchError::Numerical(
                        "adapted step size is not finite and positive",
                    ));
                }
                let mut input = Vec::with_capacity(self.config.dimensions);
                for &coordinate in &parent.evaluation.job.input {
                    let child = coordinate + sigma * normal(&mut self.candidates_rng);
                    if !child.is_finite() {
                        return Err(SearchError::Numerical("Gaussian ES mutation overflowed"));
                    }
                    input.push(child);
                }
                inputs.push(input);
                sigmas.push(sigma);
            }
        }
        let jobs = self.state.issue(inputs, &mut self.seeds_rng)?;
        self.pending_sigmas = sigmas;
        Ok(jobs)
    }

    fn select(&mut self, results: Vec<Evaluated<Vec<f64>, f64>>) -> Result<(), Self::Error> {
        self.state.validate_scalar(&results)?;
        if self.pending_sigmas.len() != results.len() {
            return Err(SearchError::InvalidResults(
                "strategy parameters do not match the pending batch",
            ));
        }
        update_best(&mut self.best, &results, self.config.objective);
        let mut parents: Vec<_> = results
            .into_iter()
            .zip(&self.pending_sigmas)
            .map(|(evaluation, &sigma)| EsParent { evaluation, sigma })
            .collect();
        parents.sort_by(|a, b| {
            score_cmp(
                a.evaluation.output,
                b.evaluation.output,
                self.config.objective,
            )
        });
        parents.truncate(self.config.parents);
        self.parents = parents;
        self.pending_sigmas.clear();
        self.state.commit();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_offspring_keep_their_own_adapted_sigma_after_sorting() {
        let mut config = EsConfig::new(2, 42);
        config.population = 4;
        config.parents = 2;
        let mut es = Es::new(config).unwrap();
        let jobs = es.generate().unwrap();
        es.select(
            jobs.into_iter()
                .enumerate()
                .map(|(i, job)| Evaluated {
                    job,
                    output: i as f64,
                })
                .collect(),
        )
        .unwrap();
        let jobs = es.generate().unwrap();
        let sigmas = es.pending_sigmas.clone();
        assert!(sigmas.windows(2).any(|pair| pair[0] != pair[1]));
        es.select(
            jobs.into_iter()
                .zip([3.0, 0.0, 1.0, 2.0])
                .map(|(job, output)| Evaluated { job, output })
                .collect(),
        )
        .unwrap();
        assert_eq!(es.parents()[0].evaluation.job.id, crate::JobId(5));
        assert_eq!(es.parents()[0].sigma, sigmas[1]);
        assert_eq!(es.parents()[1].evaluation.job.id, crate::JobId(6));
        assert_eq!(es.parents()[1].sigma, sigmas[2]);
    }
}
