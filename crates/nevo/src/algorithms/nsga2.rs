//! NSGA-II survival and mating with configurable real-valued variation.

use super::common::{BatchState, seeded_rng, uniform_index, uniform01};
use super::{Objective, SearchError};
use crate::multiobjective::{crowding_distances, nondominated_fronts};
use crate::operators::{gaussian_mutation, uniform_crossover};
use crate::{Algorithm, Evaluated, Job};
use rand_chacha::ChaCha8Rng;

/// Configuration for a bounded, real-valued NSGA-II reference implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct Nsga2Config {
    /// Number of genes in each candidate.
    pub dimensions: usize,
    /// Direction of each objective; at least two are required.
    pub objectives: Vec<Objective>,
    /// Parent and offspring population size (at least two).
    pub population: usize,
    /// Inclusive lower bound shared by all genes.
    pub lower_bound: f64,
    /// Inclusive upper bound shared by all genes.
    pub upper_bound: f64,
    /// Standard deviation of Gaussian mutation (may be zero).
    pub mutation_sigma: f64,
    /// Independent mutation probability per gene.
    pub mutation_probability: f64,
    /// Probability of uniform crossover instead of cloning the first parent.
    pub crossover_probability: f64,
    /// Root seed for independent search and evaluation-seed streams.
    pub seed: u64,
}

impl Nsga2Config {
    /// Defaults: 64 parents, `[0,1]` bounds, sigma 0.1, mutation `1/d`, crossover 0.9.
    pub fn new(dimensions: usize, objectives: Vec<Objective>, seed: u64) -> Self {
        Self {
            dimensions,
            objectives,
            population: 64,
            lower_bound: 0.0,
            upper_bound: 1.0,
            mutation_sigma: 0.1,
            mutation_probability: 1.0 / dimensions.max(1) as f64,
            crossover_probability: 0.9,
            seed,
        }
    }

    fn validate(&self) -> Result<(), SearchError> {
        if self.dimensions == 0 || self.population < 2 || self.objectives.len() < 2 {
            return Err(SearchError::InvalidConfig(
                "positive dimensions, population >= 2 and at least two objectives required",
            ));
        }
        if !self.lower_bound.is_finite()
            || !self.upper_bound.is_finite()
            || self.lower_bound >= self.upper_bound
            || !(self.upper_bound - self.lower_bound).is_finite()
        {
            return Err(SearchError::InvalidConfig(
                "finite bounds with a positive finite width required",
            ));
        }
        if !self.mutation_sigma.is_finite()
            || self.mutation_sigma < 0.0
            || !self.mutation_probability.is_finite()
            || !(0.0..=1.0).contains(&self.mutation_probability)
            || !self.crossover_probability.is_finite()
            || !(0.0..=1.0).contains(&self.crossover_probability)
        {
            return Err(SearchError::InvalidConfig(
                "sigma must be finite and nonnegative; probabilities must lie in [0,1]",
            ));
        }
        Ok(())
    }
}

/// Elitist multiobjective search using nondominated rank and crowding distance.
///
/// The first batch evaluates `N` uniform candidates. Each later batch evaluates
/// `N` offspring and retains `N` from parents plus offspring. Parent scores are
/// cached, so this version assumes a stationary deterministic objective; noisy
/// reevaluation policies are left to a different algorithm. Mating uses binary
/// tournaments on rank then crowding. Crowding is measured on each combined
/// front before truncation and retained for the next mating step. Exact survivor
/// ties retain parent-before-offspring order; tournament ties use a coin flip.
/// Constant objective dimensions contribute zero crowding, including an
/// all-identical front. Variation is uniform crossover plus Gaussian mutation,
/// with clipping to bounds, rather than the original paper's SBX/polynomial
/// variation. There is no constraint-domination rule beyond box bounds.
///
/// Reference: Deb et al. (2002), DOI 10.1109/4235.996017.
#[derive(Debug)]
pub struct Nsga2 {
    config: Nsga2Config,
    rng: ChaCha8Rng,
    evaluation_rng: ChaCha8Rng,
    batch: BatchState,
    parents: Vec<Evaluated<Vec<f64>, Vec<f64>>>,
    ranks: Vec<usize>,
    crowding: Vec<f64>,
}

impl Nsga2 {
    /// Validate configuration and create an unevaluated search.
    pub fn new(config: Nsga2Config) -> Result<Self, SearchError> {
        config.validate()?;
        Ok(Self {
            rng: seeded_rng(config.seed, 0),
            evaluation_rng: seeded_rng(config.seed, 1),
            config,
            batch: BatchState::new(),
            parents: Vec::new(),
            ranks: Vec::new(),
            crowding: Vec::new(),
        })
    }
    /// Immutable configuration used by this search.
    pub fn config(&self) -> &Nsga2Config {
        &self.config
    }
    /// Current selected population with cached objective vectors.
    pub fn parents(&self) -> &[Evaluated<Vec<f64>, Vec<f64>>] {
        &self.parents
    }
    /// Nondominated members of the current population, in stable selection order.
    pub fn pareto_front(&self) -> Vec<&Evaluated<Vec<f64>, Vec<f64>>> {
        self.parents
            .iter()
            .zip(&self.ranks)
            .filter_map(|(parent, &rank)| (rank == 0).then_some(parent))
            .collect()
    }
    /// Number of successfully selected batches, including initialization.
    pub fn batches(&self) -> usize {
        self.batch.batches()
    }
    /// Number of evaluated candidates accepted by selection.
    pub fn evaluations(&self) -> usize {
        self.batch.evaluations()
    }

    fn tournament(&mut self) -> usize {
        let a = uniform_index(self.parents.len(), &mut self.rng);
        let b = uniform_index(self.parents.len(), &mut self.rng);
        let order = self.ranks[a].cmp(&self.ranks[b]).then_with(|| {
            self.crowding[b]
                .partial_cmp(&self.crowding[a])
                .expect("valid distances")
        });
        if order.is_lt() || (order.is_eq() && uniform01(&mut self.rng) < 0.5) {
            a
        } else {
            b
        }
    }
}

impl Algorithm for Nsga2 {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    type Error = SearchError;

    fn generate(&mut self) -> Result<Vec<Job<Self::Input>>, Self::Error> {
        self.batch.ensure_ready()?;
        let mut inputs = Vec::with_capacity(self.config.population);
        for _ in 0..self.config.population {
            let point = if self.parents.is_empty() {
                (0..self.config.dimensions)
                    .map(|_| {
                        self.config.lower_bound
                            + uniform01(&mut self.rng)
                                * (self.config.upper_bound - self.config.lower_bound)
                    })
                    .collect()
            } else {
                let first = self.tournament();
                let second = self.tournament();
                let mut child = if uniform01(&mut self.rng) < self.config.crossover_probability {
                    uniform_crossover(
                        &self.parents[first].job.input,
                        &self.parents[second].job.input,
                        &mut self.rng,
                    )?
                } else {
                    self.parents[first].job.input.clone()
                };
                gaussian_mutation(
                    &mut child,
                    self.config.mutation_sigma,
                    self.config.mutation_probability,
                    &mut self.rng,
                )?;
                for gene in &mut child {
                    *gene = gene.clamp(self.config.lower_bound, self.config.upper_bound);
                }
                child
            };
            inputs.push(point);
        }
        self.batch.issue(inputs, &mut self.evaluation_rng)
    }

    fn select(
        &mut self,
        results: Vec<Evaluated<Self::Input, Self::Output>>,
    ) -> Result<(), Self::Error> {
        self.batch.validate(&results)?;
        if results.iter().any(|r| {
            r.output.len() != self.config.objectives.len()
                || r.output.iter().any(|v| !v.is_finite())
        }) {
            return Err(SearchError::InvalidResults(
                "wrong objective count or nonfinite objective value",
            ));
        }
        let combined: Vec<_> = self.parents.iter().cloned().chain(results).collect();
        let values: Vec<_> = combined.iter().map(|r| r.output.clone()).collect();
        let mut parents = Vec::with_capacity(self.config.population);
        let mut ranks = Vec::with_capacity(self.config.population);
        let mut crowding = Vec::with_capacity(self.config.population);
        for (rank, front) in nondominated_fronts(&values, &self.config.objectives)?
            .iter()
            .enumerate()
        {
            let remaining = self.config.population - parents.len();
            let local: Vec<_> = front.iter().map(|&i| values[i].clone()).collect();
            let distances = crowding_distances(&local)?;
            let mut local_order: Vec<_> = (0..front.len()).collect();
            if front.len() > remaining {
                local_order.sort_by(|&a, &b| {
                    distances[b]
                        .partial_cmp(&distances[a])
                        .expect("valid distances")
                });
            }
            for local_index in local_order.into_iter().take(remaining) {
                parents.push(combined[front[local_index]].clone());
                ranks.push(rank);
                // NSGA-II carries the full-front distances into mating; they
                // must not be recomputed on the truncated survivor population.
                crowding.push(distances[local_index]);
            }
            if parents.len() == self.config.population {
                break;
            }
        }
        self.parents = parents;
        self.ranks = ranks;
        self.crowding = crowding;
        self.batch.commit();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mating_keeps_crowding_measured_before_front_truncation() {
        let mut config = Nsga2Config::new(1, vec![Objective::Minimize; 2], 42);
        config.population = 4;
        let mut search = Nsga2::new(config).unwrap();
        for first_objectives in [[0.0, 4.0, 10.0, 16.0], [1.0, 5.0, 8.0, 15.0]] {
            let results = search
                .generate()
                .unwrap()
                .into_iter()
                .zip(first_objectives)
                .map(|(job, x)| Evaluated {
                    job,
                    output: vec![x, 16.0 - x],
                })
                .collect();
            search.select(results).unwrap();
        }
        let ids: Vec<_> = search
            .parents
            .iter()
            .map(|parent| parent.job.id.0)
            .collect();
        assert_eq!(ids, [0, 3, 2, 7]);
        assert_eq!(search.ranks, [0, 0, 0, 0]);
        // Full front: 0,1,4,5,8,10,15,16. For x=10, the neighbors
        // x=8 and x=15 give 2 * (15-8)/16 = 0.875. Recomputing after
        // truncation would wrongly use x=0 and x=15, giving 1.875.
        assert_eq!(search.crowding, [f64::INFINITY, f64::INFINITY, 0.875, 0.75]);
    }
}
