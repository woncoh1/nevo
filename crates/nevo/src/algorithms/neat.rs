//! Feedforward/recurrent NEAT with historical alignment and species reproduction.

use rand_chacha::ChaCha8Rng;

use super::SearchError;
use super::batch::Batch;
use super::common::{seeded_rng, symmetric, uniform_index, uniform01};
use crate::neat::{
    CompatibilityConfig, Genome, InnovationTracker, MutationConfig, NeatError, NetworkMode,
    SpeciesStats, allocate_offspring, compatibility_distance, crossover, mutate,
};
use crate::{Algorithm, Evaluated, Job};

/// Configuration for the documented feedforward or synchronous recurrent NEAT variant.
#[derive(Clone, Debug, PartialEq)]
pub struct NeatConfig {
    /// Number of external inputs, excluding the automatically added bias.
    pub inputs: usize,
    /// Number of sigmoid output nodes.
    pub outputs: usize,
    /// Graph and phenotype execution mode; defaults to feedforward.
    pub mode: NetworkMode,
    /// Number of newly evaluated genomes in every batch, including initialization.
    pub population: usize,
    /// Root seed for variation stream 0, evaluation stream 1, and representative stream 2.
    pub seed: u64,
    /// Weight and structural mutation probabilities and bounds.
    pub mutation: MutationConfig,
    /// Historical compatibility distance coefficients.
    pub compatibility: CompatibilityConfig,
    /// A genome joins the first species whose representative is strictly nearer than this threshold.
    pub compatibility_threshold: f64,
    /// Probability of crossover rather than cloning a reproductive parent.
    pub crossover_probability: f64,
    /// Probability that a matching gene disabled in either parent remains disabled.
    pub disable_probability: f64,
    /// Given crossover, probability of choosing the second parent from another reproducing species.
    pub interspecies_mating_probability: f64,
    /// Highest-fitness fraction eligible to be parents in each species; rounded upward with a minimum of one.
    pub survival_fraction: f64,
    /// Generations without strict improvement before a species loses its reproductive allocation.
    pub stagnation_generations: usize,
    /// Copy the species champion unchanged when its offspring quota is at least this large.
    pub champion_min_offspring: usize,
}

impl NeatConfig {
    /// Set 150 genomes, threshold 3, crossover 0.75, stagnation 15, and champion quota 5.
    ///
    /// Execution defaults to feedforward. Other defaults are disable probability
    /// 0.75, interspecies mating 0.001,
    /// survival fraction 0.2, and the documented mutation/distance defaults.
    /// Fitness is always maximized and must be finite and nonnegative. An
    /// evaluator must explicitly transform a loss if that is its task objective.
    pub fn new(inputs: usize, outputs: usize, seed: u64) -> Self {
        Self {
            inputs,
            outputs,
            mode: NetworkMode::FeedForward,
            population: 150,
            seed,
            mutation: MutationConfig::default(),
            compatibility: CompatibilityConfig::default(),
            compatibility_threshold: 3.0,
            crossover_probability: 0.75,
            disable_probability: 0.75,
            interspecies_mating_probability: 0.001,
            survival_fraction: 0.2,
            stagnation_generations: 15,
            champion_min_offspring: 5,
        }
    }

    fn validate(&self) -> Result<(), NeatError> {
        if self.inputs == 0
            || self.outputs == 0
            || self.population < 2
            || self.population as u128 > (1_u128 << 53)
            || self.stagnation_generations == 0
            || self.champion_min_offspring == 0
        {
            return Err(NeatError::InvalidConfig(
                "positive dimensions/stagnation/champion quota and population in 2..=2^53 required",
            ));
        }
        self.inputs
            .checked_add(1)
            .and_then(|sources| sources.checked_mul(self.outputs))
            .and_then(|connections| connections.checked_mul(self.population))
            .ok_or(NeatError::InvalidConfig(
                "initial population size overflowed",
            ))?;
        if !self.compatibility_threshold.is_finite()
            || self.compatibility_threshold <= 0.0
            || !self.survival_fraction.is_finite()
            || self.survival_fraction <= 0.0
            || self.survival_fraction > 1.0
        {
            return Err(NeatError::InvalidConfig(
                "positive finite threshold and survival fraction in (0,1] required",
            ));
        }
        if [
            self.crossover_probability,
            self.disable_probability,
            self.interspecies_mating_probability,
        ]
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return Err(NeatError::InvalidConfig(
                "reproduction probabilities must lie in [0,1]",
            ));
        }
        self.mutation.validate()?;
        self.compatibility.validate()
    }
}

#[derive(Clone, Debug)]
struct Species {
    id: u64,
    representative: Genome,
    members: Vec<usize>,
    parent_pool: Vec<usize>,
    best_ever: Option<f64>,
    stagnant: usize,
    offspring: usize,
}

/// Topology and weight evolution with NEAT's central mechanisms.
///
/// All genomes start with direct input/bias-to-output edges and no hidden
/// neurons. Innovation-aligned crossover, structural mutation, compatibility
/// species, and explicit fitness sharing jointly control reproduction.
///
/// `select` assigns evaluated genomes to the first compatible representative,
/// updates persistent species, removes stagnant species from reproduction
/// except the current champion's species, and allocates exact offspring quotas.
/// Stagnation protection does not impose a minimum reproductive quota.
/// Species weight is the sum of `fitness/species_size`, computed before culling.
/// All-zero weights give uniform species quotas. Reproductive parents are
/// sampled uniformly from each species' surviving top fraction.
///
/// `generate` creates exactly one population, including unchanged champions
/// within their quotas. All are freshly evaluated; no parent fitness is cached
/// for a new job. Innovation caches are generation-local. Representatives are
/// sampled from all current members using a separate random stream.
///
/// Execution mode is explicit. Feedforward genomes use a DAG; recurrent genomes
/// use synchronous ticks with current inputs and previous-tick hidden/output
/// sources. Both use a fixed compatibility threshold, independent structural
/// mutation events, and quota-based champion copying. Neither claims to
/// reproduce the original recurrent pole-balancing protocol or its global-
/// stagnation rescue rule. `best()` is an observed reporting record, not an
/// additional archived reproductive parent.
#[derive(Debug)]
pub struct Neat {
    config: NeatConfig,
    tracker: InnovationTracker,
    variation_rng: ChaCha8Rng,
    evaluation_rng: ChaCha8Rng,
    representative_rng: ChaCha8Rng,
    batch: Batch<Genome>,
    parents: Vec<Evaluated<Genome, f64>>,
    species: Vec<Species>,
    statistics: Vec<SpeciesStats>,
    next_species_id: Option<u64>,
    best: Option<Evaluated<Genome, f64>>,
}

impl Neat {
    /// Validate settings and initialize a shared historical registry.
    pub fn new(config: NeatConfig) -> Result<Self, NeatError> {
        config.validate()?;
        let tracker = match config.mode {
            NetworkMode::FeedForward => InnovationTracker::new(config.inputs, config.outputs)?,
            NetworkMode::Recurrent => {
                InnovationTracker::new_recurrent(config.inputs, config.outputs)?
            }
        };
        Ok(Self {
            tracker,
            variation_rng: seeded_rng(config.seed, 0),
            evaluation_rng: seeded_rng(config.seed, 1),
            representative_rng: seeded_rng(config.seed, 2),
            config,
            batch: Batch::new(),
            parents: Vec::new(),
            species: Vec::new(),
            statistics: Vec::new(),
            next_species_id: Some(0),
            best: None,
        })
    }

    /// Immutable settings used by this search.
    pub fn config(&self) -> &NeatConfig {
        &self.config
    }
    /// Best actually evaluated genome; exact fitness ties retain the earlier record.
    pub fn best(&self) -> Option<&Evaluated<Genome, f64>> {
        self.best.as_ref()
    }
    /// Entire last evaluated population, in submission order, before within-species culling.
    pub fn parents(&self) -> &[Evaluated<Genome, f64>] {
        &self.parents
    }
    /// Persistent species and next-generation quotas after the last selected batch.
    pub fn species(&self) -> &[SpeciesStats] {
        &self.statistics
    }
    /// Successfully selected batches, including initialization.
    pub fn batches(&self) -> usize {
        self.batch.batches()
    }
    /// Fitness evaluations accepted in successfully selected batches.
    pub fn evaluations(&self) -> usize {
        self.batch.evaluations()
    }
}

impl Algorithm for Neat {
    type Input = Genome;
    type Output = f64;
    type Error = NeatError;

    fn generate(&mut self) -> Result<Vec<Job<Genome>>, NeatError> {
        self.batch.ensure_ready()?;
        let mut rng = self.variation_rng.clone();
        let mut evaluation_rng = self.evaluation_rng.clone();
        let mut tracker = self.tracker.clone();
        let mut inputs = Vec::with_capacity(self.config.population);
        if self.parents.is_empty() {
            for _ in 0..self.config.population {
                let weights = (0..tracker.initial_connection_count())
                    .map(|_| symmetric(&mut rng, self.config.mutation.weight_bound))
                    .collect();
                inputs.push(tracker.initial_genome(weights)?);
            }
        } else {
            tracker.begin_generation();
            for (index, species) in self
                .species
                .iter()
                .enumerate()
                .filter(|(_, s)| s.offspring > 0)
            {
                let mut produced = 0;
                if species.offspring >= self.config.champion_min_offspring {
                    inputs.push(self.parents[species.parent_pool[0]].job.input.clone());
                    produced += 1;
                }
                let other_species: Vec<_> = self
                    .species
                    .iter()
                    .enumerate()
                    .filter_map(|(other, s)| (other != index && s.offspring > 0).then_some(other))
                    .collect();
                while produced < species.offspring {
                    let first =
                        species.parent_pool[uniform_index(species.parent_pool.len(), &mut rng)];
                    let mut child = if uniform01(&mut rng) < self.config.crossover_probability {
                        let donor = if !other_species.is_empty()
                            && uniform01(&mut rng) < self.config.interspecies_mating_probability
                        {
                            &self.species
                                [other_species[uniform_index(other_species.len(), &mut rng)]]
                        } else {
                            species
                        };
                        let second =
                            donor.parent_pool[uniform_index(donor.parent_pool.len(), &mut rng)];
                        crossover(
                            &self.parents[first].job.input,
                            self.parents[first].output,
                            &self.parents[second].job.input,
                            self.parents[second].output,
                            self.config.disable_probability,
                            &mut rng,
                        )?
                    } else {
                        self.parents[first].job.input.clone()
                    };
                    mutate(&mut child, &mut tracker, &self.config.mutation, &mut rng)?;
                    inputs.push(child);
                    produced += 1;
                }
            }
        }
        if inputs.len() != self.config.population {
            return Err(NeatError::Numerical(
                "species quotas did not fill the population",
            ));
        }
        for genome in &inputs {
            tracker.validate_genome(genome)?;
        }
        let jobs = self.batch.issue(inputs, &mut evaluation_rng)?;
        self.variation_rng = rng;
        self.evaluation_rng = evaluation_rng;
        self.tracker = tracker;
        Ok(jobs)
    }

    fn select(&mut self, results: Vec<Evaluated<Genome, f64>>) -> Result<(), NeatError> {
        self.batch.validate_scalar(&results)?;
        if results.iter().any(|result| result.output < 0.0) {
            return Err(SearchError::InvalidResults(
                "NEAT maximizes finite nonnegative fitness; transform losses explicitly",
            )
            .into());
        }
        let mut species = self.species.clone();
        for group in &mut species {
            group.members.clear();
            group.parent_pool.clear();
            group.offspring = 0;
        }
        let mut next_id = self.next_species_id;
        let mut representative_rng = self.representative_rng.clone();
        for (index, result) in results.iter().enumerate() {
            self.tracker.validate_genome(&result.job.input)?;
            let mut destination = None;
            for (i, group) in species.iter().enumerate() {
                if compatibility_distance(
                    &result.job.input,
                    &group.representative,
                    &self.config.compatibility,
                )? < self.config.compatibility_threshold
                {
                    destination = Some(i);
                    break;
                }
            }
            let destination = match destination {
                Some(index) => index,
                None => {
                    let id = next_id.ok_or(NeatError::IdExhausted)?;
                    next_id = id.checked_add(1);
                    species.push(Species {
                        id,
                        representative: result.job.input.clone(),
                        members: Vec::new(),
                        parent_pool: Vec::new(),
                        best_ever: None,
                        stagnant: 0,
                        offspring: 0,
                    });
                    species.len() - 1
                }
            };
            species[destination].members.push(index);
        }
        species.retain(|group| !group.members.is_empty());
        let mut champion = 0;
        for index in 1..results.len() {
            if results[index].output > results[champion].output {
                champion = index;
            }
        }
        let maximum = results[champion].output;
        let mut statistics = Vec::with_capacity(species.len());
        let mut eligible = Vec::new();
        let mut weights = Vec::new();
        for (index, group) in species.iter_mut().enumerate() {
            group.parent_pool = group.members.clone();
            group.parent_pool.sort_by(|&a, &b| {
                results[b]
                    .output
                    .partial_cmp(&results[a].output)
                    .expect("validated fitness")
            });
            let best_fitness = results[group.parent_pool[0]].output;
            if group.best_ever.is_some_and(|best| best_fitness <= best) {
                group.stagnant = group.stagnant.checked_add(1).ok_or(NeatError::Numerical(
                    "species stagnation counter overflowed",
                ))?;
            } else {
                group.best_ever = Some(best_fitness);
                group.stagnant = 0;
            }
            let protected = group.members.contains(&champion);
            // Scaling before summation preserves proportional sharing without
            // overflowing on otherwise valid large raw fitness values.
            let scaled_mean = if maximum == 0.0 {
                0.0
            } else {
                (group
                    .members
                    .iter()
                    .map(|&member| results[member].output / maximum)
                    .sum::<f64>()
                    / group.members.len() as f64)
                    .min(1.0)
            };
            if protected || group.stagnant < self.config.stagnation_generations {
                eligible.push(index);
                weights.push(scaled_mean);
            }
            statistics.push(SpeciesStats {
                id: group.id,
                size: group.members.len(),
                best_fitness,
                best_ever_fitness: group.best_ever.expect("species just updated"),
                stagnant_generations: group.stagnant,
                shared_fitness: scaled_mean * maximum,
                offspring: 0,
                protected,
            });
            let survivor_count = ((group.members.len() as f64 * self.config.survival_fraction)
                .ceil() as usize)
                .max(1)
                .min(group.members.len());
            group.parent_pool.truncate(survivor_count);
            let representative =
                group.members[uniform_index(group.members.len(), &mut representative_rng)];
            group.representative = results[representative].job.input.clone();
        }
        let quotas = allocate_offspring(&weights, self.config.population)?;
        for (&index, quota) in eligible.iter().zip(quotas) {
            species[index].offspring = quota;
            statistics[index].offspring = quota;
        }
        let mut best = self.best.clone();
        if best
            .as_ref()
            .is_none_or(|old| results[champion].output > old.output)
        {
            best = Some(results[champion].clone());
        }
        self.parents = results;
        self.species = species;
        self.statistics = statistics;
        self.next_species_id = next_id;
        self.representative_rng = representative_rng;
        self.best = best;
        self.batch.commit();
        Ok(())
    }
}
