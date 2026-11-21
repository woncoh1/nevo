//! Reusable operators for real-valued evolutionary search.
//!
//! These functions work independently of a complete optimizer. Randomness is
//! supplied by the caller; no global RNG or hidden objective direction is used.

use rand_chacha::rand_core::RngCore;

use crate::Evaluated;
use crate::algorithms::common::{normal, score_cmp, uniform_index, uniform01};
use crate::algorithms::{Objective, SearchError};

/// Return the index of the best of `size` uniformly sampled contestants.
///
/// Sampling is with replacement, so a tournament can be larger than the
/// population. Equal scores keep the first sampled contestant. The returned
/// index refers to the original slice; the population is never reordered.
///
/// # Errors
/// Rejects an empty population, zero tournament size, or nonfinite score.
pub fn tournament_selection(
    population: &[Evaluated<Vec<f64>, f64>],
    size: usize,
    objective: Objective,
    rng: &mut impl RngCore,
) -> Result<usize, SearchError> {
    if population.is_empty() || size == 0 {
        return Err(SearchError::InvalidConfig(
            "a tournament requires a nonempty population and positive size",
        ));
    }
    if population
        .iter()
        .any(|candidate| !candidate.output.is_finite())
    {
        return Err(SearchError::InvalidResults(
            "tournament scores must be finite",
        ));
    }
    let mut selected = uniform_index(population.len(), rng);
    for _ in 1..size {
        let challenger = uniform_index(population.len(), rng);
        if score_cmp(
            population[challenger].output,
            population[selected].output,
            objective,
        )
        .is_lt()
        {
            selected = challenger;
        }
    }
    Ok(selected)
}

/// Independently inherit each coordinate from either parent with probability 1/2.
///
/// This is uniform, discrete crossover of real-valued vectors: coordinates are
/// copied, not averaged. No mutation is performed here.
///
/// # Errors
/// Rejects empty or unequal dimensions and nonfinite parental coordinates.
pub fn uniform_crossover(
    first: &[f64],
    second: &[f64],
    rng: &mut impl RngCore,
) -> Result<Vec<f64>, SearchError> {
    if first.is_empty() || first.len() != second.len() {
        return Err(SearchError::InvalidConfig(
            "crossover requires parents of the same positive dimension",
        ));
    }
    if first.iter().chain(second).any(|value| !value.is_finite()) {
        return Err(SearchError::InvalidResults(
            "crossover parents must be finite",
        ));
    }
    Ok(first
        .iter()
        .zip(second)
        .map(|(&a, &b)| if uniform01(rng) < 0.5 { a } else { b })
        .collect())
}

/// Add `N(0, sigma^2)` independently to each coordinate selected for mutation.
///
/// Each coordinate is selected with `probability`. A zero scale or probability
/// leaves the vector unchanged without consuming randomness. On an error the
/// input remains unchanged, although the caller's RNG may have advanced.
///
/// # Errors
/// Rejects invalid parameters, an empty/nonfinite input, or numerical overflow.
pub fn gaussian_mutation(
    point: &mut [f64],
    sigma: f64,
    probability: f64,
    rng: &mut impl RngCore,
) -> Result<(), SearchError> {
    if !sigma.is_finite()
        || sigma < 0.0
        || !probability.is_finite()
        || !(0.0..=1.0).contains(&probability)
    {
        return Err(SearchError::InvalidConfig(
            "mutation requires finite sigma >= 0 and probability in [0, 1]",
        ));
    }
    if point.is_empty() || point.iter().any(|value| !value.is_finite()) {
        return Err(SearchError::InvalidResults(
            "mutation requires a nonempty finite point",
        ));
    }
    if sigma == 0.0 || probability == 0.0 {
        return Ok(());
    }
    let mut mutated = point.to_vec();
    for value in &mut mutated {
        if uniform01(rng) < probability {
            *value += sigma * normal(rng);
            if !value.is_finite() {
                return Err(SearchError::Numerical("Gaussian mutation overflowed"));
            }
        }
    }
    point.copy_from_slice(&mutated);
    Ok(())
}
