use super::NeatError;

/// Diagnostics for a species after the most recent successful selection.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeciesStats {
    /// Persistent run-local species identity; never reused after extinction.
    pub id: u64,
    /// Number of evaluated members before reproductive culling.
    pub size: usize,
    /// Highest raw fitness among current members.
    pub best_fitness: f64,
    /// Highest raw fitness observed during this species' lifetime.
    pub best_ever_fitness: f64,
    /// Consecutive selected generations without a strict fitness improvement.
    pub stagnant_generations: usize,
    /// Sum of `fitness/species_size`, equal to the mean raw fitness.
    pub shared_fitness: f64,
    /// Exact number of jobs allocated to this species in the next batch.
    pub offspring: usize,
    /// Whether this species contains the current population champion and is protected from stagnation removal.
    /// This does not reserve a minimum offspring quota.
    pub protected: bool,
}

/// Allocate an exact population by proportional largest remainders.
///
/// Input weights are finite and nonnegative. If they are all zero, species
/// receive uniform quotas. Fractional ties retain input order. The allocation
/// includes any champions; callers must not add elites on top of these quotas.
pub fn allocate_offspring(weights: &[f64], population: usize) -> Result<Vec<usize>, NeatError> {
    if weights.is_empty()
        || population == 0
        || population as u128 > (1_u128 << 53)
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(NeatError::InvalidConfig(
            "allocation requires nonempty nonnegative finite weights and population in 1..=2^53",
        ));
    }
    let maximum = weights.iter().copied().fold(0.0_f64, f64::max);
    let scaled: Vec<_> = weights
        .iter()
        .map(|weight| {
            if maximum == 0.0 {
                1.0
            } else {
                weight / maximum
            }
        })
        .collect();
    let total: f64 = scaled.iter().sum();
    let quotas: Vec<_> = scaled
        .iter()
        .map(|weight| (weight / total) * population as f64)
        .collect();
    let mut counts: Vec<_> = quotas.iter().map(|quota| quota.floor() as usize).collect();
    let allocated = counts
        .iter()
        .try_fold(0usize, |sum, &count| sum.checked_add(count))
        .ok_or(NeatError::Numerical("offspring count overflowed"))?;
    let mut order: Vec<_> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| {
        (quotas[b] - quotas[b].floor()).total_cmp(&(quotas[a] - quotas[a].floor()))
    });
    if allocated <= population {
        for i in 0..(population - allocated) {
            counts[order[i % order.len()]] += 1;
        }
    } else {
        // Very large exact populations can round a quota across an integer.
        // Correct that floating-point surplus from the smallest remainders.
        let mut surplus = allocated - population;
        for &index in order.iter().rev().cycle() {
            if surplus == 0 {
                break;
            }
            if counts[index] > 0 {
                counts[index] -= 1;
                surplus -= 1;
            }
        }
    }
    Ok(counts)
}
