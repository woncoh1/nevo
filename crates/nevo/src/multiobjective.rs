//! Pareto dominance, nondominated sorting, and crowding distance.
//!
//! These functions can be used independently of [`crate::algorithms::Nsga2`].
//! The reference sorter uses O(M N²) comparisons and O(N²) worst-case storage.

use crate::algorithms::common::score_cmp;
use crate::algorithms::{Objective, SearchError};

fn validate(values: &[Vec<f64>], dimensions: usize) -> Result<(), SearchError> {
    if dimensions == 0 {
        return Err(SearchError::InvalidConfig(
            "at least one objective is required",
        ));
    }
    if values
        .iter()
        .any(|v| v.len() != dimensions || v.iter().any(|x| !x.is_finite()))
    {
        return Err(SearchError::InvalidResults(
            "objective vectors must have the expected length and finite entries",
        ));
    }
    Ok(())
}

fn dominates_unchecked(a: &[f64], b: &[f64], directions: &[Objective]) -> bool {
    let mut strictly_better = false;
    for ((&left, &right), &direction) in a.iter().zip(b).zip(directions) {
        let order = score_cmp(left, right, direction);
        if order.is_gt() {
            return false;
        }
        strictly_better |= order.is_lt();
    }
    strictly_better
}

/// Whether `a` is no worse in every objective and strictly better in at least one.
///
/// Equal points do not dominate one another. Mixed minimize/maximize directions
/// are supported. Returns an error for empty, mismatched, or nonfinite vectors.
pub fn dominates(a: &[f64], b: &[f64], directions: &[Objective]) -> Result<bool, SearchError> {
    if directions.is_empty()
        || a.len() != directions.len()
        || b.len() != directions.len()
        || a.iter().chain(b).any(|x| !x.is_finite())
    {
        return Err(SearchError::InvalidResults(
            "dominance needs matching, finite, nonempty objective vectors",
        ));
    }
    Ok(dominates_unchecked(a, b, directions))
}

/// Partition points into nondominated fronts, best front first.
///
/// Every returned index refers to `values`. Ties retain input order. An empty
/// population yields no fronts. Directions must be nonempty; scores must be
/// finite and have exactly one entry per direction.
pub fn nondominated_fronts(
    values: &[Vec<f64>],
    directions: &[Objective],
) -> Result<Vec<Vec<usize>>, SearchError> {
    validate(values, directions.len())?;
    let mut dominates_list = vec![Vec::new(); values.len()];
    let mut dominated_by = vec![0usize; values.len()];
    for left in 0..values.len() {
        for right in (left + 1)..values.len() {
            if dominates_unchecked(&values[left], &values[right], directions) {
                dominates_list[left].push(right);
                dominated_by[right] += 1;
            } else if dominates_unchecked(&values[right], &values[left], directions) {
                dominates_list[right].push(left);
                dominated_by[left] += 1;
            }
        }
    }
    let mut current: Vec<_> = (0..values.len())
        .filter(|&i| dominated_by[i] == 0)
        .collect();
    let mut fronts = Vec::new();
    while !current.is_empty() {
        let mut next = Vec::new();
        for &i in &current {
            for &j in &dominates_list[i] {
                dominated_by[j] -= 1;
                if dominated_by[j] == 0 {
                    next.push(j);
                }
            }
        }
        fronts.push(current);
        next.sort_unstable();
        current = next;
    }
    Ok(fronts)
}

/// Normalized NSGA-II crowding distances for points in a single front.
///
/// Returns one distance per input point. Extremes of a varying objective get
/// infinity; interior distances sum normalized adjacent gaps. Constant
/// objectives contribute zero, avoiding arbitrary preference among identical
/// scores. Thus an all-identical front has all-zero distances. The function
/// checks shapes/finiteness but does not require the supplied points to be
/// nondominated. Empty input yields an empty vector.
pub fn crowding_distances(values: &[Vec<f64>]) -> Result<Vec<f64>, SearchError> {
    let Some(first) = values.first() else {
        return Ok(Vec::new());
    };
    validate(values, first.len())?;
    let mut distances = vec![0.0; values.len()];
    for objective in 0..first.len() {
        let mut order: Vec<_> = (0..values.len()).collect();
        order.sort_by(|&a, &b| {
            values[a][objective]
                .partial_cmp(&values[b][objective])
                .expect("finite values")
        });
        let low = values[order[0]][objective];
        let high = values[*order.last().expect("nonempty")][objective];
        if low == high {
            continue;
        }
        // Scaling before subtraction also handles ranges spanning ±f64::MAX.
        let scale = low.abs().max(high.abs());
        let span = high / scale - low / scale;
        distances[order[0]] = f64::INFINITY;
        distances[*order.last().expect("nonempty")] = f64::INFINITY;
        for neighbors in order.windows(3) {
            let gap =
                values[neighbors[2]][objective] / scale - values[neighbors[0]][objective] / scale;
            distances[neighbors[1]] += gap / span;
        }
    }
    Ok(distances)
}
