//! Small, trace-free episode and trial reports, separate from scalar fitness.

use super::{ControlError, ControlRollout, RolloutError};

/// An episode's outcome without retaining states or actions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpisodeSummary {
    /// Sum of rewards returned by the environment.
    pub total_reward: f64,
    /// Applied actions, including a failing final action when present.
    pub steps: usize,
    /// Whether a physical task boundary ended the episode.
    pub terminated: bool,
    /// Whether the time limit was reached.
    pub truncated: bool,
}

impl EpisodeSummary {
    /// Reached the horizon without simultaneously crossing a task boundary.
    pub fn horizon_hit(&self) -> bool {
        self.truncated && !self.terminated
    }
}

impl<S> From<&ControlRollout<S>> for EpisodeSummary {
    fn from(rollout: &ControlRollout<S>) -> Self {
        Self {
            total_reward: rollout.total_reward,
            steps: rollout.steps,
            terminated: rollout.terminated,
            truncated: rollout.truncated,
        }
    }
}

/// Descriptive statistics over independent episodes on explicit initial states.
///
/// This report is diagnostic data, not a mandatory optimizer output type.
/// Algorithms can still receive a single score such as `mean_return`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrialSummary {
    /// Number of completed episodes.
    pub episodes: usize,
    /// Total applied control actions over all episodes.
    pub control_steps: usize,
    /// Arithmetic mean of the episode returns.
    pub mean_return: f64,
    /// Sample standard deviation, using n - 1; undefined for one episode.
    pub sample_std_dev: Option<f64>,
    /// Smallest episode return.
    pub worst_return: f64,
    /// Episodes reaching the horizon without a physical failure.
    pub horizon_hits: usize,
}

/// Evaluate a nonempty suite of explicit starts without retaining episode traces.
///
/// The closure must reset the environment and any policy memory for each start.
/// Failures propagate immediately; no partial report is returned. This helper
/// accepts any start type and does not require an environment or evaluator trait.
/// Returns must be finite; count overflow and nonfinite aggregate arithmetic are
/// reported as control errors. The sample deviation is computed with Welford's
/// recurrence in suite order.
pub fn evaluate_trials<S, P, E>(
    starts: &[S],
    mut episode: P,
) -> Result<TrialSummary, RolloutError<E>>
where
    P: FnMut(&S) -> Result<EpisodeSummary, RolloutError<E>>,
{
    if starts.is_empty() {
        return Err(RolloutError::Control(ControlError::InvalidConfig(
            "a trial suite requires at least one initial state",
        )));
    }
    let mut summary = TrialSummary {
        episodes: 0,
        control_steps: 0,
        mean_return: 0.0,
        sample_std_dev: None,
        worst_return: f64::INFINITY,
        horizon_hits: 0,
    };
    let mut squared_deviations = 0.0;
    for start in starts {
        let result = episode(start)?;
        super::finite(&[result.total_reward], "trial return").map_err(RolloutError::Control)?;
        summary.episodes += 1;
        summary.control_steps =
            summary
                .control_steps
                .checked_add(result.steps)
                .ok_or(RolloutError::Control(ControlError::Numerical(
                    "trial control-step count overflow",
                )))?;
        let difference = result.total_reward - summary.mean_return;
        summary.mean_return += difference / summary.episodes as f64;
        squared_deviations += difference * (result.total_reward - summary.mean_return);
        super::finite(
            &[summary.mean_return, squared_deviations],
            "trial aggregate return",
        )
        .map_err(RolloutError::Control)?;
        summary.worst_return = summary.worst_return.min(result.total_reward);
        summary.horizon_hits += usize::from(result.horizon_hit());
    }
    if summary.episodes > 1 {
        summary.sample_std_dev =
            Some((squared_deviations.max(0.0) / (summary.episodes - 1) as f64).sqrt());
    }
    Ok(summary)
}
