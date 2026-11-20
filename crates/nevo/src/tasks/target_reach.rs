use std::error::Error;
use std::fmt;

use crate::Evaluator;

/// A one-dimensional control task with a target at the origin.
///
/// A fixed-topology neuron receives position error and velocity:
/// `action = tanh(w[0] * (-position) + w[1] * velocity + w[2])`.
/// Its three genes are two weights and a bias. The action accelerates a point
/// mass with drag. A lower mean squared position error plus action cost is better.
///
/// This is an original teaching task, not a benchmark from the textbook.
/// Its deterministic trials restart from zero velocity on every evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetReach {
    starts: Vec<f64>,
    steps: usize,
}

impl TargetReach {
    /// Two input weights and one bias.
    pub const PARAMETERS: usize = 3;

    /// Create a task with explicit starting positions and a positive horizon.
    ///
    /// # Errors
    /// Rejects empty/nonfinite starts and a zero-step horizon.
    pub fn new(starts: Vec<f64>, steps: usize) -> Result<Self, TargetReachError> {
        if starts.is_empty() || starts.iter().any(|x| !x.is_finite()) || steps == 0 {
            return Err(TargetReachError::InvalidTask);
        }
        Ok(Self { starts, steps })
    }

    /// Starting positions used to score each candidate.
    pub fn starts(&self) -> &[f64] {
        &self.starts
    }

    /// Number of simulation steps in each episode.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Evaluate one controller across all configured starting positions.
    ///
    /// The objective averages `position_after_step^2 + 0.01 * action^2` over
    /// steps and episodes. No action targets, gradients, or learning occur
    /// inside an episode. The evaluation seed is not needed for this task.
    ///
    /// # Errors
    /// Rejects invalid genomes and nonfinite simulation arithmetic.
    pub fn value(&self, weights: &[f64]) -> Result<f64, TargetReachError> {
        Self::check_weights(weights)?;
        let mut total = 0.0;
        for &start in &self.starts {
            total += self.simulate(weights, start, None)? / self.starts.len() as f64;
            if !total.is_finite() {
                return Err(TargetReachError::NonFiniteSimulation);
            }
        }
        Ok(total)
    }

    /// Record an episode from a chosen start, including the initial state.
    ///
    /// The chosen start need not be among the task's training starts. This
    /// permits held-out visualization without changing the training objective.
    ///
    /// # Errors
    /// Rejects invalid weights, a nonfinite start, or simulation overflow.
    pub fn rollout(&self, weights: &[f64], start: f64) -> Result<Rollout, TargetReachError> {
        Self::check_weights(weights)?;
        if !start.is_finite() {
            return Err(TargetReachError::InvalidTask);
        }
        let mut states = Vec::new();
        let loss = self.simulate(weights, start, Some(&mut states))?;
        Ok(Rollout { loss, states })
    }

    fn check_weights(weights: &[f64]) -> Result<(), TargetReachError> {
        if weights.len() != Self::PARAMETERS {
            return Err(TargetReachError::DimensionMismatch {
                actual: weights.len(),
            });
        }
        if weights.iter().any(|x| !x.is_finite()) {
            return Err(TargetReachError::NonFiniteWeights);
        }
        Ok(())
    }

    fn simulate(
        &self,
        weights: &[f64],
        start: f64,
        mut trace: Option<&mut Vec<State>>,
    ) -> Result<f64, TargetReachError> {
        let mut position = start;
        let mut velocity = 0.0;
        let mut loss = 0.0;
        if let Some(states) = trace.as_mut() {
            states.push(State {
                step: 0,
                position,
                velocity,
                action: 0.0,
            });
        }
        for step in 1..=self.steps {
            let activation = weights[0] * (-position) + weights[1] * velocity + weights[2];
            // Reject overflow before tanh could hide it by saturating.
            if !activation.is_finite() {
                return Err(TargetReachError::NonFiniteSimulation);
            }
            let action = activation.tanh();
            velocity = 0.9 * velocity + 0.2 * action;
            position += 0.1 * velocity;
            let cost = position * position + 0.01 * action * action;
            loss += cost / self.steps as f64;
            if !position.is_finite()
                || !velocity.is_finite()
                || !cost.is_finite()
                || !loss.is_finite()
            {
                return Err(TargetReachError::NonFiniteSimulation);
            }
            if let Some(states) = trace.as_mut() {
                states.push(State {
                    step,
                    position,
                    velocity,
                    action,
                });
            }
        }
        Ok(loss)
    }
}

impl Default for TargetReach {
    fn default() -> Self {
        Self {
            starts: vec![-1.0, -0.5, 0.5, 1.0],
            steps: 80,
        }
    }
}

impl<T: AsRef<[f64]> + ?Sized> Evaluator<T> for TargetReach {
    type Output = f64;
    type Error = TargetReachError;

    fn evaluate(&mut self, input: &T, _seed: u64) -> Result<f64, Self::Error> {
        self.value(input.as_ref())
    }
}

/// An episode trace and its mean cost.
#[derive(Clone, Debug, PartialEq)]
pub struct Rollout {
    /// The same per-episode loss used in fitness evaluation.
    pub loss: f64,
    /// Initial state followed by one state per simulation step.
    pub states: Vec<State>,
}

/// A recorded physical state after an action (step zero is initialization).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    /// Simulation step; zero denotes the initial condition.
    pub step: usize,
    /// Position relative to the target at zero.
    pub position: f64,
    /// Current velocity.
    pub velocity: f64,
    /// Action that produced this state; zero for initialization.
    pub action: f64,
}

/// Invalid configuration, genome, or simulation arithmetic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TargetReachError {
    /// Starts must be finite and nonempty, and the horizon positive.
    InvalidTask,
    /// A controller must contain exactly three genes.
    DimensionMismatch {
        /// Supplied genome length.
        actual: usize,
    },
    /// A weight or bias is NaN or infinite.
    NonFiniteWeights,
    /// A controller or trajectory calculation overflowed.
    NonFiniteSimulation,
}

impl fmt::Display for TargetReachError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTask => write!(
                f,
                "target-reaching requires finite starts and a positive horizon"
            ),
            Self::DimensionMismatch { actual } => {
                write!(f, "controller requires 3 genes, received {actual}")
            }
            Self::NonFiniteWeights => write!(f, "controller genes must be finite"),
            Self::NonFiniteSimulation => {
                write!(f, "controller or simulation arithmetic is not finite")
            }
        }
    }
}

impl Error for TargetReachError {}
