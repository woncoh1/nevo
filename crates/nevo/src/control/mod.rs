//! Deterministic, headless control environments with explicit physical state.
//!
//! [`CartPole`] follows the classic Gym cart-pole equations and discrete-force
//! actions. [`DoublePole`] follows the dynamics in the original UT NEAT
//! double-pole experiment. [`SpringWalker`] is an original three-particle
//! teaching robot; it is **not** Gym's Box2D BipedalWalker benchmark.
//!
//! All environments use fixed simulation steps and ordinary `f64` arithmetic.
//! Reproducibility means the same inputs reproduce on the same implementation
//! and platform; cross-platform bit identity of transcendental functions is
//! not promised. Rendering and neural policies are independent of the physics.

mod cart_pole;
mod double_pole;
mod spring_walker;

pub use cart_pole::{CartPole, CartPoleConfig, CartPoleState};
pub use double_pole::{DoublePole, DoublePoleConfig, DoublePoleState, PoleObservations};
pub use spring_walker::{Particle, SpringWalker, SpringWalkerConfig, SpringWalkerState};

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use std::error::Error;
use std::fmt;

/// Outcome of one control action, after all of its physics substeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlStep {
    /// Environment-specific reward for this action; larger is better.
    pub reward: f64,
    /// A physical task boundary was reached.
    pub terminated: bool,
    /// The configured control-step horizon was reached.
    pub truncated: bool,
}

impl ControlStep {
    /// Whether either a task boundary or the time limit ended the episode.
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// A recorded episode, including its initial state.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlRollout<S> {
    /// Sum of the step rewards actually returned by the environment.
    pub total_reward: f64,
    /// Number of applied actions, including a terminal transition if present.
    pub steps: usize,
    /// Whether a physical task boundary ended the episode.
    pub terminated: bool,
    /// Whether the configured time limit was reached.
    pub truncated: bool,
    /// Initial state followed by each resulting control-step state.
    pub states: Vec<S>,
    /// One action vector for each transition between recorded states.
    pub actions: Vec<Vec<f64>>,
}

/// Invalid environment parameters, actions, state, or numerical arithmetic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControlError {
    /// A physical parameter or horizon is outside its documented domain.
    InvalidConfig(&'static str),
    /// A reset state is nonfinite, degenerate, or already outside task bounds.
    InvalidState(&'static str),
    /// The action vector has the wrong number of coordinates.
    ActionLength {
        /// Required action dimension.
        expected: usize,
        /// Supplied action dimension.
        actual: usize,
    },
    /// An action coordinate is not finite or is outside the interval [-1, 1].
    InvalidAction,
    /// Stepping or rolling out requires resetting a finished episode first.
    EpisodeFinished,
    /// A physical computation produced a nonfinite or degenerate value.
    Numerical(&'static str),
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid control configuration: {message}"),
            Self::InvalidState(message) => write!(f, "invalid control state: {message}"),
            Self::ActionLength { expected, actual } => {
                write!(
                    f,
                    "action requires {expected} coordinates, received {actual}"
                )
            }
            Self::InvalidAction => write!(f, "actions must be finite and lie in [-1, 1]"),
            Self::EpisodeFinished => {
                write!(f, "reset the environment before starting another episode")
            }
            Self::Numerical(message) => write!(f, "control computation failed: {message}"),
        }
    }
}

impl Error for ControlError {}

/// Distinguishes a failed policy computation from a failed simulation.
#[derive(Clone, Debug, PartialEq)]
pub enum RolloutError<E> {
    /// The environment rejected an action or encountered a numerical failure.
    Control(ControlError),
    /// The policy returned its own typed error.
    Policy(E),
}

impl<E: fmt::Display> fmt::Display for RolloutError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Control(error) => error.fmt(f),
            Self::Policy(error) => write!(f, "policy failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for RolloutError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(match self {
            Self::Control(error) => error,
            Self::Policy(error) => error,
        })
    }
}

pub(crate) fn check_action(action: &[f64], expected: usize) -> Result<(), ControlError> {
    if action.len() != expected {
        return Err(ControlError::ActionLength {
            expected,
            actual: action.len(),
        });
    }
    if action
        .iter()
        .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
    {
        return Err(ControlError::InvalidAction);
    }
    Ok(())
}

pub(crate) fn positive(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && *v > 0.0)
}

pub(crate) fn nonnegative(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && *v >= 0.0)
}

pub(crate) fn finite(values: &[f64], context: &'static str) -> Result<(), ControlError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(ControlError::Numerical(context))
    }
}

pub(crate) fn reset_rng(seed: u64) -> ChaCha8Rng {
    // Environment randomness is independent of search/distribution sampling.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(3);
    rng
}

pub(crate) fn signed_uniform(rng: &mut ChaCha8Rng) -> f64 {
    2.0 * ((rng.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0) - 1.0
}

// Private implementation sharing; downstream code does not need another trait.
trait Environment {
    type State: Clone;
    fn state(&self) -> &Self::State;
    fn observation(&self) -> Vec<f64>;
    fn step(&mut self, action: &[f64]) -> Result<ControlStep, ControlError>;
    fn done(&self) -> bool;
}

fn rollout<Env, P, E>(
    env: &mut Env,
    mut policy: P,
) -> Result<ControlRollout<Env::State>, RolloutError<E>>
where
    Env: Environment,
    P: FnMut(&[f64]) -> Result<Vec<f64>, E>,
{
    if env.done() {
        return Err(RolloutError::Control(ControlError::EpisodeFinished));
    }
    let mut result = ControlRollout {
        total_reward: 0.0,
        steps: 0,
        terminated: false,
        truncated: false,
        states: vec![env.state().clone()],
        actions: Vec::new(),
    };
    loop {
        let action = policy(&env.observation()).map_err(RolloutError::Policy)?;
        let outcome = env.step(&action).map_err(RolloutError::Control)?;
        result.total_reward += outcome.reward;
        finite(&[result.total_reward], "episode return").map_err(RolloutError::Control)?;
        result.steps += 1;
        result.terminated = outcome.terminated;
        result.truncated = outcome.truncated;
        result.states.push(env.state().clone());
        result.actions.push(action);
        if outcome.done() {
            return Ok(result);
        }
    }
}

macro_rules! environment_impl {
    ($environment:ty, $state:ty) => {
        impl $crate::control::Environment for $environment {
            type State = $state;
            fn state(&self) -> &Self::State {
                self.state()
            }
            fn observation(&self) -> Vec<f64> {
                self.observation()
            }
            fn step(
                &mut self,
                action: &[f64],
            ) -> Result<$crate::control::ControlStep, $crate::control::ControlError> {
                self.step(action)
            }
            fn done(&self) -> bool {
                self.done()
            }
        }
        impl $environment {
            /// Record an episode from the current state using a fallible policy.
            ///
            /// Reset the environment and any recurrent policy state before each
            /// independent episode. The policy receives the documented observation
            /// vector and must return actions in [-1, 1].
            pub fn rollout<P, E>(
                &mut self,
                policy: P,
            ) -> Result<$crate::control::ControlRollout<$state>, $crate::control::RolloutError<E>>
            where
                P: FnMut(&[f64]) -> Result<Vec<f64>, E>,
            {
                $crate::control::rollout(self, policy)
            }
        }
    };
}

use environment_impl;
