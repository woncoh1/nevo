//! Fixed-topology MLP adapters for the headless control environments.

use crate::Evaluator;
use crate::control::{
    CartPole, CartPoleConfig, CartPoleState, ControlError, ControlRollout, DoublePole,
    DoublePoleConfig, DoublePoleState, RolloutError, SpringWalker, SpringWalkerConfig,
    SpringWalkerState,
};
use crate::nn::{Activation, Mlp, NnError};
use std::error::Error;
use std::fmt;

/// Invalid trial suites, neural controllers, or control simulations.
#[derive(Clone, Debug, PartialEq)]
pub enum ControlTaskError {
    /// Every candidate must be evaluated on at least one explicit initial state.
    EmptyStarts,
    /// The environment rejected parameters, an action, or numerical arithmetic.
    Control(ControlError),
    /// The neural network rejected a shape, parameter, or numerical operation.
    Neural(NnError),
}

impl fmt::Display for ControlTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyStarts => write!(f, "a control task requires at least one initial state"),
            Self::Control(error) => error.fmt(f),
            Self::Neural(error) => error.fmt(f),
        }
    }
}

impl Error for ControlTaskError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::EmptyStarts => None,
            Self::Control(error) => Some(error),
            Self::Neural(error) => Some(error),
        }
    }
}

impl From<ControlError> for ControlTaskError {
    fn from(error: ControlError) -> Self {
        Self::Control(error)
    }
}

impl From<NnError> for ControlTaskError {
    fn from(error: NnError) -> Self {
        Self::Neural(error)
    }
}

impl From<RolloutError<NnError>> for ControlTaskError {
    fn from(error: RolloutError<NnError>) -> Self {
        match error {
            RolloutError::Control(error) => Self::Control(error),
            RolloutError::Policy(error) => Self::Neural(error),
        }
    }
}

macro_rules! mlp_control_task {
    ($(#[$doc:meta])* $task:ident, $env:ty, $config:ty, $state:ty, $inputs:expr, $outputs:expr) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        pub struct $task {
            config: $config,
            starts: Vec<$state>,
            widths: [usize; 3],
            dimensions: usize,
        }

        impl $task {
            /// Construct a checked task from explicit training starts and a hidden width.
            ///
            /// Each evaluation resets the physical state for every start. Both MLP
            /// layers use tanh; the parameter order is the existing `Mlp` encoding.
            /// The trial suite is fixed, so the evaluator's job seed is not used.
            pub fn new(config: $config, starts: Vec<$state>, hidden: usize) -> Result<Self, ControlTaskError> {
                if starts.is_empty() { return Err(ControlTaskError::EmptyStarts); }
                let widths = [($inputs)(&config), hidden, $outputs];
                let dimensions = Mlp::parameter_count(&widths)?;
                let mut env = <$env>::new(config.clone())?;
                for &start in &starts { env.reset_to(start)?; }
                Ok(Self { config, starts, widths, dimensions })
            }

            /// Number of weights and biases in the evolved controller.
            pub fn dimensions(&self) -> usize { self.dimensions }

            /// Input, hidden, and output dimensions of the fixed topology.
            pub fn widths(&self) -> &[usize; 3] { &self.widths }

            /// Shared environment parameters used for all candidates.
            pub fn config(&self) -> &$config { &self.config }

            /// Explicit training initial conditions, reused for every candidate.
            pub fn starts(&self) -> &[$state] { &self.starts }

            /// Decode a flat parameter vector into the tanh policy network.
            pub fn network(&self, parameters: Vec<f64>) -> Result<Mlp, ControlTaskError> {
                Ok(Mlp::new(&self.widths, &[Activation::Tanh, Activation::Tanh], parameters)?)
            }

            /// Negative mean episode return across the configured starts; minimize it.
            ///
            /// Episodes are complete and independent. Training does not retain traces,
            /// so memory use is independent of the episode horizon. No policy gradients
            /// or neural updates occur during evaluation.
            pub fn value(&self, parameters: &[f64]) -> Result<f64, ControlTaskError> {
                let network = self.network(parameters.to_vec())?;
                let mut env = <$env>::new(self.config.clone())?;
                let mut objective = 0.0;
                for &start in &self.starts {
                    env.reset_to(start)?;
                    loop {
                        let action = network.forward(&env.observation())?;
                        let result = env.step(&action)?;
                        objective -= result.reward / self.starts.len() as f64;
                        if !objective.is_finite() {
                            return Err(ControlError::Numerical("control-task aggregate return").into());
                        }
                        if result.done() { break; }
                    }
                }
                Ok(objective)
            }

            /// Record one episode, allowing a held-out start without changing training.
            pub fn rollout(&self, parameters: &[f64], start: $state) -> Result<ControlRollout<$state>, ControlTaskError> {
                let network = self.network(parameters.to_vec())?;
                let mut env = <$env>::new(self.config.clone())?;
                env.reset_to(start)?;
                Ok(env.rollout(|observation| network.forward(observation))?)
            }
        }

        impl<T: AsRef<[f64]> + ?Sized> Evaluator<T> for $task {
            type Output = f64;
            type Error = ControlTaskError;
            fn evaluate(&mut self, input: &T, _seed: u64) -> Result<Self::Output, Self::Error> {
                self.value(input.as_ref())
            }
        }
    };
}

mlp_control_task!(
    /// Evolve a fixed-topology tanh MLP for classic discrete-force cart-pole.
    ///
    /// Four state inputs and one signed action; hidden width 4 gives 25 genes.
    /// Reward follows classic Gym's per-action +1 convention, including failure.
    CartPoleTask, CartPole, CartPoleConfig, CartPoleState, |_: &CartPoleConfig| 4, 1
);

mlp_control_task!(
    /// Evolve a fixed-topology MLP for full-state or position-only double-pole.
    ///
    /// The observation choice determines 6 versus 3 inputs; there is one
    /// continuous-force output. A position-only MLP has no recurrent memory.
    /// The objective counts valid control endpoints, not the historical
    /// non-Markov benchmark's oscillation/generalization scoring protocol.
    DoublePoleTask, DoublePole, DoublePoleConfig, DoublePoleState,
    |config: &DoublePoleConfig| config.observation.dimensions(), 1
);

mlp_control_task!(
    /// Evolve a fixed-topology MLP for the original three-particle spring robot.
    ///
    /// Thirteen observations and two rest-length actions; hidden width 4 gives
    /// 66 parameters. Reward combines centre-of-mass displacement, positive
    /// actuator-work cost, and falling penalty. It is not BipedalWalker reward.
    SpringWalkerTask, SpringWalker, SpringWalkerConfig, SpringWalkerState,
    |_: &SpringWalkerConfig| 13, 2
);
