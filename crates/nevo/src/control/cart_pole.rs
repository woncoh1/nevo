use super::{
    ControlError, ControlStep, check_action, environment_impl, finite, positive, reset_rng,
    signed_uniform,
};

/// Physical state of one cart and one pole, in SI units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CartPoleState {
    /// Cart position in metres, positive to the right.
    pub position: f64,
    /// Cart velocity in metres per second.
    pub velocity: f64,
    /// Pole angle in radians from upright, positive clockwise.
    pub angle: f64,
    /// Pole angular velocity in radians per second.
    pub angular_velocity: f64,
}

impl CartPoleState {
    fn values(self) -> [f64; 4] {
        [
            self.position,
            self.velocity,
            self.angle,
            self.angular_velocity,
        ]
    }
}

/// Parameters of the classic Gym cart-pole model.
#[derive(Clone, Debug, PartialEq)]
pub struct CartPoleConfig {
    /// Gravitational acceleration magnitude, in metres per second squared.
    pub gravity: f64,
    /// Cart mass in kilograms.
    pub cart_mass: f64,
    /// Pole mass in kilograms.
    pub pole_mass: f64,
    /// Pivot-to-centre-of-mass distance, half the full pole length, in metres.
    pub half_length: f64,
    /// Magnitude of the two discrete horizontal forces, in newtons.
    pub force: f64,
    /// Explicit Euler control interval in seconds.
    pub dt: f64,
    /// Absolute cart-position failure threshold in metres.
    pub position_limit: f64,
    /// Absolute pole-angle failure threshold in radians.
    pub angle_limit: f64,
    /// Number of control actions before time-limit truncation.
    pub max_steps: usize,
}

impl Default for CartPoleConfig {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            cart_mass: 1.0,
            pole_mass: 0.1,
            half_length: 0.5,
            force: 10.0,
            dt: 0.02,
            position_limit: 2.4,
            angle_limit: 12.0 * std::f64::consts::PI / 180.0,
            max_steps: 500,
        }
    }
}

impl CartPoleConfig {
    fn validate(&self) -> Result<(), ControlError> {
        if !positive(&[
            self.gravity,
            self.cart_mass,
            self.pole_mass,
            self.half_length,
            self.force,
            self.dt,
            self.position_limit,
            self.angle_limit,
        ]) || self.max_steps == 0
        {
            return Err(ControlError::InvalidConfig(
                "cart-pole parameters and horizon must be positive and finite",
            ));
        }
        if !((self.cart_mass + self.pole_mass).is_finite()
            && (self.pole_mass * self.half_length).is_finite())
        {
            return Err(ControlError::InvalidConfig(
                "cart-pole parameter products overflow",
            ));
        }
        Ok(())
    }
}

/// Classic cart-pole with Gym's explicit Euler update and discrete forces.
///
/// A policy supplies one bounded value: values below zero push left, while
/// zero and positive values push right. Thus a tanh policy maps to Gym's two
/// actions; this is not the continuous-force cart-pole variant. Reward is one
/// per applied action, including the transition that first crosses a failure
/// boundary. Stepping a finished episode is an error.
///
/// The reset distribution matches Gym's uniform [-0.05, 0.05] convention, but
/// uses ChaCha8 rather than NumPy, so seed numbers do not match Gym trajectories.
/// Reference: <https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py>.
#[derive(Clone, Debug)]
pub struct CartPole {
    config: CartPoleConfig,
    state: CartPoleState,
    steps: usize,
    done: bool,
}

impl CartPole {
    /// Construct a checked environment in the exactly upright, stationary state.
    pub fn new(config: CartPoleConfig) -> Result<Self, ControlError> {
        config.validate()?;
        Ok(Self {
            config,
            state: CartPoleState::default(),
            steps: 0,
            done: false,
        })
    }

    /// Validated physical parameters and horizon.
    pub fn config(&self) -> &CartPoleConfig {
        &self.config
    }

    /// Current physical state.
    pub fn state(&self) -> &CartPoleState {
        &self.state
    }

    /// Number of actions applied since the last reset.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Whether a task boundary or time limit ended the episode.
    pub fn done(&self) -> bool {
        self.done
    }

    /// Raw `[position, velocity, angle, angular_velocity]`, as in classic Gym.
    pub fn observation(&self) -> Vec<f64> {
        self.state.values().to_vec()
    }

    /// Reset with explicit randomness, uniformly sampling each coordinate.
    ///
    /// Custom thresholds smaller than 0.05 use a reduced position/angle range
    /// so every sampled initial state is valid. Default thresholds exactly use
    /// the [-0.05, 0.05] distribution.
    pub fn reset(&mut self, seed: u64) -> Vec<f64> {
        let mut rng = reset_rng(seed);
        self.state = CartPoleState {
            position: 0.05_f64.min(self.config.position_limit * 0.5) * signed_uniform(&mut rng),
            velocity: 0.05 * signed_uniform(&mut rng),
            angle: 0.05_f64.min(self.config.angle_limit * 0.5) * signed_uniform(&mut rng),
            angular_velocity: 0.05 * signed_uniform(&mut rng),
        };
        self.steps = 0;
        self.done = false;
        self.observation()
    }

    /// Reset to an explicit, finite initial state within the task boundaries.
    pub fn reset_to(&mut self, state: CartPoleState) -> Result<Vec<f64>, ControlError> {
        if state.values().iter().any(|v| !v.is_finite()) || self.outside(state) {
            return Err(ControlError::InvalidState(
                "cart-pole start must be finite and within bounds",
            ));
        }
        self.state = state;
        self.steps = 0;
        self.done = false;
        Ok(self.observation())
    }

    /// Compute `[cart acceleration, angular acceleration]` for a force in N.
    ///
    /// Unlike `step`, this inspection method accepts any finite force, including
    /// zero, without changing the state or discretizing the action.
    pub fn accelerations(&self, force: f64) -> Result<[f64; 2], ControlError> {
        if !force.is_finite() {
            return Err(ControlError::InvalidAction);
        }
        let c = self.state.angle.cos();
        let s = self.state.angle.sin();
        let total_mass = self.config.cart_mass + self.config.pole_mass;
        let pole_mass_length = self.config.pole_mass * self.config.half_length;
        let temp =
            (force + pole_mass_length * self.state.angular_velocity.powi(2) * s) / total_mass;
        let angular = (self.config.gravity * s - c * temp)
            / (self.config.half_length * (4.0 / 3.0 - self.config.pole_mass * c * c / total_mass));
        let cart = temp - pole_mass_length * angular * c / total_mass;
        finite(&[cart, angular], "cart-pole accelerations")?;
        Ok([cart, angular])
    }

    /// Apply one signed policy action using the classic explicit Euler update.
    pub fn step(&mut self, action: &[f64]) -> Result<ControlStep, ControlError> {
        if self.done {
            return Err(ControlError::EpisodeFinished);
        }
        check_action(action, 1)?;
        let force = if action[0] < 0.0 {
            -self.config.force
        } else {
            self.config.force
        };
        let [cart, angular] = self.accelerations(force)?;
        let dt = self.config.dt;
        let next = CartPoleState {
            position: self.state.position + dt * self.state.velocity,
            velocity: self.state.velocity + dt * cart,
            angle: self.state.angle + dt * self.state.angular_velocity,
            angular_velocity: self.state.angular_velocity + dt * angular,
        };
        finite(&next.values(), "cart-pole integration")?;
        let terminated = self.outside(next);
        self.state = next;
        self.steps += 1;
        let truncated = self.steps >= self.config.max_steps;
        self.done = terminated || truncated;
        Ok(ControlStep {
            reward: 1.0,
            terminated,
            truncated,
        })
    }

    fn outside(&self, state: CartPoleState) -> bool {
        state.position.abs() > self.config.position_limit
            || state.angle.abs() > self.config.angle_limit
    }
}

environment_impl!(CartPole, CartPoleState);
