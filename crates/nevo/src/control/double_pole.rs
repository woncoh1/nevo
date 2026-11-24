use super::{
    ControlError, ControlStep, check_action, environment_impl, finite, nonnegative, positive,
};

/// Whether the controller receives velocities as well as positions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PoleObservations {
    /// Six coordinates: cart position/velocity and both angles/velocities.
    #[default]
    FullState,
    /// Three coordinates: cart position and the two pole angles.
    ///
    /// This is a partially observed control task. A recurrent policy can retain
    /// information across time; the environment does not estimate velocities.
    PositionsOnly,
}

impl PoleObservations {
    /// Number of observed coordinates, excluding any network bias node.
    pub fn dimensions(self) -> usize {
        match self {
            Self::FullState => 6,
            Self::PositionsOnly => 3,
        }
    }
}

/// Two independent poles attached to one cart; coordinates use SI units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoublePoleState {
    /// Horizontal cart position in metres.
    pub position: f64,
    /// Horizontal cart velocity in metres per second.
    pub velocity: f64,
    /// Long and short pole angles in radians from upright, clockwise positive.
    pub angles: [f64; 2],
    /// Long and short pole angular velocities in radians per second.
    pub angular_velocities: [f64; 2],
}

impl Default for DoublePoleState {
    fn default() -> Self {
        Self {
            position: 0.0,
            velocity: 0.0,
            angles: [0.07, 0.0],
            angular_velocities: [0.0; 2],
        }
    }
}

impl DoublePoleState {
    fn values(self) -> [f64; 6] {
        [
            self.position,
            self.velocity,
            self.angles[0],
            self.angular_velocities[0],
            self.angles[1],
            self.angular_velocities[1],
        ]
    }

    fn from_values(v: [f64; 6]) -> Self {
        Self {
            position: v[0],
            velocity: v[1],
            angles: [v[2], v[4]],
            angular_velocities: [v[3], v[5]],
        }
    }
}

/// Checked parameters of the UT NEAT double-pole dynamics.
#[derive(Clone, Debug, PartialEq)]
pub struct DoublePoleConfig {
    /// Gravitational acceleration magnitude in metres per second squared.
    pub gravity: f64,
    /// Cart mass in kilograms.
    pub cart_mass: f64,
    /// Long and short pole masses in kilograms.
    pub pole_masses: [f64; 2],
    /// Pivot-to-centre-of-mass distances in metres, **half** the full lengths.
    pub half_lengths: [f64; 2],
    /// Viscous hinge coefficient in newton-metres per radian per second.
    pub hinge_damping: f64,
    /// Maximum magnitude of the continuous horizontal force, in newtons.
    pub force: f64,
    /// Time for which one action is held, in seconds.
    pub dt: f64,
    /// Number of RK4 substeps per control action; default 2 gives 0.01 s steps.
    pub substeps: usize,
    /// Absolute cart-position failure threshold in metres.
    pub position_limit: f64,
    /// Absolute pole-angle failure threshold in radians.
    pub angle_limit: f64,
    /// Control-step time limit; default is a short teaching episode.
    pub max_steps: usize,
    /// Full-state or position-only controller inputs.
    pub observation: PoleObservations,
}

impl Default for DoublePoleConfig {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            cart_mass: 1.0,
            pole_masses: [0.1, 0.01],
            half_lengths: [0.5, 0.05],
            hinge_damping: 0.000002,
            force: 10.0,
            dt: 0.02,
            substeps: 2,
            position_limit: 2.4,
            // Preserve the constant in the historical simulator (~36 degrees).
            angle_limit: 0.628329,
            max_steps: 1000,
            observation: PoleObservations::FullState,
        }
    }
}

impl DoublePoleConfig {
    fn validate(&self) -> Result<(), ControlError> {
        if !positive(&[
            self.cart_mass,
            self.pole_masses[0],
            self.pole_masses[1],
            self.half_lengths[0],
            self.half_lengths[1],
            self.force,
            self.dt,
            self.position_limit,
            self.angle_limit,
        ]) || !nonnegative(&[self.gravity, self.hinge_damping])
            || self.substeps == 0
            || self.max_steps == 0
        {
            return Err(ControlError::InvalidConfig(
                "double-pole masses, lengths, force, time and horizons must be positive; gravity/damping nonnegative",
            ));
        }
        if !(self.cart_mass + self.pole_masses.iter().sum::<f64>()).is_finite()
            || self
                .pole_masses
                .iter()
                .zip(self.half_lengths)
                .any(|(mass, length)| !(mass * length).is_finite())
            || self.dt / self.substeps as f64 == 0.0
        {
            return Err(ControlError::InvalidConfig(
                "double-pole parameter arithmetic overflows or underflows",
            ));
        }
        Ok(())
    }
}

/// Continuous-force double-pole model, independently implemented from equations.
///
/// Defaults reproduce the dynamics and initial condition in the original UT
/// NEAT simulator: two RK4 steps of 0.01 seconds for each policy action, no cart
/// friction, and the long pole starting at 0.07 radians. A signed action `u`
/// produces `force * u` newtons. Both poles share the cart pivot; they are not a
/// serial, jointed double pendulum.
///
/// Reward is one for each control-step endpoint that remains within bounds,
/// and zero for the first violating endpoint. The default 1,000-step survival
/// objective is a teaching task, not the complete historical non-Markov
/// benchmark protocol with oscillation scoring and generalization trials.
/// Reference: <https://nn.cs.utexas.edu/soft-view.php?SoftID=4>.
#[derive(Clone, Debug)]
pub struct DoublePole {
    config: DoublePoleConfig,
    state: DoublePoleState,
    steps: usize,
    done: bool,
}

impl DoublePole {
    /// Construct an environment at the historical initial state.
    pub fn new(config: DoublePoleConfig) -> Result<Self, ControlError> {
        config.validate()?;
        let mut env = Self {
            config,
            state: DoublePoleState::default(),
            steps: 0,
            done: false,
        };
        env.reset()?;
        Ok(env)
    }

    /// Validated parameters and observation choice.
    pub fn config(&self) -> &DoublePoleConfig {
        &self.config
    }

    /// Current physical state in SI units.
    pub fn state(&self) -> &DoublePoleState {
        &self.state
    }

    /// Number of applied actions, including any final violating action.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Whether physical failure or the horizon ended this episode.
    pub fn done(&self) -> bool {
        self.done
    }

    /// Historical input scaling, with no extra bias coordinate.
    ///
    /// Full-state order is `[x/4.8, v/2, theta1/0.52, omega1/2,
    /// theta2/0.52, omega2/2]`. Position-only order is
    /// `[x/4.8, theta1/0.52, theta2/0.52]`. Values are not clipped.
    pub fn observation(&self) -> Vec<f64> {
        let s = self.state;
        match self.config.observation {
            PoleObservations::FullState => vec![
                s.position / 4.8,
                s.velocity / 2.0,
                s.angles[0] / 0.52,
                s.angular_velocities[0] / 2.0,
                s.angles[1] / 0.52,
                s.angular_velocities[1] / 2.0,
            ],
            PoleObservations::PositionsOnly => {
                vec![s.position / 4.8, s.angles[0] / 0.52, s.angles[1] / 0.52]
            }
        }
    }

    /// Restore the deterministic historical start; no RNG is needed.
    pub fn reset(&mut self) -> Result<Vec<f64>, ControlError> {
        self.reset_to(DoublePoleState::default())
    }

    /// Start a fresh episode at an explicit finite state within bounds.
    pub fn reset_to(&mut self, state: DoublePoleState) -> Result<Vec<f64>, ControlError> {
        if state.values().iter().any(|v| !v.is_finite()) || self.outside(state) {
            return Err(ControlError::InvalidState(
                "double-pole start must be finite and within bounds",
            ));
        }
        self.state = state;
        self.steps = 0;
        self.done = false;
        Ok(self.observation())
    }

    /// Inspect `[cart, long-pole, short-pole] accelerations for a force in N.
    pub fn accelerations(&self, force: f64) -> Result<[f64; 3], ControlError> {
        if !force.is_finite() {
            return Err(ControlError::InvalidAction);
        }
        let d = self.derivative(self.state.values(), force)?;
        Ok([d[1], d[3], d[5]])
    }

    /// Apply one bounded continuous-force action using fixed-step RK4.
    ///
    /// State is committed only after all substeps have succeeded. As in the
    /// historical simulator, failure bounds are checked at the control endpoint.
    pub fn step(&mut self, action: &[f64]) -> Result<ControlStep, ControlError> {
        if self.done {
            return Err(ControlError::EpisodeFinished);
        }
        check_action(action, 1)?;
        let force = action[0] * self.config.force;
        let h = self.config.dt / self.config.substeps as f64;
        let mut y = self.state.values();
        for _ in 0..self.config.substeps {
            let k1 = self.derivative(y, force)?;
            let k2 = self.derivative(std::array::from_fn(|i| y[i] + h * 0.5 * k1[i]), force)?;
            let k3 = self.derivative(std::array::from_fn(|i| y[i] + h * 0.5 * k2[i]), force)?;
            let k4 = self.derivative(std::array::from_fn(|i| y[i] + h * k3[i]), force)?;
            y = std::array::from_fn(|i| {
                y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            });
            finite(&y, "double-pole RK4 integration")?;
        }
        let next = DoublePoleState::from_values(y);
        let terminated = self.outside(next);
        self.state = next;
        self.steps += 1;
        let truncated = self.steps >= self.config.max_steps;
        self.done = terminated || truncated;
        Ok(ControlStep {
            reward: if terminated { 0.0 } else { 1.0 },
            terminated,
            truncated,
        })
    }

    fn derivative(&self, y: [f64; 6], force: f64) -> Result<[f64; 6], ControlError> {
        finite(&y, "double-pole intermediate state")?;
        let mut numerator = force;
        let mut denominator = self.config.cart_mass;
        let mut sin = [0.0; 2];
        let mut cos = [0.0; 2];
        for i in 0..2 {
            let theta = y[2 + i * 2];
            let omega = y[3 + i * 2];
            let mass = self.config.pole_masses[i];
            let length = self.config.half_lengths[i];
            sin[i] = theta.sin();
            cos[i] = theta.cos();
            numerator += mass * length * omega * omega * sin[i]
                - 0.75 * mass * self.config.gravity * sin[i] * cos[i]
                + 0.75 * self.config.hinge_damping * omega * cos[i] / length;
            denominator += mass * (1.0 - 0.75 * cos[i] * cos[i]);
        }
        let cart = numerator / denominator;
        let angular: [f64; 2] = std::array::from_fn(|i| {
            0.75 / self.config.half_lengths[i]
                * (self.config.gravity * sin[i]
                    - cart * cos[i]
                    - self.config.hinge_damping * y[3 + 2 * i]
                        / (self.config.pole_masses[i] * self.config.half_lengths[i]))
        });
        let d = [y[1], cart, y[3], angular[0], y[5], angular[1]];
        finite(&d, "double-pole derivative")?;
        Ok(d)
    }

    fn outside(&self, state: DoublePoleState) -> bool {
        state.position.abs() > self.config.position_limit
            || state
                .angles
                .iter()
                .any(|theta| theta.abs() > self.config.angle_limit)
    }
}

environment_impl!(DoublePole, DoublePoleState);
