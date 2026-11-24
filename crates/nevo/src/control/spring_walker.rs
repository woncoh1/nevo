use super::{
    ControlError, ControlStep, check_action, environment_impl, finite, nonnegative, positive,
    reset_rng, signed_uniform,
};

type InternalAndActuatorForces = ([[f64; 2]; 3], [[f64; 2]; 2]);

/// A planar point mass with position in metres and velocity in metres/second.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Particle {
    /// Horizontal and vertical position; the flat ground is at y = 0.
    pub position: [f64; 2],
    /// Horizontal and vertical velocity.
    pub velocity: [f64; 2],
}

/// Complete mechanical state of the original three-particle robot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringWalkerState {
    /// Torso, left-labelled foot, and right-labelled foot, in that order.
    ///
    /// Labels are persistent particle identities, not a sort by x coordinate.
    pub particles: [Particle; 3],
}

impl Default for SpringWalkerState {
    fn default() -> Self {
        Self {
            particles: [
                Particle {
                    position: [0.0, (1.1_f64.powi(2) - 0.6_f64.powi(2)).sqrt() + 0.06],
                    velocity: [0.0; 2],
                },
                Particle {
                    position: [-0.6, 0.06],
                    velocity: [0.0; 2],
                },
                Particle {
                    position: [0.6, 0.06],
                    velocity: [0.0; 2],
                },
            ],
        }
    }
}

/// Parameters of the original spring robot; this is not a Gym benchmark.
#[derive(Clone, Debug, PartialEq)]
pub struct SpringWalkerConfig {
    /// Torso and two foot masses, in kilograms.
    pub masses: [f64; 3],
    /// Gravitational acceleration magnitude in metres/second squared.
    pub gravity: f64,
    /// Unactuated torso-to-foot spring rest length in metres.
    pub leg_length: f64,
    /// Passive foot-to-foot spring rest length in metres.
    pub foot_spacing: f64,
    /// Torso-to-foot spring stiffness, in newtons/metre.
    pub leg_stiffness: f64,
    /// Foot-to-foot spring stiffness, in newtons/metre.
    pub spacing_stiffness: f64,
    /// Axial viscous damping of all springs, in newton-seconds/metre.
    pub spring_damping: f64,
    /// Maximum relative change in effective leg rest length, strictly in (0, 1).
    pub actuation_fraction: f64,
    /// Whether gravity acts above a flat, frictional contact surface.
    ///
    /// Turning this off removes contact and falling termination, useful for
    /// checking momentum conservation. Gravity is controlled independently.
    pub ground: bool,
    /// Ground penalty stiffness, in newtons/metre.
    pub ground_stiffness: f64,
    /// Vertical contact damping, in newton-seconds/metre.
    pub ground_damping: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
    /// Torso contact radius, in metres; torso contact ends the episode.
    pub torso_radius: f64,
    /// Contact radius of each foot, in metres.
    pub foot_radius: f64,
    /// Duration of a policy action, in seconds.
    pub dt: f64,
    /// Fixed semi-implicit Euler substeps per policy action.
    pub substeps: usize,
    /// Maximum number of policy actions.
    pub max_steps: usize,
    /// Metres of reward deducted per joule of positive actuator work.
    pub energy_cost: f64,
    /// Reward deduction when the torso first reaches the ground.
    pub fall_penalty: f64,
}

impl Default for SpringWalkerConfig {
    fn default() -> Self {
        Self {
            masses: [1.0, 0.2, 0.2],
            gravity: 9.81,
            leg_length: 1.1,
            foot_spacing: 1.2,
            leg_stiffness: 120.0,
            spacing_stiffness: 80.0,
            spring_damping: 2.0,
            actuation_fraction: 0.25,
            ground: true,
            ground_stiffness: 2000.0,
            ground_damping: 20.0,
            friction: 0.8,
            torso_radius: 0.15,
            foot_radius: 0.06,
            dt: 0.02,
            substeps: 20,
            max_steps: 500,
            energy_cost: 0.002,
            fall_penalty: 2.0,
        }
    }
}

impl SpringWalkerConfig {
    fn validate(&self) -> Result<(), ControlError> {
        if !positive(&self.masses)
            || !positive(&[
                self.leg_length,
                self.foot_spacing,
                self.leg_stiffness,
                self.spacing_stiffness,
                self.actuation_fraction,
                self.ground_stiffness,
                self.torso_radius,
                self.foot_radius,
                self.dt,
            ])
            || !nonnegative(&[
                self.gravity,
                self.spring_damping,
                self.ground_damping,
                self.friction,
                self.energy_cost,
                self.fall_penalty,
            ])
            || self.actuation_fraction >= 1.0
            || self.substeps == 0
            || self.max_steps == 0
            || self.foot_spacing >= 2.0 * self.leg_length
        {
            return Err(ControlError::InvalidConfig(
                "invalid spring-robot mass, geometry, stiffness, action range, or horizon",
            ));
        }
        let min_mass = self.masses.iter().copied().fold(f64::INFINITY, f64::min);
        let h = self.dt / self.substeps as f64;
        let contact_stiffness = if self.ground {
            self.ground_stiffness
        } else {
            0.0
        };
        let contact_damping = if self.ground {
            self.ground_damping
        } else {
            0.0
        };
        // A conservative guard, not a proof of arbitrary contact stability.
        let stiffness_scale =
            (2.0 * self.leg_stiffness + 2.0 * self.spacing_stiffness + contact_stiffness)
                / min_mass;
        let damping_scale = (4.0 * self.spring_damping + contact_damping) / min_mass;
        if !h.is_finite()
            || h <= 0.0
            || !stiffness_scale.is_finite()
            || !damping_scale.is_finite()
            || h * stiffness_scale.sqrt() > 0.5
            || h * damping_scale > 0.5
            || !self.masses.iter().sum::<f64>().is_finite()
        {
            return Err(ControlError::InvalidConfig(
                "spring-robot substeps are too large for the configured stiffness/damping and masses",
            ));
        }
        Ok(())
    }

    /// Symmetric stationary initial geometry for these leg and foot dimensions.
    pub fn initial_state(&self) -> Result<SpringWalkerState, ControlError> {
        self.validate()?;
        let half_spacing = 0.5 * self.foot_spacing;
        let height = (self.leg_length.powi(2) - half_spacing.powi(2)).sqrt() + self.foot_radius;
        if !height.is_finite() || (self.ground && height <= self.torso_radius) {
            return Err(ControlError::InvalidConfig(
                "initial torso height must be finite and above the ground",
            ));
        }
        Ok(SpringWalkerState {
            particles: [
                Particle {
                    position: [0.0, height],
                    velocity: [0.0; 2],
                },
                Particle {
                    position: [-half_spacing, self.foot_radius],
                    velocity: [0.0; 2],
                },
                Particle {
                    position: [half_spacing, self.foot_radius],
                    velocity: [0.0; 2],
                },
            ],
        })
    }
}

/// Original planar locomotion task with three masses and two actuated springs.
///
/// Two bounded actions set effective leg rest lengths
/// `leg_length * (1 + actuation_fraction * action)`. Equivalently, each leg has
/// a fixed passive spring plus a bounded axial actuator force. This latter
/// interpretation defines actuator work: sum the positive force-displacement
/// work of each actuator over substeps. Changing an action alone does not add
/// instantaneous, fictitious mechanical energy. There are no direct horizontal
/// propulsion forces: motion comes from internal forces and ground reactions.
///
/// Ground normals are nonadhesive penalty spring/damper forces. Coulomb friction
/// is a velocity-reducing impulse capped by `mu * normal_force * dt`, with no
/// reversal through zero. Fixed semi-implicit Euler substeps update velocity
/// before position. Default physics substeps last 0.001 seconds.
///
/// Reward is horizontal centre-of-mass displacement minus `energy_cost` times
/// positive actuator work, with `fall_penalty` when the torso touches ground.
/// This robot can walk, hop, or crawl; its score is not BipedalWalker reward.
#[derive(Clone, Debug)]
pub struct SpringWalker {
    config: SpringWalkerConfig,
    state: SpringWalkerState,
    steps: usize,
    done: bool,
    energy_used: f64,
    initial_x: f64,
}

impl SpringWalker {
    /// Construct a checked robot at its symmetric unactuated initial geometry.
    pub fn new(config: SpringWalkerConfig) -> Result<Self, ControlError> {
        let state = config.initial_state()?;
        let mut env = Self {
            config,
            state,
            steps: 0,
            done: false,
            energy_used: 0.0,
            initial_x: 0.0,
        };
        env.reset_to(state)?;
        Ok(env)
    }

    /// Validated geometry, physical parameters, and objective coefficients.
    pub fn config(&self) -> &SpringWalkerConfig {
        &self.config
    }

    /// Current complete particle state.
    pub fn state(&self) -> &SpringWalkerState {
        &self.state
    }

    /// Number of applied policy actions since reset.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Whether torso contact or the time limit ended the episode.
    pub fn done(&self) -> bool {
        self.done
    }

    /// Cumulative positive actuator work, in joules.
    pub fn energy_used(&self) -> f64 {
        self.energy_used
    }

    /// Current horizontal centre-of-mass displacement from this episode's start.
    pub fn distance(&self) -> f64 {
        self.com_x(self.state) - self.initial_x
    }

    /// Thirteen observations with no absolute horizontal position.
    ///
    /// Order: torso height, torso vx/vy, left relative x/y, right relative x/y,
    /// left relative vx/vy, right relative vx/vy, left/right contact flags.
    /// Lengths are divided by `leg_length`; velocities by
    /// `sqrt(max(gravity, 1) * leg_length)`. Contacts are 0 or 1. Values are not
    /// clipped, and absolute horizontal translation does not change observations.
    pub fn observation(&self) -> Vec<f64> {
        let p = self.state.particles;
        let length = self.config.leg_length;
        let speed = (self.config.gravity.max(1.0) * length).sqrt();
        let mut values = vec![
            p[0].position[1] / length,
            p[0].velocity[0] / speed,
            p[0].velocity[1] / speed,
        ];
        for foot in &p[1..] {
            values.extend((0..2).map(|axis| (foot.position[axis] - p[0].position[axis]) / length));
        }
        for foot in &p[1..] {
            values.extend((0..2).map(|axis| (foot.velocity[axis] - p[0].velocity[axis]) / speed));
        }
        for foot in &p[1..] {
            values.push(
                if self.config.ground && foot.position[1] <= self.config.foot_radius {
                    1.0
                } else {
                    0.0
                },
            );
        }
        values
    }

    /// Seeded small torso-position/velocity perturbation of the initial geometry.
    ///
    /// This changes initial balance only, not terrain or hidden force noise.
    pub fn reset(&mut self, seed: u64) -> Result<Vec<f64>, ControlError> {
        let mut state = self.config.initial_state()?;
        let mut rng = reset_rng(seed);
        state.particles[0].position[0] += 0.02 * self.config.leg_length * signed_uniform(&mut rng);
        state.particles[0].velocity[0] = 0.02 * signed_uniform(&mut rng);
        state.particles[0].velocity[1] = 0.02 * signed_uniform(&mut rng);
        self.reset_to(state)
    }

    /// Reset mechanics, work accounting, and the horizon to an explicit state.
    pub fn reset_to(&mut self, state: SpringWalkerState) -> Result<Vec<f64>, ControlError> {
        Self::check_state(state).map_err(|_| {
            ControlError::InvalidState(
                "spring-robot coordinates must be finite with distinct connected particles",
            )
        })?;
        if self.config.ground && state.particles[0].position[1] <= self.config.torso_radius {
            return Err(ControlError::InvalidState(
                "spring-robot torso must start above ground contact",
            ));
        }
        let x = self.com_x(state);
        if !x.is_finite() {
            return Err(ControlError::InvalidState(
                "spring-robot centre of mass overflow",
            ));
        }
        self.state = state;
        self.steps = 0;
        self.done = false;
        self.energy_used = 0.0;
        self.initial_x = x;
        Ok(self.observation())
    }

    /// Internal spring, actuator, and damping forces on the three particles.
    ///
    /// Forces are in newtons and sum to zero on each axis. Gravity and ground
    /// reaction forces are deliberately excluded from this inspection method.
    pub fn internal_forces(&self, action: &[f64]) -> Result<[[f64; 2]; 3], ControlError> {
        check_action(action, 2)?;
        Ok(self.forces(self.state, [action[0], action[1]])?.0)
    }

    /// Nonnegative vertical ground reactions in newtons, before friction.
    pub fn ground_normal_forces(&self) -> Result<[f64; 3], ControlError> {
        self.normals(self.state)
    }

    /// Kinetic + gravitational + passive-spring potential energy, in joules.
    ///
    /// Ground penalty energy and actuator work are not included. This is most
    /// useful for the ground-free conservation checks, with damping disabled.
    pub fn mechanical_energy(&self) -> Result<f64, ControlError> {
        let mut energy = 0.0;
        for (particle, mass) in self.state.particles.iter().zip(self.config.masses) {
            energy += 0.5 * mass * (particle.velocity[0].powi(2) + particle.velocity[1].powi(2))
                + mass * self.config.gravity * particle.position[1];
        }
        for (a, b, rest, stiffness) in self.springs() {
            let dx = self.state.particles[b].position[0] - self.state.particles[a].position[0];
            let dy = self.state.particles[b].position[1] - self.state.particles[a].position[1];
            energy += 0.5 * stiffness * (dx.hypot(dy) - rest).powi(2);
        }
        finite(&[energy], "spring-robot mechanical energy")?;
        Ok(energy)
    }

    /// Apply two bounded rest-length actions with fixed physics substeps.
    ///
    /// The complete control step is committed only after successful arithmetic.
    /// A torso contact stops physics immediately at that substep.
    pub fn step(&mut self, action: &[f64]) -> Result<ControlStep, ControlError> {
        if self.done {
            return Err(ControlError::EpisodeFinished);
        }
        check_action(action, 2)?;
        let h = self.config.dt / self.config.substeps as f64;
        let before_x = self.com_x(self.state);
        let mut next = self.state;
        let mut work = 0.0;
        let mut terminated = false;
        for _ in 0..self.config.substeps {
            let previous = next;
            let (mut forces, actuators) = self.forces(previous, [action[0], action[1]])?;
            let normals = self.normals(previous)?;
            for i in 0..3 {
                let mass = self.config.masses[i];
                forces[i][1] += normals[i] - mass * self.config.gravity;
                for axis in 0..2 {
                    next.particles[i].velocity[axis] += h * forces[i][axis] / mass;
                }
                // Friction acts after forces and cannot reverse horizontal velocity.
                let vx = next.particles[i].velocity[0];
                let reduction = self.config.friction * normals[i] * h / mass;
                finite(&next.particles[i].velocity, "spring-robot velocity update")?;
                finite(&[reduction], "spring-robot friction impulse")?;
                next.particles[i].velocity[0] = vx.signum() * (vx.abs() - reduction).max(0.0);
                for axis in 0..2 {
                    next.particles[i].position[axis] += h * next.particles[i].velocity[axis];
                }
            }
            Self::check_state(next)?;
            for (leg, actuator) in actuators.iter().enumerate() {
                let mut leg_work = 0.0;
                for (axis, force) in actuator.iter().enumerate() {
                    let torso_displacement =
                        next.particles[0].position[axis] - previous.particles[0].position[axis];
                    let foot_displacement = next.particles[leg + 1].position[axis]
                        - previous.particles[leg + 1].position[axis];
                    // actuators[leg] is the actuator force on the torso.
                    leg_work += force * (torso_displacement - foot_displacement);
                }
                finite(&[leg_work], "spring-robot actuator work")?;
                work += leg_work.max(0.0);
            }
            terminated =
                self.config.ground && next.particles[0].position[1] <= self.config.torso_radius;
            if terminated {
                break;
            }
        }
        let reward = self.com_x(next)
            - before_x
            - self.config.energy_cost * work
            - if terminated {
                self.config.fall_penalty
            } else {
                0.0
            };
        let energy = self.energy_used + work;
        finite(&[reward, energy], "spring-robot work or reward")?;
        self.state = next;
        self.energy_used = energy;
        self.steps += 1;
        let truncated = self.steps >= self.config.max_steps;
        self.done = terminated || truncated;
        Ok(ControlStep {
            reward,
            terminated,
            truncated,
        })
    }

    fn springs(&self) -> [(usize, usize, f64, f64); 3] {
        [
            (0, 1, self.config.leg_length, self.config.leg_stiffness),
            (0, 2, self.config.leg_length, self.config.leg_stiffness),
            (
                1,
                2,
                self.config.foot_spacing,
                self.config.spacing_stiffness,
            ),
        ]
    }

    fn forces(
        &self,
        state: SpringWalkerState,
        action: [f64; 2],
    ) -> Result<InternalAndActuatorForces, ControlError> {
        let mut forces = [[0.0; 2]; 3];
        let mut actuators = [[0.0; 2]; 2];
        for (index, (a, b, rest, stiffness)) in self.springs().into_iter().enumerate() {
            let pa = state.particles[a];
            let pb = state.particles[b];
            let offset = [
                pb.position[0] - pa.position[0],
                pb.position[1] - pa.position[1],
            ];
            let distance = offset[0].hypot(offset[1]);
            if !distance.is_finite() || distance <= 1e-12 {
                return Err(ControlError::Numerical(
                    "spring-robot connected particles coincide or overflow",
                ));
            }
            let n = [offset[0] / distance, offset[1] / distance];
            let axial_speed =
                (pb.velocity[0] - pa.velocity[0]) * n[0] + (pb.velocity[1] - pa.velocity[1]) * n[1];
            let active = if index < 2 {
                -stiffness * rest * self.config.actuation_fraction * action[index]
            } else {
                0.0
            };
            let tension =
                stiffness * (distance - rest) + self.config.spring_damping * axial_speed + active;
            finite(&[tension, active], "spring-robot internal force")?;
            for (axis, direction) in n.into_iter().enumerate() {
                let force = tension * direction;
                forces[a][axis] += force;
                forces[b][axis] -= force;
                if index < 2 {
                    actuators[index][axis] = active * direction;
                }
            }
        }
        for force in forces {
            finite(&force, "spring-robot summed force")?;
        }
        Ok((forces, actuators))
    }

    fn normals(&self, state: SpringWalkerState) -> Result<[f64; 3], ControlError> {
        if !self.config.ground {
            return Ok([0.0; 3]);
        }
        let mut normals = [0.0; 3];
        for (i, particle) in state.particles.iter().enumerate() {
            let radius = if i == 0 {
                self.config.torso_radius
            } else {
                self.config.foot_radius
            };
            let penetration = radius - particle.position[1];
            if penetration > 0.0 {
                normals[i] = (self.config.ground_stiffness * penetration
                    - self.config.ground_damping * particle.velocity[1])
                    .max(0.0);
            }
        }
        finite(&normals, "spring-robot ground contact")?;
        Ok(normals)
    }

    fn com_x(&self, state: SpringWalkerState) -> f64 {
        let total_mass: f64 = self.config.masses.iter().sum();
        state
            .particles
            .iter()
            .zip(self.config.masses)
            .map(|(particle, mass)| (mass / total_mass) * particle.position[0])
            .sum()
    }

    fn check_state(state: SpringWalkerState) -> Result<(), ControlError> {
        for particle in state.particles {
            finite(&particle.position, "spring-robot position")?;
            finite(&particle.velocity, "spring-robot velocity")?;
        }
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let offset = [
                state.particles[b].position[0] - state.particles[a].position[0],
                state.particles[b].position[1] - state.particles[a].position[1],
            ];
            let distance = offset[0].hypot(offset[1]);
            if !distance.is_finite() || distance <= 1e-12 {
                return Err(ControlError::Numerical(
                    "spring-robot connected particles coincide or overflow",
                ));
            }
        }
        Ok(())
    }
}

environment_impl!(SpringWalker, SpringWalkerState);
