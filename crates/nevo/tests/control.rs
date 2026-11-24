//! Physical reference checks, integration checks, and control-task protocols.

use nevo::Evaluator;
use nevo::control::{
    CartPole, CartPoleConfig, CartPoleState, ControlError, DoublePole, DoublePoleConfig,
    DoublePoleState, PoleObservations, RolloutError, SpringWalker, SpringWalkerConfig,
    SpringWalkerState,
};
use nevo::tasks::{CartPoleTask, ControlTaskError, DoublePoleTask, SpringWalkerTask};
use std::convert::Infallible;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.17e} != {expected:.17e}, tolerance {tolerance:e}"
    );
}

fn pole_values(s: &DoublePoleState) -> [f64; 6] {
    [
        s.position,
        s.velocity,
        s.angles[0],
        s.angular_velocities[0],
        s.angles[1],
        s.angular_velocities[1],
    ]
}

#[test]
fn cart_pole_matches_classic_zero_state_acceleration_and_explicit_euler() {
    let mut env = CartPole::new(CartPoleConfig::default()).unwrap();
    assert_eq!(env.accelerations(0.0).unwrap(), [0.0, 0.0]);
    let [cart, angular] = env.accelerations(10.0).unwrap();
    close(cart, 9.75609756097561, 1e-13);
    close(angular, -14.634146341463415, 1e-13);
    let step = env.step(&[1.0]).unwrap();
    assert_eq!(env.state().position, 0.0); // Explicit, not semi-implicit Euler.
    assert_eq!(env.state().angle, 0.0);
    close(env.state().velocity, 0.1951219512195122, 1e-14);
    close(env.state().angular_velocity, -0.2926829268292683, 1e-14);
    assert_eq!(step.reward, 1.0);
    assert!(!step.done());
}

#[test]
fn cart_pole_seeded_resets_are_reproducible_and_use_explicit_randomness() {
    let mut a = CartPole::new(CartPoleConfig::default()).unwrap();
    let mut b = a.clone();
    let observation = a.reset(42);
    assert_eq!(observation, b.reset(42));
    assert!(observation.iter().all(|x| (-0.05..=0.05).contains(x)));
    assert_ne!(observation, b.reset(43));
    a.step(&[1.0]).unwrap();
    assert_eq!(a.reset(42), observation);
    assert_eq!(a.steps(), 0);
    assert!(!a.done());
}

#[test]
fn cart_pole_actions_select_discrete_directions_not_force_magnitude() {
    let env = CartPole::new(CartPoleConfig::default()).unwrap();
    let mut left = env.clone();
    let mut weak_left = env.clone();
    let mut right = env.clone();
    let mut zero = env;
    left.step(&[-1.0]).unwrap();
    weak_left.step(&[-0.01]).unwrap();
    right.step(&[1.0]).unwrap();
    zero.step(&[0.0]).unwrap();
    assert_eq!(left.state(), weak_left.state());
    assert_eq!(right.state(), zero.state());
    close(left.state().velocity, -right.state().velocity, 0.0);
    close(
        left.state().angular_velocity,
        -right.state().angular_velocity,
        0.0,
    );
}

#[test]
fn cart_pole_failure_reward_and_horizon_are_explicit() {
    let mut env = CartPole::new(CartPoleConfig {
        max_steps: 1,
        ..Default::default()
    })
    .unwrap();
    env.reset_to(CartPoleState {
        position: 2.39,
        velocity: 1.0,
        ..Default::default()
    })
    .unwrap();
    let last = env.step(&[1.0]).unwrap();
    assert!(last.terminated);
    assert!(last.truncated);
    assert_eq!(last.reward, 1.0); // Gym includes its first failed transition.
    assert_eq!(env.step(&[1.0]), Err(ControlError::EpisodeFinished));
    env.reset_to(CartPoleState::default()).unwrap();
    assert!(!env.done());
    assert_eq!(env.steps(), 0);
}

#[test]
fn cart_pole_checks_actions_and_rejects_bad_states_without_mutation() {
    let mut env = CartPole::new(CartPoleConfig::default()).unwrap();
    let original = *env.state();
    for action in [vec![], vec![0.0, 1.0], vec![1.1], vec![f64::NAN]] {
        assert!(env.step(&action).is_err());
        assert_eq!(*env.state(), original);
        assert_eq!(env.steps(), 0);
    }
    assert!(
        env.reset_to(CartPoleState {
            position: f64::INFINITY,
            ..original
        })
        .is_err()
    );
    assert!(
        env.reset_to(CartPoleState {
            angle: 1.0,
            ..original
        })
        .is_err()
    );
    assert_eq!(*env.state(), original);
}

#[test]
fn numerical_failure_does_not_commit_a_cart_or_double_pole_step() {
    let mut cart = CartPole::new(CartPoleConfig {
        force: 1.7e308,
        ..Default::default()
    })
    .unwrap();
    let cart_before = *cart.state();
    assert!(matches!(cart.step(&[1.0]), Err(ControlError::Numerical(_))));
    assert_eq!(*cart.state(), cart_before);
    assert_eq!(cart.steps(), 0);
    let mut double = DoublePole::new(DoublePoleConfig {
        force: 1e308,
        ..Default::default()
    })
    .unwrap();
    let before = *double.state();
    assert!(matches!(
        double.step(&[1.0]),
        Err(ControlError::Numerical(_))
    ));
    assert_eq!(*double.state(), before);
    assert_eq!(double.steps(), 0);
}

#[test]
fn double_pole_upright_equilibrium_and_known_force_have_correct_units() {
    let mut env = DoublePole::new(DoublePoleConfig::default()).unwrap();
    env.reset_to(DoublePoleState {
        angles: [0.0; 2],
        ..Default::default()
    })
    .unwrap();
    assert_eq!(env.accelerations(0.0).unwrap(), [0.0; 3]);
    let a = env.accelerations(10.0).unwrap();
    close(a[0], 9.732360097323602, 1e-13);
    close(a[1], -14.598540145985403, 1e-13);
    close(a[2], -145.98540145985402, 1e-12);
    close(a[2] / a[1], 10.0, 1e-14); // Tenfold shorter half-length.
    env.step(&[0.0]).unwrap();
    assert_eq!(pole_values(env.state()), [0.0; 6]);
}

#[test]
fn double_pole_one_step_matches_independent_rk4_reference() {
    // Equations evaluated independently with the UT defaults and two h=.01 steps.
    let expected = [
        -9.983824566971115e-6,
        -9.98931470706497e-4,
        0.0702206787340885,
        0.02207947227723079,
        0.00015043182472776782,
        0.015122033536002971,
    ];
    let mut env = DoublePole::new(DoublePoleConfig::default()).unwrap();
    env.step(&[0.0]).unwrap();
    for (actual, reference) in pole_values(env.state()).into_iter().zip(expected) {
        close(actual, reference, 2e-15);
    }
}

#[test]
fn double_pole_reflects_state_and_force_symmetrically() {
    let state = DoublePoleState {
        position: 0.1,
        velocity: -0.2,
        angles: [0.04, -0.03],
        angular_velocities: [0.1, -0.2],
    };
    let mirror = DoublePoleState {
        position: -state.position,
        velocity: -state.velocity,
        angles: state.angles.map(|x| -x),
        angular_velocities: state.angular_velocities.map(|x| -x),
    };
    let mut a = DoublePole::new(DoublePoleConfig::default()).unwrap();
    let mut b = a.clone();
    a.reset_to(state).unwrap();
    b.reset_to(mirror).unwrap();
    a.step(&[0.3]).unwrap();
    b.step(&[-0.3]).unwrap();
    for (x, y) in pole_values(a.state())
        .into_iter()
        .zip(pole_values(b.state()))
    {
        close(x, -y, 1e-15);
    }
}

#[test]
fn double_pole_positions_only_does_not_leak_velocity_information() {
    let state = DoublePoleState {
        position: 1.2,
        velocity: 0.4,
        angles: [0.2, -0.1],
        angular_velocities: [0.5, -0.3],
    };
    let mut full = DoublePole::new(DoublePoleConfig::default()).unwrap();
    let mut partial = DoublePole::new(DoublePoleConfig {
        observation: PoleObservations::PositionsOnly,
        ..Default::default()
    })
    .unwrap();
    full.reset_to(state).unwrap();
    partial.reset_to(state).unwrap();
    let f = full.observation();
    let p = partial.observation();
    assert_eq!(f.len(), 6);
    assert_eq!(p.len(), 3);
    assert_eq!(p, vec![f[0], f[2], f[4]]);
    partial
        .reset_to(DoublePoleState {
            velocity: -99.0,
            angular_velocities: [10.0, 20.0],
            ..state
        })
        .unwrap();
    assert_eq!(p, partial.observation());
}

#[test]
fn double_pole_zero_action_survival_counts_valid_endpoints() {
    let mut env = DoublePole::new(DoublePoleConfig::default()).unwrap();
    let result = env.rollout(|_| Ok::<_, Infallible>(vec![0.0])).unwrap();
    assert_eq!(result.steps, 23);
    assert_eq!(result.total_reward, 22.0);
    assert!(result.terminated);
    assert!(!result.truncated);
    assert_eq!(result.states.len(), 24);
    assert_eq!(result.actions.len(), 23);
    assert!(result.states.last().unwrap().angles[1] > env.config().angle_limit);
}

#[test]
fn double_pole_rk4_error_decreases_when_substeps_are_halved() {
    let start = DoublePoleState {
        position: 0.1,
        velocity: 0.01,
        angles: [0.03, -0.02],
        angular_velocities: [-0.02, 0.01],
    };
    let simulate = |substeps| {
        let mut env = DoublePole::new(DoublePoleConfig {
            substeps,
            ..Default::default()
        })
        .unwrap();
        env.reset_to(start).unwrap();
        for _ in 0..3 {
            env.step(&[0.1]).unwrap();
        }
        pole_values(env.state())
    };
    let reference = simulate(128);
    let error = |values: [f64; 6]| {
        values
            .into_iter()
            .zip(reference)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let coarse = error(simulate(2));
    let fine = error(simulate(4));
    assert!(coarse > 0.0 && fine > 0.0);
    assert!(
        coarse / fine > 12.0,
        "RK4 convergence ratio {}",
        coarse / fine
    );
}

#[test]
fn environments_reject_invalid_physical_configurations() {
    assert!(
        CartPole::new(CartPoleConfig {
            half_length: 0.0,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        CartPole::new(CartPoleConfig {
            max_steps: 0,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        DoublePole::new(DoublePoleConfig {
            pole_masses: [0.1, -0.01],
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        DoublePole::new(DoublePoleConfig {
            substeps: 0,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        SpringWalker::new(SpringWalkerConfig {
            substeps: 1,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        SpringWalker::new(SpringWalkerConfig {
            actuation_fraction: 1.0,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        SpringWalker::new(SpringWalkerConfig {
            foot_spacing: 3.0,
            ..Default::default()
        })
        .is_err()
    );
}

#[test]
fn spring_robot_internal_forces_are_equal_and_opposite() {
    let mut env = SpringWalker::new(SpringWalkerConfig::default()).unwrap();
    let mut state = *env.state();
    state.particles[0].position[0] = 0.1;
    state.particles[1].velocity = [0.3, -0.2];
    env.reset_to(state).unwrap();
    let forces = env.internal_forces(&[0.7, -0.4]).unwrap();
    for axis in 0..2 {
        close(forces.iter().map(|f| f[axis]).sum(), 0.0, 1e-13);
    }
}

#[test]
fn spring_robot_unstretched_springs_at_rest_exert_no_internal_force() {
    let env = SpringWalker::new(SpringWalkerConfig::default()).unwrap();
    for force in env.internal_forces(&[0.0, 0.0]).unwrap() {
        close(force[0], 0.0, 1e-12);
        close(force[1], 0.0, 1e-12);
    }
}

#[test]
fn spring_robot_has_no_horizontal_propulsion_without_ground() {
    let mut env = SpringWalker::new(SpringWalkerConfig {
        ground: false,
        gravity: 0.0,
        ..Default::default()
    })
    .unwrap();
    for _ in 0..20 {
        env.step(&[0.8, -0.3]).unwrap();
    }
    let momentum: f64 = env
        .state()
        .particles
        .iter()
        .zip(env.config().masses)
        .map(|(p, mass)| mass * p.velocity[0])
        .sum();
    close(momentum, 0.0, 1e-12);
    close(env.distance(), 0.0, 1e-12);
    assert!(env.energy_used() > 0.0);
}

#[test]
fn spring_robot_symmetric_actions_preserve_left_right_symmetry() {
    let mut env = SpringWalker::new(SpringWalkerConfig::default()).unwrap();
    for _ in 0..50 {
        env.step(&[0.3, 0.3]).unwrap();
    }
    let p = env.state().particles;
    close(p[0].position[0], 0.0, 1e-13);
    close(p[1].position[0], -p[2].position[0], 1e-13);
    close(p[1].position[1], p[2].position[1], 1e-13);
    close(env.distance(), 0.0, 1e-13);
}

#[test]
fn spring_robot_positive_actuator_work_is_charged_and_zero_action_is_free() {
    let config = SpringWalkerConfig {
        ground: false,
        gravity: 0.0,
        max_steps: 5,
        ..Default::default()
    };
    let mut env = SpringWalker::new(config.clone()).unwrap();
    let result = env
        .rollout(|_| Ok::<_, Infallible>(vec![1.0, -1.0]))
        .unwrap();
    assert!(env.energy_used() > 0.0);
    close(
        result.total_reward,
        env.distance() - config.energy_cost * env.energy_used(),
        1e-12,
    );
    let mut passive = SpringWalker::new(config).unwrap();
    passive.step(&[0.0, 0.0]).unwrap();
    assert_eq!(passive.energy_used(), 0.0);
}

#[test]
fn spring_robot_ground_force_has_correct_units_and_never_pulls_down() {
    let mut env = SpringWalker::new(SpringWalkerConfig::default()).unwrap();
    let mut state = *env.state();
    state.particles[1].position[1] = env.config().foot_radius - 0.001;
    state.particles[2].position[1] = env.config().foot_radius + 0.001;
    env.reset_to(state).unwrap();
    let normals = env.ground_normal_forces().unwrap();
    close(normals[1], 2.0, 1e-12); // 2000 N/m * 0.001 m.
    assert_eq!(normals[0], 0.0);
    assert_eq!(normals[2], 0.0);
    state.particles[1].velocity[1] = 100.0;
    env.reset_to(state).unwrap();
    assert_eq!(env.ground_normal_forces().unwrap()[1], 0.0);
}

#[test]
fn spring_robot_ground_friction_reduces_slide_without_reversing_it() {
    let config = SpringWalkerConfig {
        gravity: 0.0,
        dt: 0.001,
        substeps: 1,
        ..Default::default()
    };
    let mut frictional = SpringWalker::new(config.clone()).unwrap();
    let mut frictionless = SpringWalker::new(SpringWalkerConfig {
        friction: 0.0,
        ..config
    })
    .unwrap();
    let mut state = *frictional.state();
    for particle in &mut state.particles {
        particle.velocity[0] = 1.0;
    }
    state.particles[1].position[1] -= 0.005;
    state.particles[2].position[1] -= 0.005;
    frictional.reset_to(state).unwrap();
    frictionless.reset_to(state).unwrap();
    frictional.step(&[0.0, 0.0]).unwrap();
    frictionless.step(&[0.0, 0.0]).unwrap();
    for foot in 1..3 {
        let vx = frictional.state().particles[foot].velocity[0];
        assert!(vx >= 0.0);
        assert!(vx < frictionless.state().particles[foot].velocity[0]);
    }
}

#[test]
fn spring_robot_observation_is_translation_invariant_and_has_thirteen_values() {
    let mut env = SpringWalker::new(SpringWalkerConfig::default()).unwrap();
    let before = env.observation();
    assert_eq!(before.len(), 13);
    let mut shifted = *env.state();
    for particle in &mut shifted.particles {
        particle.position[0] += 10.0;
    }
    env.reset_to(shifted).unwrap();
    for (a, b) in before.into_iter().zip(env.observation()) {
        close(a, b, 1e-14);
    }
    close(env.distance(), 0.0, 0.0);
}

#[test]
fn spring_robot_reset_clears_energy_and_repeats_seeded_state() {
    let mut env = SpringWalker::new(SpringWalkerConfig::default()).unwrap();
    let first = env.reset(7).unwrap();
    env.step(&[1.0, -0.5]).unwrap();
    assert!(env.energy_used() > 0.0);
    assert_eq!(env.reset(7).unwrap(), first);
    assert_eq!(env.steps(), 0);
    assert_eq!(env.energy_used(), 0.0);
    assert_eq!(env.distance(), 0.0);
}

#[test]
fn spring_robot_invalid_actions_do_not_change_the_state() {
    let mut env = SpringWalker::new(SpringWalkerConfig::default()).unwrap();
    let before = *env.state();
    for action in [vec![0.0], vec![0.0, f64::NAN], vec![-1.01, 0.0]] {
        assert!(env.step(&action).is_err());
        assert_eq!(*env.state(), before);
        assert_eq!(env.steps(), 0);
    }
}

#[test]
fn spring_robot_bounded_changing_actions_stay_numerically_valid() {
    let mut env = SpringWalker::new(SpringWalkerConfig {
        max_steps: 250,
        ..Default::default()
    })
    .unwrap();
    for seed in 0..16 {
        env.reset(seed).unwrap();
        for step in 0..250 {
            let phase = step as f64 * 0.37 + seed as f64;
            let result = env.step(&[phase.sin(), (phase * 1.31).cos()]).unwrap();
            assert!(result.reward.is_finite());
            assert!(env.energy_used().is_finite() && env.energy_used() >= 0.0);
            if result.done() {
                break;
            }
        }
        assert!(env.done());
    }
}

#[test]
fn spring_robot_smaller_substeps_reduce_short_trajectory_error() {
    let simulate = |substeps| {
        let mut env = SpringWalker::new(SpringWalkerConfig {
            substeps,
            ..Default::default()
        })
        .unwrap();
        for _ in 0..10 {
            env.step(&[0.25, -0.1]).unwrap();
        }
        *env.state()
    };
    let reference = simulate(160);
    let error = |state: SpringWalkerState| {
        state
            .particles
            .iter()
            .zip(reference.particles)
            .flat_map(|(a, b)| {
                [
                    a.position[0] - b.position[0],
                    a.position[1] - b.position[1],
                    a.velocity[0] - b.velocity[0],
                    a.velocity[1] - b.velocity[1],
                ]
            })
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    };
    assert!(error(simulate(20)) < error(simulate(10)));
}

#[test]
fn recorded_rollout_preserves_policy_errors_and_exact_horizon() {
    let mut env = CartPole::new(CartPoleConfig {
        max_steps: 3,
        ..Default::default()
    })
    .unwrap();
    let failure = env.rollout(|_| Err::<Vec<f64>, _>("policy failure"));
    assert_eq!(failure, Err(RolloutError::Policy("policy failure")));
    assert_eq!(env.steps(), 0);
    let result = env.rollout(|_| Ok::<_, Infallible>(vec![1.0])).unwrap();
    assert_eq!(result.steps, 3);
    assert_eq!(result.total_reward, 3.0);
    assert_eq!(result.states.len(), 4);
    assert_eq!(result.actions.len(), 3);
    assert!(result.truncated);
    assert!(!result.terminated);
    assert!(matches!(
        env.rollout(|_| Ok::<_, Infallible>(vec![1.0])),
        Err(RolloutError::Control(ControlError::EpisodeFinished))
    ));
}

#[test]
fn mlp_control_tasks_have_declared_dimensions_and_reject_bad_suites() {
    let cart =
        CartPoleTask::new(CartPoleConfig::default(), vec![CartPoleState::default()], 4).unwrap();
    assert_eq!(cart.dimensions(), 25);
    let double = DoublePoleTask::new(
        DoublePoleConfig::default(),
        vec![DoublePoleState::default()],
        4,
    )
    .unwrap();
    assert_eq!(double.dimensions(), 33);
    let partial = DoublePoleTask::new(
        DoublePoleConfig {
            observation: PoleObservations::PositionsOnly,
            ..Default::default()
        },
        vec![DoublePoleState::default()],
        4,
    )
    .unwrap();
    assert_eq!(partial.dimensions(), 21);
    let walker = SpringWalkerTask::new(
        SpringWalkerConfig::default(),
        vec![SpringWalkerState::default()],
        4,
    )
    .unwrap();
    assert_eq!(walker.dimensions(), 66);
    assert!(matches!(
        CartPoleTask::new(CartPoleConfig::default(), vec![], 4),
        Err(ControlTaskError::EmptyStarts)
    ));
    assert!(
        CartPoleTask::new(CartPoleConfig::default(), vec![CartPoleState::default()], 0).is_err()
    );
    assert!(cart.value(&[0.0]).is_err());
}

#[test]
fn mlp_task_objective_is_negative_mean_of_independent_episode_returns() {
    let starts = vec![
        CartPoleState::default(),
        CartPoleState {
            angle: -0.04,
            ..Default::default()
        },
    ];
    let mut task = CartPoleTask::new(
        CartPoleConfig {
            max_steps: 20,
            ..Default::default()
        },
        starts.clone(),
        4,
    )
    .unwrap();
    let weights = vec![0.0; task.dimensions()];
    let expected = -starts
        .iter()
        .map(|&start| task.rollout(&weights, start).unwrap().total_reward)
        .sum::<f64>()
        / starts.len() as f64;
    close(task.value(&weights).unwrap(), expected, 1e-12);
    assert_eq!(
        task.evaluate(&weights, 1).unwrap(),
        task.evaluate(&weights, 99).unwrap()
    );
    let double = DoublePoleTask::new(
        DoublePoleConfig {
            max_steps: 30,
            ..Default::default()
        },
        vec![DoublePoleState::default()],
        4,
    )
    .unwrap();
    assert_eq!(
        double.value(&vec![0.0; double.dimensions()]).unwrap(),
        -22.0
    );
    let walker = SpringWalkerTask::new(
        SpringWalkerConfig {
            max_steps: 5,
            ..Default::default()
        },
        vec![SpringWalkerState::default()],
        4,
    )
    .unwrap();
    let weights = vec![0.0; walker.dimensions()];
    close(
        walker.value(&weights).unwrap(),
        -walker
            .rollout(&weights, SpringWalkerState::default())
            .unwrap()
            .total_reward,
        1e-12,
    );
}
