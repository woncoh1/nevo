//! Physical equations, reset semantics, and numerical failures of the teaching task.

use nevo::Evaluator;
use nevo::tasks::{TargetReach, TargetReachError};

#[test]
fn zero_controller_is_the_no_action_baseline() {
    let task = TargetReach::default();
    assert!((task.value(&[0.0; 3]).unwrap() - 0.625).abs() < 1e-12);
    let rollout = task.rollout(&[0.0; 3], -0.75).unwrap();
    assert_eq!(rollout.states.len(), task.steps() + 1);
    assert!(
        rollout
            .states
            .iter()
            .all(|s| s.position == -0.75 && s.velocity == 0.0 && s.action == 0.0)
    );
}

#[test]
fn one_step_agrees_with_the_published_equations() {
    let task = TargetReach::new(vec![1.0], 1).unwrap();
    let trace = task.rollout(&[1.0, 0.0, 0.0], 1.0).unwrap();
    let action = (-1.0_f64).tanh();
    let velocity = 0.2 * action;
    let position = 1.0 + 0.1 * velocity;
    let expected_loss = position * position + 0.01 * action * action;
    assert_eq!(trace.states[1].action, action);
    assert_eq!(trace.states[1].velocity, velocity);
    assert_eq!(trace.states[1].position, position);
    assert_eq!(trace.loss, expected_loss);
    assert_eq!(task.value(&[1.0, 0.0, 0.0]).unwrap(), expected_loss);
}

#[test]
fn evaluation_matches_recorded_episodes_and_resets_each_trial() {
    let mut task = TargetReach::default();
    let weights = [2.0, -1.0, 0.1];
    let expected = task
        .starts()
        .iter()
        .map(|&start| task.rollout(&weights, start).unwrap().loss)
        .sum::<f64>()
        / task.starts().len() as f64;
    let first = task.evaluate(&weights, 42).unwrap();
    task.evaluate(&[0.0; 3], 42).unwrap();
    let second = task.evaluate(&weights, u64::MAX).unwrap();
    assert_eq!(first, second);
    assert!((first - expected).abs() < 1e-12);
}

#[test]
fn symmetric_controller_produces_mirrored_trajectories() {
    let task = TargetReach::default();
    let left = task.rollout(&[2.0, -1.0, 0.0], -1.0).unwrap();
    let right = task.rollout(&[2.0, -1.0, 0.0], 1.0).unwrap();
    for (a, b) in left.states.iter().zip(&right.states) {
        assert_eq!(a.position, -b.position);
        assert_eq!(a.velocity, -b.velocity);
        assert_eq!(a.action, -b.action);
        assert!(a.action.abs() <= 1.0);
    }
    assert_eq!(left.loss, right.loss);
}

#[test]
fn invalid_tasks_and_genomes_are_errors() {
    assert_eq!(
        TargetReach::new(vec![], 80),
        Err(TargetReachError::InvalidTask)
    );
    assert_eq!(
        TargetReach::new(vec![1.0], 0),
        Err(TargetReachError::InvalidTask)
    );
    assert_eq!(
        TargetReach::new(vec![f64::NAN], 80),
        Err(TargetReachError::InvalidTask)
    );
    let task = TargetReach::default();
    assert_eq!(
        task.value(&[1.0, 2.0]),
        Err(TargetReachError::DimensionMismatch { actual: 2 })
    );
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            task.value(&[invalid, 0.0, 0.0]),
            Err(TargetReachError::NonFiniteWeights)
        );
    }
    assert_eq!(
        task.rollout(&[0.0; 3], f64::INFINITY),
        Err(TargetReachError::InvalidTask)
    );
}

#[test]
fn nonfinite_arithmetic_is_not_hidden_by_activation_saturation() {
    let task = TargetReach::new(vec![2.0], 1).unwrap();
    assert_eq!(
        task.value(&[f64::MAX, 0.0, 0.0]),
        Err(TargetReachError::NonFiniteSimulation)
    );
    let huge_start = TargetReach::new(vec![f64::MAX], 1).unwrap();
    assert_eq!(
        huge_start.value(&[0.0; 3]),
        Err(TargetReachError::NonFiniteSimulation)
    );
}
