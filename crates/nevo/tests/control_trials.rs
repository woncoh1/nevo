//! Episode diagnostics, explicit trial suites, and recurrent reset isolation.

use nevo::control::{
    CartPole, CartPoleConfig, ControlError, DoublePole, DoublePoleConfig, DoublePoleState,
    EpisodeSummary, PoleObservations, RolloutError, SpringWalker, SpringWalkerConfig,
    evaluate_trials,
};
use nevo::neat::{ConnectionGene, Genome, NodeGene, NodeKind};
use nevo::tasks::{ControlTaskError, DoublePoleTask, NeatDoublePoleTask};
use std::convert::Infallible;

fn record(reward: f64, steps: usize, terminated: bool, truncated: bool) -> EpisodeSummary {
    EpisodeSummary {
        total_reward: reward,
        steps,
        terminated,
        truncated,
    }
}

#[test]
fn trial_aggregation_uses_sample_sd_and_unambiguous_horizon_success() {
    let records = [
        record(2.0, 3, false, true),
        record(4.0, 5, true, true),
        record(6.0, 7, true, false),
    ];
    let summary =
        evaluate_trials(&records, |&value| Ok::<_, RolloutError<Infallible>>(value)).unwrap();
    assert_eq!(summary.episodes, 3);
    assert_eq!(summary.control_steps, 15);
    assert_eq!(summary.mean_return, 4.0);
    assert_eq!(summary.sample_std_dev, Some(2.0));
    assert_eq!(summary.worst_return, 2.0);
    assert_eq!(summary.horizon_hits, 1);
    let singleton = evaluate_trials(&records[..1], |&value| {
        Ok::<_, RolloutError<Infallible>>(value)
    })
    .unwrap();
    assert_eq!(singleton.sample_std_dev, None);
}

#[test]
fn invalid_suites_and_policy_errors_are_not_silent_partial_reports() {
    let empty: &[EpisodeSummary] = &[];
    let result = evaluate_trials(empty, |_| -> Result<_, RolloutError<Infallible>> {
        panic!("empty suites must not invoke the policy")
    });
    assert!(matches!(
        result,
        Err(RolloutError::Control(ControlError::InvalidConfig(_)))
    ));
    let result = evaluate_trials(&[0, 1, 2], |&start| {
        if start == 1 {
            Err(RolloutError::Policy("policy failure"))
        } else {
            Ok(record(1.0, 1, false, true))
        }
    });
    assert_eq!(result, Err(RolloutError::Policy("policy failure")));
    let result = evaluate_trials(&[()], |_| {
        Ok::<_, RolloutError<Infallible>>(record(f64::NAN, 1, false, true))
    });
    assert!(matches!(
        result,
        Err(RolloutError::Control(ControlError::Numerical(_)))
    ));
    let result = evaluate_trials(&[usize::MAX, 1], |&steps| {
        Ok::<_, RolloutError<Infallible>>(record(0.0, steps, false, true))
    });
    assert!(matches!(
        result,
        Err(RolloutError::Control(ControlError::Numerical(_)))
    ));
}

#[test]
fn no_trace_episodes_match_recorded_rollouts_in_all_environments() {
    let config = CartPoleConfig {
        max_steps: 12,
        ..CartPoleConfig::default()
    };
    let mut unrecorded = CartPole::new(config.clone()).unwrap();
    let mut recorded = CartPole::new(config).unwrap();
    let summary = unrecorded
        .run_episode(|_| Ok::<_, Infallible>(vec![0.0]))
        .unwrap();
    let trace = recorded
        .rollout(|_| Ok::<_, Infallible>(vec![0.0]))
        .unwrap();
    assert_eq!(summary, EpisodeSummary::from(&trace));
    assert_eq!(unrecorded.state(), recorded.state());
    assert!(matches!(
        unrecorded.run_episode(|_| Ok::<_, Infallible>(vec![0.0])),
        Err(RolloutError::Control(ControlError::EpisodeFinished))
    ));

    let config = DoublePoleConfig {
        max_steps: 12,
        ..DoublePoleConfig::default()
    };
    let mut unrecorded = DoublePole::new(config.clone()).unwrap();
    let mut recorded = DoublePole::new(config).unwrap();
    let summary = unrecorded
        .run_episode(|_| Ok::<_, Infallible>(vec![0.0]))
        .unwrap();
    let trace = recorded
        .rollout(|_| Ok::<_, Infallible>(vec![0.0]))
        .unwrap();
    assert_eq!(summary, EpisodeSummary::from(&trace));
    assert_eq!(unrecorded.state(), recorded.state());

    let config = SpringWalkerConfig {
        max_steps: 12,
        ..SpringWalkerConfig::default()
    };
    let mut unrecorded = SpringWalker::new(config.clone()).unwrap();
    let mut recorded = SpringWalker::new(config).unwrap();
    let summary = unrecorded
        .run_episode(|_| Ok::<_, Infallible>(vec![0.0, 0.0]))
        .unwrap();
    let trace = recorded
        .rollout(|_| Ok::<_, Infallible>(vec![0.0, 0.0]))
        .unwrap();
    assert_eq!(summary, EpisodeSummary::from(&trace));
    assert_eq!(unrecorded.state(), recorded.state());
}

fn recurrent_policy() -> Genome {
    Genome::new_recurrent(
        vec![
            NodeGene {
                id: 0,
                kind: NodeKind::Input,
            },
            NodeGene {
                id: 1,
                kind: NodeKind::Input,
            },
            NodeGene {
                id: 2,
                kind: NodeKind::Input,
            },
            NodeGene {
                id: 3,
                kind: NodeKind::Bias,
            },
            NodeGene {
                id: 4,
                kind: NodeKind::Output,
            },
        ],
        vec![
            ConnectionGene {
                innovation: 0,
                from: 3,
                to: 4,
                weight: 0.01,
                enabled: true,
            },
            ConnectionGene {
                innovation: 1,
                from: 4,
                to: 4,
                weight: 0.4,
                enabled: true,
            },
        ],
    )
    .unwrap()
}

#[test]
fn recurrent_trial_task_resets_memory_for_every_start_and_evaluation() {
    let config = DoublePoleConfig {
        max_steps: 100,
        observation: PoleObservations::PositionsOnly,
        ..DoublePoleConfig::default()
    };
    let start = DoublePoleState::default();
    let genome = recurrent_policy();
    let task = NeatDoublePoleTask::new(config.clone(), vec![start; 3]).unwrap();
    let episode = task.episode(&genome, start).unwrap();
    let summary = task.summary(&genome).unwrap();
    assert_eq!(summary.mean_return, episode.total_reward);
    assert_eq!(summary.control_steps, 3 * episode.steps);
    assert_eq!(summary.sample_std_dev, Some(0.0));
    assert_eq!(summary, task.summary(&genome).unwrap());
    assert_eq!(
        task.rollout(&genome, start).unwrap(),
        task.rollout(&genome, start).unwrap()
    );

    // This fixture really is sensitive to state carryover: a deliberately
    // unreset phenotype changes the second episode's initial action.
    let mut env = DoublePole::new(config).unwrap();
    let mut network = genome.compile_recurrent().unwrap();
    let fresh_action = network.step(&env.observation()).unwrap();
    env.run_episode(|observation| {
        Ok::<_, nevo::neat::NeatError>(vec![2.0 * network.step(observation)?[0] - 1.0])
    })
    .unwrap();
    env.reset_to(start).unwrap();
    let leaked_action = network.step(&env.observation()).unwrap();
    assert_ne!(fresh_action, leaked_action);
    network.reset();
    assert_eq!(network.step(&env.observation()).unwrap(), fresh_action);
}

#[test]
fn task_diagnostics_preserve_scalar_objectives_and_validate_shapes() {
    let config = DoublePoleConfig {
        max_steps: 40,
        ..DoublePoleConfig::default()
    };
    let starts = vec![
        DoublePoleState::default(),
        DoublePoleState {
            angles: [-0.07, 0.0],
            ..DoublePoleState::default()
        },
    ];
    let task = DoublePoleTask::new(config.clone(), starts.clone(), 4).unwrap();
    let parameters = vec![0.0; task.dimensions()];
    assert_eq!(
        task.value(&parameters).unwrap(),
        -task.summary(&parameters).unwrap().mean_return
    );
    assert!(matches!(
        NeatDoublePoleTask::new(config.clone(), vec![]),
        Err(ControlTaskError::EmptyStarts)
    ));
    let neat = NeatDoublePoleTask::new(config, starts).unwrap();
    assert!(matches!(
        neat.summary(&recurrent_policy()),
        Err(ControlTaskError::Neat(_))
    ));
}
