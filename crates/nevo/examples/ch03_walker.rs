//! A frozen, reproducible CMA-ES experiment on the original SpringWalker task.
//!
//! Run from the workspace root:
//! `cargo run --release -p nevo --example ch03_walker -- book/results/ch03`
//! The output describes measured locomotion; it does not claim Gym BipedalWalker
//! performance or assume that the evolved behavior is a walking gait.

use nevo::algorithms::{CmaEs, CmaEsConfig};
use nevo::control::{ControlStep, SpringWalker, SpringWalkerConfig, SpringWalkerState};
use nevo::nn::{Activation, Mlp};
use nevo::step;
use nevo::tasks::SpringWalkerTask;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const SEED: u64 = 42;
const POPULATION: usize = 32;
const BATCHES: usize = 100;
const SIGMA: f64 = 0.5;
const HIDDEN: usize = 4;

struct Case {
    name: &'static str,
    split: &'static str,
    config: SpringWalkerConfig,
    start: SpringWalkerState,
}

fn shifted_start(
    config: &SpringWalkerConfig,
    x: f64,
    vx: f64,
) -> Result<SpringWalkerState, Box<dyn Error>> {
    let mut state = config.initial_state()?;
    // Only the torso changes: shifting all three particles would be a symmetry.
    state.particles[0].position[0] = x;
    state.particles[0].velocity[0] = vx;
    Ok(state)
}

fn write_trace(
    out: &mut impl Write,
    policy: &str,
    case: &Case,
    env: &SpringWalker,
    action: &[f64],
    total_return: f64,
    outcome: ControlStep,
) -> Result<(), Box<dyn Error>> {
    let terminated = outcome.terminated;
    let truncated = outcome.truncated;
    write!(
        out,
        "{policy},{},{},{},{total_return:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{terminated},{truncated}",
        case.split,
        case.name,
        env.steps(),
        env.distance(),
        env.energy_used(),
        action[0],
        action[1]
    )?;
    for particle in env.state().particles {
        write!(
            out,
            ",{:.17e},{:.17e},{:.17e},{:.17e}",
            particle.position[0], particle.position[1], particle.velocity[0], particle.velocity[1]
        )?;
    }
    writeln!(out)?;
    Ok(())
}

fn evaluate_case(
    policy: &str,
    parameters: Option<&[f64]>,
    case: &Case,
    runs: &mut impl Write,
    traces: &mut impl Write,
) -> Result<(), Box<dyn Error>> {
    let network = parameters
        .map(|weights| {
            Mlp::new(
                &[13, HIDDEN, 2],
                &[Activation::Tanh, Activation::Tanh],
                weights.to_vec(),
            )
        })
        .transpose()?;
    let mut env = SpringWalker::new(case.config.clone())?;
    env.reset_to(case.start)?;
    let mut total_return = 0.0;
    write_trace(
        traces,
        policy,
        case,
        &env,
        &[0.0, 0.0],
        total_return,
        ControlStep {
            reward: 0.0,
            terminated: false,
            truncated: false,
        },
    )?;
    let result = loop {
        let action = match &network {
            Some(network) => network.forward(&env.observation())?,
            None => vec![0.0, 0.0],
        };
        let outcome = env.step(&action)?;
        total_return += outcome.reward;
        write_trace(traces, policy, case, &env, &action, total_return, outcome)?;
        if outcome.done() {
            break outcome;
        }
    };
    writeln!(
        runs,
        "{policy},{},{},{SEED},{},{:.17e},{:.17e},{:.17e},{},{total_return:.17e},{:.17e},{:.17e},{},{}",
        case.split,
        case.name,
        case.config.max_steps,
        case.config.friction,
        case.start.particles[0].position[0],
        case.start.particles[0].velocity[0],
        env.steps(),
        env.distance(),
        env.energy_used(),
        result.terminated,
        result.truncated
    )?;
    println!(
        "{policy:11} {:22} steps={:3} return={:9.5} COM={:9.5} m work={:9.5} J terminated={} truncated={}",
        case.name,
        env.steps(),
        total_return,
        env.distance(),
        env.energy_used(),
        result.terminated,
        result.truncated
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("book/results/ch03"), PathBuf::from);
    fs::create_dir_all(&output)?;
    let training_config = SpringWalkerConfig {
        max_steps: 250,
        ..Default::default()
    };
    let starts = [-0.015, 0.0, 0.015]
        .into_iter()
        .map(|x| shifted_start(&training_config, x, 0.0))
        .collect::<Result<Vec<_>, _>>()?;
    let mut task = SpringWalkerTask::new(training_config.clone(), starts.clone(), HIDDEN)?;
    let zero_parameters = vec![0.0; task.dimensions()];
    let baseline = -task.value(&zero_parameters)?;
    let mut config = CmaEsConfig::new(zero_parameters, SEED);
    config.population = POPULATION;
    config.sigma = SIGMA;
    let mut search = CmaEs::new(config)?;
    let mut curves = BufWriter::new(File::create(output.join("walker_curves.csv"))?);
    writeln!(curves, "batch,evaluations,best_training_return,sigma")?;
    println!("SpringWalker is an original three-particle task, not Gym BipedalWalker.");
    println!(
        "Frozen CMA-ES: seed={SEED}, population={POPULATION}, batches={BATCHES}, sigma={SIGMA}, parameters={}",
        task.dimensions()
    );
    println!("Three 250-step training starts; zero-action mean return {baseline:.9}.");
    for _ in 0..BATCHES {
        step(&mut search, &mut task)?;
        let best = search.best().expect("one successful nonempty batch");
        writeln!(
            curves,
            "{},{},{:.17e},{:.17e}",
            search.batches(),
            search.evaluations(),
            -best.output,
            search.sigma()
        )?;
        if search.batches() % 20 == 0 {
            println!(
                "batch {:3}: {} objective calls; best training return {:.9}",
                search.batches(),
                search.evaluations(),
                -best.output
            );
        }
    }
    curves.flush()?;
    let best = search.best().expect("100 successful nonempty batches");
    let parameters = &best.job.input;
    let mut parameter_file = BufWriter::new(File::create(output.join("walker_parameters.txt"))?);
    for parameter in parameters {
        writeln!(parameter_file, "{parameter:.17e}")?;
    }
    parameter_file.flush()?;

    let mut cases: Vec<_> = [
        "train_torso_x_minus_015",
        "train_neutral",
        "train_torso_x_plus_015",
    ]
    .into_iter()
    .zip(starts)
    .map(|(name, start)| Case {
        name,
        split: "training_replay",
        config: training_config.clone(),
        start,
    })
    .collect();
    let extended = SpringWalkerConfig {
        max_steps: 500,
        ..training_config.clone()
    };
    for (name, split, x, vx, friction) in [
        ("neutral_500", "extended", 0.0, 0.0, 0.8),
        ("torso_x_minus_03", "held_out", -0.03, 0.0, 0.8),
        ("torso_x_plus_03", "held_out", 0.03, 0.0, 0.8),
        ("torso_vx_minus_05", "held_out", 0.0, -0.05, 0.8),
        ("torso_vx_plus_05", "held_out", 0.0, 0.05, 0.8),
        ("friction_06", "held_out", 0.0, 0.0, 0.6),
        ("friction_10", "held_out", 0.0, 0.0, 1.0),
    ] {
        let config = SpringWalkerConfig {
            friction,
            ..extended.clone()
        };
        let start = shifted_start(&config, x, vx)?;
        cases.push(Case {
            name,
            split,
            config,
            start,
        });
    }
    let mut runs = BufWriter::new(File::create(output.join("walker_runs.csv"))?);
    writeln!(
        runs,
        "policy,split,case,search_seed,horizon,friction,initial_torso_x,initial_torso_vx,steps,return,com_distance,actuator_work,terminated,truncated"
    )?;
    let mut traces = BufWriter::new(File::create(output.join("walker_trajectories.csv"))?);
    writeln!(
        traces,
        "policy,split,case,control_step,cumulative_return,com_distance,actuator_work,action_left,action_right,terminated,truncated,torso_x,torso_y,torso_vx,torso_vy,left_x,left_y,left_vx,left_vy,right_x,right_y,right_vx,right_vy"
    )?;
    for case in &cases {
        evaluate_case("zero_action", None, case, &mut runs, &mut traces)?;
        evaluate_case("selected", Some(parameters), case, &mut runs, &mut traces)?;
    }
    runs.flush()?;
    traces.flush()?;
    let manifest = format!(
        r#"{{
  "experiment": "original SpringWalker; no Gym BipedalWalker equivalence",
  "algorithm": "CMA-ES, positive weights, full covariance, minimize negative mean return",
  "seed": {SEED},
  "population": {POPULATION},
  "batches": {BATCHES},
  "objective_evaluations": {},
  "episodes_per_training_evaluation": 3,
  "training_episode_horizon": 250,
  "training_torso_x": [-0.015, 0.0, 0.015],
  "held_out_episode_horizon": 500,
  "held_out_cases": ["torso_x=-0.03", "torso_x=0.03", "torso_vx=-0.05", "torso_vx=0.05", "friction=0.6", "friction=1.0"],
  "held_out_cases_combined": false,
  "extended_neutral_case": true,
  "architecture": [13, 4, 2],
  "activations": ["tanh", "tanh"],
  "parameter_count": {},
  "initial_mean": "all zero",
  "initial_sigma": {SIGMA},
  "initial_covariance": "identity",
  "chosen_policy": "best evaluated sample on the three training starts only",
  "zero_action_training_mean_return": {baseline:.17e},
  "chosen_training_mean_return": {:.17e},
  "return_definition": "COM horizontal displacement - energy_cost * positive actuator work - fall_penalty if terminated",
  "integration": "semi-implicit Euler; fixed substeps; capped dissipative friction impulse",
  "control_dt_seconds": {},
  "physics_substeps_per_action": {},
  "masses_kg": [{}, {}, {}],
  "gravity_m_per_s2": {},
  "leg_length_m": {},
  "foot_spacing_m": {},
  "leg_stiffness_N_per_m": {},
  "spacing_stiffness_N_per_m": {},
  "spring_damping_Ns_per_m": {},
  "actuation_fraction": {},
  "ground_stiffness_N_per_m": {},
  "ground_damping_Ns_per_m": {},
  "training_friction": {},
  "torso_radius_m": {},
  "foot_radius_m": {},
  "energy_cost_m_per_J": {},
  "fall_penalty": {},
  "initial_baseline_summary_episodes": 3,
  "recorded_baseline_and_selected_replay_episodes": 20,
  "curves_include_extra_evaluations": false,
  "hyperparameter_tuning_on_results": false,
  "notes": "A single prespecified seed and budget, not a benchmark of optimizer reliability. Time-limit truncation is not success. Baseline and replay calls are outside the optimizer objective-call count. Final falling actions can contain fewer physics substeps; traces therefore index control steps, not inferred elapsed time."
}}
"#,
        search.evaluations(),
        task.dimensions(),
        -best.output,
        training_config.dt,
        training_config.substeps,
        training_config.masses[0],
        training_config.masses[1],
        training_config.masses[2],
        training_config.gravity,
        training_config.leg_length,
        training_config.foot_spacing,
        training_config.leg_stiffness,
        training_config.spacing_stiffness,
        training_config.spring_damping,
        training_config.actuation_fraction,
        training_config.ground_stiffness,
        training_config.ground_damping,
        training_config.friction,
        training_config.torso_radius,
        training_config.foot_radius,
        training_config.energy_cost,
        training_config.fall_penalty
    );
    fs::write(output.join("walker_manifest.json"), manifest)?;
    println!(
        "Saved frozen settings, selected parameters, curves, all episode metrics, and complete trajectories to {}",
        output.display()
    );
    Ok(())
}
