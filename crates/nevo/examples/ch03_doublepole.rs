//! Fixed-topology CMA-ES and feedforward NEAT on fully observed double-pole.
//!
//! Run: cargo run -p nevo --release --example ch03_doublepole -- book/results/ch03
//! Budgets and the held-out suite are fixed below; this is not a paper replication.

use nevo::algorithms::{CmaEs, CmaEsConfig, Neat, NeatConfig};
use nevo::control::{ControlRollout, DoublePole, DoublePoleConfig, DoublePoleState, RolloutError};
use nevo::neat::{Genome, NeatError, NodeKind};
use nevo::tasks::{ControlTaskError, DoublePoleTask};
use nevo::{Evaluator, step};
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{RngCore, SeedableRng},
};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const SEED: u64 = 42;
const HORIZON: usize = 500;
const BATCH_CAP: usize = 150;
const CMA_POPULATION: usize = 32;
const NEAT_POPULATION: usize = 150;

fn survival<E: Error + 'static>(
    config: &DoublePoleConfig,
    start: DoublePoleState,
    mut policy: impl FnMut(&[f64]) -> Result<Vec<f64>, E>,
) -> Result<(f64, usize), RolloutError<E>> {
    let mut environment = DoublePole::new(config.clone()).map_err(RolloutError::Control)?;
    environment.reset_to(start).map_err(RolloutError::Control)?;
    let mut reward = 0.0;
    loop {
        let action = policy(&environment.observation()).map_err(RolloutError::Policy)?;
        let outcome = environment.step(&action).map_err(RolloutError::Control)?;
        reward += outcome.reward;
        if outcome.done() {
            return Ok((reward, environment.steps()));
        }
    }
}

struct FixedPole {
    task: DoublePoleTask,
    evaluations: usize,
    control_steps: usize,
}

impl Evaluator<Vec<f64>> for FixedPole {
    type Output = f64;
    type Error = ControlTaskError;

    fn evaluate(&mut self, parameters: &Vec<f64>, _seed: u64) -> Result<f64, Self::Error> {
        let network = self.task.network(parameters.clone())?;
        let (reward, steps) = survival(self.task.config(), self.task.starts()[0], |input| {
            network.forward(input)
        })?;
        self.evaluations += 1;
        self.control_steps += steps;
        // DoublePoleTask and the scalar-search examples minimize negative return.
        Ok(-reward)
    }
}

// A graph genome plugs directly into the same GES evaluator interface. The
// environment and algorithm do not need to know each other's concrete types.
struct NeatPole {
    config: DoublePoleConfig,
    start: DoublePoleState,
    evaluations: usize,
    control_steps: usize,
}

impl Evaluator<Genome> for NeatPole {
    type Output = f64;
    type Error = RolloutError<NeatError>;

    fn evaluate(&mut self, genome: &Genome, _seed: u64) -> Result<f64, Self::Error> {
        let network = genome.compile().map_err(RolloutError::Policy)?;
        if network.input_count() != 6 || network.output_count() != 1 {
            return Err(RolloutError::Policy(NeatError::InvalidGenome(
                "double-pole needs six inputs and one output",
            )));
        }
        let (reward, steps) = survival(&self.config, self.start, |input| {
            Ok(vec![2.0 * network.forward(input)?[0] - 1.0])
        })?;
        self.evaluations += 1;
        self.control_steps += steps;
        // Canonical sharing uses finite nonnegative maximizing fitness.
        Ok(reward)
    }
}

fn heldout_starts() -> Vec<DoublePoleState> {
    let mut starts = Vec::new();
    for position in [-0.1, 0.1] {
        for long in [-0.07, -0.03, 0.03, 0.07] {
            for short in [-0.003, 0.003] {
                starts.push(DoublePoleState {
                    position,
                    velocity: 0.0,
                    angles: [long, short],
                    angular_velocities: [0.0; 2],
                });
            }
        }
    }
    starts
}

fn save_trace(
    writer: &mut impl Write,
    algorithm: &str,
    partition: &str,
    start_id: usize,
    episode: &ControlRollout<DoublePoleState>,
    config: &DoublePoleConfig,
) -> Result<(), Box<dyn Error>> {
    for (step, state) in episode.states.iter().enumerate() {
        let action = episode
            .actions
            .get(step)
            .map(|a| format!("{:.17e}", a[0]))
            .unwrap_or_default();
        let force = episode
            .actions
            .get(step)
            .map(|a| format!("{:.17e}", a[0] * config.force))
            .unwrap_or_default();
        writeln!(
            writer,
            "{algorithm},{SEED},{partition},{start_id},{step},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{action},{force}",
            step as f64 * config.dt,
            state.position,
            state.velocity,
            state.angles[0],
            state.angular_velocities[0],
            state.angles[1],
            state.angular_velocities[1]
        )?;
    }
    Ok(())
}

struct Report {
    mean: f64,
    minimum: f64,
    horizon_hits: usize,
}

fn report_trials(
    algorithm: &str,
    config: &DoublePoleConfig,
    heldout: &[DoublePoleState],
    mut rollout: impl FnMut(DoublePoleState) -> Result<ControlRollout<DoublePoleState>, Box<dyn Error>>,
    trials: &mut impl Write,
    traces: &mut impl Write,
) -> Result<Report, Box<dyn Error>> {
    let mut total = 0.0;
    let mut minimum = f64::INFINITY;
    let mut horizon_hits = 0;
    for (partition, states) in [
        ("training", vec![DoublePoleState::default()]),
        ("heldout", heldout.to_vec()),
    ] {
        for (start_id, &start) in states.iter().enumerate() {
            let episode = rollout(start)?;
            writeln!(
                trials,
                "{algorithm},{SEED},{partition},{start_id},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{},{}",
                start.position,
                start.velocity,
                start.angles[0],
                start.angular_velocities[0],
                start.angles[1],
                start.angular_velocities[1],
                episode.total_reward,
                episode.steps,
                episode.terminated,
                episode.truncated
            )?;
            if partition == "heldout" {
                total += episode.total_reward;
                minimum = minimum.min(episode.total_reward);
                horizon_hits += usize::from(episode.truncated && !episode.terminated);
            }
            // Choose the first held-out case in advance, not the best-looking trial.
            if partition == "training" || start_id == 0 {
                save_trace(traces, algorithm, partition, start_id, &episode, config)?;
            }
        }
    }
    Ok(Report {
        mean: total / heldout.len() as f64,
        minimum,
        horizon_hits,
    })
}

fn save_neat_genome(directory: &Path, genome: &Genome) -> Result<(), Box<dyn Error>> {
    let mut nodes = BufWriter::new(File::create(directory.join("doublepole_neat_nodes.csv"))?);
    let mut connections = BufWriter::new(File::create(
        directory.join("doublepole_neat_connections.csv"),
    )?);
    writeln!(nodes, "id,kind")?;
    writeln!(connections, "innovation,from,to,weight,enabled")?;
    for node in genome.nodes() {
        let kind = match node.kind {
            NodeKind::Input => "input",
            NodeKind::Bias => "bias",
            NodeKind::Hidden => "hidden",
            NodeKind::Output => "output",
        };
        writeln!(nodes, "{},{kind}", node.id)?;
    }
    for connection in genome.connections() {
        writeln!(
            connections,
            "{},{},{},{:.17e},{}",
            connection.innovation,
            connection.from,
            connection.to,
            connection.weight,
            connection.enabled
        )?;
    }
    nodes.flush()?;
    connections.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: ch03_doublepole [output-directory]".into());
    }
    let directory = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("results/ch03"));
    fs::create_dir_all(&directory)?;
    let config = DoublePoleConfig {
        max_steps: HORIZON,
        ..DoublePoleConfig::default()
    };
    let training = DoublePoleState::default();
    let heldout = heldout_starts();
    let task = DoublePoleTask::new(config.clone(), vec![training], 4)?;

    // Record the complete fixed protocol before running either search. Neither
    // held-out scores nor trajectories influence configuration or stopping.
    fs::write(
        directory.join("doublepole_manifest.json"),
        r#"{
  "experiment": "Chapter 3 full-state double-pole teaching comparison",
  "search_seed": 42,
  "physics": {"gravity":9.8,"cart_mass":1.0,"pole_masses":[0.1,0.01],"half_lengths":[0.5,0.05],"hinge_damping":0.000002,"force_newtons":10.0,"dt_seconds":0.02,"rk4_substeps":2,"position_limit":2.4,"angle_limit_radians":0.628329,"max_steps":500},
  "observation": "Full state [x/4.8,v/2,long_angle/0.52,long_omega/2,short_angle/0.52,short_omega/2], not clipped",
  "training_start": {"position":0,"velocity":0,"angles":[0.07,0],"angular_velocities":[0,0]},
  "heldout_suite": {"cart_positions":[-0.1,0.1],"long_angles":[-0.07,-0.03,0.03,0.07],"short_angles":[-0.003,0.003],"all_velocities":0,"construction":"Cartesian product, position outermost then long angle then short angle","trials":16},
  "fitness": "Number of valid control endpoints; a failing final action earns zero",
  "stopping": "Best observed training return reaches 500 or 150 selected batches, including initialization",
  "cma_es": {"network":[6,4,1],"activations":["tanh","tanh"],"parameters":33,"population":32,"evaluation_cap":4800,"initial_mean":"ChaCha8 seed42 stream2, uniform [-0.5,0.5)","sigma":0.5,"objective":"minimize negative training return","other_settings":"CmaEsConfig defaults","action":"tanh output is signed force fraction"},
  "neat": {"initial_network":"six inputs, one bias, one output, zero hidden nodes","population":150,"evaluation_cap":22500,"config":"NeatConfig::new(6,1,42), all defaults","objective":"maximize nonnegative training return","action":"2*sigmoid_output-1 is signed force fraction","recurrent":false},
  "accounting": "One training fitness call contains one episode. Training control steps count applied actions, including a failing final action. Final reporting adds one training and sixteen held-out episodes per selected policy; those are excluded from search counts.",
  "heldout_policy": "Only final training-selected policies are evaluated; no held-out tuning or reselection",
  "trace_selection": "Training start and held-out start_id 0, fixed before evaluation; action on each state row applies to the following state; final action is blank",
  "limitations": "Single search seed, different architectures and unequal budgets. No equal-cost algorithm ranking. This is not a partial-observation/recurrent experiment or a reproduction of the 2002 paper's benchmark protocol."
}
"#,
    )?;

    let mut curves = BufWriter::new(File::create(directory.join("doublepole_curves.csv"))?);
    let mut runs = BufWriter::new(File::create(directory.join("doublepole_runs.csv"))?);
    let mut trials = BufWriter::new(File::create(directory.join("doublepole_trials.csv"))?);
    let mut traces = BufWriter::new(File::create(directory.join("doublepole_trajectories.csv"))?);
    writeln!(
        curves,
        "algorithm,seed,batch,fitness_evaluations,training_control_steps,best_training_valid_steps,hidden_nodes,enabled_connections"
    )?;
    writeln!(
        runs,
        "algorithm,seed,batches,fitness_evaluations,training_control_steps,training_valid_steps,training_solved,heldout_mean_valid_steps,heldout_min_valid_steps,heldout_horizon_hits,heldout_trials,hidden_nodes,enabled_connections"
    )?;
    writeln!(
        trials,
        "algorithm,seed,partition,start_id,start_position,start_velocity,start_long_angle,start_long_omega,start_short_angle,start_short_omega,valid_steps,applied_actions,terminated,truncated"
    )?;
    writeln!(
        traces,
        "algorithm,seed,partition,start_id,step,time,position,velocity,long_angle,long_omega,short_angle,short_omega,action,force"
    )?;

    let mut initialization = ChaCha8Rng::seed_from_u64(SEED);
    initialization.set_stream(2);
    let mean = (0..task.dimensions())
        .map(|_| (initialization.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0 - 0.5)
        .collect();
    let mut cma_config = CmaEsConfig::new(mean, SEED);
    cma_config.population = CMA_POPULATION;
    let mut cma = CmaEs::new(cma_config)?;
    let mut fixed = FixedPole {
        task,
        evaluations: 0,
        control_steps: 0,
    };
    for batch in 1..=BATCH_CAP {
        let count = step(&mut cma, &mut fixed)?;
        if count != CMA_POPULATION {
            return Err("CMA batch size changed".into());
        }
        let reward = -cma.best().ok_or("missing CMA policy")?.output;
        writeln!(
            curves,
            "cma_es,{SEED},{batch},{},{},{reward:.17e},4,33",
            fixed.evaluations, fixed.control_steps
        )?;
        if reward >= HORIZON as f64 {
            break;
        }
    }
    let fixed_best = cma.best().ok_or("missing CMA policy")?;
    let fixed_training = -fixed_best.output;
    let fixed_report = report_trials(
        "cma_es",
        &config,
        &heldout,
        |start| Ok(fixed.task.rollout(&fixed_best.job.input, start)?),
        &mut trials,
        &mut traces,
    )?;
    let mut parameters = BufWriter::new(File::create(
        directory.join("doublepole_cma_parameters.txt"),
    )?);
    for parameter in &fixed_best.job.input {
        writeln!(parameters, "{parameter:.17e}")?;
    }
    parameters.flush()?;
    writeln!(
        runs,
        "cma_es,{SEED},{},{},{},{fixed_training:.17e},{},{:.17e},{:.17e},{},16,4,33",
        cma.batches(),
        fixed.evaluations,
        fixed.control_steps,
        fixed_training >= HORIZON as f64,
        fixed_report.mean,
        fixed_report.minimum,
        fixed_report.horizon_hits
    )?;
    println!(
        "DoublePole CMA seed42: {} fitness calls, {} applied training actions, train={fixed_training}/500, heldout_mean={:.3}, heldout_full={}/16",
        fixed.evaluations, fixed.control_steps, fixed_report.mean, fixed_report.horizon_hits
    );

    let mut neat = Neat::new(NeatConfig::new(6, 1, SEED))?;
    let mut evolved = NeatPole {
        config: config.clone(),
        start: training,
        evaluations: 0,
        control_steps: 0,
    };
    for batch in 1..=BATCH_CAP {
        let count = step(&mut neat, &mut evolved)?;
        if count != NEAT_POPULATION {
            return Err("NEAT batch size changed".into());
        }
        let best = neat.best().ok_or("missing NEAT policy")?;
        writeln!(
            curves,
            "neat,{SEED},{batch},{},{},{:.17e},{},{}",
            evolved.evaluations,
            evolved.control_steps,
            best.output,
            best.job.input.hidden_node_count(),
            best.job.input.enabled_connection_count()
        )?;
        if best.output >= HORIZON as f64 {
            break;
        }
    }
    let neat_best = neat.best().ok_or("missing NEAT policy")?;
    let network = neat_best.job.input.compile()?;
    let neat_report = report_trials(
        "neat",
        &config,
        &heldout,
        |start| {
            let mut environment = DoublePole::new(config.clone())?;
            environment.reset_to(start)?;
            Ok(environment.rollout(|input| {
                Ok::<_, NeatError>(vec![2.0 * network.forward(input)?[0] - 1.0])
            })?)
        },
        &mut trials,
        &mut traces,
    )?;
    save_neat_genome(&directory, &neat_best.job.input)?;
    writeln!(
        runs,
        "neat,{SEED},{},{},{},{:.17e},{},{:.17e},{:.17e},{},16,{},{}",
        neat.batches(),
        evolved.evaluations,
        evolved.control_steps,
        neat_best.output,
        neat_best.output >= HORIZON as f64,
        neat_report.mean,
        neat_report.minimum,
        neat_report.horizon_hits,
        neat_best.job.input.hidden_node_count(),
        neat_best.job.input.enabled_connection_count()
    )?;
    println!(
        "DoublePole NEAT seed42: {} fitness calls, {} applied training actions, train={}/500, heldout_mean={:.3}, heldout_full={}/16",
        evolved.evaluations,
        evolved.control_steps,
        neat_best.output,
        neat_report.mean,
        neat_report.horizon_hits
    );

    for writer in [&mut curves, &mut runs, &mut trials, &mut traces] {
        writer.flush()?;
    }
    println!(
        "Results retain training failures and all fixed held-out trials; budgets and policy classes differ."
    );
    Ok(())
}
