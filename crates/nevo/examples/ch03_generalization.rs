//! Matched-budget, multi-seed double-pole generalization experiment.
//!
//! `cargo run --release -p nevo --example ch03_generalization -- OUTPUT`
//! Add `--seed 0` to run one independent seed, or `--smoke` for a small wiring check.
//! Existing seed directories are never overwritten. See each pre-run manifest.

use nevo::algorithms::{CmaEs, CmaEsConfig, Neat, NeatConfig};
use nevo::control::{
    ControlRollout, DoublePole, DoublePoleConfig, DoublePoleState, EpisodeSummary,
    PoleObservations, RolloutError, TrialSummary, evaluate_trials,
};
use nevo::neat::{Genome, NetworkMode};
use nevo::tasks::{DoublePoleTask, NeatDoublePoleTask};
use nevo::{Algorithm, Evaluated, JobId};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use std::convert::Infallible;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Csv = BufWriter<File>;

const TRAIN_SEED: u64 = 0x4e45_564f_5452_4149;
const TEST_SEED: u64 = 0x4e45_564f_5445_5354;
const SUITE_STREAM: u64 = 4;

#[derive(Clone, Copy)]
struct Protocol {
    smoke: bool,
    population: usize,
    training_budget: usize,
    horizon: usize,
    heldout_count: usize,
}

impl Protocol {
    fn new(smoke: bool) -> Self {
        if smoke {
            Self {
                smoke,
                population: 8,
                training_budget: 256,
                horizon: 40,
                heldout_count: 8,
            }
        } else {
            Self {
                smoke,
                population: 64,
                training_budget: 32_768,
                horizon: 500,
                heldout_count: 100,
            }
        }
    }

    fn name(self) -> &'static str {
        if self.smoke {
            "smoke_only"
        } else {
            "controlled_v1"
        }
    }

    fn batches(self, starts: usize) -> Result<usize> {
        let batch_episodes = self
            .population
            .checked_mul(starts)
            .ok_or("batch budget overflow")?;
        if batch_episodes == 0
            || self.training_budget == 0
            || self.training_budget % batch_episodes != 0
        {
            return Err(
                "training budget must contain a positive integer number of complete batches".into(),
            );
        }
        Ok(self.training_budget / batch_episodes)
    }
}

#[derive(Clone, Copy)]
enum Method {
    Cma,
    Neat(NetworkMode),
}

#[derive(Clone, Copy)]
struct Condition {
    id: &'static str,
    starts: usize,
    observation: PoleObservations,
    method: Method,
}

const CONDITIONS: [Condition; 6] = [
    Condition {
        id: "full_cma_1",
        starts: 1,
        observation: PoleObservations::FullState,
        method: Method::Cma,
    },
    Condition {
        id: "full_cma_16",
        starts: 16,
        observation: PoleObservations::FullState,
        method: Method::Cma,
    },
    Condition {
        id: "full_ff_neat_1",
        starts: 1,
        observation: PoleObservations::FullState,
        method: Method::Neat(NetworkMode::FeedForward),
    },
    Condition {
        id: "full_ff_neat_16",
        starts: 16,
        observation: PoleObservations::FullState,
        method: Method::Neat(NetworkMode::FeedForward),
    },
    Condition {
        id: "partial_ff_neat_16",
        starts: 16,
        observation: PoleObservations::PositionsOnly,
        method: Method::Neat(NetworkMode::FeedForward),
    },
    Condition {
        id: "partial_recurrent_neat_16",
        starts: 16,
        observation: PoleObservations::PositionsOnly,
        method: Method::Neat(NetworkMode::Recurrent),
    },
];

fn signed(rng: &mut ChaCha8Rng) -> f64 {
    2.0 * ((rng.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0) - 1.0
}

fn suite(seed: u64, count: usize) -> Vec<DoublePoleState> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(SUITE_STREAM);
    (0..count)
        .map(|_| DoublePoleState {
            position: 0.1 * signed(&mut rng),
            velocity: 0.02 * signed(&mut rng),
            angles: [0.07 * signed(&mut rng), 0.005 * signed(&mut rng)],
            angular_velocities: [0.02 * signed(&mut rng), 0.05 * signed(&mut rng)],
        })
        .collect()
}

fn suites(protocol: Protocol) -> Result<(Vec<DoublePoleState>, Vec<DoublePoleState>)> {
    let mut training = vec![DoublePoleState::default()];
    training.extend(suite(TRAIN_SEED, 15));
    let heldout = suite(TEST_SEED, protocol.heldout_count);
    if heldout.iter().any(|state| training.contains(state)) {
        return Err("training and held-out suites intersect".into());
    }
    for (index, state) in heldout.iter().enumerate() {
        if heldout[..index].contains(state) {
            return Err("held-out starts are duplicated".into());
        }
    }
    Ok((training, heldout))
}

fn physics(protocol: Protocol, condition: Condition) -> DoublePoleConfig {
    DoublePoleConfig {
        max_steps: protocol.horizon,
        observation: condition.observation,
        ..DoublePoleConfig::default()
    }
}

fn cma_config(protocol: Protocol, seed: u64) -> CmaEsConfig {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(2);
    let initial_mean = (0..33).map(|_| 0.5 * signed(&mut rng)).collect();
    let mut config = CmaEsConfig::new(initial_mean, seed);
    config.population = protocol.population;
    config.sigma = 0.5;
    config
}

fn neat_config(protocol: Protocol, seed: u64, condition: Condition) -> NeatConfig {
    let mut config = NeatConfig::new(condition.observation.dimensions(), 1, seed);
    config.population = protocol.population;
    if let Method::Neat(mode) = condition.method {
        config.mode = mode;
    }
    config
}

fn csv(path: &Path, header: &str) -> Result<Csv> {
    let mut writer = BufWriter::new(File::create_new(path)?);
    writeln!(writer, "{header}")?;
    Ok(writer)
}

fn optional<T: ToString>(value: Option<T>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn write_setup(
    directory: &Path,
    protocol: Protocol,
    seed: u64,
    training: &[DoublePoleState],
    heldout: &[DoublePoleState],
) -> Result<()> {
    let mut manifest = BufWriter::new(File::create_new(directory.join("manifest.md"))?);
    writeln!(
        manifest,
        "# Double-pole controlled protocol: {}\n",
        protocol.name()
    )?;
    writeln!(
        manifest,
        "This file and starts.csv were written before any optimization or held-out evaluation.\n"
    )?;
    writeln!(
        manifest,
        "- Optimizer seed: {seed}; full study uses seeds 0 through 9."
    )?;
    writeln!(
        manifest,
        "- Population: {}; training rollout episodes per condition: {}; horizon: {}; held-out starts: {}.",
        protocol.population, protocol.training_budget, protocol.horizon, protocol.heldout_count
    )?;
    writeln!(
        manifest,
        "- One-start conditions use {} complete batches; 16-start conditions use {}. Initialization is batch 1 and counts toward the budget. No early stopping.",
        protocol.batches(1)?,
        protocol.batches(16)?
    )?;
    writeln!(
        manifest,
        "- Fixed training suite: historical default first, then 15 pseudorandom starts. A single-start condition uses only the first entry."
    )?;
    writeln!(
        manifest,
        "- Suite RNG: ChaCha8; stream {SUITE_STREAM}; training seed {TRAIN_SEED}; independent held-out seed {TEST_SEED}."
    )?;
    writeln!(
        manifest,
        "- Sampling draw order: position, velocity, long angle, short angle, long angular velocity, short angular velocity. Uniform half-open intervals [-b,b), b=[0.1,0.02,0.07,0.005,0.02,0.05]. Exact train/test disjointness and held-out uniqueness are checked."
    )?;
    writeln!(
        manifest,
        "- Selection: highest mean training return actually observed at the fixed budget; exact ties retain the earliest evaluated record. CMA minimizes its negative. Fixed suites permit historical comparison. Only selected final policies see held-out starts; no held-out tuning, selection or stopping."
    )?;
    writeln!(
        manifest,
        "- Reward: one per physically valid control endpoint, zero for a failing endpoint. Step counts include a failing action. Horizon success requires truncated=true AND terminated=false."
    )?;
    writeln!(
        manifest,
        "- Every episode resets environment and recurrent phenotype. Recurrent mode uses synchronous ticks: current inputs/bias, previous hidden/output activations. Initial hidden/output state is zero."
    )?;
    writeln!(
        manifest,
        "- Recurrent diagnostic: the same selected recurrent genome is additionally evaluated with state reset before every action, on the same held-out starts. This also suppresses delayed acyclic hidden-to-output paths; it is not an edge-deletion or equivalent feedforward baseline. These {} extra episodes are reported separately and never used for selection.",
        protocol.heldout_count
    )?;
    writeln!(
        manifest,
        "- Predeclared traces: only optimizer seed 0 and held-out case 0, for all six selected policies plus the recurrent state-reset diagnostic. These reporting replays occur after selection and are counted separately in trace_replay_episodes/control_steps. The trace return column is cumulative; step 0 is the initial state with no action."
    )?;
    writeln!(
        manifest,
        "- Sample SD uses n-1, with a blank CSV value for n=1. Trial SD is variability across starts; optimizer-run uncertainty must be computed across independent seed rows, not pooled trials."
    )?;
    writeln!(
        manifest,
        "- CMA controller: 6 -> 4 -> 1 tanh/tanh MLP, 33 parameters. Initial mean uniform [-0.5,0.5) from optimizer seed, stream 2; sigma 0.5. NEAT: direct sigmoid output mapped to 2*y-1; defaults except population and mode."
    )?;
    writeln!(
        manifest,
        "- Fairness scope: equal rollout-episode budgets and common populations; fitness-call counts differ with suite size. Control steps and runtime are separately reported. Architectures, activations, search methods, initializations and adaptation horizons differ. This is not a benchmark of optimizers with an identical representation, and the initial-state distribution is deliberately local."
    )?;
    writeln!(
        manifest,
        "- Reported seconds cover the whole condition: setup, training, policy serialization, held-out/ablation evaluation and trace replays, with some I/O. It is not training-only timing. Across-seed uncertainty is conditional on this one fixed training and held-out suite."
    )?;
    writeln!(
        manifest,
        "- This is a modern controlled teaching experiment, not a reproduction of the original 2002 NEAT non-Markov oscillation/generalization protocol, not a reproduction of all settings in the book, and not evidence from a smoke run.\n"
    )?;
    writeln!(
        manifest,
        "Sources: <https://neuroevolutionbook.com/>; <https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf>; <https://arxiv.org/abs/1604.00772>.\n"
    )?;
    for condition in CONDITIONS {
        writeln!(
            manifest,
            "## {}\n\nTraining starts: {}; physics:\n\n```rust\n{:#?}\n```\n",
            condition.id,
            condition.starts,
            physics(protocol, condition)
        )?;
        match condition.method {
            Method::Cma => writeln!(
                manifest,
                "Search configuration:\n\n```rust\n{:#?}\n```\n",
                cma_config(protocol, seed)
            )?,
            Method::Neat(_) => writeln!(
                manifest,
                "Search configuration:\n\n```rust\n{:#?}\n```\n",
                neat_config(protocol, seed, condition)
            )?,
        }
    }
    manifest.flush()?;
    let mut starts = csv(
        &directory.join("starts.csv"),
        "suite,trial,position,velocity,long_angle,short_angle,long_angular_velocity,short_angular_velocity",
    )?;
    for (name, states) in [("training", training), ("heldout", heldout)] {
        for (index, state) in states.iter().enumerate() {
            writeln!(
                starts,
                "{name},{index},{},{},{},{},{},{}",
                state.position,
                state.velocity,
                state.angles[0],
                state.angles[1],
                state.angular_velocities[0],
                state.angular_velocities[1]
            )?;
        }
    }
    starts.flush()?;
    Ok(())
}

#[derive(Default)]
struct Diagnostics {
    best_hidden: Option<usize>,
    best_enabled: Option<usize>,
    best_connections: Option<usize>,
    population_hidden: Option<f64>,
    population_enabled: Option<f64>,
    population_connections: Option<f64>,
    species: Option<usize>,
}

struct Selected<I> {
    input: I,
    id: JobId,
    batch: usize,
    summary: TrialSummary,
}

struct Trained<I> {
    selected: Selected<I>,
    fitness_calls: usize,
    rollouts: usize,
    control_steps: usize,
    diagnostics: Diagnostics,
}

struct Output {
    runs: Csv,
    training: Csv,
    trials: Csv,
    species: Csv,
    traces: Csv,
}

impl Output {
    fn new(directory: &Path) -> Result<Self> {
        Ok(Self {
            runs: csv(
                &directory.join("runs.csv"),
                "protocol,condition,seed,training_starts,population,horizon,training_budget,batches,fitness_calls,training_rollouts,training_control_steps,selected_job_id,selected_batch,train_mean,train_sample_sd,train_worst,train_horizon_hits,train_episodes,heldout_mean,heldout_sample_sd,heldout_worst,heldout_horizon_hits,heldout_episodes,heldout_control_steps,ablation_mean,ablation_sample_sd,ablation_worst,ablation_horizon_hits,ablation_episodes,ablation_control_steps,best_hidden_nodes,best_enabled_connections,best_total_connections,final_species_count,trace_replay_episodes,trace_replay_control_steps,seconds",
            )?,
            training: csv(
                &directory.join("training.csv"),
                "protocol,condition,seed,batch,fitness_calls,training_rollouts,training_control_steps,batch_mean_return,batch_worst_return,batch_horizon_hits,batch_episodes,best_train_mean,best_train_sample_sd,best_train_worst,best_train_horizon_hits,selected_job_id,selected_batch,best_hidden_nodes,best_enabled_connections,best_total_connections,population_mean_hidden_nodes,population_mean_enabled_connections,population_mean_total_connections,species_count",
            )?,
            trials: csv(
                &directory.join("trials.csv"),
                "protocol,condition,seed,suite,trial,return,control_steps,terminated,truncated,horizon_hit",
            )?,
            species: csv(
                &directory.join("species.csv"),
                "protocol,condition,seed,batch,species_id,size,best_fitness,best_ever_fitness,stagnant_generations,shared_fitness,offspring,protected",
            )?,
            traces: csv(
                &directory.join("traces.csv"),
                "condition,seed,suite,trial,step,position,velocity,long_angle,short_angle,long_angular_velocity,short_angular_velocity,action,return,terminated,truncated",
            )?,
        })
    }

    fn flush(&mut self) -> Result<()> {
        self.runs.flush()?;
        self.training.flush()?;
        self.trials.flush()?;
        self.species.flush()?;
        self.traces.flush()?;
        Ok(())
    }
}

struct Run<'a> {
    protocol: Protocol,
    condition: Condition,
    seed: u64,
    output: &'a mut Output,
    trace_episodes: usize,
    trace_steps: usize,
}

impl Run<'_> {
    fn trace(&mut self, suite_name: &str, trace: &ControlRollout<DoublePoleState>) -> Result<()> {
        for (step, state) in trace.states.iter().enumerate() {
            let last = step == trace.steps;
            let action = if step == 0 {
                None
            } else {
                Some(trace.actions[step - 1][0])
            };
            // Double-pole returns exactly one per valid endpoint; only a final
            // failing endpoint can return zero. Step zero has cumulative return 0.
            let cumulative_return = if last {
                trace.total_reward
            } else {
                step as f64
            };
            writeln!(
                self.output.traces,
                "{},{},{suite_name},0,{step},{},{},{},{},{},{},{},{cumulative_return},{},{}",
                self.condition.id,
                self.seed,
                state.position,
                state.velocity,
                state.angles[0],
                state.angles[1],
                state.angular_velocities[0],
                state.angular_velocities[1],
                optional(action),
                last && trace.terminated,
                last && trace.truncated
            )?;
        }
        self.trace_episodes += 1;
        self.trace_steps += trace.steps;
        Ok(())
    }

    fn train<A, E, D>(
        &mut self,
        search: &mut A,
        mut evaluate: E,
        mut diagnose: D,
    ) -> Result<Trained<A::Input>>
    where
        A: Algorithm<Output = f64>,
        A::Input: Clone,
        E: FnMut(&A::Input) -> Result<TrialSummary>,
        D: FnMut(&A, &A::Input, usize, &mut Csv) -> Result<Diagnostics>,
    {
        let mut selected: Option<Selected<A::Input>> = None;
        let mut fitness_calls = 0;
        let mut rollouts = 0;
        let mut control_steps = 0;
        let mut diagnostics = Diagnostics::default();
        for batch in 1..=self.protocol.batches(self.condition.starts)? {
            let jobs = search.generate()?;
            if jobs.len() != self.protocol.population {
                return Err("search emitted an unexpected population size".into());
            }
            let mut evaluated = Vec::with_capacity(jobs.len());
            let mut batch_mean = 0.0;
            let mut batch_worst = f64::INFINITY;
            let mut batch_hits = 0;
            for job in jobs {
                let summary = evaluate(&job.input)?;
                if summary.episodes != self.condition.starts || !summary.mean_return.is_finite() {
                    return Err("invalid training evaluation or episode accounting".into());
                }
                fitness_calls += 1;
                rollouts += summary.episodes;
                control_steps += summary.control_steps;
                batch_mean += summary.mean_return / self.protocol.population as f64;
                batch_worst = batch_worst.min(summary.worst_return);
                batch_hits += summary.horizon_hits;
                if selected
                    .as_ref()
                    .is_none_or(|best| summary.mean_return > best.summary.mean_return)
                {
                    selected = Some(Selected {
                        input: job.input.clone(),
                        id: job.id,
                        batch,
                        summary,
                    });
                }
                let output = match self.condition.method {
                    Method::Cma => -summary.mean_return,
                    Method::Neat(_) => summary.mean_return,
                };
                evaluated.push(Evaluated { job, output });
            }
            search.select(evaluated)?;
            let best = selected.as_ref().ok_or("empty training batch")?;
            diagnostics = diagnose(search, &best.input, batch, &mut self.output.species)?;
            let d = &diagnostics;
            let s = best.summary;
            writeln!(
                self.output.training,
                "{},{},{},{batch},{fitness_calls},{rollouts},{control_steps},{batch_mean},{batch_worst},{batch_hits},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                self.protocol.name(),
                self.condition.id,
                self.seed,
                self.protocol.population * self.condition.starts,
                s.mean_return,
                optional(s.sample_std_dev),
                s.worst_return,
                s.horizon_hits,
                best.id,
                best.batch,
                optional(d.best_hidden),
                optional(d.best_enabled),
                optional(d.best_connections),
                optional(d.population_hidden),
                optional(d.population_enabled),
                optional(d.population_connections),
                optional(d.species)
            )?;
        }
        if rollouts != self.protocol.training_budget
            || fitness_calls * self.condition.starts != rollouts
        {
            return Err("training budget accounting mismatch".into());
        }
        self.output.training.flush()?;
        self.output.species.flush()?;
        Ok(Trained {
            selected: selected.ok_or("no selected policy")?,
            fitness_calls,
            rollouts,
            control_steps,
            diagnostics,
        })
    }

    fn heldout<F>(
        &mut self,
        suite_name: &str,
        starts: &[DoublePoleState],
        mut episode: F,
    ) -> Result<TrialSummary>
    where
        F: FnMut(DoublePoleState) -> Result<EpisodeSummary>,
    {
        let mut records = Vec::with_capacity(starts.len());
        for (index, &start) in starts.iter().enumerate() {
            let summary = episode(start)?;
            writeln!(
                self.output.trials,
                "{},{},{},{suite_name},{index},{},{},{},{},{}",
                self.protocol.name(),
                self.condition.id,
                self.seed,
                summary.total_reward,
                summary.steps,
                summary.terminated,
                summary.truncated,
                summary.horizon_hit()
            )?;
            records.push(summary);
        }
        // Summarize already evaluated episodes; this does not call any policy again.
        Ok(evaluate_trials(&records, |&record| {
            Ok::<_, RolloutError<Infallible>>(record)
        })?)
    }

    fn report<I>(
        &mut self,
        trained: &Trained<I>,
        heldout: TrialSummary,
        ablation: Option<TrialSummary>,
        seconds: f64,
    ) -> Result<()> {
        let best = &trained.selected;
        let train = best.summary;
        let d = &trained.diagnostics;
        writeln!(
            self.output.runs,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{seconds}",
            self.protocol.name(),
            self.condition.id,
            self.seed,
            self.condition.starts,
            self.protocol.population,
            self.protocol.horizon,
            self.protocol.training_budget,
            self.protocol.batches(self.condition.starts)?,
            trained.fitness_calls,
            trained.rollouts,
            trained.control_steps,
            best.id,
            best.batch,
            train.mean_return,
            optional(train.sample_std_dev),
            train.worst_return,
            train.horizon_hits,
            train.episodes,
            heldout.mean_return,
            optional(heldout.sample_std_dev),
            heldout.worst_return,
            heldout.horizon_hits,
            heldout.episodes,
            heldout.control_steps,
            optional(ablation.map(|s| s.mean_return)),
            optional(ablation.and_then(|s| s.sample_std_dev)),
            optional(ablation.map(|s| s.worst_return)),
            optional(ablation.map(|s| s.horizon_hits)),
            ablation.map_or(0, |s| s.episodes),
            ablation.map_or(0, |s| s.control_steps),
            optional(d.best_hidden),
            optional(d.best_enabled),
            optional(d.best_connections),
            optional(d.species),
            self.trace_episodes,
            self.trace_steps
        )?;
        self.output.flush()?;
        println!(
            "{} seed {}: train {:.3}, held-out {:.3}, successes {}/{}, {} training episodes, {:.2}s",
            self.condition.id,
            self.seed,
            train.mean_return,
            heldout.mean_return,
            heldout.horizon_hits,
            heldout.episodes,
            trained.rollouts,
            seconds
        );
        Ok(())
    }
}

fn save_weights(
    directory: &Path,
    condition: Condition,
    selected: &Selected<Vec<f64>>,
) -> Result<()> {
    let mut weights = csv(
        &directory.join(format!("{}.weights.csv", condition.id)),
        "index,value",
    )?;
    for (index, value) in selected.input.iter().enumerate() {
        writeln!(weights, "{index},{value}")?;
    }
    weights.flush()?;
    fs::write(
        directory.join(format!("{}.metadata.txt", condition.id)),
        format!(
            "mode=FeedForward\nencoding=Mlp\nwidths=6,4,1\nactivations=Tanh,Tanh\nselected_job_id={}\nselected_batch={}\n",
            selected.id, selected.batch
        ),
    )?;
    Ok(())
}

fn save_genome(directory: &Path, condition: Condition, selected: &Selected<Genome>) -> Result<()> {
    let mut nodes = csv(
        &directory.join(format!("{}.nodes.csv", condition.id)),
        "id,kind",
    )?;
    for node in selected.input.nodes() {
        writeln!(nodes, "{},{:?}", node.id, node.kind)?;
    }
    nodes.flush()?;
    let mut connections = csv(
        &directory.join(format!("{}.connections.csv", condition.id)),
        "innovation,from,to,weight,enabled",
    )?;
    for gene in selected.input.connections() {
        writeln!(
            connections,
            "{},{},{},{},{}",
            gene.innovation, gene.from, gene.to, gene.weight, gene.enabled
        )?;
    }
    connections.flush()?;
    fs::write(
        directory.join(format!("{}.metadata.txt", condition.id)),
        format!(
            "mode={:?}\nencoding=NeatGenome\ninputs={}\noutputs=1\nactivation=Sigmoid4.9\naction=2*output-1\nselected_job_id={}\nselected_batch={}\n",
            selected.input.mode(),
            condition.observation.dimensions(),
            selected.id,
            selected.batch
        ),
    )?;
    Ok(())
}

fn run_seed(root: &Path, protocol: Protocol, seed: u64) -> Result<()> {
    let (training, heldout) = suites(protocol)?;
    fs::create_dir_all(root)?;
    let directory = root.join(format!("seed_{seed:03}"));
    fs::create_dir(&directory)?;
    let policies = directory.join("policies");
    fs::create_dir(&policies)?;
    write_setup(&directory, protocol, seed, &training, &heldout)?;
    let mut output = Output::new(&directory)?;
    for condition in CONDITIONS {
        let started = Instant::now();
        let config = physics(protocol, condition);
        let starts = training[..condition.starts].to_vec();
        let mut run = Run {
            protocol,
            condition,
            seed,
            output: &mut output,
            trace_episodes: 0,
            trace_steps: 0,
        };
        match condition.method {
            Method::Cma => {
                let task = DoublePoleTask::new(config, starts, 4)?;
                let mut search = CmaEs::new(cma_config(protocol, seed))?;
                let trained = run.train(
                    &mut search,
                    |input| Ok(task.summary(input)?),
                    |_, _, _, _| Ok(Diagnostics::default()),
                )?;
                if search.best().map(|best| best.job.id) != Some(trained.selected.id) {
                    return Err("CMA champion selection mismatch".into());
                }
                save_weights(&policies, condition, &trained.selected)?;
                let report = run.heldout("heldout", &heldout, |start| {
                    Ok(task.episode(&trained.selected.input, start)?)
                })?;
                if seed == 0 {
                    run.trace(
                        "heldout",
                        &task.rollout(&trained.selected.input, heldout[0])?,
                    )?;
                }
                run.report(&trained, report, None, started.elapsed().as_secs_f64())?;
            }
            Method::Neat(mode) => {
                let task = NeatDoublePoleTask::new(config.clone(), starts)?;
                let mut search = Neat::new(neat_config(protocol, seed, condition))?;
                let trained = run.train(
                    &mut search,
                    |input| Ok(task.summary(input)?),
                    |search, best, batch, species| {
                        for entry in search.species() {
                            writeln!(
                                species,
                                "{},{},{seed},{batch},{},{},{},{},{},{},{},{}",
                                protocol.name(),
                                condition.id,
                                entry.id,
                                entry.size,
                                entry.best_fitness,
                                entry.best_ever_fitness,
                                entry.stagnant_generations,
                                entry.shared_fitness,
                                entry.offspring,
                                entry.protected
                            )?;
                        }
                        let n = search.parents().len() as f64;
                        Ok(Diagnostics {
                            best_hidden: Some(best.hidden_node_count()),
                            best_enabled: Some(best.enabled_connection_count()),
                            best_connections: Some(best.connections().len()),
                            population_hidden: Some(
                                search
                                    .parents()
                                    .iter()
                                    .map(|p| p.job.input.hidden_node_count() as f64)
                                    .sum::<f64>()
                                    / n,
                            ),
                            population_enabled: Some(
                                search
                                    .parents()
                                    .iter()
                                    .map(|p| p.job.input.enabled_connection_count() as f64)
                                    .sum::<f64>()
                                    / n,
                            ),
                            population_connections: Some(
                                search
                                    .parents()
                                    .iter()
                                    .map(|p| p.job.input.connections().len() as f64)
                                    .sum::<f64>()
                                    / n,
                            ),
                            species: Some(search.species().len()),
                        })
                    },
                )?;
                if search.best().map(|best| best.job.id) != Some(trained.selected.id) {
                    return Err("NEAT champion selection mismatch".into());
                }
                save_genome(&policies, condition, &trained.selected)?;
                let report = run.heldout("heldout", &heldout, |start| {
                    Ok(task.episode(&trained.selected.input, start)?)
                })?;
                if seed == 0 {
                    run.trace(
                        "heldout",
                        &task.rollout(&trained.selected.input, heldout[0])?,
                    )?;
                }
                let ablation = if mode == NetworkMode::Recurrent {
                    let mut network = trained.selected.input.compile_recurrent()?;
                    let mut env = DoublePole::new(config)?;
                    let summary = run.heldout("memory_reset", &heldout, |start| {
                        env.reset_to(start)?;
                        network.reset();
                        Ok(env.run_episode(|observation| {
                            network.reset();
                            let output = network.step(observation)?;
                            Ok::<_, nevo::neat::NeatError>(vec![2.0 * output[0] - 1.0])
                        })?)
                    })?;
                    if seed == 0 {
                        env.reset_to(heldout[0])?;
                        network.reset();
                        let trace = env.rollout(|observation| {
                            network.reset();
                            let output = network.step(observation)?;
                            Ok::<_, nevo::neat::NeatError>(vec![2.0 * output[0] - 1.0])
                        })?;
                        run.trace("memory_reset", &trace)?;
                    }
                    Some(summary)
                } else {
                    None
                };
                run.report(&trained, report, ablation, started.elapsed().as_secs_f64())?;
            }
        }
    }
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let mut directory = None;
    let mut smoke = false;
    let mut seed = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => smoke = true,
            "--seed" => {
                seed = Some(
                    args.next()
                        .ok_or("--seed requires an integer")?
                        .parse::<u64>()?,
                )
            }
            "--out" => {
                directory = Some(PathBuf::from(
                    args.next().ok_or("--out requires a directory")?,
                ))
            }
            "--help" | "-h" => {
                println!(
                    "ch03_generalization [OUTPUT] [--out OUTPUT] [--seed 0..9] [--smoke]\nFull default: all ten seeds, six conditions. Existing seed directories are rejected."
                );
                return Ok(());
            }
            _ if !arg.starts_with('-') && directory.is_none() => {
                directory = Some(PathBuf::from(arg))
            }
            _ => return Err(format!("unknown or duplicate argument: {arg}").into()),
        }
    }
    let protocol = Protocol::new(smoke);
    let directory = directory.unwrap_or_else(|| {
        PathBuf::from(if smoke {
            "results/ch03_generalization_smoke"
        } else {
            "results/ch03_generalization"
        })
    });
    let seeds: Vec<_> = match seed {
        Some(seed) if seed <= 9 => vec![seed],
        Some(_) => return Err("optimizer seed must be in 0..=9 for this frozen protocol".into()),
        None if smoke => vec![0],
        None => (0..10).collect(),
    };
    for seed in seeds {
        run_seed(&directory, protocol, seed)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_budgets_include_initialization_and_complete_batches() {
        let protocol = Protocol::new(false);
        assert_eq!(protocol.batches(1).unwrap(), 512);
        assert_eq!(protocol.batches(16).unwrap(), 32);
        for smoke in [false, true] {
            let protocol = Protocol::new(smoke);
            for condition in CONDITIONS {
                assert_eq!(
                    protocol.batches(condition.starts).unwrap()
                        * protocol.population
                        * condition.starts,
                    protocol.training_budget
                );
            }
        }
        assert!(
            Protocol {
                training_budget: 100,
                ..protocol
            }
            .batches(16)
            .is_err()
        );
        assert!(protocol.batches(0).is_err());
    }

    #[test]
    fn suites_are_fixed_disjoint_and_independent_of_optimizer_seed() {
        let protocol = Protocol::new(false);
        let (train, heldout) = suites(protocol).unwrap();
        assert_eq!(train.len(), 16);
        assert_eq!(heldout.len(), 100);
        assert_eq!(train[0], DoublePoleState::default());
        assert_eq!(suites(protocol).unwrap(), (train.clone(), heldout.clone()));
        assert!(heldout.iter().all(|state| !train.contains(state)));
        assert!(
            train
                .iter()
                .chain(&heldout)
                .all(|state| state.position.abs() <= 0.1
                    && state.velocity.abs() <= 0.02
                    && state.angles[0].abs() <= 0.07
                    && state.angles[1].abs() <= 0.005
                    && state.angular_velocities[0].abs() <= 0.02
                    && state.angular_velocities[1].abs() <= 0.05)
        );
    }
}
