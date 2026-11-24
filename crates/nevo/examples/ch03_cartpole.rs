//! The same fixed-topology controller, evolved by GA and self-adaptive ES.

use nevo::algorithms::{Es, EsConfig, Ga, GaConfig};
use nevo::control::{CartPole, CartPoleConfig, CartPoleState};
use nevo::tasks::{CartPoleTask, ControlTaskError};
use nevo::{Algorithm, Evaluator, step};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

struct Observed {
    task: CartPoleTask,
    best: Option<(Vec<f64>, f64)>,
    evaluations: usize,
}

impl Evaluator<Vec<f64>> for Observed {
    type Output = f64;
    type Error = ControlTaskError;
    fn evaluate(&mut self, input: &Vec<f64>, seed: u64) -> Result<f64, Self::Error> {
        let loss = self.task.evaluate(input, seed)?;
        self.evaluations += 1;
        if self.best.as_ref().is_none_or(|(_, old)| loss < *old) {
            self.best = Some((input.clone(), loss));
        }
        Ok(loss)
    }
}

fn starts(
    seeds: std::ops::Range<u64>,
    config: &CartPoleConfig,
) -> Result<Vec<CartPoleState>, Box<dyn Error>> {
    let mut env = CartPole::new(config.clone())?;
    Ok(seeds
        .map(|seed| {
            env.reset(seed);
            *env.state()
        })
        .collect())
}

fn train<A>(
    mut search: A,
    task: CartPoleTask,
    name: &str,
    seed: u64,
    curves: &mut impl Write,
) -> Result<Observed, Box<dyn Error>>
where
    A: Algorithm<Input = Vec<f64>, Output = f64>,
{
    let target = task.config().max_steps as f64;
    let mut observed = Observed {
        task,
        best: None,
        evaluations: 0,
    };
    for batch in 1..=60 {
        step(&mut search, &mut observed)?;
        let score = -observed.best.as_ref().ok_or("missing control policy")?.1;
        writeln!(
            curves,
            "{name},{seed},{batch},{},{score:.17e}",
            observed.evaluations
        )?;
        if score >= target - 1e-9 {
            break;
        }
    }
    Ok(observed)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: ch03_cartpole [output-directory]".into());
    }
    let directory = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("results/ch03"));
    fs::create_dir_all(&directory)?;
    let config = CartPoleConfig {
        max_steps: 500,
        ..CartPoleConfig::default()
    };
    let training = starts(100..103, &config)?;
    let heldout = starts(200..220, &config)?;
    let task = CartPoleTask::new(config, training, 4)?;
    let mut curves = BufWriter::new(File::create(directory.join("cartpole_curves.csv"))?);
    let mut runs = BufWriter::new(File::create(directory.join("cartpole_runs.csv"))?);
    let mut trials = BufWriter::new(File::create(directory.join("cartpole_trials.csv"))?);
    let mut traces = BufWriter::new(File::create(directory.join("cartpole_trajectories.csv"))?);
    writeln!(
        curves,
        "algorithm,seed,batch,evaluations,best_training_return"
    )?;
    writeln!(
        runs,
        "algorithm,seed,evaluations,training_mean,heldout_mean,heldout_full_horizon"
    )?;
    writeln!(
        trials,
        "algorithm,seed,partition,start_seed,return,steps,terminated,truncated"
    )?;
    writeln!(
        traces,
        "algorithm,seed,start_seed,step,position,velocity,angle,angular_velocity,action"
    )?;
    for seed in 0..3 {
        let mut ga = GaConfig::new(task.dimensions(), seed);
        ga.initial_bound = 0.5;
        ga.mutation_sigma = 0.2;
        let mut es = EsConfig::new(task.dimensions(), seed);
        es.initial_bound = 0.5;
        es.initial_sigma = 0.2;
        let searches = [
            (
                "ga",
                train(Ga::new(ga)?, task.clone(), "ga", seed, &mut curves)?,
            ),
            (
                "es",
                train(Es::new(es)?, task.clone(), "es", seed, &mut curves)?,
            ),
        ];
        for (name, search) in searches {
            let (genes, loss) = search.best.ok_or("missing control policy")?;
            let mut total = 0.0;
            let mut survived = 0;
            for (partition, first, states) in [
                ("training", 100_u64, task.starts()),
                ("heldout", 200_u64, heldout.as_slice()),
            ] {
                for (index, &start) in states.iter().enumerate() {
                    let start_seed = first + index as u64;
                    let episode = task.rollout(&genes, start)?;
                    writeln!(
                        trials,
                        "{name},{seed},{partition},{start_seed},{:.17e},{},{},{}",
                        episode.total_reward, episode.steps, episode.terminated, episode.truncated
                    )?;
                    if partition == "heldout" {
                        total += episode.total_reward;
                        survived += usize::from(!episode.terminated && episode.truncated);
                    }
                    if start_seed == 200 {
                        for (step, state) in episode.states.iter().enumerate() {
                            let action = episode
                                .actions
                                .get(step)
                                .map(|a| format!("{:.17e}", a[0]))
                                .unwrap_or_default();
                            writeln!(
                                traces,
                                "{name},{seed},{start_seed},{step},{:.17e},{:.17e},{:.17e},{:.17e},{action}",
                                state.position, state.velocity, state.angle, state.angular_velocity
                            )?;
                        }
                    }
                }
            }
            let mean = total / heldout.len() as f64;
            writeln!(
                runs,
                "{name},{seed},{},{:.17e},{mean:.17e},{survived}",
                search.evaluations, -loss
            )?;
            let mut parameters = BufWriter::new(File::create(
                directory.join(format!("cartpole_{name}_{seed}_parameters.txt")),
            )?);
            for weight in &genes {
                writeln!(parameters, "{weight:.17e}")?;
            }
            parameters.flush()?;
            println!(
                "CartPole {name} seed={seed}: fitness calls={}, training_mean={:.2}, heldout_mean={mean:.2}, full-horizon={survived}/20",
                search.evaluations, -loss
            );
        }
    }
    for writer in [&mut curves, &mut runs, &mut trials, &mut traces] {
        writer.flush()?;
    }
    fs::write(
        directory.join("cartpole_manifest.json"),
        r#"{
  "environment":"Native Rust CartPole with classic Gym equations and discrete +/-10 N actions",
  "horizon":500,"dt":0.02,"network":[4,4,1],"activations":["tanh","tanh"],"parameters":25,
  "training_start_seeds":[100,101,102],"heldout_start_seeds":"200..219",
  "reset":"Independent ChaCha8 stream3, each of four coordinates uniform [-0.05,0.05)",
  "search_seeds":[0,1,2],"population":64,"batch_cap_including_initialization":60,
  "stop":"Best observed mean training return reaches 500",
  "fitness":"Negative mean complete episode return over the same three training starts; minimize",
  "reward":"+1 per action including failure; termination and time-limit truncation reported separately",
  "ga":{"initial_bound":0.5,"mutation_sigma":0.2,"other_settings":"GaConfig defaults"},
  "es":{"initial_bound":0.5,"initial_sigma":0.2,"other_settings":"EsConfig defaults"},
  "holdout":"Evaluate final training-selected policy once per held-out start; no holdout tuning",
  "limitations":"Teaching experiments, not the standard Gym evaluation protocol; native RNG is not Gym's NumPy RNG. Fitness calls each contain three episodes; reporting rollouts are additional."
}
"#,
    )?;
    Ok(())
}
