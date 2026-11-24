//! Chapter 1 §1.1 population structure, applied to the original TargetReach task.
//!
//! Run `cargo run -p nevo --example ch01_population --locked -- results/ch01_population`.
//! Numerical settings and per-gene Gaussian mutation are explicit companion choices.

use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use nevo::algorithms::{Ga, GaConfig};
use nevo::step;
use nevo::tasks::TargetReach;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() > 1 {
        return Err("usage: ch01_population [output-directory]".into());
    }
    let output_dir = arguments.first().map(Path::new);
    let mut task = TargetReach::default();
    let mut config = GaConfig::new(TargetReach::PARAMETERS, 42);
    config.population = 100;
    config.elite_count = 20;
    config.parent_pool_size = Some(20);
    config.tournament_size = 1;
    config.crossover_probability = 1.0;
    config.initial_bound = 1.0;
    config.mutation_sigma = 0.2;
    // The default independent probability 1/d can mutate zero or several genes.
    // It is our chosen operator, not an exact one-coordinate mutation rule.
    let mut search = Ga::new(config)?;
    let mut history = Vec::new();

    println!("nevo Chapter 1 population companion: Generate -> Evaluate -> Select");
    println!("100 evaluated candidates; 20 retained elites; 80 offspring of the top 20.");
    println!("Uniform parent sampling and crossover; per-gene Gaussian mutation p=1/3, sigma=0.2.");
    println!("seed=42, batches=32 (initial batch included), search evaluations=3200");
    println!("batch,evaluations,best_training_loss");
    for _ in 0..32 {
        step(&mut search, &mut task)?;
        let best = search
            .best()
            .ok_or("selection did not retain a candidate")?;
        history.push((search.batches(), search.evaluations(), best.output));
        println!(
            "{},{},{:.9}",
            search.batches(),
            search.evaluations(),
            best.output
        );
    }
    let best = search.best().ok_or("search has no selected controller")?;
    let held_out = TargetReach::new(vec![-1.25, -0.75, 0.75, 1.25], task.steps())?;
    println!("Weights [error, velocity, bias]: {:?}", best.job.input);
    println!("Final training loss: {:.9}", best.output);
    println!(
        "Held-out-start loss: {:.9}",
        held_out.value(&best.job.input)?
    );
    println!("Every candidate, including the 20 copied elites, is evaluated each batch.");

    if let Some(dir) = output_dir {
        fs::create_dir_all(dir)?;
        let mut curve = BufWriter::new(File::create(dir.join("learning.csv"))?);
        writeln!(curve, "batch,evaluations,best_training_loss")?;
        for (batch, evaluations, loss) in history {
            writeln!(curve, "{batch},{evaluations},{loss:.17}")?;
        }
        curve.flush()?;
        println!("Saved learning.csv to {}", dir.display());
    }
    Ok(())
}
