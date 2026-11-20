//! Original Chapter 1 companion: evolve a three-gene neural controller.
//!
//! Run `cargo run -p nevo --example ch01_introduction --locked -- results/ch01`.
//! The optional output directory receives curves and trajectories as CSV.

use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use nevo::algorithms::{ElitistConfig, ElitistSearch};
use nevo::step;
use nevo::tasks::TargetReach;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() > 1 {
        return Err("usage: ch01_introduction [output-directory]".into());
    }
    let output_dir = arguments.first().map(Path::new);
    let mut task = TargetReach::default();
    let mut search = ElitistSearch::new(ElitistConfig::new(TargetReach::PARAMETERS, 42))?;
    let baseline = task.value(&[0.0; 3])?;
    let mut history = Vec::new();

    println!("nevo Chapter 1 companion: Generate -> Evaluate -> Select");
    println!("Original teaching task; not an official book exercise.");
    println!("seed=42, genes=3, offspring=32, batches=100 (initial batch included)");
    println!("Objective: mean squared target error + 0.01 * action^2; minimize.");
    println!("No-action baseline: {baseline:.9}");
    println!("batch,evaluations,best_training_loss");

    for _ in 0..100 {
        // The same public execution path is available to any downstream task.
        step(&mut search, &mut task)?;
        let best = search
            .best()
            .ok_or("selection did not retain a candidate")?;
        history.push((search.batches(), search.evaluations(), best.output));
        if search.batches() == 1 || search.batches() % 10 == 0 {
            println!(
                "{},{},{:.9}",
                search.batches(),
                search.evaluations(),
                best.output
            );
        }
    }

    let best = search.best().ok_or("search has no selected controller")?;
    let held_out = TargetReach::new(vec![-1.25, -0.75, 0.75, 1.25], task.steps())?;
    let held_out_loss = held_out.value(&best.job.input)?;
    println!("Weights [error, velocity, bias]: {:?}", best.job.input);
    println!("Final training loss: {:.9}", best.output);
    println!("Held-out-start loss: {held_out_loss:.9}");
    println!(
        "Training search evaluations: {} (4 episodes of 80 steps per evaluation)",
        search.evaluations()
    );

    if let Some(dir) = output_dir {
        fs::create_dir_all(dir)?;
        let mut curve = BufWriter::new(File::create(dir.join("learning.csv"))?);
        writeln!(curve, "batch,evaluations,best_training_loss")?;
        for (batch, evaluations, loss) in history {
            writeln!(curve, "{batch},{evaluations},{loss:.17}")?;
        }
        curve.flush()?;

        let mut trajectories = BufWriter::new(File::create(dir.join("trajectories.csv"))?);
        writeln!(
            trajectories,
            "controller,start,step,position,velocity,action"
        )?;
        for start in [-1.25, 1.25] {
            for (label, weights) in [
                ("zero", &[0.0; 3][..]),
                ("evolved", best.job.input.as_slice()),
            ] {
                for state in task.rollout(weights, start)?.states {
                    writeln!(
                        trajectories,
                        "{label},{start},{},{:.17},{:.17},{:.17}",
                        state.step, state.position, state.velocity, state.action
                    )?;
                }
            }
        }
        trajectories.flush()?;
        println!(
            "Saved learning.csv and trajectories.csv to {}",
            dir.display()
        );
    }
    Ok(())
}
