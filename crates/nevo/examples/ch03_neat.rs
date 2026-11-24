//! NEAT evolves XOR topology from direct input-to-output connections.
//! Run: cargo run -p nevo --example ch03_neat --release -- book/results/ch03

use nevo::algorithms::{Neat, NeatConfig};
use nevo::neat::NodeKind;
use nevo::step;
use nevo::tasks::NeatXor;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: ch03_neat [output-directory]".into());
    }
    let directory = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("results/ch03"));
    fs::create_dir_all(&directory)?;
    let mut curves = BufWriter::new(File::create(directory.join("neat_xor_curves.csv"))?);
    let mut runs = BufWriter::new(File::create(directory.join("neat_xor_runs.csv"))?);
    let mut predictions = BufWriter::new(File::create(directory.join("neat_xor_predictions.csv"))?);
    let mut nodes = BufWriter::new(File::create(directory.join("neat_xor_nodes.csv"))?);
    let mut edges = BufWriter::new(File::create(directory.join("neat_xor_connections.csv"))?);
    writeln!(
        curves,
        "seed,batch,evaluations,best_fitness,mse,hidden_nodes,enabled_connections,species"
    )?;
    writeln!(
        runs,
        "seed,evaluations,solved,fitness,mse,hidden_nodes,enabled_connections"
    )?;
    writeln!(predictions, "seed,x1,x2,target,prediction")?;
    writeln!(nodes, "seed,id,kind")?;
    writeln!(edges, "seed,innovation,from,to,weight,enabled")?;
    let targets = [0.0, 1.0, 1.0, 0.0];
    let mut successes = 0;
    for seed in 0..10 {
        let mut search = Neat::new(NeatConfig::new(2, 1, seed))?;
        let mut task = NeatXor;
        let mut solved = false;
        for batch in 1..=300 {
            step(&mut search, &mut task)?;
            let best = search.best().ok_or("missing evaluated genome")?;
            let output = task.predictions(&best.job.input)?;
            solved = output
                .iter()
                .zip(targets)
                .all(|(a, b)| (a - b).abs() <= 0.1);
            writeln!(
                curves,
                "{seed},{batch},{},{:.17e},{:.17e},{},{},{}",
                search.evaluations(),
                best.output,
                task.mse(&best.job.input)?,
                best.job.input.hidden_node_count(),
                best.job.input.enabled_connection_count(),
                search.species().len()
            )?;
            if solved {
                break;
            }
        }
        successes += usize::from(solved);
        let best = search.best().ok_or("missing evaluated genome")?;
        let mse = task.mse(&best.job.input)?;
        let output = task.predictions(&best.job.input)?;
        writeln!(
            runs,
            "{seed},{},{solved},{:.17e},{mse:.17e},{},{}",
            search.evaluations(),
            best.output,
            best.job.input.hidden_node_count(),
            best.job.input.enabled_connection_count()
        )?;
        for (index, (&prediction, target)) in output.iter().zip(targets).enumerate() {
            writeln!(
                predictions,
                "{seed},{},{},{target},{prediction:.17e}",
                index / 2,
                index % 2
            )?;
        }
        for node in best.job.input.nodes() {
            let kind = match node.kind {
                NodeKind::Input => "input",
                NodeKind::Bias => "bias",
                NodeKind::Hidden => "hidden",
                NodeKind::Output => "output",
            };
            writeln!(nodes, "{seed},{},{kind}", node.id)?;
        }
        for gene in best.job.input.connections() {
            writeln!(
                edges,
                "{seed},{},{},{},{:.17e},{}",
                gene.innovation, gene.from, gene.to, gene.weight, gene.enabled
            )?;
        }
        println!(
            "NEAT XOR seed={seed}: solved={solved}, evaluations={}, MSE={mse:.6e}, hidden={}, enabled_edges={}",
            search.evaluations(),
            best.job.input.hidden_node_count(),
            best.job.input.enabled_connection_count()
        );
    }
    for writer in [
        &mut curves,
        &mut runs,
        &mut predictions,
        &mut nodes,
        &mut edges,
    ] {
        writer.flush()?;
    }
    fs::write(
        directory.join("neat_manifest.json"),
        r#"{
  "task": "XOR, all four truth-table rows; no held-out data",
  "seeds": [0,1,2,3,4,5,6,7,8,9],
  "population": 150,
  "batch_cap_including_initialization": 300,
  "stop": "Every prediction of the best-fitness observed genome is within 0.1 of its target",
  "fitness": "(4 - sum_absolute_error)^2, maximize",
  "network": "Feedforward DAG, steepened sigmoid 4.9, bias 1; no hidden nodes initially",
  "config": "NeatConfig::new(2,1,seed), all other defaults",
  "reported_genome": "Best actually evaluated fitness; earliest evaluation wins ties",
  "reporting_forward_passes": "Additional prediction/MSE passes are outside the fitness-evaluation count",
  "limitations": "Teaching-scale experiment; not a replication of the 2002 paper's budgets or success definition"
}
"#,
    )?;
    println!("Solved {successes}/10 declared seeds within the 45,000-evaluation cap per run.");
    Ok(())
}
