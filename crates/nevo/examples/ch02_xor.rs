//! Evolve or differentiate the same XOR network; costs are reported separately.

use nevo::algorithms::{CmaEs, CmaEsConfig};
use nevo::step;
use nevo::tasks::Xor;
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{RngCore, SeedableRng},
};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() > 1 {
        return Err("usage: ch02_xor [output-directory]".into());
    }
    let directory = arguments
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("results/ch02"));
    fs::create_dir_all(&directory)?;
    let mut task = Xor::new(4)?;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    rng.set_stream(2);
    let initial: Vec<_> = (0..task.dimensions())
        .map(|_| (rng.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0 - 0.5)
        .collect();
    let baseline = task.value(&initial)?;
    let mut network = task.network(initial.clone())?;
    let mut config = CmaEsConfig::new(initial, 42);
    config.population = 32;
    let mut search = CmaEs::new(config)?;
    let mut curves = BufWriter::new(File::create(directory.join("xor_curves.csv"))?);
    writeln!(
        curves,
        "method,updates,search_evaluations,full_batch_gradient_steps,mse"
    )?;
    writeln!(curves, "cma_es,0,0,0,{baseline:.17e}")?;
    writeln!(curves, "backprop,0,0,0,{baseline:.17e}")?;
    const TARGET: f64 = 1e-3;
    for batch in 1..=400 {
        step(&mut search, &mut task)?;
        writeln!(
            curves,
            "cma_es,{batch},{},0,{:.17e}",
            search.evaluations(),
            search.best().ok_or("missing evaluated network")?.output
        )?;
        if search.best().ok_or("missing evaluated network")?.output <= TARGET {
            break;
        }
    }
    let mut gradient_steps = 0;
    let mut learned_mse = baseline;
    for update in 1..=5000 {
        network.train_step(task.data(), 1.0)?;
        gradient_steps = update;
        learned_mse = network.mse(task.data())?;
        let loss = learned_mse;
        if update % 10 == 0 || loss <= TARGET {
            writeln!(curves, "backprop,{update},0,{update},{loss:.17e}")?;
        }
        if loss <= TARGET {
            break;
        }
    }
    curves.flush()?;
    let best = search.best().ok_or("missing evaluated network")?;
    let evolved = task.network(best.job.input.clone())?;
    let mut predictions = BufWriter::new(File::create(directory.join("xor_predictions.csv"))?);
    writeln!(predictions, "x1,x2,target,evolved,backprop")?;
    println!("XOR: 2 -> 4 tanh -> 1 sigmoid, 17 parameters, all 4 truth-table rows.");
    println!("seed=42, initial mean weights in [-0.5,0.5), initial MSE={baseline:.9e}");
    println!("Stop at MSE <= {TARGET} or the configured budget cap.");
    println!(
        "CMA-ES: {} search evaluations ({} batches x 32), best MSE={:.9e}",
        search.evaluations(),
        search.batches(),
        best.output
    );
    println!(
        "Backprop: {gradient_steps} full-batch gradient steps, learning_rate=1, MSE={learned_mse:.9e}"
    );
    println!("These are different work units, not an equal-cost speed comparison.");
    for row in task.data() {
        let a = evolved.forward(&row.input)?[0];
        let b = network.forward(&row.input)?[0];
        writeln!(
            predictions,
            "{},{},{},{a:.17e},{b:.17e}",
            row.input[0], row.input[1], row.target[0]
        )?;
        println!(
            "{:?} target={} evolved={a:.8} backprop={b:.8}",
            row.input, row.target[0]
        );
    }
    predictions.flush()?;
    fs::write(
        directory.join("xor_parameters.txt"),
        format!(
            "evolved={:?}\nbackprop={:?}\n",
            best.job.input,
            network.parameters()
        ),
    )?;
    fs::write(
        directory.join("xor_manifest.json"),
        format!(
            r#"{{
  "architecture": [2,4,1],
  "activations": ["tanh","sigmoid"],
  "parameter_order": "each layer, each output neuron: input weights then bias",
  "parameters": 17,
  "seed": 42,
  "initial_mean": "ChaCha8 stream2 uniform [-0.5,0.5), independent of search stream0",
  "loss": "MSE on all four XOR truth-table examples",
  "target_mse": 0.001,
  "cma_es": {{"population":32,"sigma":0.5,"batch_cap":400,"actual_batches":{},"search_evaluations":{}}},
  "backprop": {{"learning_rate":1.0,"step_cap":5000,"actual_steps":{gradient_steps},"reporting_forward_loss_passes":{gradient_steps}}},
  "reporting": "One common baseline loss before search; final prediction forward passes are outside search budget. No held-out set. Unequal compute work."
}}
"#,
            search.batches(),
            search.evaluations()
        ),
    )?;
    Ok(())
}
