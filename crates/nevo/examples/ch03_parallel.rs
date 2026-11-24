//! Seed-encoded deep neural weights and deterministic optional CPU parallelism.
//! Run: cargo run -p nevo --example ch03_parallel --features parallel --release -- book/results/ch03

use nevo::algorithms::{DeepGa, DeepGaConfig};
use nevo::encoding::{SeedGenome, SeededEncoding};
use nevo::execution::step_parallel;
use nevo::nn::{Activation, Mlp, TrainingExample};
use nevo::{Evaluator, step};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const WIDTHS: [usize; 5] = [2, 16, 16, 16, 1];
const ACTIVATIONS: [Activation; 4] = [
    Activation::Tanh,
    Activation::Tanh,
    Activation::Tanh,
    Activation::Sigmoid,
];

struct SeededXor {
    encoding: SeededEncoding,
    data: Arc<[TrainingExample]>,
}

impl Evaluator<SeedGenome> for SeededXor {
    type Output = f64;
    type Error = io::Error;
    fn evaluate(&mut self, genome: &SeedGenome, _seed: u64) -> Result<f64, Self::Error> {
        let weights = self.encoding.decode(genome).map_err(io::Error::other)?;
        let network = Mlp::new(&WIDTHS, &ACTIVATIONS, weights).map_err(io::Error::other)?;
        network.mse(&self.data).map_err(io::Error::other)
    }
}

#[derive(Debug, PartialEq)]
struct ReplayRecord {
    genome: SeedGenome,
    weight_bits: Vec<u64>,
    best_scores: Vec<u64>,
    evaluations: usize,
}

fn run(
    mode: &str,
    pool: Option<&ThreadPool>,
    data: Arc<[TrainingExample]>,
    curves: &mut impl Write,
    runs: &mut impl Write,
    directory: &Path,
) -> Result<ReplayRecord, Box<dyn Error>> {
    let dimensions = Mlp::parameter_count(&WIDTHS)?;
    let mut config = DeepGaConfig::new(dimensions, 42);
    config.initial_bound = 0.5;
    config.mutation_sigma = 0.1;
    let mut search = DeepGa::new(config)?;
    let encoding = *search.encoding();
    let factory = || SeededXor {
        encoding,
        data: Arc::clone(&data),
    };
    let mut serial = factory();
    let started = Instant::now();
    let mut best_scores = Vec::new();
    for batch in 1..=100 {
        if let Some(pool) = pool {
            pool.install(|| step_parallel(&mut search, factory))?;
        } else {
            step(&mut search, &mut serial)?;
        }
        let best = search.best().ok_or("missing evaluated seed lineage")?;
        best_scores.push(best.output.to_bits());
        let mutations = best.job.input.mutation_seeds.len();
        writeln!(
            curves,
            "{mode},{batch},{},{:.17e},{mutations},{},{}",
            search.evaluations(),
            best.output,
            8 * (1 + mutations),
            8 * dimensions
        )?;
        if best.output <= 0.01 {
            break;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let best = search.best().ok_or("missing evaluated seed lineage")?;
    let weights = encoding.decode(&best.job.input)?;
    let mutations = best.job.input.mutation_seeds.len();
    writeln!(
        runs,
        "{mode},{},{},{:.17e},{mutations},{},{},{elapsed:.9}",
        search.evaluations(),
        search.batches(),
        best.output,
        8 * (1 + mutations),
        8 * dimensions
    )?;
    println!(
        "Deep GA {mode}: {} weights, {} evaluations, MSE={:.6e}, {} mutation seeds, payload={} bytes vs {} dense bytes, elapsed={elapsed:.3}s",
        dimensions,
        search.evaluations(),
        best.output,
        mutations,
        8 * (1 + mutations),
        8 * dimensions
    );
    if mode == "serial" {
        let network = Mlp::new(&WIDTHS, &ACTIVATIONS, weights.clone())?;
        let mut predictions =
            BufWriter::new(File::create(directory.join("deep_ga_predictions.csv"))?);
        writeln!(predictions, "x1,x2,target,prediction")?;
        for row in data.iter() {
            writeln!(
                predictions,
                "{},{},{},{:.17e}",
                row.input[0],
                row.input[1],
                row.target[0],
                network.forward(&row.input)?[0]
            )?;
        }
        predictions.flush()?;
        fs::write(
            directory.join("deep_ga_lineage.json"),
            format!(
                "{{\n  \"initial_seed\": {},\n  \"mutation_seeds\": {:?},\n  \"dimensions\": {dimensions},\n  \"initial_bound\": 0.5,\n  \"mutation_sigma\": 0.1,\n  \"rng\": \"ChaCha8, stream0 per seed; nevo SeededEncoding\"\n}}\n",
                best.job.input.initial_seed, best.job.input.mutation_seeds
            ),
        )?;
        let mut parameters =
            BufWriter::new(File::create(directory.join("deep_ga_parameters.txt"))?);
        for weight in &weights {
            writeln!(parameters, "{weight:.17e}")?;
        }
        parameters.flush()?;
    }
    Ok(ReplayRecord {
        genome: best.job.input.clone(),
        weight_bits: weights.iter().map(|w| w.to_bits()).collect(),
        best_scores,
        evaluations: search.evaluations(),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: ch03_parallel [output-directory]".into());
    }
    let directory = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("results/ch03"));
    fs::create_dir_all(&directory)?;
    let data: Arc<[TrainingExample]> = vec![
        TrainingExample {
            input: vec![0., 0.],
            target: vec![0.],
        },
        TrainingExample {
            input: vec![0., 1.],
            target: vec![1.],
        },
        TrainingExample {
            input: vec![1., 0.],
            target: vec![1.],
        },
        TrainingExample {
            input: vec![1., 1.],
            target: vec![0.],
        },
    ]
    .into();
    let mut curves = BufWriter::new(File::create(directory.join("deep_ga_curves.csv"))?);
    let mut runs = BufWriter::new(File::create(directory.join("parallel_runs.csv"))?);
    writeln!(
        curves,
        "mode,batch,evaluations,best_mse,mutations,seed_payload_bytes,dense_payload_bytes"
    )?;
    writeln!(
        runs,
        "mode,evaluations,batches,best_mse,mutations,seed_payload_bytes,dense_payload_bytes,wall_seconds"
    )?;
    let serial = run(
        "serial",
        None,
        Arc::clone(&data),
        &mut curves,
        &mut runs,
        &directory,
    )?;
    for workers in [1, 4] {
        let pool = ThreadPoolBuilder::new().num_threads(workers).build()?;
        let parallel = run(
            &format!("rayon_{workers}"),
            Some(&pool),
            Arc::clone(&data),
            &mut curves,
            &mut runs,
            &directory,
        )?;
        if serial != parallel {
            return Err("serial/parallel replay changed the run".into());
        }
    }
    curves.flush()?;
    runs.flush()?;
    fs::write(
        directory.join("parallel_manifest.json"),
        r#"{
  "task":"XOR, all four truth-table rows; no held-out set",
  "network":[2,16,16,16,1],"activations":["tanh","tanh","tanh","sigmoid"],"parameters":609,
  "algorithm":"Mutation-only DeepGa teaching variant; top-parent truncation and seeded dense Gaussian mutations",
  "seed":42,"population":64,"parents":16,"elite_count":1,"initial_bound":0.5,"mutation_sigma":0.1,
  "batch_cap_including_initialization":100,"stop":"Best observed MSE <= 0.01",
  "execution":["serial","Rayon1worker","Rayon4workers"],
  "checked":"Every batch best-score bits, final seed genome, decoded weight bits, evaluation count match all three runs",
  "payload":"8 bytes per seed versus 8 bytes per f64; excludes Vec metadata, allocation, network architecture/settings, and issued-batch copies",
  "decode_cost":"O(dimensions * lineage_length); no shared prefix cache or noise table",
  "timing":"One local demonstration trial per mode; includes search and curve logging, excludes final decoding/reporting. Not a controlled throughput benchmark.",
  "limitations":"Small deep MLP, not Atari or large-model reproduction; CPU job-level parallelism, not SIMD, GPU, distributed execution, or batched tensors."
}
"#,
    )?;
    println!("Serial, 1-worker, and 4-worker runs reproduced exactly in the checked fields.");
    Ok(())
}
