//! Genuine 8x8 Optdigits classification with explicit disjoint partitions.
//! Run: cargo run -p nevo --example ch03_classification --release -- book/results/ch03

use nevo::algorithms::{Es, EsConfig, Ga, GaConfig, Neat, NeatConfig};
use nevo::tasks::{
    ClassificationData, ClassificationError, ClassificationMetrics, MlpClassifier, NeatClassifier,
};
use nevo::{Algorithm, Evaluator, step};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const BATCHES: usize = 80;
const POPULATION: usize = 64;
const DIGITS: &str = include_str!("../../../data/optdigits.csv");

struct Observed {
    task: MlpClassifier,
    best: Option<(Vec<f64>, f64)>,
    evaluations: usize,
}

impl Evaluator<Vec<f64>> for Observed {
    type Output = f64;
    type Error = ClassificationError;
    fn evaluate(&mut self, genes: &Vec<f64>, seed: u64) -> Result<f64, Self::Error> {
        let loss = self.task.evaluate(genes, seed)?;
        self.evaluations += 1;
        if self.best.as_ref().is_none_or(|(_, old)| loss < *old) {
            self.best = Some((genes.clone(), loss));
        }
        Ok(loss)
    }
}

fn save_metrics(
    runs: &mut impl Write,
    name: &str,
    seed: u64,
    evaluations: usize,
    partition: &str,
    metrics: ClassificationMetrics,
) -> Result<(), Box<dyn Error>> {
    writeln!(
        runs,
        "{name},{seed},{evaluations},{partition},{},{:.17e},{:.17e}",
        metrics.samples, metrics.mse, metrics.accuracy
    )?;
    println!(
        "digits {name:5} seed={seed} {partition:10}: MSE={:.6}, accuracy={:.2}% ({} samples)",
        metrics.mse,
        100.0 * metrics.accuracy,
        metrics.samples
    );
    Ok(())
}

fn run_fixed<A>(
    name: &str,
    seed: u64,
    mut search: A,
    tasks: &[MlpClassifier; 3],
    curves: &mut impl Write,
    runs: &mut impl Write,
    directory: &Path,
) -> Result<(), Box<dyn Error>>
where
    A: Algorithm<Input = Vec<f64>, Output = f64>,
{
    let mut evaluator = Observed {
        task: tasks[0].clone(),
        best: None,
        evaluations: 0,
    };
    for batch in 1..=BATCHES {
        let count = step(&mut search, &mut evaluator)?;
        if count != POPULATION {
            return Err("unexpected experiment population".into());
        }
        writeln!(
            curves,
            "{name},{seed},{batch},{},{:.17e}",
            evaluator.evaluations,
            evaluator.best.as_ref().ok_or("missing classifier")?.1
        )?;
    }
    let (genes, _) = evaluator.best.ok_or("missing classifier")?;
    // Only training loss selects a candidate. Holdouts are inspected after search.
    for (partition, task) in ["training", "validation", "test"].into_iter().zip(tasks) {
        save_metrics(
            runs,
            name,
            seed,
            evaluator.evaluations,
            partition,
            task.metrics(&genes)?,
        )?;
    }
    let mut parameters = BufWriter::new(File::create(
        directory.join(format!("digits_{name}_{seed}_parameters.txt")),
    )?);
    for weight in &genes {
        writeln!(parameters, "{weight:.17e}")?;
    }
    parameters.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: ch03_classification [output-directory]".into());
    }
    let directory = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("results/ch03"));
    fs::create_dir_all(&directory)?;
    let data = ClassificationData::from_optdigits_csv(DIGITS)?;
    let split = data.stratified_split(60, 30, 2026)?;
    let datasets = [split.training, split.validation, split.test];
    let tasks = [
        MlpClassifier::new(datasets[0].clone(), &[8])?,
        MlpClassifier::new(datasets[1].clone(), &[8])?,
        MlpClassifier::new(datasets[2].clone(), &[8])?,
    ];
    let mut partitions = BufWriter::new(File::create(directory.join("digits_split.csv"))?);
    for (name, partition) in ["training", "validation", "test"]
        .into_iter()
        .zip(&datasets)
    {
        if name == "training" {
            writeln!(partitions, "sample_id,label,partition")?;
        }
        for sample in partition.samples() {
            writeln!(partitions, "{},{},{name}", sample.id, sample.label)?;
        }
    }
    partitions.flush()?;
    let mut curves = BufWriter::new(File::create(directory.join("digits_curves.csv"))?);
    let mut runs = BufWriter::new(File::create(directory.join("digits_runs.csv"))?);
    let mut graphs = BufWriter::new(File::create(directory.join("digits_neat_genomes.csv"))?);
    writeln!(curves, "algorithm,seed,batch,evaluations,best_training_mse")?;
    writeln!(
        runs,
        "algorithm,seed,evaluations,partition,samples,mse,accuracy"
    )?;
    writeln!(graphs, "seed,record,id,kind,from,to,weight,enabled")?;
    println!("Optdigits, not MNIST: 1,797 genuine 8x8 images; split 600/300/897, split seed 2026.");
    println!("GA/ES: 64->8 tanh->10 sigmoid (610 parameters). NEAT starts 64->10 (650 edges).");
    println!(
        "Each run: 80 batches including initialization x 64 candidates = 5,120 training fitness calls."
    );
    for seed in 0..3 {
        let mut ga = GaConfig::new(tasks[0].dimensions(), seed);
        ga.initial_bound = 0.5;
        ga.mutation_sigma = 0.2;
        run_fixed(
            "ga",
            seed,
            Ga::new(ga)?,
            &tasks,
            &mut curves,
            &mut runs,
            &directory,
        )?;
        let mut es = EsConfig::new(tasks[0].dimensions(), seed);
        es.initial_bound = 0.5;
        es.initial_sigma = 0.1;
        run_fixed(
            "es",
            seed,
            Es::new(es)?,
            &tasks,
            &mut curves,
            &mut runs,
            &directory,
        )?;
        let mut config = NeatConfig::new(64, 10, seed);
        config.population = POPULATION;
        config.mutation.weight_bound = 0.3;
        config.mutation.perturb_bound = 0.1;
        config.mutation.perturb_probability = 0.95;
        let mut search = Neat::new(config)?;
        let mut task = NeatClassifier::new(datasets[0].clone());
        for batch in 1..=BATCHES {
            step(&mut search, &mut task)?;
            writeln!(
                curves,
                "neat,{seed},{batch},{},{:.17e}",
                search.evaluations(),
                1.0 - search.best().ok_or("missing NEAT classifier")?.output
            )?;
        }
        let genome = &search.best().ok_or("missing NEAT classifier")?.job.input;
        for (name, partition) in ["training", "validation", "test"]
            .into_iter()
            .zip(&datasets)
        {
            save_metrics(
                &mut runs,
                "neat",
                seed,
                search.evaluations(),
                name,
                NeatClassifier::new(partition.clone()).metrics(genome)?,
            )?;
        }
        for node in genome.nodes() {
            writeln!(graphs, "{seed},node,{},{:?},,,,", node.id, node.kind)?;
        }
        for gene in genome.connections() {
            writeln!(
                graphs,
                "{seed},connection,{},,{},{},{:.17e},{}",
                gene.innovation, gene.from, gene.to, gene.weight, gene.enabled
            )?;
        }
    }
    curves.flush()?;
    runs.flush()?;
    graphs.flush()?;
    fs::write(
        directory.join("digits_manifest.json"),
        r#"{
  "dataset": "UCI Optdigits optdigits.tes, 1797 images, 64 integer features in 0..16; normalized by 16",
  "source_sha256": "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8",
  "split": {"seed":2026,"method":"per-class Fisher-Yates; 60 training, 30 validation, remainder test","sizes":[600,300,897]},
  "seeds":[0,1,2], "population":64, "batches_including_initialization":80,
  "fitness_evaluations_per_run":5120, "training_examples_per_evaluation":600,
  "loss":"Mean squared error over samples and 10 one-hot outputs",
  "metric":"Argmax accuracy; first class wins exact ties",
  "selection":"Best actually evaluated training score; holdouts inspected only after search",
  "ga":{"widths":[64,8,10],"activations":["tanh","sigmoid"],"initial_bound":0.5,"mutation_sigma":0.2,"other_settings":"GaConfig defaults"},
  "es":{"widths":[64,8,10],"activations":["tanh","sigmoid"],"initial_bound":0.5,"initial_sigma":0.1,"other_settings":"EsConfig defaults"},
  "neat":{"initial_widths":[64,10],"activation":"sigmoid 4.9","fitness":"1-MSE","weight_bound":0.3,"perturb_bound":0.1,"perturb_probability":0.95,"other_settings":"NeatConfig defaults except population64"},
  "limitations":"Small fixed budgets and architectures; no tuned model ranking, writer-independent split, official UCI split, MNIST result, or paper reproduction."
}
"#,
    )?;
    Ok(())
}
