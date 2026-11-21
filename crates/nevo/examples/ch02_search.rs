//! Recorded Chapter 2 search experiments; all search/evaluation code is Rust.
//! Run: cargo run -p nevo --example ch02_search --release -- book/results/ch02

use nevo::algorithms::{
    CmaEs, CmaEsConfig, Es, EsConfig, Ga, GaConfig, Nsga2, Nsga2Config, Objective, OpenAiEs,
    OpenAiEsConfig,
};
use nevo::benchmarks::{Rastrigin, Sphere, Zdt1};
use nevo::{Algorithm, Evaluator, step};
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{RngCore, SeedableRng},
};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

const DIMENSIONS: usize = 8;
const POPULATION: usize = 64;
const BATCHES: usize = 100;
const SEEDS: u64 = 20;
const TARGET: f64 = 1e-8;

struct Counted<E> {
    inner: E,
    evaluations: usize,
    best: f64,
    first_hit: Option<usize>,
}

impl<E: Evaluator<Vec<f64>, Output = f64>> Evaluator<Vec<f64>> for Counted<E> {
    type Output = f64;
    type Error = E::Error;
    fn evaluate(&mut self, input: &Vec<f64>, seed: u64) -> Result<f64, Self::Error> {
        let value = self.inner.evaluate(input, seed)?;
        self.evaluations += 1;
        self.best = self.best.min(value);
        if value <= TARGET && self.first_hit.is_none() {
            self.first_hit = Some(self.evaluations);
        }
        Ok(value)
    }
}

fn initial_mean(dimensions: usize, seed: u64) -> Vec<f64> {
    // Random center on its own stream, independent of the algorithms' stream 0
    // perturbations and stream 1 evaluation seeds. No objective is evaluated here.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(2);
    (0..dimensions)
        .map(|_| 5.0 * (2.0 * (rng.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0 - 1.0))
        .collect()
}

fn run<A, E>(
    name: &str,
    problem: &str,
    seed: u64,
    mut algorithm: A,
    evaluator: E,
    curves: &mut impl Write,
    runs: &mut impl Write,
) -> Result<f64, Box<dyn Error>>
where
    A: Algorithm<Input = Vec<f64>, Output = f64>,
    E: Evaluator<Vec<f64>, Output = f64>,
{
    let mut counted = Counted {
        inner: evaluator,
        evaluations: 0,
        best: f64::INFINITY,
        first_hit: None,
    };
    let mut initial = 0.0;
    for batch in 1..=BATCHES {
        let count = step(&mut algorithm, &mut counted)?;
        if count != POPULATION {
            return Err("experiment requires exactly 64 evaluations per batch".into());
        }
        if batch == 1 {
            initial = counted.best;
        }
        writeln!(
            curves,
            "{problem},{name},{seed},{batch},{},{:.17e}",
            counted.evaluations, counted.best
        )?;
    }
    let first_hit = counted.first_hit.map(|n| n.to_string()).unwrap_or_default();
    writeln!(
        runs,
        "{problem},{name},{seed},{},{initial:.17e},{:.17e},{first_hit}",
        counted.evaluations, counted.best
    )?;
    Ok(counted.best)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() > 1 {
        return Err("usage: ch02_search [output-directory]".into());
    }
    let directory = arguments
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("results/ch02"));
    fs::create_dir_all(&directory)?;
    let mut curves = BufWriter::new(File::create(directory.join("search_curves.csv"))?);
    let mut runs = BufWriter::new(File::create(directory.join("search_runs.csv"))?);
    writeln!(
        curves,
        "problem,algorithm,seed,batch,evaluations,best_observed"
    )?;
    writeln!(
        runs,
        "problem,algorithm,seed,evaluations,initial_best,final_best,first_target_evaluation"
    )?;
    println!("Chapter 2: 8 dimensions, 64 evaluations/batch, 100 batches, seeds 0..19.");
    println!(
        "Budget includes initialization. Scores are best evaluated candidates, not distribution means."
    );
    println!(
        "GA/ES share the full initial population. CMA/OpenAI use a Gaussian around an independent uniform center."
    );
    for problem in ["sphere", "rastrigin"] {
        let mut scores = vec![Vec::new(); 4];
        for seed in 0..SEEDS {
            let ga = Ga::new(GaConfig::new(DIMENSIONS, seed))?;
            let es = Es::new(EsConfig::new(DIMENSIONS, seed))?;
            let mut cma_config = CmaEsConfig::new(initial_mean(DIMENSIONS, seed), seed);
            cma_config.population = POPULATION;
            let cma = CmaEs::new(cma_config)?;
            let openai = OpenAiEs::new(OpenAiEsConfig::new(initial_mean(DIMENSIONS, seed), seed))?;
            // Separate calls keep error types and algorithm types concrete.
            if problem == "sphere" {
                scores[0].push(run(
                    "ga",
                    problem,
                    seed,
                    ga,
                    Sphere::new(DIMENSIONS)?,
                    &mut curves,
                    &mut runs,
                )?);
                scores[1].push(run(
                    "es",
                    problem,
                    seed,
                    es,
                    Sphere::new(DIMENSIONS)?,
                    &mut curves,
                    &mut runs,
                )?);
                scores[2].push(run(
                    "cma_es",
                    problem,
                    seed,
                    cma,
                    Sphere::new(DIMENSIONS)?,
                    &mut curves,
                    &mut runs,
                )?);
                scores[3].push(run(
                    "openai_es",
                    problem,
                    seed,
                    openai,
                    Sphere::new(DIMENSIONS)?,
                    &mut curves,
                    &mut runs,
                )?);
            } else {
                scores[0].push(run(
                    "ga",
                    problem,
                    seed,
                    ga,
                    Rastrigin::new(DIMENSIONS)?,
                    &mut curves,
                    &mut runs,
                )?);
                scores[1].push(run(
                    "es",
                    problem,
                    seed,
                    es,
                    Rastrigin::new(DIMENSIONS)?,
                    &mut curves,
                    &mut runs,
                )?);
                scores[2].push(run(
                    "cma_es",
                    problem,
                    seed,
                    cma,
                    Rastrigin::new(DIMENSIONS)?,
                    &mut curves,
                    &mut runs,
                )?);
                scores[3].push(run(
                    "openai_es",
                    problem,
                    seed,
                    openai,
                    Rastrigin::new(DIMENSIONS)?,
                    &mut curves,
                    &mut runs,
                )?);
            }
        }
        for (algorithm, values) in ["ga", "es", "cma_es", "openai_es"].into_iter().zip(scores) {
            let hits = values.iter().filter(|&&v| v <= TARGET).count();
            println!(
                "{problem:10} {algorithm:10} median_final={:.9e} target_hits={hits}/20",
                median(values)
            );
        }
    }
    curves.flush()?;
    runs.flush()?;

    let mut pareto = Nsga2::new(Nsga2Config::new(30, vec![Objective::Minimize; 2], 42))?;
    let mut zdt = Zdt1::new(30)?;
    let mut pareto_csv = BufWriter::new(File::create(directory.join("zdt1_population.csv"))?);
    writeln!(pareto_csv, "batch,evaluations,job_id,f1,f2")?;
    for batch in 1..=200 {
        step(&mut pareto, &mut zdt)?;
        if [1, 50, 100, 200].contains(&batch) {
            for individual in pareto.parents() {
                writeln!(
                    pareto_csv,
                    "{batch},{},{},{:.17e},{:.17e}",
                    pareto.evaluations(),
                    individual.job.id.0,
                    individual.output[0],
                    individual.output[1]
                )?;
            }
        }
    }
    pareto_csv.flush()?;
    let front = pareto.pareto_front();
    let mean_gap = front
        .iter()
        .map(|r| r.output[1] - (1.0 - r.output[0].sqrt()))
        .sum::<f64>()
        / front.len() as f64;
    println!(
        "zdt1: 30 dimensions, seed 42, {} evaluations, {} nondominated parents, mean vertical gap={mean_gap:.9e}",
        pareto.evaluations(),
        front.len()
    );
    fs::write(
        directory.join("search_manifest.json"),
        r#"{
  "experiment": "chapter2_reference_searches",
  "dimensions": 8,
  "problems": ["sphere", "rastrigin"],
  "seeds": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
  "population": 64,
  "batches_including_initialization": 100,
  "evaluations_per_run": 6400,
  "target": 1e-8,
  "objective": "minimize",
  "reported_score": "best actually evaluated candidate",
  "ga": {"initial_bound":5,"mutation_sigma":0.1,"mutation_probability":0.125,"tournament_size":3,"elite_count":1,"crossover_probability":0.9,"reevaluate_elites":true},
  "es": {"initial_bound":5,"parents":16,"initial_sigma":0.5,"tau":"1/sqrt(2*d)","sigma_min":1e-12,"sigma_max":1e3,"replacement":"comma","recombination":false},
  "cma_es": {"initial_mean":"uniform [-5,5) on independent ChaCha8 stream2","sigma":0.5,"population":64,"variant":"positive-weight non-active full covariance"},
  "openai_es": {"initial_mean":"uniform [-5,5) on independent ChaCha8 stream2","directions":32,"sigma":0.5,"learning_rate":0.1,"score_transform":"centered average-tie ranks","update":"SGD with 1/(2*K*sigma)"},
  "zdt1": {"dimensions":30,"seed":42,"population":64,"batches":200,"evaluations":12800,"lower_bound":0,"upper_bound":1,"mutation_sigma":0.1,"mutation_probability":"1/30","crossover_probability":0.9,"replacement":"NSGA-II with cached parents"},
  "execution": "serial f64; no early termination; no held-out tuning",
  "limitations": "Equal evaluation budgets do not imply identical initial distributions or universal algorithm rankings. ZDT1 is one demonstration seed."
}
"#,
    )?;
    println!("Saved experiment files to {}", directory.display());
    Ok(())
}
