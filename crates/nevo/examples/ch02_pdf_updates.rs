//! Reproducible small experiments for the additions prompted by the Chapter 2 PDF.
//!
//! Run `cargo run -p nevo --example ch02_pdf_updates --release --locked -- results/pdf_review`.
//! These declared Sphere runs are companion experiments, not the book's Rastrigin study.

use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use nevo::algorithms::{De, DeConfig, Es, EsConfig, EsSelection, EsStepSize};
use nevo::benchmarks::Sphere;
use nevo::nn::{max_pool2d, multi_head_attention, softmax};
use nevo::{Algorithm, step};

const DIMENSIONS: usize = 8;
const BATCHES: usize = 100;

fn run<A, F>(
    label: &str,
    seed: u64,
    mut search: A,
    best_score: F,
    curves: &mut impl Write,
    runs: &mut impl Write,
) -> Result<(), Box<dyn Error>>
where
    A: Algorithm<Input = Vec<f64>, Output = f64>,
    F: Fn(&A) -> f64,
{
    let mut problem = Sphere::new(DIMENSIONS)?;
    let mut evaluations = 0;
    let mut initial = 0.0;
    for batch in 1..=BATCHES {
        evaluations += step(&mut search, &mut problem)?;
        let score = best_score(&search);
        if batch == 1 {
            initial = score;
        }
        writeln!(curves, "{label},{seed},{batch},{evaluations},{score:.17e}")?;
    }
    let final_score = best_score(&search);
    writeln!(
        runs,
        "{label},{seed},{evaluations},{initial:.17e},{final_score:.17e}"
    )?;
    println!(
        "{label:15} seed={seed} evaluations={evaluations} initial={initial:.6e} final={final_score:.6e}"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() > 1 {
        return Err("usage: ch02_pdf_updates [output-directory]".into());
    }
    let directory = arguments
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("results/pdf_review"));
    fs::create_dir_all(&directory)?;
    let mut curves = BufWriter::new(File::create(directory.join("search_curves.csv"))?);
    let mut runs = BufWriter::new(File::create(directory.join("search_runs.csv"))?);
    writeln!(curves, "algorithm,seed,batch,evaluations,best_sphere")?;
    writeln!(
        runs,
        "algorithm,seed,evaluations,initial_sphere,final_sphere"
    )?;
    for seed in 0..5 {
        run(
            "de_rand_1_bin",
            seed,
            De::new(DeConfig::new(DIMENSIONS, seed))?,
            |s| s.best().unwrap().output,
            &mut curves,
            &mut runs,
        )?;
        for (label, selection) in [
            ("es_comma_fixed", EsSelection::Comma),
            ("es_plus_fixed", EsSelection::Plus),
        ] {
            let mut config = EsConfig::new(DIMENSIONS, seed);
            config.step_size = EsStepSize::Fixed;
            config.selection = selection;
            config.initial_sigma = 0.2;
            run(
                label,
                seed,
                Es::new(config)?,
                |s| s.best().unwrap().output,
                &mut curves,
                &mut runs,
            )?;
        }
    }
    curves.flush()?;
    runs.flush()?;
    fs::write(
        directory.join("search_manifest.json"),
        concat!(
            "{\n",
            "  \"description\": \"PDF-audit additions; original Sphere companion, not a textbook benchmark reproduction\",\n",
            "  \"source_sections\": [\"Chapter 2 section 2.2.2 pp.24-25\", \"Chapter 2 section 2.2.6 pp.34-35\"],\n",
            "  \"objective\": \"minimize sum(x_j^2)\", \"dimensions\": 8,\n",
            "  \"seeds\": [0,1,2,3,4], \"population\": 64,\n",
            "  \"batches_including_initialization\": 100, \"evaluations_per_run\": 6400,\n",
            "  \"initialization\": \"uniform [-5,5), same initial jobs for all three variants at each seed\",\n",
            "  \"de\": {\"variant\": \"rand/1/bin\", \"F\": 0.8, \"CR\": 0.9, \"donors_exclude_target\": true, \"forced_mutant_coordinate\": true, \"strict_slotwise_survival\": true},\n",
            "  \"es\": {\"mu\": 16, \"lambda\": 64, \"sigma\": 0.2, \"scale\": \"fixed isotropic\", \"survival\": [\"comma\",\"plus\"]},\n",
            "  \"parent_fitness_reuse\": \"DE and plus ES reuse deterministic cached scores\",\n",
            "  \"bounds_after_initialization\": null, \"score_transform\": null,\n",
            "  \"stopping\": \"fixed budget; no seed or hyperparameter search\",\n",
            "  \"rng\": \"ChaCha8; search stream 0; evaluation seed stream 1\"\n",
            "}\n"
        ),
    )?;

    let pooled = max_pool2d(
        &[vec![1.0, 4.0, 2.0, 3.0], vec![6.0, 5.0, 8.0, 7.0]],
        [2, 2],
        [2, 2],
    )?;
    let probabilities = softmax(&[1000.0, 1000.0 + 3.0_f64.ln()])?;
    let attended = multi_head_attention(
        &[vec![1.0, -1.0]],
        &[vec![3.0_f64.ln(), 3.0_f64.ln()], vec![0.0, 0.0]],
        &[vec![4.0, 10.0], vec![0.0, 2.0]],
        2,
        &[vec![1.0, 2.0], vec![-1.0, 1.0]],
        false,
    )?;
    println!("max pooling: {pooled:?}");
    println!("stable softmax: {probabilities:?}");
    println!("two-head projected output: {:?}", attended.values);
    println!("Saved search_curves.csv, search_runs.csv and search_manifest.json.");
    Ok(())
}
