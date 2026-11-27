//! Small, repeatable performance baselines for the public API.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use nevo::algorithms::{Ga, GaConfig};
use nevo::benchmarks::Sphere;
use nevo::neat::{Genome, InnovationTracker, split_connection};
use nevo::nn::{Activation, Mlp};
use nevo::step;

fn ga_sphere(c: &mut Criterion) {
    const POPULATION: usize = 64;
    let mut group = c.benchmark_group("ga_sphere_generation");
    group.throughput(Throughput::Elements(POPULATION as u64));
    for dimensions in [8, 32, 128] {
        group.bench_with_input(
            BenchmarkId::from_parameter(dimensions),
            &dimensions,
            |b, &dimensions| {
                b.iter_batched(
                    || {
                        let mut config = GaConfig::new(dimensions, 42);
                        config.population = POPULATION;
                        let mut search = Ga::new(config).unwrap();
                        let mut problem = Sphere::new(dimensions).unwrap();
                        // Initialization is setup; time the second complete GES cycle.
                        step(&mut search, &mut problem).unwrap();
                        (search, problem)
                    },
                    |(mut search, mut problem)| {
                        step(black_box(&mut search), black_box(&mut problem)).unwrap();
                        // Returning the state keeps its final destruction outside timing.
                        black_box(search)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn mlp_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_forward");
    let input = [0.25; 8];
    for hidden in [8, 32, 128] {
        let widths = [8, hidden, 2];
        let parameters = vec![0.05; Mlp::parameter_count(&widths).unwrap()];
        let model = Mlp::new(&widths, &[Activation::Tanh, Activation::Tanh], parameters).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(hidden), &model, |b, model| {
            b.iter(|| black_box(model).forward(black_box(&input)).unwrap());
        });
    }
    group.finish();
}

fn neat_genome(hidden: usize) -> Genome {
    let mut tracker = InnovationTracker::new(8, 1).unwrap();
    let mut genome = tracker
        .initial_genome(vec![0.05; tracker.initial_connection_count()])
        .unwrap();
    for innovation in 0..hidden {
        assert!(split_connection(&mut genome, &mut tracker, innovation as u64).unwrap());
    }
    genome
}

fn neat(c: &mut Criterion) {
    let genomes: Vec<_> = [0, 8]
        .into_iter()
        .map(|hidden| (hidden, neat_genome(hidden)))
        .collect();
    let mut group = c.benchmark_group("neat_compile");
    for (hidden, genome) in &genomes {
        group.bench_with_input(BenchmarkId::from_parameter(hidden), genome, |b, genome| {
            b.iter(|| black_box(genome).compile().unwrap());
        });
    }
    group.finish();

    let input = [0.25; 8];
    let mut group = c.benchmark_group("neat_forward");
    for (hidden, genome) in &genomes {
        let network = genome.compile().unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(hidden),
            &network,
            |b, network| {
                b.iter(|| black_box(network).forward(black_box(&input)).unwrap());
            },
        );
    }
    group.finish();
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    ga_sphere(&mut criterion);
    mlp_forward(&mut criterion);
    neat(&mut criterion);
    criterion.final_summary();
}
