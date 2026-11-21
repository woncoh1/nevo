//! Equation, covariance, estimator, and protocol checks for Gaussian searches.

use nevo::algorithms::{
    CmaEs, CmaEsConfig, Objective, OpenAiEs, OpenAiEsConfig, ScoreTransform, SearchError,
};
use nevo::{Algorithm, Evaluated, Job};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn scored(
    jobs: Vec<Job<Vec<f64>>>,
    fitness: impl Fn(&[f64]) -> f64,
) -> Vec<Evaluated<Vec<f64>, f64>> {
    jobs.into_iter()
        .map(|job| {
            let output = fitness(&job.input);
            Evaluated { job, output }
        })
        .collect()
}

fn cycle<A>(algorithm: &mut A, fitness: impl Fn(&[f64]) -> f64)
where
    A: Algorithm<Input = Vec<f64>, Output = f64, Error = SearchError>,
{
    let results = scored(algorithm.generate().unwrap(), fitness);
    algorithm.select(results).unwrap();
}

#[test]
fn cma_first_generation_matches_positive_weight_equations_including_odd_population() {
    let initial = [1.5, -2.0];
    let sigma = 0.7;
    let mut config = CmaEsConfig::new(initial.to_vec(), 13);
    config.population = 5;
    config.sigma = sigma;
    let mut search = CmaEs::new(config).unwrap();
    let jobs = search.generate().unwrap();
    let raw_weights = [3.0_f64.ln(), 1.5_f64.ln()];
    let sum = raw_weights.iter().sum::<f64>();
    let weights = raw_weights.map(|w| w / sum);
    let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    let selected = [1, 3];
    let mut weighted_mean = [0.0; 2];
    let mut y_mean = [0.0; 2];
    let mut ys = [[0.0; 2]; 2];
    for (rank, &index) in selected.iter().enumerate() {
        for j in 0..2 {
            ys[rank][j] = (jobs[index].input[j] - initial[j]) / sigma;
            y_mean[j] += weights[rank] * ys[rank][j];
            weighted_mean[j] += weights[rank] * jobs[index].input[j];
        }
    }
    let n = 2.0;
    let cs = (mu_eff + 2.0) / (n + mu_eff + 5.0);
    let cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n);
    let c1 = 2.0 / ((n + 1.3).powi(2) + mu_eff);
    let cmu = (1.0 - c1).min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0).powi(2) + mu_eff));
    let damping = 1.0 + 2.0 * (((mu_eff - 1.0) / (n + 1.0)).sqrt() - 1.0).max(0.0) + cs;
    let chi = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
    let ps = y_mean.map(|y| (cs * (2.0 - cs) * mu_eff).sqrt() * y);
    let ps_norm = ps.iter().map(|v| v * v).sum::<f64>().sqrt();
    let h = ps_norm / (1.0 - (1.0 - cs).powi(2)).sqrt() < (1.4 + 2.0 / (n + 1.0)) * chi;
    let pc = y_mean.map(|y| {
        if h {
            (cc * (2.0 - cc) * mu_eff).sqrt() * y
        } else {
            0.0
        }
    });
    let mut covariance = [[0.0; 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            let old = if row == column {
                1.0 - c1 - cmu + if h { 0.0 } else { c1 * cc * (2.0 - cc) }
            } else {
                0.0
            };
            let rank_mu = (0..2)
                .map(|i| weights[i] * ys[i][row] * ys[i][column])
                .sum::<f64>();
            covariance[row][column] = old + c1 * pc[row] * pc[column] + cmu * rank_mu;
        }
    }
    let expected_sigma = sigma * ((cs / damping) * (ps_norm / chi - 1.0)).exp();
    let results = jobs
        .into_iter()
        .zip([4.0, 0.0, 3.0, 1.0, 2.0])
        .map(|(job, output)| Evaluated { job, output })
        .collect();
    search.select(results).unwrap();
    for (&actual, expected) in search.mean().iter().zip(weighted_mean) {
        close(actual, expected, 1e-13);
    }
    for (actual, expected) in search.covariance().iter().zip(covariance) {
        for (&a, e) in actual.iter().zip(expected) {
            close(a, e, 1e-13);
        }
    }
    close(search.sigma(), expected_sigma, 1e-13);
    assert_eq!(search.evaluations(), 5);
    assert_eq!(search.best().unwrap().output, 0.0);
}

#[test]
fn cma_adapts_a_full_symmetric_positive_covariance_on_a_rotated_problem() {
    let mut config = CmaEsConfig::new(vec![3.0, 1.0], 42);
    config.population = 12;
    config.sigma = 1.0;
    let mut search = CmaEs::new(config).unwrap();
    let fitness = |x: &[f64]| (x[0] + x[1]).powi(2) + 100.0 * (x[0] - x[1]).powi(2);
    cycle(&mut search, fitness);
    let initial = search.best().unwrap().output;
    for _ in 1..70 {
        cycle(&mut search, fitness);
        let c = search.covariance();
        assert_eq!(c[0][1], c[1][0]);
        assert!(c[0][0] > 0.0 && c[1][1] > 0.0);
        assert!(c[0][0] * c[1][1] - c[0][1] * c[1][0] > 0.0);
        assert!(search.sigma() > 0.0 && search.sigma().is_finite());
    }
    let c = search.covariance();
    assert!(c[0][1].abs() > 0.25 * (c[0][0] * c[1][1]).sqrt());
    assert!(search.best().unwrap().output < initial * 1e-4);
}

#[test]
fn cma_reduces_sphere_loss_across_fixed_seeds() {
    for seed in [1, 7, 42] {
        let mut config = CmaEsConfig::new(vec![3.0, -2.0, 1.0], seed);
        config.population = 12;
        let mut search = CmaEs::new(config).unwrap();
        cycle(&mut search, |x| x.iter().map(|x| x * x).sum());
        let initial = search.best().unwrap().output;
        for _ in 1..80 {
            cycle(&mut search, |x| x.iter().map(|x| x * x).sum());
        }
        assert!(search.best().unwrap().output < initial * 1e-4);
        assert_eq!(search.evaluations(), 960);
    }
}

#[test]
fn cma_objective_direction_and_seeded_replay_are_consistent() {
    let config = CmaEsConfig::new(vec![1.0, -1.0], 17);
    let mut maximization = config.clone();
    maximization.objective = Objective::Maximize;
    let mut first = CmaEs::new(config.clone()).unwrap();
    let mut replay = CmaEs::new(config).unwrap();
    let mut opposite = CmaEs::new(maximization).unwrap();
    for _ in 0..8 {
        let jobs = first.generate().unwrap();
        assert_eq!(jobs, replay.generate().unwrap());
        assert_eq!(jobs, opposite.generate().unwrap());
        let results = scored(jobs, |x| x.iter().map(|x| x * x).sum());
        first.select(results.clone()).unwrap();
        replay.select(results.clone()).unwrap();
        opposite
            .select(
                results
                    .into_iter()
                    .map(|mut r| {
                        r.output = -r.output;
                        r
                    })
                    .collect(),
            )
            .unwrap();
        assert_eq!(first.mean(), replay.mean());
        assert_eq!(first.mean(), opposite.mean());
        assert_eq!(first.covariance(), replay.covariance());
        assert_eq!(first.sigma(), opposite.sigma());
    }
}

#[test]
fn cma_supports_one_dimension_and_two_offspring() {
    let mut config = CmaEsConfig::new(vec![1.0], 7);
    config.population = 2;
    let mut search = CmaEs::new(config).unwrap();
    for _ in 0..10 {
        cycle(&mut search, |x| x[0] * x[0]);
    }
    assert!(search.covariance()[0][0] > 0.0);
    assert_eq!(search.evaluations(), 20);
}

#[test]
fn openai_samples_mirrored_pairs_with_common_evaluation_seeds() {
    let config = OpenAiEsConfig::new(vec![1.5, -2.0, 0.0], 42);
    let initial = config.initial_mean.clone();
    let mut search = OpenAiEs::new(config).unwrap();
    let jobs = search.generate().unwrap();
    assert_eq!(jobs.len(), 64);
    for pair in jobs.chunks_exact(2) {
        assert_eq!(pair[0].seed, pair[1].seed);
        assert_ne!(pair[0].id, pair[1].id);
        for (j, &center) in initial.iter().enumerate() {
            close((pair[0].input[j] + pair[1].input[j]) / 2.0, center, 1e-14);
        }
    }
    assert_ne!(jobs[0].seed, jobs[2].seed);
}

#[test]
fn openai_raw_linear_update_matches_the_sampled_gradient_and_objective_direction() {
    for objective in [Objective::Minimize, Objective::Maximize] {
        let mut config = OpenAiEsConfig::new(vec![0.0], 23);
        config.directions = 8;
        config.sigma = 0.3;
        config.learning_rate = 0.05;
        config.score_transform = ScoreTransform::Raw;
        config.objective = objective;
        let mut search = OpenAiEs::new(config.clone()).unwrap();
        let jobs = search.generate().unwrap();
        let noise_square_mean = jobs
            .chunks_exact(2)
            .map(|pair| (pair[0].input[0] / config.sigma).powi(2))
            .sum::<f64>()
            / config.directions as f64;
        let sign = if objective == Objective::Minimize {
            -1.0
        } else {
            1.0
        };
        let expected = sign * config.learning_rate * 2.0 * noise_square_mean;
        search.select(scored(jobs, |x| 2.0 * x[0])).unwrap();
        close(search.mean()[0], expected, 1e-13);
        assert_eq!(search.evaluations(), 16);
    }
}

#[test]
fn openai_rank_ties_have_no_artificial_gradient_including_signed_zero() {
    let mut search = OpenAiEs::new(OpenAiEsConfig::new(vec![1.0, -2.0], 2)).unwrap();
    let jobs = search.generate().unwrap();
    let results = jobs
        .into_iter()
        .enumerate()
        .map(|(i, job)| Evaluated {
            job,
            output: if i % 2 == 0 { 0.0 } else { -0.0 },
        })
        .collect();
    search.select(results).unwrap();
    assert_eq!(search.mean(), &[1.0, -2.0]);
}

#[test]
fn openai_centered_ranks_are_invariant_to_positive_affine_score_changes() {
    let config = OpenAiEsConfig::new(vec![1.0, -2.0], 19);
    let mut first = OpenAiEs::new(config.clone()).unwrap();
    let mut transformed = OpenAiEs::new(config).unwrap();
    let jobs = first.generate().unwrap();
    assert_eq!(jobs, transformed.generate().unwrap());
    let results = scored(jobs, |x| x[0] * x[0] + x[1] * x[1]);
    first.select(results.clone()).unwrap();
    transformed
        .select(
            results
                .into_iter()
                .map(|mut r| {
                    r.output = 3.0 * r.output + 10.0;
                    r
                })
                .collect(),
        )
        .unwrap();
    assert_eq!(first.mean(), transformed.mean());
}

#[test]
fn openai_replays_and_moves_toward_the_sphere_minimum() {
    for seed in [1, 7, 42] {
        let mut config = OpenAiEsConfig::new(vec![3.0, -2.0, 1.0], seed);
        config.score_transform = ScoreTransform::Raw;
        let mut search = OpenAiEs::new(config.clone()).unwrap();
        let mut replay = OpenAiEs::new(config).unwrap();
        for _ in 0..40 {
            let jobs = search.generate().unwrap();
            assert_eq!(jobs, replay.generate().unwrap());
            let results = scored(jobs, |x| x.iter().map(|x| x * x).sum());
            search.select(results.clone()).unwrap();
            replay.select(results).unwrap();
            assert_eq!(search.mean(), replay.mean());
            assert_eq!(search.best(), replay.best());
        }
        assert!(search.mean().iter().map(|x| x * x).sum::<f64>() < 1e-4);
        assert_eq!(search.evaluations(), 2560);
    }
}

#[test]
fn invalid_gaussian_search_configs_are_rejected() {
    for mean in [vec![], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(CmaEs::new(CmaEsConfig::new(mean.clone(), 0)).is_err());
        assert!(OpenAiEs::new(OpenAiEsConfig::new(mean, 0)).is_err());
    }
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut cma = CmaEsConfig::new(vec![0.0], 0);
        cma.sigma = invalid;
        assert!(CmaEs::new(cma).is_err());
        let mut es = OpenAiEsConfig::new(vec![0.0], 0);
        es.learning_rate = invalid;
        assert!(OpenAiEs::new(es.clone()).is_err());
        es.learning_rate = 0.1;
        es.sigma = invalid;
        assert!(OpenAiEs::new(es).is_err());
    }
    let mut cma = CmaEsConfig::new(vec![0.0], 0);
    cma.population = 1;
    assert!(CmaEs::new(cma).is_err());
    let mut es = OpenAiEsConfig::new(vec![0.0], 0);
    es.directions = 0;
    assert!(OpenAiEs::new(es).is_err());
}

fn rejects_bad_batches<A>(make: impl Fn() -> A)
where
    A: Algorithm<Input = Vec<f64>, Output = f64, Error = SearchError>,
{
    let mut search = make();
    assert!(matches!(
        search.select(vec![]),
        Err(SearchError::NoPendingBatch)
    ));
    search.generate().unwrap();
    assert!(matches!(search.generate(), Err(SearchError::PendingBatch)));
    for mode in 0..5 {
        let mut search = make();
        let mut results = scored(search.generate().unwrap(), |x| x.iter().sum());
        match mode {
            0 => {
                results.pop();
            }
            1 => results.swap(0, 1),
            2 => results[0].job.input[0] += 1.0,
            3 => results[0].job.seed ^= 1,
            _ => results.last_mut().unwrap().output = f64::NAN,
        }
        assert!(matches!(
            search.select(results),
            Err(SearchError::InvalidResults(_))
        ));
    }
}

#[test]
fn both_algorithms_reject_protocol_violations() {
    rejects_bad_batches(|| CmaEs::new(CmaEsConfig::new(vec![1.0, 2.0], 0)).unwrap());
    rejects_bad_batches(|| OpenAiEs::new(OpenAiEsConfig::new(vec![1.0, 2.0], 0)).unwrap());
}

#[test]
fn invalid_results_do_not_change_distributions_best_or_accounting() {
    let mut cma = CmaEs::new(CmaEsConfig::new(vec![1.0, 2.0], 0)).unwrap();
    let mut results = scored(cma.generate().unwrap(), |_| 0.0);
    results.last_mut().unwrap().output = f64::NAN;
    assert!(cma.select(results).is_err());
    assert_eq!(cma.mean(), &[1.0, 2.0]);
    assert_eq!(cma.sigma(), 0.5);
    assert_eq!(cma.covariance(), vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert!(cma.best().is_none());
    assert_eq!(cma.evaluations(), 0);

    let mut es = OpenAiEs::new(OpenAiEsConfig::new(vec![1.0, 2.0], 0)).unwrap();
    let mut results = scored(es.generate().unwrap(), |_| 0.0);
    results.last_mut().unwrap().output = f64::NAN;
    assert!(es.select(results).is_err());
    assert_eq!(es.mean(), &[1.0, 2.0]);
    assert!(es.best().is_none());
    assert_eq!(es.evaluations(), 0);
}

#[test]
fn raw_es_reports_numeric_overflow_without_committing_a_mean() {
    let mut config = OpenAiEsConfig::new(vec![0.0], 42);
    config.score_transform = ScoreTransform::Raw;
    let mut search = OpenAiEs::new(config).unwrap();
    let results = search
        .generate()
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, job)| Evaluated {
            job,
            output: if i % 2 == 0 { f64::MAX } else { -f64::MAX },
        })
        .collect();
    assert!(matches!(
        search.select(results),
        Err(SearchError::Numerical(_))
    ));
    assert_eq!(search.mean(), &[0.0]);
    assert_eq!(search.evaluations(), 0);
    assert!(search.best().is_none());
}
