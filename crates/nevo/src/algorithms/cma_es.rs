//! Classic non-active full-covariance CMA-ES.
//!
//! Equations follow Hansen's tutorial, Appendix C (positive recombination
//! weights): <https://arxiv.org/html/1604.00772v2>. This implementation performs
//! a symmetric eigendecomposition after every accepted batch. It omits active
//! negative weights, restarts, constraints, and automatic stopping heuristics.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand_chacha::ChaCha8Rng;

use super::Objective;
use super::common::{BatchState, SearchError, normal, score_cmp, seeded_rng, update_best};
use crate::{Algorithm, Evaluated, Job};

/// Configuration for full-covariance [`CmaEs`].
#[derive(Clone, Debug, PartialEq)]
pub struct CmaEsConfig {
    /// Initial distribution mean; at least one finite coordinate is required.
    pub initial_mean: Vec<f64>,
    /// Initial positive, finite global step size.
    pub sigma: f64,
    /// Offspring per batch; at least two. The best half supply positive weights.
    pub population: usize,
    /// Preference for the raw objective values.
    pub objective: Objective,
    /// Root seed; stream 0 samples candidates and stream 1 supplies job seeds.
    pub seed: u64,
}

impl CmaEsConfig {
    /// Choose sigma 0.5, minimization, and population `4 + floor(3 ln(n))`.
    ///
    /// An empty mean is retained here and rejected by [`CmaEs::new`].
    pub fn new(initial_mean: Vec<f64>, seed: u64) -> Self {
        let n = initial_mean.len().max(1) as f64;
        Self {
            initial_mean,
            sigma: 0.5,
            population: 4 + (3.0 * n.ln()).floor() as usize,
            objective: Objective::Minimize,
            seed,
        }
    }

    fn validate(&self) -> Result<(), SearchError> {
        if self.initial_mean.is_empty() || self.initial_mean.iter().any(|x| !x.is_finite()) {
            return Err(SearchError::InvalidConfig(
                "CMA-ES needs a nonempty finite mean",
            ));
        }
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err(SearchError::InvalidConfig(
                "CMA-ES sigma must be positive and finite",
            ));
        }
        if self.population < 2 {
            return Err(SearchError::InvalidConfig(
                "CMA-ES needs at least two offspring",
            ));
        }
        self.initial_mean
            .len()
            .checked_mul(self.initial_mean.len())
            .and_then(|size| size.checked_mul(100))
            .ok_or(SearchError::InvalidConfig(
                "CMA-ES matrix size is too large",
            ))?;
        self.initial_mean
            .len()
            .checked_mul(self.population)
            .ok_or(SearchError::InvalidConfig(
                "CMA-ES population size is too large",
            ))?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Parameters {
    weights: Vec<f64>,
    mu_eff: f64,
    c_c: f64,
    c_sigma: f64,
    c_one: f64,
    c_mu: f64,
    damping: f64,
    expected_norm: f64,
}

impl Parameters {
    fn new(dimensions: usize, population: usize) -> Self {
        let n = dimensions as f64;
        let mu = population / 2;
        // The half-population is not rounded before computing the logarithm.
        // This matches the positive weights for odd as well as even lambda.
        let log_half = ((population as f64 + 1.0) / 2.0).ln();
        let mut weights: Vec<_> = (1..=mu).map(|rank| log_half - (rank as f64).ln()).collect();
        let weight_sum: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= weight_sum;
        }
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        let c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n);
        let c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0);
        let c_one = 2.0 / ((n + 1.3).powi(2) + mu_eff);
        let c_mu =
            (1.0 - c_one).min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0).powi(2) + mu_eff));
        let damping = 1.0 + 2.0 * (((mu_eff - 1.0) / (n + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let expected_norm = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
        Self {
            weights,
            mu_eff,
            c_c,
            c_sigma,
            c_one,
            c_mu,
            damping,
            expected_norm,
        }
    }
}

/// A Gaussian search with cumulative paths, covariance and step-size adaptation.
///
/// Every batch samples `x = mean + sigma * B * D * z`, with independent standard
/// normal `z`. Positive logarithmic weights recombine the best half. Two paths
/// adapt the global step size and the full covariance, including off-diagonal
/// terms. Ties preserve submission order.
///
/// The distribution mean is not automatically evaluated. [`Self::best`] is a
/// separate best-ever *evaluated sample*, not a score attached to the mean.
/// No objective call is hidden inside the optimizer. Matrix work is pure Rust
/// through nalgebra; covariance storage is quadratic and each eigensolve cubic.
#[derive(Debug)]
pub struct CmaEs {
    config: CmaEsConfig,
    parameters: Parameters,
    mean: DVector<f64>,
    sigma: f64,
    covariance: DMatrix<f64>,
    transform: DMatrix<f64>,
    inverse_root: DMatrix<f64>,
    path_covariance: DVector<f64>,
    path_sigma: DVector<f64>,
    pending_steps: Vec<DVector<f64>>,
    candidates_rng: ChaCha8Rng,
    seeds_rng: ChaCha8Rng,
    batch: BatchState,
    best: Option<Evaluated<Vec<f64>, f64>>,
}

impl CmaEs {
    /// Initialize identity covariance and zero evolution paths.
    ///
    /// # Errors
    /// Rejects invalid means, step sizes, and populations.
    pub fn new(config: CmaEsConfig) -> Result<Self, SearchError> {
        config.validate()?;
        let n = config.initial_mean.len();
        Ok(Self {
            parameters: Parameters::new(n, config.population),
            mean: DVector::from_vec(config.initial_mean.clone()),
            sigma: config.sigma,
            covariance: DMatrix::identity(n, n),
            transform: DMatrix::identity(n, n),
            inverse_root: DMatrix::identity(n, n),
            path_covariance: DVector::zeros(n),
            path_sigma: DVector::zeros(n),
            pending_steps: Vec::new(),
            candidates_rng: seeded_rng(config.seed, 0),
            seeds_rng: seeded_rng(config.seed, 1),
            batch: BatchState::new(),
            best: None,
            config,
        })
    }

    /// Initial configuration; adapted state is exposed separately.
    pub fn config(&self) -> &CmaEsConfig {
        &self.config
    }

    /// Current mean; no score is implied for this generally unevaluated point.
    pub fn mean(&self) -> &[f64] {
        self.mean.as_slice()
    }

    /// Current global step size.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Copy the current covariance shape as rows, excluding the `sigma^2` scale.
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        (0..self.mean.len())
            .map(|row| self.covariance.row(row).iter().copied().collect())
            .collect()
    }

    /// Best evaluated sample so far; never a fabricated mean evaluation.
    pub fn best(&self) -> Option<&Evaluated<Vec<f64>, f64>> {
        self.best.as_ref()
    }

    /// Number of successfully selected batches.
    pub fn batches(&self) -> usize {
        self.batch.batches()
    }

    /// Evaluations incorporated by successful selection, excluding failed batches.
    pub fn evaluations(&self) -> usize {
        self.batch.evaluations()
    }
}

impl Algorithm for CmaEs {
    type Input = Vec<f64>;
    type Output = f64;
    type Error = SearchError;

    fn generate(&mut self) -> Result<Vec<Job<Vec<f64>>>, SearchError> {
        self.batch.ensure_ready()?;
        let n = self.mean.len();
        let mut steps = Vec::with_capacity(self.config.population);
        let mut inputs = Vec::with_capacity(self.config.population);
        for _ in 0..self.config.population {
            let z = DVector::from_iterator(n, (0..n).map(|_| normal(&mut self.candidates_rng)));
            let y = &self.transform * z;
            let input = &self.mean + self.sigma * &y;
            if input.iter().chain(y.iter()).any(|x| !x.is_finite()) {
                return Err(SearchError::Numerical("CMA-ES sampling overflowed"));
            }
            inputs.push(input.as_slice().to_vec());
            steps.push(y);
        }
        let jobs = self.batch.issue(inputs, &mut self.seeds_rng)?;
        self.pending_steps = steps;
        Ok(jobs)
    }

    fn select(&mut self, results: Vec<Evaluated<Vec<f64>, f64>>) -> Result<(), SearchError> {
        self.batch.validate_scalar(&results)?;
        let mut order: Vec<_> = (0..results.len()).collect();
        order.sort_by(|&a, &b| {
            score_cmp(results[a].output, results[b].output, self.config.objective)
        });
        let n = self.mean.len();
        let p = &self.parameters;
        let mut mean = DVector::zeros(n);
        let mut y_mean = DVector::zeros(n);
        for (&index, &weight) in order.iter().zip(&p.weights) {
            mean += weight * DVector::from_column_slice(&results[index].job.input);
            y_mean += weight * &self.pending_steps[index];
        }

        let path_sigma = (1.0 - p.c_sigma) * &self.path_sigma
            + (p.c_sigma * (2.0 - p.c_sigma) * p.mu_eff).sqrt() * (&self.inverse_root * &y_mean);
        let path_norm = path_sigma.norm();
        let generation = self
            .batch
            .batches()
            .checked_add(1)
            .ok_or(SearchError::CountOverflow)?;
        let correction = (1.0 - (1.0 - p.c_sigma).powf(2.0 * generation as f64)).sqrt();
        let h_sigma = path_norm / correction < (1.4 + 2.0 / (n as f64 + 1.0)) * p.expected_norm;
        let path_covariance = (1.0 - p.c_c) * &self.path_covariance
            + if h_sigma {
                (p.c_c * (2.0 - p.c_c) * p.mu_eff).sqrt() * &y_mean
            } else {
                DVector::zeros(n)
            };
        let old_covariance_weight = 1.0 - p.c_one - p.c_mu
            + if h_sigma {
                0.0
            } else {
                p.c_one * p.c_c * (2.0 - p.c_c)
            };
        let mut covariance = old_covariance_weight * &self.covariance
            + p.c_one * (&path_covariance * path_covariance.transpose());
        for (&index, &weight) in order.iter().zip(&p.weights) {
            let y = &self.pending_steps[index];
            covariance += (p.c_mu * weight) * (y * y.transpose());
        }
        // Avoid allowing numerical asymmetry to enter the next eigensolve.
        for row in 0..n {
            for column in 0..row {
                let value = 0.5 * covariance[(row, column)] + 0.5 * covariance[(column, row)];
                covariance[(row, column)] = value;
                covariance[(column, row)] = value;
            }
        }
        let sigma =
            self.sigma * ((p.c_sigma / p.damping) * (path_norm / p.expected_norm - 1.0)).exp();
        if !sigma.is_finite()
            || sigma <= 0.0
            || mean
                .iter()
                .chain(path_sigma.iter())
                .chain(path_covariance.iter())
                .chain(covariance.iter())
                .any(|x| !x.is_finite())
        {
            return Err(SearchError::Numerical(
                "CMA-ES adaptation became nonfinite or sigma vanished",
            ));
        }
        let (transform, inverse_root) = decompose(&covariance)?;
        // All protocol and numerical checks precede this state commit.
        update_best(&mut self.best, &results, self.config.objective);
        self.mean = mean;
        self.sigma = sigma;
        self.path_sigma = path_sigma;
        self.path_covariance = path_covariance;
        self.covariance = covariance;
        self.transform = transform;
        self.inverse_root = inverse_root;
        self.pending_steps.clear();
        self.batch.commit();
        Ok(())
    }
}

fn decompose(covariance: &DMatrix<f64>) -> Result<(DMatrix<f64>, DMatrix<f64>), SearchError> {
    let n = covariance.nrows();
    let scale = covariance
        .diagonal()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(SearchError::Numerical(
            "CMA-ES covariance has no positive scale",
        ));
    }
    // Normalize the matrix so an absolute eigensolver tolerance remains useful
    // when covariance entries become very small or very large.
    let eigen = SymmetricEigen::try_new(covariance / scale, f64::EPSILON, 100 * n * n).ok_or(
        SearchError::Numerical("CMA-ES eigendecomposition did not converge"),
    )?;
    if eigen
        .eigenvalues
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(SearchError::Numerical(
            "CMA-ES covariance lost positive definiteness",
        ));
    }
    let axes = eigen.eigenvalues.map(|value| value.sqrt() * scale.sqrt());
    let inverse_axes = axes.map(|value| 1.0 / value);
    let transform = &eigen.eigenvectors * DMatrix::from_diagonal(&axes);
    let inverse_root = &eigen.eigenvectors
        * DMatrix::from_diagonal(&inverse_axes)
        * eigen.eigenvectors.transpose();
    if transform
        .iter()
        .chain(inverse_root.iter())
        .any(|x| !x.is_finite())
    {
        return Err(SearchError::Numerical(
            "CMA-ES covariance factors are not finite",
        ));
    }
    Ok((transform, inverse_root))
}
