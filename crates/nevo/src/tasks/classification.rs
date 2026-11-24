//! Checked classification data and evaluators for fixed and evolving topologies.

use crate::Evaluator;
use crate::algorithms::common::{seeded_rng, uniform_index};
use crate::neat::{Genome, NeatError};
use crate::nn::{Activation, Mlp, NnError};
use std::{collections::BTreeSet, error::Error, fmt, sync::Arc};

/// A labeled input and its original row identity.
#[derive(Clone, Debug, PartialEq)]
pub struct LabeledSample {
    /// Stable identity used to audit disjoint data splits.
    pub id: usize,
    /// Finite input coordinates.
    pub input: Vec<f64>,
    /// Zero-based class index.
    pub label: usize,
}

/// Validated immutable data; clones share rows through `Arc`.
#[derive(Clone, Debug)]
pub struct ClassificationData {
    samples: Arc<[LabeledSample]>,
    features: usize,
    classes: usize,
}

/// Disjoint partitions with explicit sample identities.
#[derive(Clone, Debug)]
pub struct DataSplit {
    /// Data permitted to affect evolutionary selection.
    pub training: ClassificationData,
    /// Separate diagnostics; examples do not tune settings on this partition.
    pub validation: ClassificationData,
    /// Final evaluation partition, untouched during optimization.
    pub test: ClassificationData,
}

/// Shape, input, decoding, or numerical failure in classification.
#[derive(Debug)]
pub enum ClassificationError {
    /// Invalid dataset, split, or prediction contract.
    Invalid(&'static str),
    /// Malformed Optdigits CSV row.
    Csv {
        /// One-based physical line number.
        line: usize,
        /// Explanation of the malformed value.
        message: &'static str,
    },
    /// Fixed network could not be constructed or evaluated.
    Network(NnError),
    /// Evolved topology could not be compiled or evaluated.
    Topology(NeatError),
}
impl fmt::Display for ClassificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "classification: {s}"),
            Self::Csv { line, message } => write!(f, "Optdigits CSV line {line}: {message}"),
            Self::Network(e) => write!(f, "classification network: {e}"),
            Self::Topology(e) => write!(f, "classification topology: {e}"),
        }
    }
}
impl Error for ClassificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Network(e) => Some(e),
            Self::Topology(e) => Some(e),
            _ => None,
        }
    }
}
impl From<NnError> for ClassificationError {
    fn from(e: NnError) -> Self {
        Self::Network(e)
    }
}
impl From<NeatError> for ClassificationError {
    fn from(e: NeatError) -> Self {
        Self::Topology(e)
    }
}

impl ClassificationData {
    /// Validate nonempty, equally sized finite inputs, unique IDs and class labels.
    pub fn new(samples: Vec<LabeledSample>, classes: usize) -> Result<Self, ClassificationError> {
        let features = samples.first().map_or(0, |row| row.input.len());
        if features == 0 || classes < 2 {
            return Err(ClassificationError::Invalid(
                "nonempty inputs and at least two classes required",
            ));
        }
        let mut ids = BTreeSet::new();
        for row in &samples {
            if row.input.len() != features
                || row.input.iter().any(|v| !v.is_finite())
                || row.label >= classes
                || !ids.insert(row.id)
            {
                return Err(ClassificationError::Invalid(
                    "inconsistent features, nonfinite input, invalid class or duplicate row ID",
                ));
            }
        }
        Ok(Self {
            samples: samples.into(),
            features,
            classes,
        })
    }
    /// Parse 64 integer pixels in `[0,16]` followed by a class in `[0,9]`.
    /// Pixels are normalized by 16. Blank lines are ignored; row IDs are their
    /// zero-based physical line positions. No header or missing values are accepted.
    pub fn from_optdigits_csv(csv: &str) -> Result<Self, ClassificationError> {
        let mut rows = Vec::new();
        for (index, line) in csv.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let values: Vec<_> = line.split(',').map(str::trim).collect();
            let fail = |message| ClassificationError::Csv {
                line: index + 1,
                message,
            };
            if values.len() != 65 {
                return Err(fail("expected 64 pixels and one label"));
            }
            let mut input = Vec::with_capacity(64);
            for raw in &values[..64] {
                let value: u8 = raw
                    .parse()
                    .map_err(|_| fail("pixels must be integers in [0,16]"))?;
                if value > 16 {
                    return Err(fail("pixel outside [0,16]"));
                }
                input.push(f64::from(value) / 16.0);
            }
            let label: usize = values[64]
                .parse()
                .map_err(|_| fail("label must be an integer in [0,9]"))?;
            if label > 9 {
                return Err(fail("label outside [0,9]"));
            }
            rows.push(LabeledSample {
                id: index,
                input,
                label,
            });
        }
        Self::new(rows, 10)
    }
    /// Original immutable rows, in deterministic evaluation order.
    pub fn samples(&self) -> &[LabeledSample] {
        &self.samples
    }
    /// Number of input coordinates.
    pub fn features(&self) -> usize {
        self.features
    }
    /// Number of output classes.
    pub fn classes(&self) -> usize {
        self.classes
    }
    /// Sample count.
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    /// Always false for a successfully constructed dataset.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    /// Shuffle each class with a specified seed, then take fixed counts for
    /// training/validation and all remaining rows for test. All classes must
    /// have at least one test row. This is a teaching split, not UCI's original
    /// train/test partition. Each partition is returned in ascending row-ID order.
    pub fn stratified_split(
        &self,
        train_per_class: usize,
        validation_per_class: usize,
        seed: u64,
    ) -> Result<DataSplit, ClassificationError> {
        if train_per_class == 0 || validation_per_class == 0 {
            return Err(ClassificationError::Invalid(
                "training and validation class counts must be positive",
            ));
        }
        let used = train_per_class
            .checked_add(validation_per_class)
            .ok_or(ClassificationError::Invalid("split count overflow"))?;
        let mut rng = seeded_rng(seed, 0);
        let mut training = Vec::new();
        let mut validation = Vec::new();
        let mut test = Vec::new();
        for class in 0..self.classes {
            let mut group: Vec<_> = self
                .samples
                .iter()
                .filter(|row| row.label == class)
                .cloned()
                .collect();
            if group.len() <= used {
                return Err(ClassificationError::Invalid(
                    "each class needs rows for all three partitions",
                ));
            }
            for end in (1..group.len()).rev() {
                let other = uniform_index(end + 1, &mut rng);
                group.swap(end, other);
            }
            test.extend(group.drain(used..));
            validation.extend(group.drain(train_per_class..));
            training.extend(group);
        }
        for rows in [&mut training, &mut validation, &mut test] {
            rows.sort_by_key(|row| row.id);
        }
        Ok(DataSplit {
            training: Self::new(training, self.classes)?,
            validation: Self::new(validation, self.classes)?,
            test: Self::new(test, self.classes)?,
        })
    }
}

/// One-hot MSE and classification accuracy, computed in a single pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassificationMetrics {
    /// Squared error averaged over rows and output coordinates; lower is better.
    pub mse: f64,
    /// Fraction of correct argmax predictions; exact ties choose the first class.
    pub accuracy: f64,
    /// Number of rows evaluated.
    pub samples: usize,
}

fn measure<F>(
    data: &ClassificationData,
    mut forward: F,
) -> Result<ClassificationMetrics, ClassificationError>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, ClassificationError>,
{
    let mut total = 0.0;
    let mut correct = 0;
    for row in data.samples() {
        let prediction = forward(&row.input)?;
        if prediction.len() != data.classes
            || prediction
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(ClassificationError::Invalid(
                "predictions require one finite [0,1] output per class",
            ));
        }
        let mut selected = 0;
        for (class, &value) in prediction.iter().enumerate() {
            if value > prediction[selected] {
                selected = class;
            }
            let error = value - if class == row.label { 1.0 } else { 0.0 };
            total += error * error / data.classes as f64;
        }
        correct += usize::from(selected == row.label);
    }
    Ok(ClassificationMetrics {
        mse: total / data.len() as f64,
        accuracy: correct as f64 / data.len() as f64,
        samples: data.len(),
    })
}

/// Minimize one-hot MSE by evolving weights of a fixed tanh/sigmoid MLP.
#[derive(Clone, Debug)]
pub struct MlpClassifier {
    data: ClassificationData,
    widths: Vec<usize>,
    activations: Vec<Activation>,
    dimensions: usize,
}
impl MlpClassifier {
    /// Insert the requested positive hidden widths between data input and output.
    /// An empty list gives a single affine/sigmoid output layer.
    pub fn new(data: ClassificationData, hidden: &[usize]) -> Result<Self, ClassificationError> {
        let mut widths = vec![data.features];
        widths.extend_from_slice(hidden);
        widths.push(data.classes);
        let dimensions = Mlp::parameter_count(&widths)?;
        let mut activations = vec![Activation::Tanh; hidden.len()];
        activations.push(Activation::Sigmoid);
        Ok(Self {
            data,
            widths,
            activations,
            dimensions,
        })
    }
    /// Number of evolvable weights and biases.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    /// Architecture widths, including input and output layers.
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }
    /// Dataset used for evolutionary selection.
    pub fn data(&self) -> &ClassificationData {
        &self.data
    }
    /// Decode checked flat weights using the existing MLP parameter layout.
    pub fn network(&self, parameters: Vec<f64>) -> Result<Mlp, ClassificationError> {
        Ok(Mlp::new(&self.widths, &self.activations, parameters)?)
    }
    /// Compute MSE and accuracy on the configured dataset.
    pub fn metrics(
        &self,
        parameters: &[f64],
    ) -> Result<ClassificationMetrics, ClassificationError> {
        let network = self.network(parameters.to_vec())?;
        measure(&self.data, |input| Ok(network.forward(input)?))
    }
}
impl<T: AsRef<[f64]> + ?Sized> Evaluator<T> for MlpClassifier {
    type Output = f64;
    type Error = ClassificationError;
    fn evaluate(&mut self, input: &T, _seed: u64) -> Result<f64, Self::Error> {
        Ok(self.metrics(input.as_ref())?.mse)
    }
}

/// Maximize `1 - one_hot_MSE` with a sigmoid-output NEAT genome.
#[derive(Clone, Debug)]
pub struct NeatClassifier {
    data: ClassificationData,
}
impl NeatClassifier {
    /// Use checked classification data for topology evolution.
    pub fn new(data: ClassificationData) -> Self {
        Self { data }
    }
    /// Dataset used to evaluate genomes.
    pub fn data(&self) -> &ClassificationData {
        &self.data
    }
    /// Compile once, then score all rows. The genome must match data dimensions.
    pub fn metrics(&self, genome: &Genome) -> Result<ClassificationMetrics, ClassificationError> {
        let network = genome.compile()?;
        measure(&self.data, |input| Ok(network.forward(input)?))
    }
}
impl Evaluator<Genome> for NeatClassifier {
    type Output = f64;
    type Error = ClassificationError;
    fn evaluate(&mut self, genome: &Genome, _seed: u64) -> Result<f64, Self::Error> {
        Ok(1.0 - self.metrics(genome)?.mse)
    }
}

/// NEAT's small binary XOR exercise with fitness `(4 - sum_absolute_error)^2`.
///
/// All four truth-table rows are training data. The single output must be in
/// `[0,1]`. This is a structure-learning exercise, not a generalization test.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeatXor;
impl NeatXor {
    /// Predict the truth table in order 00, 01, 10, 11.
    pub fn predictions(&self, genome: &Genome) -> Result<[f64; 4], ClassificationError> {
        let network = genome.compile()?;
        let mut result = [0.0; 4];
        for (index, input) in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
            .iter()
            .enumerate()
        {
            let output = network.forward(input)?;
            if output.len() != 1 || !output[0].is_finite() || !(0.0..=1.0).contains(&output[0]) {
                return Err(ClassificationError::Invalid(
                    "XOR requires a single [0,1] output",
                ));
            }
            result[index] = output[0];
        }
        Ok(result)
    }
    /// Mean squared error for comparison with the fixed-topology XOR task.
    pub fn mse(&self, genome: &Genome) -> Result<f64, ClassificationError> {
        Ok(self
            .predictions(genome)?
            .iter()
            .zip([0.0, 1.0, 1.0, 0.0])
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / 4.0)
    }
}
impl Evaluator<Genome> for NeatXor {
    type Output = f64;
    type Error = ClassificationError;
    fn evaluate(&mut self, genome: &Genome, _seed: u64) -> Result<f64, Self::Error> {
        let error = self
            .predictions(genome)?
            .iter()
            .zip([0.0, 1.0, 1.0, 0.0])
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>();
        Ok((4.0 - error).powi(2))
    }
}
