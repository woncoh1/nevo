use crate::Evaluator;
use crate::nn::{Activation, Mlp, NnError, TrainingExample};

/// Four-point XOR regression with an evolvable `2 -> hidden -> 1` MLP.
///
/// Hidden units use tanh, the output uses sigmoid, and fitness is mean squared
/// error on the entire four-row truth table. This is a tiny representation and
/// optimization exercise, not a held-out generalization benchmark. Parameter
/// layout is exactly [`Mlp`]'s neuron-major weights-followed-by-bias layout.
#[derive(Clone, Debug)]
pub struct Xor {
    hidden: usize,
    dimensions: usize,
    data: Vec<TrainingExample>,
}

impl Xor {
    /// Construct an XOR evaluator with a positive hidden-layer width.
    pub fn new(hidden: usize) -> Result<Self, NnError> {
        let dimensions = Mlp::parameter_count(&[2, hidden, 1])?;
        Ok(Self {
            hidden,
            dimensions,
            data: vec![
                TrainingExample {
                    input: vec![0.0, 0.0],
                    target: vec![0.0],
                },
                TrainingExample {
                    input: vec![0.0, 1.0],
                    target: vec![1.0],
                },
                TrainingExample {
                    input: vec![1.0, 0.0],
                    target: vec![1.0],
                },
                TrainingExample {
                    input: vec![1.0, 1.0],
                    target: vec![0.0],
                },
            ],
        })
    }
    /// Number of evolvable weights and biases.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    /// The four training examples, in truth-table order.
    pub fn data(&self) -> &[TrainingExample] {
        &self.data
    }
    /// Decode flat genes into a checked network with this task's architecture.
    pub fn network(&self, parameters: Vec<f64>) -> Result<Mlp, NnError> {
        Mlp::new(
            &[2, self.hidden, 1],
            &[Activation::Tanh, Activation::Sigmoid],
            parameters,
        )
    }
    /// Compute the mean squared error of a flat parameter vector.
    pub fn value(&self, parameters: &[f64]) -> Result<f64, NnError> {
        self.network(parameters.to_vec())?.mse(&self.data)
    }
}
impl<T: AsRef<[f64]> + ?Sized> Evaluator<T> for Xor {
    type Output = f64;
    type Error = NnError;
    fn evaluate(&mut self, parameters: &T, _seed: u64) -> Result<f64, Self::Error> {
        self.value(parameters.as_ref())
    }
}
