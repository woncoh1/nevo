use super::{NnError, affine, check_finite, check_len, dense_count, finite};

/// Elementwise activation applied after an affine layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    /// Leave the affine output unchanged.
    Identity,
    /// Hyperbolic tangent.
    Tanh,
    /// Logistic sigmoid, evaluated without exponentiating a positive argument.
    Sigmoid,
    /// Rectified linear unit; its derivative at zero is defined as zero.
    Relu,
}

impl Activation {
    pub(crate) fn apply(self, value: f64) -> f64 {
        match self {
            Self::Identity => value,
            Self::Tanh => value.tanh(),
            Self::Sigmoid if value >= 0.0 => 1.0 / (1.0 + (-value).exp()),
            Self::Sigmoid => {
                let exp = value.exp();
                exp / (1.0 + exp)
            }
            Self::Relu => value.max(0.0),
        }
    }

    fn derivative(self, before: f64, after: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Tanh => 1.0 - after * after,
            Self::Sigmoid => after * (1.0 - after),
            Self::Relu => {
                if before > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// One supervised example. Dataset methods validate both vectors' dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingExample {
    /// Network input.
    pub input: Vec<f64>,
    /// Desired output, with one value per output neuron.
    pub target: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Layer {
    inputs: usize,
    outputs: usize,
    offset: usize,
    activation: Activation,
}

type ForwardPass = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// A fully connected multilayer perceptron with explicit flat parameters.
///
/// Parameter order is **layer, output neuron, input weights, bias**. For
/// `[2, 2, 1]`, the first hidden neuron uses indices `0..3` (two weights and a
/// bias), the second uses `3..6`, and the output neuron uses `6..9`.
/// This layout is shared by forward execution, gradients, and evolution.
///
/// ```
/// use nevo::nn::{Activation, Mlp};
///
/// let model = Mlp::new(&[2, 1], &[Activation::Identity], vec![2.0, -1.0, 0.5])?;
/// assert_eq!(model.forward(&[3.0, 4.0])?, vec![2.5]);
/// # Ok::<(), nevo::nn::NnError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Mlp {
    widths: Vec<usize>,
    layers: Vec<Layer>,
    parameters: Vec<f64>,
}

impl Mlp {
    /// Number of weights and biases needed for a nonempty sequence of layers.
    ///
    /// At least input and output widths are required; every width is positive.
    pub fn parameter_count(widths: &[usize]) -> Result<usize, NnError> {
        if widths.len() < 2 {
            return Err(NnError::InvalidShape(
                "an MLP needs input and output widths",
            ));
        }
        widths.windows(2).try_fold(0_usize, |total, pair| {
            total
                .checked_add(dense_count(pair[0], pair[1])?)
                .ok_or(NnError::Overflow("MLP parameter count"))
        })
    }

    /// Construct an MLP with one activation per affine layer.
    ///
    /// The supplied parameters are used as-is; no initialization is hidden in
    /// this constructor. All dimensions, lengths, and parameter values are checked.
    pub fn new(
        widths: &[usize],
        activations: &[Activation],
        parameters: Vec<f64>,
    ) -> Result<Self, NnError> {
        let count = Self::parameter_count(widths)?;
        if activations.len() != widths.len() - 1 {
            return Err(NnError::InvalidShape(
                "one activation is required per MLP layer",
            ));
        }
        check_len(&parameters, count, "MLP parameters")?;
        let mut offset = 0;
        let mut layers = Vec::with_capacity(activations.len());
        for (pair, &activation) in widths.windows(2).zip(activations) {
            layers.push(Layer {
                inputs: pair[0],
                outputs: pair[1],
                offset,
                activation,
            });
            offset += dense_count(pair[0], pair[1])?;
        }
        Ok(Self {
            widths: widths.to_vec(),
            layers,
            parameters,
        })
    }

    /// Input, hidden, and output widths in order.
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }

    /// Flat weights and biases in the documented encoding order.
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// Replace weights without rebuilding the architecture or allocating a vector.
    ///
    /// Validation finishes before any value is changed.
    pub fn set_parameters(&mut self, parameters: &[f64]) -> Result<(), NnError> {
        check_len(parameters, self.parameters.len(), "MLP parameters")?;
        self.parameters.copy_from_slice(parameters);
        Ok(())
    }

    /// Evaluate one finite input vector.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>, NnError> {
        check_len(input, self.widths[0], "MLP input")?;
        let mut values = input.to_vec();
        for layer in &self.layers {
            values = affine(&values, self.layer_parameters(layer))?
                .into_iter()
                .map(|value| layer.activation.apply(value))
                .collect();
        }
        Ok(values)
    }

    /// Mean squared error averaged over **examples and output coordinates**.
    ///
    /// An empty dataset is rejected. This method computes no gradients and is
    /// suitable for evaluating parameter vectors during evolution.
    pub fn mse(&self, examples: &[TrainingExample]) -> Result<f64, NnError> {
        let divisor = self.validate_examples(examples)? as f64;
        let mut loss = 0.0;
        for example in examples {
            let output = self.forward(&example.input)?;
            for (&prediction, &target) in output.iter().zip(&example.target) {
                let error = finite(prediction - target, "MSE residual")?;
                loss = finite(loss + error * error / divisor, "MSE loss")?;
            }
        }
        Ok(loss)
    }

    /// Return mean squared error and its analytic gradient in parameter order.
    ///
    /// Gradients use the same mean-over-all-output-values convention as [`Self::mse`].
    /// Backpropagation is specific to this dense MLP; no graph or autograd engine
    /// is involved. Weights remain unchanged.
    pub fn mse_gradient(&self, examples: &[TrainingExample]) -> Result<(f64, Vec<f64>), NnError> {
        let divisor = self.validate_examples(examples)? as f64;
        let mut gradient = vec![0.0; self.parameters.len()];
        let mut loss = 0.0;
        for example in examples {
            let (activations, before_activation) = self.forward_pass(&example.input)?;
            let output = &activations[self.layers.len()];
            let mut adjoint = Vec::with_capacity(output.len());
            for (&prediction, &target) in output.iter().zip(&example.target) {
                let error = finite(prediction - target, "MSE residual")?;
                loss = finite(loss + error * error / divisor, "MSE loss")?;
                adjoint.push(finite(2.0 * error / divisor, "MSE derivative")?);
            }
            for (layer_index, layer) in self.layers.iter().enumerate().rev() {
                let input = &activations[layer_index];
                let mut previous_adjoint = vec![0.0; layer.inputs];
                for (neuron, &upstream) in adjoint.iter().enumerate() {
                    let delta = finite(
                        upstream
                            * layer.activation.derivative(
                                before_activation[layer_index][neuron],
                                activations[layer_index + 1][neuron],
                            ),
                        "activation derivative",
                    )?;
                    let start = layer.offset + neuron * (layer.inputs + 1);
                    for (index, &value) in input.iter().enumerate() {
                        gradient[start + index] =
                            finite(gradient[start + index] + delta * value, "weight gradient")?;
                        previous_adjoint[index] = finite(
                            previous_adjoint[index] + delta * self.parameters[start + index],
                            "backpropagated derivative",
                        )?;
                    }
                    let bias = start + layer.inputs;
                    gradient[bias] = finite(gradient[bias] + delta, "bias gradient")?;
                }
                adjoint = previous_adjoint;
            }
        }
        Ok((loss, gradient))
    }

    /// Perform one full-batch gradient-descent step and return the **pre-step** loss.
    ///
    /// New parameters are computed and checked before replacement, so an error
    /// leaves the model unchanged. Decreasing loss requires a suitable learning
    /// rate; this method does not perform line search or promise convergence.
    pub fn train_step(
        &mut self,
        examples: &[TrainingExample],
        learning_rate: f64,
    ) -> Result<f64, NnError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(NnError::InvalidLearningRate);
        }
        let (loss, gradient) = self.mse_gradient(examples)?;
        let parameters = self
            .parameters
            .iter()
            .zip(&gradient)
            .map(|(&value, &derivative)| {
                finite(value - learning_rate * derivative, "gradient descent step")
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.parameters = parameters;
        Ok(loss)
    }

    fn layer_parameters(&self, layer: &Layer) -> &[f64] {
        let length = (layer.inputs + 1) * layer.outputs;
        &self.parameters[layer.offset..layer.offset + length]
    }

    fn forward_pass(&self, input: &[f64]) -> Result<ForwardPass, NnError> {
        check_len(input, self.widths[0], "MLP input")?;
        let mut activations = vec![input.to_vec()];
        let mut before_activation = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let previous = &activations[activations.len() - 1];
            let before = affine(previous, self.layer_parameters(layer))?;
            activations.push(
                before
                    .iter()
                    .map(|&value| layer.activation.apply(value))
                    .collect(),
            );
            before_activation.push(before);
        }
        Ok((activations, before_activation))
    }

    fn validate_examples(&self, examples: &[TrainingExample]) -> Result<usize, NnError> {
        if examples.is_empty() {
            return Err(NnError::InvalidShape("training examples cannot be empty"));
        }
        let outputs = self.widths[self.widths.len() - 1];
        for example in examples {
            check_len(&example.input, self.widths[0], "training input")?;
            check_len(&example.target, outputs, "training target")?;
        }
        check_finite(&self.parameters, "MLP parameters")?;
        examples
            .len()
            .checked_mul(outputs)
            .ok_or(NnError::Overflow("MSE output count"))
    }
}
