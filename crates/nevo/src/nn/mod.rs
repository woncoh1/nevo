//! Small scalar neural-network reference implementations.
//!
//! These transparent `f64` building blocks illustrate Chapter 2's neural
//! foundations. They use ordinary Rust data and explicit parameters. Only
//! [`Mlp`] includes analytic gradients; recurrence, convolution, and attention
//! are forward computations, not a general differentiation framework.
//!
//! References for conventions:
//! - [Affine layers](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
//!   and [mean squared error](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html).
//! - [RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
//!   and [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) equations.
//! - [CNN cross-correlation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).
//! - [Vaswani et al.](https://arxiv.org/abs/1706.03762), attention and Transformer blocks.

mod attention;
mod mlp;
mod recurrent;
mod spatial;

pub use attention::{
    AttentionOutput, TransformerBlock, layer_norm, scaled_dot_product_attention,
    sinusoidal_positions,
};
pub use mlp::{Activation, Mlp, TrainingExample};
pub use recurrent::{LstmCell, LstmState, RnnCell};
pub use spatial::conv2d_valid;

use std::error::Error;
use std::fmt;

/// A checked shape, parameter, or numerical error in a neural operation.
#[derive(Clone, Debug, PartialEq)]
pub enum NnError {
    /// A dimension is zero, a matrix is ragged, or shapes are incompatible.
    InvalidShape(&'static str),
    /// A vector has the wrong length for an otherwise valid shape.
    LengthMismatch {
        /// The value being checked.
        context: &'static str,
        /// Required length.
        expected: usize,
        /// Received length.
        actual: usize,
    },
    /// An input or parameter is NaN or infinite.
    NonFinite {
        /// The value being checked.
        context: &'static str,
        /// Index in the vector, or row-major index in a matrix.
        index: usize,
    },
    /// A dimension calculation or numerical computation overflowed.
    Overflow(&'static str),
    /// Gradient descent requires a finite, strictly positive learning rate.
    InvalidLearningRate,
    /// Normalization requires a finite, strictly positive epsilon.
    InvalidEpsilon,
}

impl fmt::Display for NnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(message) => write!(f, "invalid neural shape: {message}"),
            Self::LengthMismatch {
                context,
                expected,
                actual,
            } => write!(f, "{context} requires {expected} values, received {actual}"),
            Self::NonFinite { context, index } => {
                write!(f, "{context} value {index} must be finite")
            }
            Self::Overflow(context) => write!(f, "neural computation overflowed: {context}"),
            Self::InvalidLearningRate => write!(f, "learning rate must be finite and positive"),
            Self::InvalidEpsilon => write!(f, "normalization epsilon must be finite and positive"),
        }
    }
}

impl Error for NnError {}

pub(crate) fn check_len(
    values: &[f64],
    expected: usize,
    context: &'static str,
) -> Result<(), NnError> {
    if values.len() != expected {
        return Err(NnError::LengthMismatch {
            context,
            expected,
            actual: values.len(),
        });
    }
    check_finite(values, context)
}

pub(crate) fn check_finite(values: &[f64], context: &'static str) -> Result<(), NnError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(NnError::NonFinite { context, index });
    }
    Ok(())
}

pub(crate) fn finite(value: f64, context: &'static str) -> Result<f64, NnError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(NnError::Overflow(context))
    }
}

pub(crate) fn dense_count(inputs: usize, outputs: usize) -> Result<usize, NnError> {
    if inputs == 0 || outputs == 0 {
        return Err(NnError::InvalidShape("layer dimensions must be positive"));
    }
    inputs
        .checked_add(1)
        .and_then(|width| width.checked_mul(outputs))
        .ok_or(NnError::Overflow("parameter count"))
}

/// Internal affine operation; callers have checked the shape and inputs.
pub(crate) fn affine(input: &[f64], parameters: &[f64]) -> Result<Vec<f64>, NnError> {
    let width = input.len() + 1;
    parameters
        .chunks_exact(width)
        .map(|row| {
            let mut value = row[input.len()];
            for (&weight, &coordinate) in row[..input.len()].iter().zip(input) {
                value = finite(value + weight * coordinate, "affine layer")?;
            }
            Ok(value)
        })
        .collect()
}

pub(crate) fn matrix_shape(
    matrix: &[Vec<f64>],
    context: &'static str,
) -> Result<(usize, usize), NnError> {
    let columns = matrix
        .first()
        .map(Vec::len)
        .filter(|&length| length > 0)
        .ok_or(NnError::InvalidShape(
            "matrices need at least one row and column",
        ))?;
    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != columns {
            return Err(NnError::InvalidShape("matrix rows must have equal lengths"));
        }
        if let Some(column) = row.iter().position(|value| !value.is_finite()) {
            return Err(NnError::NonFinite {
                context,
                index: row_index * columns + column,
            });
        }
    }
    Ok((matrix.len(), columns))
}
