use super::{Activation, Mlp, NnError, check_finite, check_len, dense_count, finite, matrix_shape};

/// Convert a nonempty finite vector of logits to probabilities.
///
/// Subtracting the largest logit before exponentiation avoids positive
/// overflow and leaves the mathematical softmax unchanged. Very small
/// probabilities may underflow to zero. This is Chapter 2, equation 2.26;
/// it does not change the MLP's existing sigmoid/MSE classification convention.
pub fn softmax(logits: &[f64]) -> Result<Vec<f64>, NnError> {
    if logits.is_empty() {
        return Err(NnError::InvalidShape("softmax input cannot be empty"));
    }
    check_finite(logits, "softmax logits")?;
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities: Vec<f64> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let denominator: f64 = probabilities.iter().sum();
    for probability in &mut probabilities {
        *probability /= denominator;
    }
    Ok(probabilities)
}

/// Multi-head values and each head's query-to-key probability matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiHeadAttentionOutput {
    /// Concatenated head outputs after the output projection: `[queries, output_width]`.
    pub values: Vec<Vec<f64>>,
    /// Attention probabilities: `[heads, queries, keys]`.
    pub weights: Vec<Vec<Vec<f64>>>,
}

/// Attend in separate feature subspaces, concatenate, then apply `W_o`.
///
/// Queries, keys and values are **already projected** by the caller. Each row's
/// features contain `heads` equal contiguous slices; query/key widths match,
/// while the value width may differ. Each head scales its scores by the square
/// root of its own key width. `output_projection` is a row-major matrix of
/// shape `[total_value_width, output_width]`, with no implicit bias.
///
/// This implements Chapter 2, equation 2.29, as a composable forward operation.
/// Separate query and key sequences support cross-attention. Causal mode
/// requires aligned query/key rows and masks future keys within every head.
/// Q/K/V projections, embeddings, a full decoder, and training are caller-owned.
pub fn multi_head_attention(
    queries: &[Vec<f64>],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
    heads: usize,
    output_projection: &[Vec<f64>],
    causal: bool,
) -> Result<MultiHeadAttentionOutput, NnError> {
    let (query_count, query_width) = matrix_shape(queries, "multi-head queries")?;
    let (key_count, key_width) = matrix_shape(keys, "multi-head keys")?;
    let (value_count, value_width) = matrix_shape(values, "multi-head values")?;
    let (projection_rows, output_width) =
        matrix_shape(output_projection, "attention output projection")?;
    if heads == 0 || query_width % heads != 0 || value_width % heads != 0 {
        return Err(NnError::InvalidShape(
            "attention widths must divide into a positive number of heads",
        ));
    }
    if query_width != key_width || key_count != value_count || projection_rows != value_width {
        return Err(NnError::InvalidShape(
            "multi-head query/key widths, key/value rows and projection rows must match",
        ));
    }
    if causal && query_count != key_count {
        return Err(NnError::InvalidShape(
            "causal attention needs aligned query and key rows",
        ));
    }
    let key_head_width = key_width / heads;
    let value_head_width = value_width / heads;
    let mut concatenated = vec![Vec::with_capacity(value_width); query_count];
    let mut weights = Vec::with_capacity(heads);
    for head in 0..heads {
        let slice = |matrix: &[Vec<f64>], width: usize| {
            matrix
                .iter()
                .map(|row| row[head * width..(head + 1) * width].to_vec())
                .collect::<Vec<_>>()
        };
        let attended = scaled_dot_product_attention(
            &slice(queries, key_head_width),
            &slice(keys, key_head_width),
            &slice(values, value_head_width),
            causal,
        )?;
        for (row, head_values) in concatenated.iter_mut().zip(attended.values) {
            row.extend(head_values);
        }
        weights.push(attended.weights);
    }
    let mut output = vec![vec![0.0; output_width]; query_count];
    for (row, head_values) in output.iter_mut().zip(concatenated) {
        for (&value, projection) in head_values.iter().zip(output_projection) {
            for (coordinate, &weight) in row.iter_mut().zip(projection) {
                *coordinate = finite(*coordinate + value * weight, "multi-head output projection")?;
            }
        }
    }
    Ok(MultiHeadAttentionOutput {
        values: output,
        weights,
    })
}

/// Outputs and attention weights from a scaled dot-product attention operation.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionOutput {
    /// One weighted value vector per query: shape `[queries, value_width]`.
    pub values: Vec<Vec<f64>>,
    /// Row-normalized attention: shape `[queries, keys]`.
    /// Masked future positions have exactly zero weight.
    pub weights: Vec<Vec<f64>>,
}

/// Compute `softmax(Q K^T / sqrt(key_width)) V` with stable row softmax.
///
/// Matrices are nonempty rectangular row-major vectors. Queries and keys
/// need matching feature widths; keys and values need matching row counts.
/// When `causal` is true, queries and keys must also have equal row counts:
/// query `t` can then attend only to keys `0..=t`. There is no dropout or
/// implicit projection. Nonfinite inputs and overflowing dot products fail.
pub fn scaled_dot_product_attention(
    queries: &[Vec<f64>],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
    causal: bool,
) -> Result<AttentionOutput, NnError> {
    let (query_count, query_width) = matrix_shape(queries, "attention queries")?;
    let (key_count, key_width) = matrix_shape(keys, "attention keys")?;
    let (value_count, value_width) = matrix_shape(values, "attention values")?;
    if query_width != key_width || key_count != value_count {
        return Err(NnError::InvalidShape(
            "attention query/key widths and key/value rows must match",
        ));
    }
    if causal && query_count != key_count {
        return Err(NnError::InvalidShape(
            "causal attention needs aligned query and key rows",
        ));
    }
    let scale = (key_width as f64).sqrt();
    let mut output = AttentionOutput {
        values: vec![vec![0.0; value_width]; query_count],
        weights: vec![vec![0.0; key_count]; query_count],
    };
    for (query_index, query) in queries.iter().enumerate() {
        let visible = if causal { query_index + 1 } else { key_count };
        let mut scores = Vec::with_capacity(visible);
        for key in &keys[..visible] {
            let mut dot = 0.0;
            for (&q, &k) in query.iter().zip(key) {
                dot = finite(dot + q * k, "attention dot product")?;
            }
            scores.push(dot / scale);
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let row = &mut output.weights[query_index];
        let mut sum = 0.0;
        for (index, score) in scores.iter().enumerate() {
            // A very negative shifted logit may underflow to zero, as intended.
            row[index] = (*score - maximum).exp();
            sum += row[index];
        }
        for weight in &mut row[..visible] {
            *weight /= sum;
        }
        for (value_index, &weight) in row[..visible].iter().enumerate() {
            for (coordinate, &value) in values[value_index].iter().enumerate() {
                let accumulated = &mut output.values[query_index][coordinate];
                *accumulated = finite(*accumulated + weight * value, "attention weighted value")?;
            }
        }
    }
    Ok(output)
}

/// Normalize one token across its coordinates using population variance.
///
/// Returns `(x - mean(x)) / sqrt(variance(x) + epsilon)` with no learned
/// affine scale or bias. A positive finite epsilon is required. This function
/// normalizes a single vector, not a batch or an entire sequence.
pub fn layer_norm(input: &[f64], epsilon: f64) -> Result<Vec<f64>, NnError> {
    if input.is_empty() {
        return Err(NnError::InvalidShape("normalization input cannot be empty"));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(NnError::InvalidEpsilon);
    }
    check_finite(input, "normalization input")?;
    let count = input.len() as f64;
    let mut mean = 0.0;
    for &value in input {
        mean = finite(mean + value / count, "normalization mean")?;
    }
    let mut centered = Vec::with_capacity(input.len());
    let mut variance = 0.0;
    for &value in input {
        let difference = finite(value - mean, "normalization centering")?;
        variance = finite(
            variance + difference * difference / count,
            "normalization variance",
        )?;
        centered.push(difference);
    }
    let denominator = finite(variance + epsilon, "normalization denominator")?.sqrt();
    Ok(centered
        .into_iter()
        .map(|value| value / denominator)
        .collect())
}

/// Sinusoidal position encodings of shape `[length, width]`.
///
/// Coordinates `2i` and `2i+1` use sine and cosine of
/// `position / 10000^(2i/width)`. An odd width simply omits the last cosine.
/// Both dimensions must be positive. Callers explicitly add these encodings
/// to their token vectors; [`TransformerBlock`] does not add them itself.
pub fn sinusoidal_positions(length: usize, width: usize) -> Result<Vec<Vec<f64>>, NnError> {
    if length == 0 || width == 0 {
        return Err(NnError::InvalidShape(
            "position encoding dimensions must be positive",
        ));
    }
    length
        .checked_mul(width)
        .ok_or(NnError::Overflow("position encoding size"))?;
    let mut result = vec![vec![0.0; width]; length];
    for (position, row) in result.iter_mut().enumerate() {
        for (coordinate, value) in row.iter_mut().enumerate() {
            let even_coordinate = coordinate - coordinate % 2;
            let denominator = 10_000.0_f64.powf(even_coordinate as f64 / width as f64);
            let angle = position as f64 / denominator;
            *value = if coordinate % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            };
        }
    }
    Ok(result)
}

/// One small single-head Transformer block for forward demonstrations.
///
/// The sequence is Q/K/V projections, scaled attention, output projection,
/// residual plus layer normalization, two-layer ReLU feedforward network,
/// and another residual plus layer normalization. This uses **post-norm**,
/// with fixed epsilon `1e-5` and no learned normalization scale/bias.
///
/// Parameters are Q, K, V, output projections (each width-to-width), then a
/// `[width, hidden_width, width]` feedforward MLP. Every affine layer uses
/// [`Mlp`]'s neuron-wise weight-then-bias ordering. There is no dropout,
/// multihead split, tokenizer, embedding table, decoder stack, or training.
/// Positional encoding is an explicit separate operation.
#[derive(Clone, Debug)]
pub struct TransformerBlock {
    width: usize,
    query: Mlp,
    key: Mlp,
    value: Mlp,
    output: Mlp,
    feedforward: Mlp,
}

impl TransformerBlock {
    /// Required number of projection and feedforward parameters.
    pub fn parameter_count(width: usize, hidden_width: usize) -> Result<usize, NnError> {
        let projections = dense_count(width, width)?
            .checked_mul(4)
            .ok_or(NnError::Overflow("Transformer projection parameters"))?;
        projections
            .checked_add(Mlp::parameter_count(&[width, hidden_width, width])?)
            .ok_or(NnError::Overflow("Transformer parameter count"))
    }

    /// Construct a block from finite parameters in documented component order.
    pub fn new(width: usize, hidden_width: usize, parameters: Vec<f64>) -> Result<Self, NnError> {
        check_len(
            &parameters,
            Self::parameter_count(width, hidden_width)?,
            "Transformer parameters",
        )?;
        let projection_count = dense_count(width, width)?;
        let mut offset = 0;
        let mut projection = || {
            let model = Mlp::new(
                &[width, width],
                &[Activation::Identity],
                parameters[offset..offset + projection_count].to_vec(),
            );
            offset += projection_count;
            model
        };
        let query = projection()?;
        let key = projection()?;
        let value = projection()?;
        let output = projection()?;
        let feedforward = Mlp::new(
            &[width, hidden_width, width],
            &[Activation::Relu, Activation::Identity],
            parameters[offset..].to_vec(),
        )?;
        Ok(Self {
            width,
            query,
            key,
            value,
            output,
            feedforward,
        })
    }

    /// Width of each input and output token vector.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Compute one block, optionally masking attention to future tokens.
    pub fn forward(&self, tokens: &[Vec<f64>], causal: bool) -> Result<Vec<Vec<f64>>, NnError> {
        let (_, width) = matrix_shape(tokens, "Transformer tokens")?;
        if width != self.width {
            return Err(NnError::InvalidShape(
                "token width must match Transformer width",
            ));
        }
        let project = |model: &Mlp| {
            tokens
                .iter()
                .map(|token| model.forward(token))
                .collect::<Result<Vec<_>, _>>()
        };
        let queries = project(&self.query)?;
        let keys = project(&self.key)?;
        let values = project(&self.value)?;
        let attention = scaled_dot_product_attention(&queries, &keys, &values, causal)?;
        tokens
            .iter()
            .zip(attention.values)
            .map(|(token, attended)| {
                let projected = self.output.forward(&attended)?;
                let residual = checked_sum(token, &projected)?;
                let normalized = layer_norm(&residual, 1.0e-5)?;
                let transformed = self.feedforward.forward(&normalized)?;
                layer_norm(&checked_sum(&normalized, &transformed)?, 1.0e-5)
            })
            .collect()
    }
}

fn checked_sum(left: &[f64], right: &[f64]) -> Result<Vec<f64>, NnError> {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| finite(a + b, "Transformer residual"))
        .collect()
}
