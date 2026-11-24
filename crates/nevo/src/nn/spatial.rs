use super::{NnError, check_finite, finite, matrix_shape};

/// One-channel valid max pooling with an explicit window and stride.
///
/// `window` and `stride` are `[height, width]`, with strictly positive entries.
/// Only complete windows are used: output size is
/// `[(H - Kh) / Sh + 1, (W - Kw) / Sw + 1]` (integer division).
/// There is no padding, implicit activation, or backward pass. This is the
/// local max operation of Chapter 2, equation 2.24.
pub fn max_pool2d(
    input: &[Vec<f64>],
    window: [usize; 2],
    stride: [usize; 2],
) -> Result<Vec<Vec<f64>>, NnError> {
    let (height, width) = matrix_shape(input, "pooling input")?;
    if window.contains(&0) || stride.contains(&0) {
        return Err(NnError::InvalidShape(
            "pooling window and stride must be positive",
        ));
    }
    if window[0] > height || window[1] > width {
        return Err(NnError::InvalidShape(
            "pooling window must fit inside the input",
        ));
    }
    let rows = (height - window[0]) / stride[0] + 1;
    let columns = (width - window[1]) / stride[1] + 1;
    let mut output = vec![vec![0.0; columns]; rows];
    for (row_index, row) in output.iter_mut().enumerate() {
        let top = row_index * stride[0];
        for (column_index, value) in row.iter_mut().enumerate() {
            let left = column_index * stride[1];
            *value = input[top..top + window[0]]
                .iter()
                .flat_map(|window_row| &window_row[left..left + window[1]])
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
        }
    }
    Ok(output)
}

/// One-channel, one-filter, valid 2D CNN cross-correlation with stride one.
///
/// Input and kernel are nonempty rectangular row-major matrices. The kernel
/// must fit inside the input; output dimensions are `(H-Kh+1, W-Kw+1)`.
/// As in conventional CNN layers, the kernel is **not reversed**. No padding,
/// dilation, channels, pooling, or training is implicit in this operation.
pub fn conv2d_valid(
    input: &[Vec<f64>],
    kernel: &[Vec<f64>],
    bias: f64,
) -> Result<Vec<Vec<f64>>, NnError> {
    let (height, width) = matrix_shape(input, "convolution input")?;
    let (kernel_height, kernel_width) = matrix_shape(kernel, "convolution kernel")?;
    check_finite(&[bias], "convolution bias")?;
    if kernel_height > height || kernel_width > width {
        return Err(NnError::InvalidShape("kernel must fit inside the input"));
    }
    let mut result = vec![vec![0.0; width - kernel_width + 1]; height - kernel_height + 1];
    for (row_index, row) in result.iter_mut().enumerate() {
        for (column_index, value) in row.iter_mut().enumerate() {
            *value = bias;
            for (kernel_row, weights) in kernel.iter().enumerate() {
                for (kernel_column, &weight) in weights.iter().enumerate() {
                    *value = finite(
                        *value
                            + input[row_index + kernel_row][column_index + kernel_column] * weight,
                        "convolution",
                    )?;
                }
            }
        }
    }
    Ok(result)
}
