use super::{NnError, check_finite, finite, matrix_shape};

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
