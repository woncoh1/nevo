//! Numerical reference tests for the Chapter 2 neural building blocks.

use nevo::nn::{
    Activation, LstmCell, LstmState, Mlp, NnError, RnnCell, TrainingExample, TransformerBlock,
    conv2d_valid, layer_norm, scaled_dot_product_attention, sinusoidal_positions,
};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

#[test]
fn mlp_flat_encoding_and_affine_forward_have_known_values() {
    assert_eq!(Mlp::parameter_count(&[2, 2, 1]).unwrap(), 9);
    let model = Mlp::new(
        &[2, 2, 1],
        &[Activation::Identity, Activation::Identity],
        vec![1.0, 2.0, 0.5, -1.0, 0.5, -0.5, 2.0, -3.0, 1.0],
    )
    .unwrap();
    assert_eq!(model.forward(&[2.0, 3.0]).unwrap(), [21.0]);
    assert_eq!(model.widths(), [2, 2, 1]);
}

#[test]
fn sigmoid_is_stable_for_large_positive_and_negative_inputs() {
    let model = Mlp::new(
        &[1, 3],
        &[Activation::Sigmoid],
        vec![1000.0, 0.0, 0.0, 0.0, -1000.0, 0.0],
    )
    .unwrap();
    assert_eq!(model.forward(&[1.0]).unwrap(), [1.0, 0.5, 0.0]);
}

#[test]
fn neural_shapes_and_nonfinite_parameters_are_rejected() {
    for widths in [&[][..], &[1][..], &[0, 2][..], &[2, 0][..]] {
        assert!(Mlp::parameter_count(widths).is_err());
    }
    assert!(Mlp::parameter_count(&[usize::MAX, 1]).is_err());
    assert!(Mlp::new(&[1, 1], &[], vec![0.0, 0.0]).is_err());
    assert!(Mlp::new(&[1, 1], &[Activation::Identity], vec![0.0]).is_err());
    assert!(Mlp::new(&[1, 1], &[Activation::Identity], vec![f64::NAN, 0.0]).is_err());
}

#[test]
fn parameter_replacement_is_validated_before_mutation() {
    let mut model = Mlp::new(&[1, 1], &[Activation::Identity], vec![2.0, 1.0]).unwrap();
    assert!(model.set_parameters(&[f64::INFINITY, 3.0]).is_err());
    assert!(model.set_parameters(&[3.0]).is_err());
    assert_eq!(model.parameters(), [2.0, 1.0]);
    model.set_parameters(&[3.0, -1.0]).unwrap();
    assert_eq!(model.forward(&[2.0]).unwrap(), [5.0]);
}

#[test]
fn mlp_rejects_bad_inputs_and_affine_overflow() {
    let model = Mlp::new(&[1, 1], &[Activation::Identity], vec![2.0, 0.0]).unwrap();
    assert!(model.forward(&[]).is_err());
    assert!(model.forward(&[f64::NEG_INFINITY]).is_err());
    assert!(matches!(
        model.forward(&[f64::MAX]),
        Err(NnError::Overflow(_))
    ));
}

#[test]
fn mse_and_gradient_average_over_examples_and_output_coordinates() {
    let model = Mlp::new(&[1, 2], &[Activation::Identity], vec![1.0, 0.0, 2.0, 0.0]).unwrap();
    let examples = vec![
        TrainingExample {
            input: vec![2.0],
            target: vec![0.0, 0.0],
        },
        TrainingExample {
            input: vec![0.0],
            target: vec![0.0, 0.0],
        },
    ];
    let (loss, gradient) = model.mse_gradient(&examples).unwrap();
    assert_eq!(loss, 5.0);
    assert_eq!(model.mse(&examples).unwrap(), 5.0);
    assert_eq!(gradient, [2.0, 1.0, 4.0, 2.0]);
}

#[test]
fn analytic_backprop_matches_central_finite_differences() {
    let count = Mlp::parameter_count(&[2, 3, 2]).unwrap();
    let parameters = (0..count)
        .map(|i| (i as f64 - 8.0) * 0.07)
        .collect::<Vec<_>>();
    let mut model = Mlp::new(
        &[2, 3, 2],
        &[Activation::Tanh, Activation::Sigmoid],
        parameters.clone(),
    )
    .unwrap();
    let examples = vec![
        TrainingExample {
            input: vec![0.2, -0.5],
            target: vec![0.7, 0.2],
        },
        TrainingExample {
            input: vec![-0.4, 0.7],
            target: vec![0.0, 0.9],
        },
        TrainingExample {
            input: vec![0.1, 0.8],
            target: vec![0.5, 0.8],
        },
    ];
    let (loss, gradient) = model.mse_gradient(&examples).unwrap();
    close(loss, model.mse(&examples).unwrap(), 1.0e-14);
    let epsilon = 1.0e-6;
    for (index, &analytic) in gradient.iter().enumerate() {
        let mut perturbed = parameters.clone();
        perturbed[index] += epsilon;
        model.set_parameters(&perturbed).unwrap();
        let plus = model.mse(&examples).unwrap();
        perturbed[index] -= 2.0 * epsilon;
        model.set_parameters(&perturbed).unwrap();
        let minus = model.mse(&examples).unwrap();
        close(analytic, (plus - minus) / (2.0 * epsilon), 3.0e-8);
    }
}

#[test]
fn relu_gradient_uses_zero_derivative_at_the_kink() {
    let model = Mlp::new(&[1, 1], &[Activation::Relu], vec![0.0, 0.0]).unwrap();
    let examples = [TrainingExample {
        input: vec![1.0],
        target: vec![1.0],
    }];
    assert_eq!(
        model.mse_gradient(&examples).unwrap(),
        (1.0, vec![0.0, 0.0])
    );
}

#[test]
fn gradient_step_returns_pre_step_loss_and_changes_parameters_correctly() {
    let mut model = Mlp::new(&[1, 1], &[Activation::Identity], vec![0.0, 0.0]).unwrap();
    let examples = [
        TrainingExample {
            input: vec![1.0],
            target: vec![2.0],
        },
        TrainingExample {
            input: vec![2.0],
            target: vec![4.0],
        },
    ];
    assert_eq!(model.train_step(&examples, 0.1).unwrap(), 10.0);
    close(model.parameters()[0], 1.0, 1.0e-14);
    close(model.parameters()[1], 0.6, 1.0e-14);
    close(model.mse(&examples).unwrap(), 1.06, 1.0e-14);
}

#[test]
fn failed_gradient_step_leaves_parameters_unchanged() {
    let mut model = Mlp::new(&[1, 1], &[Activation::Identity], vec![0.0, 0.0]).unwrap();
    let examples = [TrainingExample {
        input: vec![2.0],
        target: vec![4.0],
    }];
    for rate in [0.0, -1.0, f64::INFINITY, f64::NAN, f64::MAX] {
        assert!(model.train_step(&examples, rate).is_err());
        assert_eq!(model.parameters(), [0.0, 0.0]);
    }
}

#[test]
fn supervised_data_requires_nonempty_matching_finite_examples() {
    let model = Mlp::new(&[1, 1], &[Activation::Identity], vec![1.0, 0.0]).unwrap();
    assert!(model.mse(&[]).is_err());
    for example in [
        TrainingExample {
            input: vec![],
            target: vec![0.0],
        },
        TrainingExample {
            input: vec![0.0],
            target: vec![0.0, 0.0],
        },
        TrainingExample {
            input: vec![0.0],
            target: vec![f64::NAN],
        },
    ] {
        assert!(model.mse_gradient(&[example]).is_err());
    }
}

#[test]
fn rnn_propagates_explicit_state_according_to_tanh_recurrence() {
    let model = RnnCell::new(1, 1, vec![1.0, 0.5, 0.0]).unwrap();
    let initial = model.zero_state();
    let first = model.step(&[1.0], &initial).unwrap();
    let second = model.step(&[0.0], &first).unwrap();
    close(first[0], 1.0_f64.tanh(), 1.0e-14);
    close(second[0], (0.5 * 1.0_f64.tanh()).tanh(), 1.0e-14);
    assert_eq!(initial, [0.0]);
    assert!(model.step(&[1.0], &[]).is_err());
    assert!(RnnCell::parameter_count(0, 1).is_err());
}

#[test]
fn zero_lstm_gates_have_a_hand_computable_memory_update() {
    let model = LstmCell::new(1, 1, vec![0.0; 12]).unwrap();
    let previous = LstmState {
        hidden: vec![0.0],
        cell: vec![2.0],
    };
    let next = model.step(&[0.0], &previous).unwrap();
    assert_eq!(next.cell, [1.0]);
    close(next.hidden[0], 0.5 * 1.0_f64.tanh(), 1.0e-14);
    assert_eq!(previous.cell, [2.0]);
}

#[test]
fn lstm_gate_encoding_and_state_checks_are_explicit() {
    let mut parameters = vec![0.0; LstmCell::parameter_count(1, 1).unwrap()];
    parameters[8] = 1.0; // Candidate-gate bias in i, f, g, o ordering.
    let model = LstmCell::new(1, 1, parameters).unwrap();
    let next = model.step(&[0.0], &model.zero_state()).unwrap();
    close(next.cell[0], 0.5 * 1.0_f64.tanh(), 1.0e-14);
    close(next.hidden[0], 0.5 * next.cell[0].tanh(), 1.0e-14);
    let invalid = LstmState {
        hidden: vec![0.0],
        cell: vec![f64::NAN],
    };
    assert!(model.step(&[0.0], &invalid).is_err());
    assert!(LstmState::zeros(0).is_err());
    assert!(LstmCell::parameter_count(usize::MAX, 1).is_err());
}

#[test]
fn convolution_is_cross_correlation_without_kernel_reversal() {
    let image = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let kernel = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    assert_eq!(
        conv2d_valid(&image, &kernel, 0.5).unwrap(),
        [vec![37.5, 47.5], vec![67.5, 77.5]]
    );
}

#[test]
fn convolution_rejects_ragged_oversized_and_overflowing_inputs() {
    assert!(conv2d_valid(&[vec![1.0], vec![2.0, 3.0]], &[vec![1.0]], 0.0).is_err());
    assert!(conv2d_valid(&[vec![1.0]], &[vec![1.0, 2.0]], 0.0).is_err());
    assert!(conv2d_valid(&[vec![1.0]], &[vec![f64::NAN]], 0.0).is_err());
    assert!(conv2d_valid(&[vec![f64::MAX]], &[vec![2.0]], 0.0).is_err());
}

#[test]
fn uniform_attention_returns_value_means() {
    let result = scaled_dot_product_attention(
        &[vec![0.0], vec![0.0]],
        &[vec![0.0], vec![0.0]],
        &[vec![2.0, 4.0], vec![6.0, 8.0]],
        false,
    )
    .unwrap();
    assert_eq!(result.weights, [vec![0.5, 0.5], vec![0.5, 0.5]]);
    assert_eq!(result.values, [vec![4.0, 6.0], vec![4.0, 6.0]]);
}

#[test]
fn causal_attention_masks_future_values_and_renormalizes_each_prefix() {
    let zeros = vec![vec![0.0]; 3];
    let result =
        scaled_dot_product_attention(&zeros, &zeros, &[vec![2.0], vec![4.0], vec![6.0]], true)
            .unwrap();
    assert_eq!(result.values, [vec![2.0], vec![3.0], vec![4.0]]);
    assert_eq!(result.weights[0], [1.0, 0.0, 0.0]);
    assert_eq!(result.weights[1], [0.5, 0.5, 0.0]);
    for row in &result.weights {
        close(row.iter().sum(), 1.0, 1.0e-14);
    }
}

#[test]
fn attention_uses_sqrt_dimension_scaling_and_stable_softmax() {
    let result = scaled_dot_product_attention(
        &[vec![1.0, 0.0]],
        &[vec![1.0, 0.0], vec![0.0, 1.0]],
        &[vec![10.0], vec![0.0]],
        false,
    )
    .unwrap();
    let exponential = (1.0 / 2.0_f64.sqrt()).exp();
    close(
        result.values[0][0],
        10.0 * exponential / (exponential + 1.0),
        1.0e-12,
    );
    let large = scaled_dot_product_attention(
        &[vec![1000.0]],
        &[vec![1000.0], vec![999.0]],
        &[vec![10.0], vec![0.0]],
        false,
    )
    .unwrap();
    assert_eq!(large.weights, [vec![1.0, 0.0]]);
    assert_eq!(large.values, [vec![10.0]]);
}

#[test]
fn attention_checks_matrix_shapes_and_nonfinite_values() {
    let one = [vec![1.0]];
    assert!(scaled_dot_product_attention(&[], &one, &one, false).is_err());
    assert!(scaled_dot_product_attention(&[vec![1.0, 2.0]], &one, &one, false).is_err());
    assert!(scaled_dot_product_attention(&one, &[vec![1.0], vec![2.0]], &one, false).is_err());
    assert!(scaled_dot_product_attention(&[vec![f64::NAN]], &one, &one, false).is_err());
    assert!(
        scaled_dot_product_attention(&one, &[vec![1.0], vec![2.0]], &[vec![1.0], vec![2.0]], true)
            .is_err()
    );
}

#[test]
fn layer_norm_uses_population_variance_and_handles_constant_vectors() {
    let epsilon = 1.0e-6;
    let output = layer_norm(&[1.0, 3.0], epsilon).unwrap();
    close(output[0], -1.0 / (1.0_f64 + epsilon).sqrt(), 1.0e-14);
    close(output[1], 1.0 / (1.0_f64 + epsilon).sqrt(), 1.0e-14);
    assert_eq!(layer_norm(&[5.0; 3], epsilon).unwrap(), [0.0; 3]);
    assert!(layer_norm(&[1.0], 0.0).is_err());
    assert!(layer_norm(&[], epsilon).is_err());
}

#[test]
fn sinusoidal_positions_match_known_even_and_odd_coordinates() {
    let positions = sinusoidal_positions(2, 4).unwrap();
    assert_eq!(positions[0], [0.0, 1.0, 0.0, 1.0]);
    close(positions[1][0], 1.0_f64.sin(), 1.0e-14);
    close(positions[1][1], 1.0_f64.cos(), 1.0e-14);
    close(positions[1][2], 0.01_f64.sin(), 1.0e-14);
    close(positions[1][3], 0.01_f64.cos(), 1.0e-14);
    assert_eq!(sinusoidal_positions(1, 3).unwrap()[0], [0.0, 1.0, 0.0]);
    assert!(sinusoidal_positions(0, 4).is_err());
    assert!(sinusoidal_positions(2, 0).is_err());
}

#[test]
fn transformer_block_preserves_shape_and_causal_prefix_independence() {
    let count = TransformerBlock::parameter_count(3, 4).unwrap();
    let parameters = (0..count)
        .map(|index| (index as f64 * 0.31).sin() * 0.3)
        .collect();
    let model = TransformerBlock::new(3, 4, parameters).unwrap();
    let original = vec![vec![0.1, -0.7, 0.9], vec![2.0, -3.0, 4.0]];
    let changed = vec![original[0].clone(), vec![100.0, 300.0, -50.0]];
    let first = model.forward(&original, true).unwrap();
    let second = model.forward(&changed, true).unwrap();
    assert_eq!(first.len(), 2);
    assert!(
        first
            .iter()
            .all(|row| row.len() == 3 && row.iter().all(|x| x.is_finite()))
    );
    assert_eq!(first[0], second[0]);
    for row in &first {
        close(row.iter().sum::<f64>(), 0.0, 1.0e-12);
    }
    assert!(model.forward(&[vec![0.0; 2]], false).is_err());
}

#[test]
fn zero_transformer_sublayers_still_pass_through_normalized_residuals() {
    let model = TransformerBlock::new(
        2,
        3,
        vec![0.0; TransformerBlock::parameter_count(2, 3).unwrap()],
    )
    .unwrap();
    let result = model.forward(&[vec![1.0, 3.0]], false).unwrap();
    let epsilon = 1.0e-5;
    let first_magnitude = 1.0 / (1.0_f64 + epsilon).sqrt();
    let expected = first_magnitude / (first_magnitude * first_magnitude + epsilon).sqrt();
    close(result[0][0], -expected, 1.0e-14);
    close(result[0][1], expected, 1.0e-14);
}
