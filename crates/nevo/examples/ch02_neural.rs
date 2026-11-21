//! Inspect the Chapter 2 neural building blocks through the public API.
//!
//! These are small numerical demonstrations, not pretrained models or a
//! reconstruction of the book's experiments. MLP training uses analytic
//! backpropagation; the other architectures demonstrate forward computation.

use std::error::Error;

use nevo::nn::{
    Activation, LstmCell, LstmState, Mlp, RnnCell, TrainingExample, TransformerBlock, conv2d_valid,
    layer_norm, scaled_dot_product_attention, sinusoidal_positions,
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Chapter 2: scalar neural building blocks");

    let affine = Mlp::new(&[2, 1], &[Activation::Identity], vec![2.0, -1.0, 0.5])?;
    println!(
        "Affine MLP: 2*3 - 4 + 0.5 = {:?}",
        affine.forward(&[3.0, 4.0])?
    );

    let mut learner = Mlp::new(&[1, 1], &[Activation::Identity], vec![0.0, 0.0])?;
    let data = [
        TrainingExample {
            input: vec![1.0],
            target: vec![2.0],
        },
        TrainingExample {
            input: vec![2.0],
            target: vec![4.0],
        },
    ];
    let (_, gradient) = learner.mse_gradient(&data)?;
    let before = learner.train_step(&data, 0.1)?;
    println!("Analytic MSE gradient [weight, bias]: {gradient:?}");
    println!(
        "One gradient step: loss {before:.6} -> {:.6}",
        learner.mse(&data)?
    );
    println!("Updated [weight, bias]: {:?}", learner.parameters());

    let recurrent = RnnCell::new(1, 1, vec![1.0, 0.5, 0.0])?;
    let first = recurrent.step(&[1.0], &recurrent.zero_state())?;
    let second = recurrent.step(&[0.0], &first)?;
    println!("RNN h'=tanh(x+0.5*h): {first:?} -> {second:?}");

    let lstm = LstmCell::new(1, 1, vec![0.0; LstmCell::parameter_count(1, 1)?])?;
    let previous = LstmState {
        hidden: vec![0.0],
        cell: vec![2.0],
    };
    let next = lstm.step(&[0.0], &previous)?;
    println!("Zero LSTM gates: i=f=o=0.5, g=0; next state {next:?}");

    let image = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let kernel = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    println!(
        "Valid CNN cross-correlation, bias=0.5: {:?}",
        conv2d_valid(&image, &kernel, 0.5)?
    );

    let zero_queries_and_keys = vec![vec![0.0]; 3];
    let values = vec![vec![2.0], vec![4.0], vec![6.0]];
    for causal in [false, true] {
        let attended = scaled_dot_product_attention(
            &zero_queries_and_keys,
            &zero_queries_and_keys,
            &values,
            causal,
        )?;
        println!("Attention causal={causal}: values={:?}", attended.values);
        println!("Attention weights: {:?}", attended.weights);
    }
    println!(
        "Layer normalization [1,3]: {:?}",
        layer_norm(&[1.0, 3.0], 1.0e-5)?
    );
    println!(
        "Sinusoidal positions (2 tokens, width 4): {:?}",
        sinusoidal_positions(2, 4)?
    );

    let count = TransformerBlock::parameter_count(3, 4)?;
    let fixed_parameters = (0..count)
        .map(|index| (index as f64 * 0.31).sin() * 0.3)
        .collect();
    let transformer = TransformerBlock::new(3, 4, fixed_parameters)?;
    let tokens = vec![vec![0.1, -0.7, 0.9], vec![2.0, -3.0, 4.0]];
    println!("Single-head post-norm Transformer with fixed illustrative weights:");
    for token in transformer.forward(&tokens, true)? {
        println!("  {token:?}");
    }
    println!("Forward references only for RNN/LSTM/CNN/Transformer; no training was run for them.");
    Ok(())
}
