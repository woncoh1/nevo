//! Equation and shape checks for the additional Chapter 2 forward primitives.

use nevo::nn::{NnError, max_pool2d, multi_head_attention, scaled_dot_product_attention, softmax};

fn near(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

#[test]
fn softmax_known_probabilities_and_translation_invariance() {
    let actual = softmax(&[0.0, 3.0_f64.ln()]).unwrap();
    near(actual[0], 0.25);
    near(actual[1], 0.75);
    let shifted = softmax(&[1000.0, 1000.0 + 3.0_f64.ln()]).unwrap();
    for (a, b) in actual.iter().zip(shifted) {
        near(*a, b);
    }
    near(actual.iter().sum(), 1.0);
    assert_eq!(softmax(&[f64::MAX, -f64::MAX]).unwrap(), vec![1.0, 0.0]);
    assert_eq!(softmax(&[3.0]).unwrap(), vec![1.0]);
}

#[test]
fn softmax_rejects_empty_and_nonfinite_logits() {
    assert!(matches!(softmax(&[]), Err(NnError::InvalidShape(_))));
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            softmax(&[0.0, x]),
            Err(NnError::NonFinite { index: 1, .. })
        ));
    }
}

#[test]
fn pooling_uses_actual_negative_maxima_and_only_complete_windows() {
    let input = vec![
        vec![-1.0, -2.0, -3.0, -4.0, 100.0],
        vec![-5.0, -6.0, -7.0, -8.0, 200.0],
        vec![-9.0, -10.0, -11.0, -12.0, 300.0],
    ];
    assert_eq!(
        max_pool2d(&input, [2, 2], [2, 2]).unwrap(),
        vec![vec![-1.0, -3.0]]
    );
    assert_eq!(
        max_pool2d(&input, [2, 3], [1, 2]).unwrap(),
        vec![vec![-1.0, 200.0], vec![-5.0, 300.0]]
    );
    assert_eq!(
        max_pool2d(&input, [3, 5], [usize::MAX, usize::MAX]).unwrap(),
        vec![vec![300.0]]
    );
}

#[test]
fn pooling_rejects_invalid_shapes_and_nonfinite_input() {
    let input = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    for (window, stride) in [([0, 1], [1, 1]), ([1, 1], [0, 1]), ([3, 1], [1, 1])] {
        assert!(max_pool2d(&input, window, stride).is_err());
    }
    assert!(max_pool2d(&[vec![1.0], vec![]], [1, 1], [1, 1]).is_err());
    assert!(max_pool2d(&[vec![f64::NAN]], [1, 1], [1, 1]).is_err());
}

#[test]
fn two_heads_scale_separately_and_project_in_concatenation_order() {
    // Each head has two key features. Division by sqrt(2), rather than
    // sqrt(total width=4), produces the hand-computed 3:1 probability ratios.
    let q = vec![vec![2.0_f64.sqrt(), 0.0, 2.0_f64.sqrt(), 0.0]];
    let k = vec![
        vec![3.0_f64.ln(), 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 3.0_f64.ln(), 0.0],
    ];
    let v = vec![vec![4.0, 10.0], vec![0.0, 2.0]];
    let projection = vec![vec![1.0, 2.0, 0.5], vec![-1.0, 1.0, 2.0]];
    let actual = multi_head_attention(&q, &k, &v, 2, &projection, false).unwrap();
    near(actual.weights[0][0][0], 0.75);
    near(actual.weights[1][0][0], 0.25);
    // A rectangular projection maps two concatenated values to three outputs.
    // [3,4] * [[1,2,0.5],[-1,1,2]] = [-1,10,9.5].
    assert_eq!(actual.values[0].len(), 3);
    near(actual.values[0][0], -1.0);
    near(actual.values[0][1], 10.0);
    near(actual.values[0][2], 9.5);
}

#[test]
fn one_head_with_identity_projection_matches_scaled_attention() {
    let q = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
    let k = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let v = vec![vec![2.0, 3.0], vec![5.0, 7.0]];
    let single = scaled_dot_product_attention(&q, &k, &v, false).unwrap();
    let multi =
        multi_head_attention(&q, &k, &v, 1, &[vec![1.0, 0.0], vec![0.0, 1.0]], false).unwrap();
    assert_eq!(multi.values, single.values);
    assert_eq!(multi.weights, vec![single.weights]);
}

#[test]
fn causal_mask_hides_future_keys_in_every_head() {
    let q = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let projection = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let a = multi_head_attention(
        &q,
        &q,
        &[vec![3.0, 5.0], vec![10.0, 20.0]],
        2,
        &projection,
        true,
    )
    .unwrap();
    let b = multi_head_attention(
        &q,
        &[vec![1.0, 1.0], vec![1000.0, -1000.0]],
        &[vec![3.0, 5.0], vec![-90.0, 90.0]],
        2,
        &projection,
        true,
    )
    .unwrap();
    assert_eq!(a.values[0], vec![3.0, 5.0]);
    assert_eq!(a.values[0], b.values[0]);
    for head in &a.weights {
        assert_eq!(head[0], vec![1.0, 0.0]);
    }
}

#[test]
fn multi_head_checks_shapes_and_projection_overflow() {
    let x = vec![vec![1.0, 2.0]];
    let projection = vec![vec![1.0], vec![1.0]];
    assert!(multi_head_attention(&x, &x, &x, 0, &projection, false).is_err());
    assert!(multi_head_attention(&x, &x, &x, 3, &projection, false).is_err());
    assert!(multi_head_attention(&x, &x, &x, 1, &[vec![1.0]], false).is_err());
    assert!(multi_head_attention(&x, &x, &[vec![1.0]], 2, &[vec![1.0]], false).is_err());
    assert!(
        multi_head_attention(
            &x,
            &[vec![1.0, 1.0], vec![2.0, 2.0]],
            &[vec![1.0, 1.0], vec![2.0, 2.0]],
            2,
            &projection,
            true
        )
        .is_err()
    );
    assert!(matches!(
        multi_head_attention(
            &[vec![0.0]],
            &[vec![0.0]],
            &[vec![f64::MAX]],
            1,
            &[vec![2.0]],
            false
        ),
        Err(NnError::Overflow(_))
    ));
}
