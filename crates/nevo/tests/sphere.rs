//! Numerical and input-validation tests for the Sphere reference problem.

use nevo::Evaluator;
use nevo::benchmarks::{Sphere, SphereError};

#[test]
fn rejects_zero_dimensions() {
    assert_eq!(Sphere::new(0), Err(SphereError::ZeroDimensions));
}

#[test]
fn reports_configured_dimension() {
    assert_eq!(Sphere::new(8).unwrap().dimensions(), 8);
}

#[test]
fn origin_has_zero_objective() {
    for dimensions in 1..=16 {
        let sphere = Sphere::new(dimensions).unwrap();
        assert_eq!(sphere.value(&vec![0.0; dimensions]).unwrap(), 0.0);
    }
}

#[test]
fn ones_have_objective_equal_to_dimension() {
    assert_eq!(Sphere::new(8).unwrap().value(&[1.0; 8]).unwrap(), 8.0);
}

#[test]
fn evaluates_mixed_sign_coordinates() {
    let point = [1.0, -2.0, 3.0, -4.0, 0.0, 0.5, -0.5, 2.0];
    assert_eq!(Sphere::new(8).unwrap().value(&point).unwrap(), 34.5);
}

#[test]
fn sign_reversal_does_not_change_objective() {
    let sphere = Sphere::new(4).unwrap();
    assert_eq!(
        sphere.value(&[1.0, -2.0, 3.0, -4.0]).unwrap(),
        sphere.value(&[-1.0, 2.0, -3.0, 4.0]).unwrap()
    );
}

#[test]
fn doubling_coordinates_quadruples_objective() {
    let sphere = Sphere::new(3).unwrap();
    let original = sphere.value(&[1.0, -2.0, 0.5]).unwrap();
    let doubled = sphere.value(&[2.0, -4.0, 1.0]).unwrap();
    assert_eq!(doubled, 4.0 * original);
}

#[test]
fn initialization_bounds_do_not_restrict_the_objective() {
    assert_eq!(Sphere::new(2).unwrap().value(&[6.0, 8.0]).unwrap(), 100.0);
}

#[test]
fn rejects_wrong_lengths_including_empty_input() {
    let sphere = Sphere::new(2).unwrap();
    for point in [vec![], vec![1.0], vec![1.0; 3]] {
        assert_eq!(
            sphere.value(&point),
            Err(SphereError::DimensionMismatch {
                expected: 2,
                actual: point.len(),
            })
        );
    }
}

#[test]
fn rejects_nan_and_both_infinities() {
    let sphere = Sphere::new(2).unwrap();
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            sphere.value(&[1.0, invalid]),
            Err(SphereError::NonFiniteCoordinate { index: 1 })
        );
    }
}

#[test]
fn rejects_square_overflow() {
    assert_eq!(
        Sphere::new(1).unwrap().value(&[f64::MAX]),
        Err(SphereError::Overflow)
    );
}

#[test]
fn rejects_accumulation_overflow() {
    assert_eq!(
        Sphere::new(2).unwrap().value(&[1.0e154, 1.0e154]),
        Err(SphereError::Overflow)
    );
}

#[test]
fn deterministic_evaluation_ignores_seed() {
    let mut sphere = Sphere::new(2).unwrap();
    let point = vec![3.0, 4.0];
    assert_eq!(
        sphere.evaluate(&point, 0).unwrap(),
        sphere.evaluate(&point, u64::MAX).unwrap()
    );
}

#[test]
fn supports_arrays_slices_and_vectors_without_changing_inputs() {
    let mut sphere = Sphere::new(2).unwrap();
    let array = [3.0, 4.0];
    let vector = array.to_vec();
    let slice: &[f64] = &array;
    assert_eq!(sphere.evaluate(&array, 42).unwrap(), 25.0);
    assert_eq!(sphere.evaluate(&vector, 42).unwrap(), 25.0);
    assert_eq!(sphere.evaluate(slice, 42).unwrap(), 25.0);
    assert_eq!(array, [3.0, 4.0]);
    assert_eq!(vector, vec![3.0, 4.0]);
}

#[test]
fn matches_integer_reference_for_64_deterministic_points() {
    let sphere = Sphere::new(8).unwrap();
    // Exact binary fractions: x[j] = q[j]/8, so f(x) = sum(q[j]^2)/64.
    // These are correlated test fixtures, not uniform random samples.
    for candidate in 0_i32..64 {
        let q: Vec<i32> = (0_i32..8)
            .map(|coordinate| (17 * candidate + 13 * coordinate) % 80 - 40)
            .collect();
        let point: Vec<f64> = q.iter().map(|&value| f64::from(value) / 8.0).collect();
        let exact_numerator: i32 = q.iter().map(|value| value * value).sum();
        assert_eq!(
            sphere.value(&point).unwrap(),
            f64::from(exact_numerator) / 64.0
        );
    }
}

#[test]
fn floating_point_underflow_is_documented_not_hidden_by_an_epsilon() {
    assert_eq!(
        Sphere::new(1).unwrap().value(&[f64::MIN_POSITIVE]).unwrap(),
        0.0
    );
}
