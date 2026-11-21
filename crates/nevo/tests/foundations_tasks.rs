//! Closed-form values and domain checks for the Chapter 2 tasks.

use nevo::benchmarks::{Rastrigin, Zdt1};
use nevo::tasks::Xor;

#[test]
fn rastrigin_known_values_and_domains() {
    let f = Rastrigin::new(2).unwrap();
    assert_eq!(f.value(&[0.0, 0.0]).unwrap(), 0.0);
    assert_eq!(f.value(&[1.0, -2.0]).unwrap(), 5.0);
    assert!((f.value(&[0.5, -0.5]).unwrap() - 40.5).abs() < 1e-12);
    assert!(Rastrigin::new(0).is_err());
    assert!(f.value(&[0.0]).is_err());
    assert!(f.value(&[f64::NAN, 0.0]).is_err());
    assert!(f.value(&[f64::MAX, 0.0]).is_err());
}

#[test]
fn zdt1_known_values_and_domains() {
    let f = Zdt1::new(3).unwrap();
    assert_eq!(f.value(&[0.0, 0.0, 0.0]).unwrap(), vec![0.0, 1.0]);
    assert_eq!(f.value(&[1.0, 0.0, 0.0]).unwrap(), vec![1.0, 0.0]);
    assert_eq!(f.value(&[0.25, 0.0, 0.0]).unwrap(), vec![0.25, 0.5]);
    let values = f.value(&[0.25, 1.0, 1.0]).unwrap();
    assert!((values[1] - 10.0 * (1.0 - (0.025f64).sqrt())).abs() < 1e-12);
    assert!(Zdt1::new(1).is_err());
    assert!(f.value(&[0.0, -0.01, 1.0]).is_err());
    assert!(f.value(&[0.0, f64::INFINITY, 1.0]).is_err());
}

#[test]
fn xor_decodes_and_scores_a_known_solution() {
    let xor = Xor::new(2).unwrap();
    assert_eq!(xor.dimensions(), 9);
    assert_eq!(xor.value(&[0.0; 9]).unwrap(), 0.25);
    // Hidden unit 1 detects OR; unit 2 detects NAND. Both active means XOR.
    let genes = vec![10.0, 10.0, -5.0, -10.0, -10.0, 15.0, 10.0, 10.0, -10.0];
    assert!(xor.value(&genes).unwrap() < 1e-7);
    assert!(xor.value(&[0.0; 8]).is_err());
    assert!(Xor::new(0).is_err());
}
