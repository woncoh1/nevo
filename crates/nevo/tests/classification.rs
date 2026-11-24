//! Dataset, split, and phenotype checks independent of evolutionary success.

use nevo::tasks::{ClassificationData, LabeledSample, MlpClassifier};
use std::collections::BTreeSet;

#[test]
fn optdigits_parser_checks_pixels_labels_and_shape() {
    let row = format!("{},7", vec!["16"; 64].join(","));
    let data = ClassificationData::from_optdigits_csv(&row).unwrap();
    assert_eq!(data.features(), 64);
    assert_eq!(data.classes(), 10);
    assert_eq!(data.samples()[0].input, vec![1.0; 64]);
    assert_eq!(data.samples()[0].label, 7);
    assert!(ClassificationData::from_optdigits_csv("").is_err());
    assert!(ClassificationData::from_optdigits_csv("1,2,3").is_err());
    assert!(ClassificationData::from_optdigits_csv(&row.replacen("16", "17", 1)).is_err());
    assert!(ClassificationData::from_optdigits_csv(&row.replacen("16", "NaN", 1)).is_err());
    assert!(
        ClassificationData::from_optdigits_csv(&format!("{},10", vec!["0"; 64].join(","))).is_err()
    );
}

#[test]
fn stratified_split_is_disjoint_complete_balanced_and_reproducible() {
    let data = ClassificationData::new(
        (0..30)
            .map(|id| LabeledSample {
                id,
                input: vec![id as f64],
                label: id % 3,
            })
            .collect(),
        3,
    )
    .unwrap();
    let split = data.stratified_split(3, 2, 42).unwrap();
    assert_eq!(
        (
            split.training.len(),
            split.validation.len(),
            split.test.len()
        ),
        (9, 6, 15)
    );
    let sets: Vec<BTreeSet<_>> = [&split.training, &split.validation, &split.test]
        .iter()
        .map(|part| part.samples().iter().map(|r| r.id).collect())
        .collect();
    assert!(
        sets[0].is_disjoint(&sets[1])
            && sets[0].is_disjoint(&sets[2])
            && sets[1].is_disjoint(&sets[2])
    );
    assert_eq!(
        sets.iter().flatten().copied().collect::<BTreeSet<_>>(),
        (0..30).collect()
    );
    for class in 0..3 {
        assert_eq!(
            split
                .training
                .samples()
                .iter()
                .filter(|r| r.label == class)
                .count(),
            3
        );
        assert_eq!(
            split
                .validation
                .samples()
                .iter()
                .filter(|r| r.label == class)
                .count(),
            2
        );
    }
    assert_eq!(
        split.training.samples(),
        data.stratified_split(3, 2, 42).unwrap().training.samples()
    );
    assert_ne!(
        split.training.samples(),
        data.stratified_split(3, 2, 43).unwrap().training.samples()
    );
    assert!(data.stratified_split(9, 1, 42).is_err());
    assert!(data.stratified_split(0, 1, 42).is_err());
}

#[test]
fn classifier_averages_over_outputs_and_uses_stable_argmax_ties() {
    let data = ClassificationData::new(
        vec![
            LabeledSample {
                id: 0,
                input: vec![0.0],
                label: 0,
            },
            LabeledSample {
                id: 1,
                input: vec![1.0],
                label: 1,
            },
        ],
        2,
    )
    .unwrap();
    let task = MlpClassifier::new(data, &[]).unwrap();
    assert_eq!(task.dimensions(), 4);
    let metrics = task.metrics(&[0.0; 4]).unwrap();
    assert_eq!(metrics.mse, 0.25);
    assert_eq!(metrics.accuracy, 0.5);
    assert_eq!(metrics.samples, 2);
    let accurate = task.metrics(&[-20.0, 10.0, 20.0, -10.0]).unwrap();
    assert_eq!(accurate.accuracy, 1.0);
    assert!(accurate.mse < 1e-7);
    assert!(task.metrics(&[0.0; 3]).is_err());
    assert!(task.metrics(&[f64::NAN; 4]).is_err());
}

#[test]
fn bundled_dataset_has_expected_domain_and_reproducible_teaching_split() {
    let data = ClassificationData::from_optdigits_csv(include_str!("../../../data/optdigits.csv"))
        .unwrap();
    assert_eq!(data.len(), 1797);
    let split = data.stratified_split(60, 30, 2026).unwrap();
    assert_eq!(
        (
            split.training.len(),
            split.validation.len(),
            split.test.len()
        ),
        (600, 300, 897)
    );
    let task = MlpClassifier::new(split.training, &[8]).unwrap();
    assert_eq!(task.widths(), &[64, 8, 10]);
    assert_eq!(task.dimensions(), 610);
}
