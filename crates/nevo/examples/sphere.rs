//! Deterministic reference evaluations, NOT an optimization experiment.
//!
//! Run: cargo run -p nevo --example sphere --locked

use nevo::Evaluator;
use nevo::benchmarks::Sphere;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sphere = Sphere::new(8)?;
    let seed = 42;
    let probes = [
        ("origin", vec![0.0; 8], 0.0),
        ("ones", vec![1.0; 8], 8.0),
        ("minus_fives", vec![-5.0; 8], 200.0),
        (
            "mixed",
            vec![1.0, -2.0, 3.0, -4.0, 0.0, 0.5, -0.5, 2.0],
            34.5,
        ),
    ];

    println!("Sphere reference benchmark: dimensions=8, objective=minimize");
    println!("Evaluation seed={seed}; ignored by this deterministic evaluator.");
    println!("probe,objective,expected");
    for (name, point, expected) in probes {
        let objective = sphere.evaluate(&point, seed)?;
        assert_eq!(objective, expected, "reference probe failed: {name}");
        println!("{name},{objective:.6},{expected:.6}");
    }
    println!("All reference probes passed. No optimizer was run.");
    Ok(())
}
