//! Regression data captured from the unmodified v4 feedforward implementation.

use nevo::algorithms::{Neat, NeatConfig};
use nevo::neat::{Genome, NodeKind};
use nevo::{Algorithm, Evaluated, Job};

fn fingerprint(jobs: &[Job<Genome>]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut add = |word: u64| {
        for byte in word.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    };
    for job in jobs {
        add(job.id.0);
        add(job.seed);
        add(job.input.nodes().len() as u64);
        for node in job.input.nodes() {
            add(node.id);
            add(match node.kind {
                NodeKind::Input => 0,
                NodeKind::Bias => 1,
                NodeKind::Hidden => 2,
                NodeKind::Output => 3,
            });
        }
        add(job.input.connections().len() as u64);
        for link in job.input.connections() {
            for word in [
                link.innovation,
                link.from,
                link.to,
                link.weight.to_bits(),
                u64::from(link.enabled),
            ] {
                add(word);
            }
        }
    }
    hash
}

#[test]
fn feedforward_jobs_remain_identical_to_the_v4_baseline() {
    let mut config = NeatConfig::new(2, 1, 1977);
    config.population = 16;
    config.mutation.add_node_probability = 0.4;
    config.mutation.add_connection_probability = 0.6;
    config.compatibility_threshold = 1.4;
    config.crossover_probability = 0.9;
    let mut search = Neat::new(config).unwrap();
    // Generated from the unmodified v4 starter before the recurrent patch.
    // Each FNV-1a fingerprint includes all job IDs/seeds and bit-exact node,
    // innovation, endpoint, weight, and enable data for a complete batch.
    // Structural mutations and crossover are deliberately frequent so a new
    // branch or RNG draw cannot silently change the default feedforward run.
    let expected = [
        0xbb1afe20237e0782,
        0x9d19d1db5899d3cb,
        0x28c8f52f0fbc0aed,
        0xbe629176cef93459,
        0x9e9f76cc45b8cf4f,
        0x147ea03a25e83b46,
        0xc4954c65bd4510a8,
        0xbb50f7df031acc5f,
        0x3395572925e6910c,
        0x9d6cbafb1384d11c,
        0x9488970f59c740e1,
        0xa5ce392edf769742,
    ];
    for expected_batch in expected {
        let jobs = search.generate().unwrap();
        assert!(
            jobs.iter()
                .all(|job| job.input.mode() == nevo::neat::NetworkMode::FeedForward)
        );
        assert_eq!(fingerprint(&jobs), expected_batch);
        search
            .select(
                jobs.into_iter()
                    .map(|job| {
                        let output = 1.0 + job.input.enabled_connection_count() as f64;
                        Evaluated { job, output }
                    })
                    .collect(),
            )
            .unwrap();
    }
}
