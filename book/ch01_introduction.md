# Chapter 1 companion: evolve a neural controller

This original `nevo` example illustrates the idea of evolving neural behavior.
The book's Chapter 1 is introductory; its official listed coding assignments
start in Chapter 2. This walkthrough adds a small runnable example rather than
claiming to reproduce a Chapter 1 algorithm, exercise solution, or review question.
The full online chapter was unavailable when checked on 2026-11-19.

## What you will build

A point moves on a line and should reach the origin. Its controller has just
three genes: a weight for position error, a weight for velocity, and a bias.
The topology stays fixed. Evolution changes those three numbers by evaluating
complete trajectories. No target action labels, backpropagation, or within-life
learning are used.

| Concept | Concrete representation |
| --- | --- |
| Genotype | Three `f64` values in a `Vec<f64>`. |
| Phenotype | A tanh neuron that maps observed error and velocity to acceleration. |
| Environment | A deterministic one-dimensional dynamical system. |
| Fitness information | Mean position-error cost plus a small action penalty. |
| Variation | Independent uniform perturbations of the retained controller. |
| Selection | Retain the lowest-cost controller and advance search state. |

The mapping from genotype to executable behavior is deliberately direct. There
is no additional `Genome` or `Decoder` trait for this three-number example.

## The controller and world

At each simulation step, the observation is position error and velocity:

$$e_t = 0 - x_t,$$
$$a_t = \tanh(w_e e_t + w_v v_t + b).$$

The environment then applies:

$$v_{t+1} = 0.9v_t + 0.2a_t,$$
$$x_{t+1} = x_t + 0.1v_{t+1}.$$

The target is zero and the initial velocity is always zero. The first coefficient
retains 90% of velocity; the action supplies acceleration. The position update
uses the newly computed velocity. The action lies in `[-1, 1]`.

The episode cost is:

$$J = \frac{1}{T}\sum_{t=0}^{T-1}
\left(x_{t+1}^2 + 0.01a_t^2\right).$$

Smaller is better. Every controller is scored on four starts, `[-1, -0.5, 0.5, 1]`,
with 80 steps per episode. Its score is the mean of those four episode costs.
The world resets for every episode and every candidate. Invalid genomes and
nonfinite arithmetic return errors rather than invented scores.

These equations are implemented in `tasks::TargetReach`. The recorded rollout
and fast evaluation use the same simulation function, preventing discrepancies
between the visualization and the training environment.

## Generate → Evaluate → Select

`Select` has a broad meaning in `nevo`: incorporate evaluations into the next
search state. In this example that means retaining the best controller and
updating counters. Other algorithms may revise an archive or a distribution.

The teaching optimizer is `ElitistSearch`:

1. **Generate:** sample 32 random controllers for the first batch. In later
   batches, create 32 mutated copies of the retained controller.
2. **Evaluate:** measure each controller's full trajectory cost using `TargetReach`.
3. **Select:** compare the new candidates with the incumbent and retain the best.

Initialization samples genes from `[-1, 1)`. Later mutations add values from
`[-0.2, 0.2)` independently to all three genes. Search is unbounded after
initialization. Exact ties keep the incumbent; initialization ties keep the
first issued candidate. The incumbent's deterministic score is cached.

This is mutation-only `(1 + lambda)` elitism after initialization. It is a small
original teaching variant, not a canonical GA, Gaussian ES, or adaptive search
method. It makes no promise of global convergence. The uniform mutation scale
remains constant, so precision can eventually plateau.

## Run it

From the repository root:

```sh
cargo run -p nevo --example ch01_introduction --locked -- results/ch01
```

The public code is compact:

```rust
use nevo::algorithms::{ElitistConfig, ElitistSearch};
use nevo::tasks::TargetReach;
use nevo::step;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut task = TargetReach::default();
    let mut search = ElitistSearch::new(
        ElitistConfig::new(TargetReach::PARAMETERS, 42),
    )?;

    for _ in 0..100 {
        step(&mut search, &mut task)?;
    }

    if let Some(best) = search.best() {
        println!("Best loss: {}", best.output);
        println!("Genome: {:?}", best.job.input);
    }
    Ok(())
}
```

`step()` is the serial runner. It calls `generate()`, evaluates each issued job,
and passes the complete result batch to `select()`. Its associated-type bounds
ensure the algorithm and evaluator agree about input and output types.

The complete executable additionally prints progress, evaluates held-out starts,
and optionally writes `learning.csv` and `trajectories.csv`. It runs 100 batches
including initialization: 32 initial controllers plus 99 batches of descendants.

## Interpret the result

The recorded seed-42 run produced:

| Measurement | Loss |
| --- | ---: |
| Zero-weight controller, training starts | 0.625000000 |
| Best initially sampled controller | 0.062843333 |
| Best after all 100 batches | 0.036473747 |
| Selected controller, held-out starts | 0.070859762 |

The learned genome was approximately `[8.27748, -2.14063, 0.00175]`. Its signs
make intuitive sense: it accelerates toward the target and opposes velocity.
The small bias is consistent with the symmetric training starts. This is an
interpretation of one run, not evidence that a general reasoning process emerged.

The held-out starts `[-1.25, -0.75, 0.75, 1.25]` are evaluated only after search.
They include different distances from the origin, so their raw loss should not
be read as an apples-to-apples comparison with training loss. A no-action
controller has cost 1.0625 on these starts. This small check is not a broad
statistical generalization study.

**Budget:** 3,200 training search evaluations, or 12,800 episodes and 1,024,000
simulation steps. Baseline evaluation, held-out scoring, and CSV trajectory
recordings are additional diagnostics and excluded from that search budget.

See [recorded data](results/ch01/) and [validation](../VALIDATION.md). The Rust
notebook plots the same public experiment and the trajectories inline.

## Why this helps the library design

The search works on vectors and scalar scores; it does not know the world
contains a moving point. `TargetReach` does not know how candidates were created.
Replacing the task with `Sphere::new(3)` reuses the algorithm and runner without
modification. Replacing the algorithm can reuse the target-reaching task.

The current scope keeps these boundaries visible. Burn will become useful for
larger neural computation and learning, while Bevy will support richer external
worlds. Neither is required for this scalar demonstration.

## Original exercises

1. Change the random seed and compare final scores over several runs.
2. Compare two mutation scales at equal objective-evaluation budgets.
3. Evaluate the selected controller at new starts without changing training.
4. Replace the task with Sphere, keeping three dimensions, to inspect reuse.
5. Explain why retaining a cached incumbent score would need reconsideration
   if evaluation included random noise.

These questions are written for this companion and are not copied from the book.

## References and scope

- [Official book contents and assignments](https://neuroevolutionbook.com/)
- [Rust trait overview](https://doc.rust-lang.org/book/ch10-02-traits.html)
- [ChaCha RNG used by the search](https://docs.rs/rand_chacha/0.9.0/rand_chacha/)

The next formal algorithm milestone is Chapter 2's GA and ES. This companion
prepares the software interface without claiming that later coverage is complete.
