# Validation status

Verified on **2026-11-19**, Linux `x86_64-unknown-linux-gnu`.

## Executed workspace validation

- `rustc 1.85.1 (4eb161250 2025-03-15)`
- `cargo 1.85.1 (d73d2caf9 2024-12-31)`
- Workspace `Cargo.lock` accepted by Cargo, including rand_chacha 0.9.0 and
  nalgebra 0.33.2. The core has no external dependencies.

| Check | Recorded result |
| --- | --- |
| Formatting | `cargo fmt --all -- --check` passed. |
| Strict linting | `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` passed. |
| Debug tests | **118 unit/integration tests and 3 documentation tests passed.** |
| Release tests | The same **118 tests and 3 documentation tests passed.** |
| Documentation | `cargo doc --workspace --no-deps --locked --offline` passed with `RUSTDOCFLAGS=-D warnings`. |
| Sphere reference example | All four reference probes passed. |
| Chapter 1 example | Completed; regenerated data matches the prior Chapter 1 result. |
| Chapter 2 search example | Completed all 160 scalar runs and the 30-gene ZDT1 run. |
| Chapter 2 XOR example | Both evolution and backpropagation reached the declared loss target. |
| Chapter 2 neural example | Executed inspectable MLP, recurrent, convolutional, attention and Transformer computations. |

Captured command output is in `book/results/ch02/validation/`. The runtime
measurements there are check durations, not controlled performance benchmarks.
The Chapter 1 output directory in the command record is represented as a
portable placeholder for the temporary regression output.

## What the new tests establish

The Chapter 1 protocol and numerical tests remain. Chapter 2 adds 68 tests and
one documentation example, including:

- Exact operator behavior, matching GA/ES initial populations, fresh elite IDs,
  objective direction, deterministic replay, and complete-batch rejection.
- Self-adaptive sigma inheritance and association with the correct selected
  genome; offspring-only comma replacement; bounds on strategy scales.
- Independently calculated first-generation CMA mean, covariance and sigma
  updates, including odd population size; positive definiteness and off-diagonal
  adaptation on a rotated objective.
- Mirrored OpenAI pairing and shared pair seeds, analytic raw-reward gradient
  scaling, sign handling, centered-rank tie behavior and numerical failures.
- Hand-calculated dominance/front/crowding cases, mixed objectives, extreme
  finite and subnormal values, stable selection and cached-parent accounting.
- An NSGA-II regression ensuring survivors retain crowding computed on the
  complete combined front before truncation for subsequent mating.
- MLP gradients checked against central finite differences for multiple
  examples and outputs, exact MSE normalization, and atomic parameter updates.
- Known RNN/LSTM steps, manually calculated cross-correlation, causal/scaled
  attention, normalization/positions, Transformer residual behavior and shapes.
- Closed-form Sphere/Rastrigin/ZDT1 values, domain errors, and a known XOR model.

These checks support the documented implementations. They are not a claim of
exhaustive correctness for every numerical scale or full differential validation
against another maintained library. See `coverage.md` for validation levels.

## Recorded search results

All four scalar methods use **8 dimensions, 64 calls per batch, 100 batches and
20 seeds**, with initialization inside the 6,400-call budget. GA/ES have the
same initial population. CMA/OpenAI means are sampled using independent stream 2,
then perturbations use stream 0. Initial distributions differ across families.
No hyperparameter search or held-out benchmark tuning is included.

| Method | Median Sphere objective | Sphere target hits | Median Rastrigin objective |
| --- | ---: | ---: | ---: |
| GA | 8.365754199e-5 | 0/20 | 5.478136737 |
| Self-adaptive ES | 8.697074293e-10 | 19/20 | 19.89914084 |
| CMA-ES | 5.045378547e-13 | 20/20 | 2.984956143 |
| OpenAI-style ES | 1.695175775 | 0/20 | 53.76754152 |

The target is objective at most `1e-8`. No Rastrigin run reaches it within the
budget. All scores are actually evaluated candidates, not free mean diagnostics.
Full IQRs, first-hit counts, raw curves and configuration manifests are in
[the recorded results](book/results/ch02/README.md). These values are specific
to the chosen variants and settings, not rankings of the algorithm families.

NSGA-II's seed-42, 30-gene ZDT1 demonstration uses 12,800 calls and ends with
64 nondominated selected parents. Their mean vertical gap above the known
front is 9.464360436e-5. Separate tests use an explicitly smaller six-gene
problem over three seeds; they are not substituted for the recorded 30-gene run.

The 17-parameter XOR example stops at MSE at most 0.001. CMA-ES reaches
0.0005741047324 after 1,056 evaluations; backpropagation reaches 0.0009990828383
after 1,100 full-batch steps. Backprop also performs 1,100 post-update
forward-loss checks. These are different work units. All four truth-table
rows are used for training and reporting; no generalization claim is made.

## Real notebook execution

Both notebooks were executed in fresh Rust kernels against the Chapter 2 library:

- Evcxr Jupyter **0.15.1**, Rust **1.85.1**, and `rust-src`.
- `nbformat 5.11.1`, `nbclient 0.11.0`, `jupyter_client 8.10.0`.
- Plotters **0.3.7**, with `evcxr` and `line_series` features.
- **Chapter 2: all 12 code cells passed; 3 inline SVG figures.** The saved
  outputs agree with the final CLI configuration, including stream-2
  initialization and the target-based XOR stopping rule.
- **Chapter 1: all 7 code cells passed; 2 inline SVG figures.** Its selected
  genes and training/held-out losses match the prior notebook. The original
  delivered Chapter 1 notebook remains unchanged.

The validation harness drains late stdout briefly after Evcxr 0.15.1's idle
message so complete output is captured. It does not replace computations,
rewrite results, or fabricate display output. Notebook figures were inspected.
Notebook builds use Evcxr's generated project, not the workspace lockfile.
The setup guide offers a current stable-toolchain installation route.

## Remaining limits

- RNN/LSTM, convolution and attention/Transformer components are forward
  reference computations; only MLP includes backpropagation here.
- Full covariance CMA-ES, copies for strict identity checks, and scalar
  arithmetic target clarity and small problems. GPU, SIMD, parallel/distributed
  evaluation, checkpointing, Burn and Bevy adapters are not implemented.
- The concrete method variants omit documented extensions such as CMA restarts,
  active negative weights, distributed OpenAI training infrastructure, and
  NSGA-II constraint-domination/SBX/polynomial variation.
- The GitHub Actions workflow is prepared for Linux and Windows but has not run
  on a remote repository. Windows/macOS execution was not tested locally.
- Seeded replay is checked on the recorded toolchain; cross-platform bitwise
  identity of transcendental functions and eigendecomposition is not promised.
- The textbook's full HTML chapters are currently unavailable. This is an
  original companion to the public outline, not an exhaustive full-text port or
  a reproduction of the papers' experiments.
- The source license remains a project decision; Cargo publication is disabled.

## Repeat checks

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo test --workspace --release --locked
cargo doc --workspace --no-deps --locked
cargo run -p nevo --example sphere --locked
cargo run -p nevo --example ch01_introduction --locked
cargo run -p nevo --example ch02_search --release --locked -- results/ch02
cargo run -p nevo --example ch02_xor --release --locked -- results/ch02
cargo run -p nevo --example ch02_neural --locked
```

See `notebooks/README.md` for interactive and headless notebook commands.
