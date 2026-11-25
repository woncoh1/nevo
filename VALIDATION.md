# Validation status

## Current: recurrent NEAT and controlled generalization

Verified **2026-11-25**, Linux x86_64, with **Rust 1.100.0**. The unchanged
`Cargo.lock` and the separately tested **1.85.1 MSRV** are retained. Exact
commands, logs, data audits and notebook records are in
[the current validation directory](book/results/ch03_generalization/validation/README.md).

| Check | Actual result |
| --- | --- |
| rustfmt | Passed with the pinned formatter. |
| Strict Clippy, all targets/features | Passed with `-D warnings`; the unnecessary-clone test is fixed. |
| Default tests | **228 unit/integration + 3 doctests = 231 passed.** |
| All-feature tests | **234 unit/integration + 3 doctests = 237 passed.** |
| Release, all features | **237 passed**, including doctests. |
| Controlled-driver protocol tests | **2 additional example tests passed.** |
| API documentation | Passed with `RUSTDOCFLAGS=-D warnings`. |
| Rust 1.85.1 compatibility | All-feature suite: **237 passed**. |
| Controlled study | **60/60 runs completed**; 1,966,080 training episodes, 6,000 held-out, 1,000 state-reset diagnostics and seven trace replays. |
| Data audit | Raw trials, summaries, policy identities, fixed budgets and species quotas agree. |
| Historical results | **141 files unchanged**; all eight regenerated old double-pole artifacts match byte for byte. |
| Feedforward NEAT regression | Twelve complete pre-change batch fingerprints match, including RNG/job/graph/weight identity. |
| Chapter 3 notebook | Fresh **Evcxr 0.22.0 / Rust 1.100.0** execution: **22 code cells**, zero errors, five live SVG plots and a saved topology diagram. |
| Report figures | Six PNG/SVG pairs generated from actual data and visually inspected. |

The 16 new library/integration tests cover recurrent timing, reset and failure
atomicity; cycle/feedback evolution and historical identity; unchanged default
feedforward generation; reusable trial statistics and control adapter resets.
Two additional example tests cover exact budget arithmetic and suite generation.
No new mandatory core trait or dependency was introduced.

The updated notebook ran in a fresh kernel in approximately **50.9 seconds**.
The old Evcxr 0.15.1 kernel could not start with Rust 1.100.0; upgrading to the
verified 0.22.0 release fixed the setup. A persistent borrowed value in a new
notebook analysis cell was also corrected to owned data. Failed setup/cell
attempts are retained separately from the final successful execution record.
The unchanged Chapter 1–2 notebooks keep their previous saved outputs and
validation dates; they are not presented as newly executed on this toolchain.

The controlled study does not reproduce the historical 100,000-step non-Markov
NEAT protocol. Recurrence is now implemented with explicit synchronous semantics,
but BipedalWalkerHardcore, Deep GA's extra elite-contender trials, full neural
training frameworks, Burn/Bevy backends and accelerator/distributed support remain
outside the current scope. See [coverage](coverage.md) and
[the actual results](book/results/ch03_generalization/README.md).

Linux execution is established. GitHub Actions includes Linux/Windows gates,
a Rust 1.85.1 compatibility job and the new smoke experiment; remote CI and
Windows/macOS execution have not been performed here. No cross-platform bitwise
or general runtime guarantee follows from these checks.

The dated sections below preserve earlier evidence; their scope and toolchain
statements describe those earlier milestones.

## Historical full-PDF review of Chapters 1–3 (2026-11-24)

Verified **2026-11-24** on Linux x86_64 with Rust/Cargo **1.85.1**.
All command logs, exact arguments, counts and regression hashes are in
[the PDF-review validation record](book/results/pdf_review/validation/README.md).

| Check | Actual result |
| --- | --- |
| Formatting | Passed. |
| Strict Clippy, all targets/features | Passed with `-D warnings`. |
| Default-feature tests | **212 unit/integration + 3 doctests = 215 passed.** |
| All-feature tests | **218 unit/integration + 3 doctests = 221 passed.** |
| Release, all features | **218 unit/integration + 3 doctests = 221 passed.** |
| API documentation | Passed with `RUSTDOCFLAGS=-D warnings`. |
| Previous results | **109 files unchanged**; six regenerated Chapter 1/2 artifacts match exactly. |
| Chapter 1 notebook | Fresh execution: **8 code cells**, zero errors, 2 unchanged SVG figures. |
| Chapter 2 notebook | Fresh execution: **14 code cells**, zero errors, 3 unchanged SVG figures. |
| Chapter 3 notebook | Markdown updated; all **17 code cells and their saved outputs unchanged**. |

The update adds 31 semantic tests for restricted GA mating, ES scale/survival
choices, differential evolution, softmax, max pooling and multi-head attention.
Tests preserve parent/scale associations, target identity, frozen generations,
forced crossover, complete-batch rejection, objective direction, numerical/shape
errors and deterministic replay. Existing defaults and core interfaces remain.
The new CLI experiments and their actual results are in
[book/results/pdf_review](book/results/pdf_review/README.md).

The full supplied PDFs have now been read. Earlier outline-only source claims
below are dated history, superseded by [the source audit](book/source-audit-ch01-ch03.md).
Current coverage remains deliberately explicit: feedforward NEAT is implemented;
recurrent NEAT, BipedalWalkerHardcore and its exact trial protocol, Deep GA's
extra elite-contender trials, full neural-training frameworks and accelerator/
distributed backends are not. The new components are tested reference variants,
not a claim of exhaustive coverage of all named methods in Chapters 1–3.

Only Linux execution is established. Remote GitHub Actions and Windows/macOS
runs have not been performed. The CI configuration includes the new examples.
No cross-platform bitwise or general performance guarantee follows from these
local checks.

The sections below preserve earlier milestones' evidence and counts. Their
"current" wording and source-access limitations apply to their recorded date.

## Historical Chapter 3 milestone (2026-11-23)

Verified on **2026-11-23**, Linux `x86_64-unknown-linux-gnu`, with Rust/Cargo
**1.85.1**. Full command arguments, toolchain versions, exit codes, elapsed
wall times, and captured output are in
[the Chapter 3 validation record](book/results/ch03/validation/README.md).

| Check | Recorded result |
| --- | --- |
| Formatting | `cargo fmt --all -- --check` passed. |
| Strict linting | `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` passed. |
| Default-feature tests | **181 unit/integration tests + 3 doctests = 184 passed.** |
| All-feature tests | **187 unit/integration tests + 3 doctests = 190 passed.** |
| Release, all features | **187 unit/integration tests + 3 doctests = 190 passed.** |
| Documentation | `cargo doc --workspace --all-features --no-deps --locked` passed with `RUSTDOCFLAGS=-D warnings`. |
| Sphere regression | All four exact reference probes passed; stdout matches the prior Chapter 2 run. |
| Chapter 1 regression | Both regenerated CSVs exactly match the recorded Chapter 1 files. |
| Chapter 2 neural regression | Program stdout exactly matches the prior Chapter 2 run. |

There are no failed or ignored tests in the successful runs. The first Clippy
attempt exposed mechanical example lint issues; its actual nonzero exit code
and output remain in the log, alongside the successful rerun after correction.
No Chapter 3 experiment was rerun merely to obtain these validation counts.
Durations in the command log include build/cache effects, not controlled speed
measurements.

### What the Chapter 3 checks establish

The tests cover historical NEAT genes and innovations, DAG compilation and
cycle guards, structural mutation and crossover, compatibility and fitness
sharing, exact species allocations, complete-batch failures, and deterministic
replay. They also cover control dynamics and task adapters, valid and invalid
classification data/splits, known network outputs, seed-lineage reconstruction,
truncation selection and reevaluated elites, and finite arithmetic failures.

The optional Rayon executor adds six tests. They verify submission-order
results, preserved job IDs/inputs/seeds, one fresh evaluator per job, deterministic
error choice with no partial selection, and serial/one-worker/four-worker replay.
The tests also establish that the algorithm and locally constructed evaluator
need not be `Send`, and inputs need not be `Clone` or `Sync`.

A SHA-256 comparison found all **31** prior Chapter 1/2 result artifacts and
**30** selected preexisting source/test files unchanged. The full comparison,
including exact regenerated Chapter 1 CSV hashes, is in
[prior_chapters_comparison.json](book/results/ch03/validation/prior_chapters_comparison.json).
Algorithm module exports intentionally change to expose the new implementations.

### Experiments and notebook

The six Chapter 3 example programs have been executed. Their exact settings,
budgets, failures as well as successes, and recorded metrics belong to
[the Chapter 3 results report](book/results/ch03/README.md). Those experiments
are teaching demonstrations; equal candidate counts do not imply equal compute,
and the classification runs do not establish a tuned algorithm ranking.

The Chapter 3 notebook completed a fresh **Rust/Evcxr execution** in **57.25 s**:
29 total cells, **17 code cells** with consecutive execution counts 1–17,
**zero error outputs**, and **four inline SVG figures**, all visually inspected.
The live runs reproduce the recorded NEAT seed-1, GA classifier and CartPole
results. Neutral and low-friction walker replays match the frozen CSV to printed
precision; Deep GA serial/four-worker results match exactly. The dataset checksum
is unchanged. See [notebook_checks.json](book/results/ch03/validation/notebook_checks.json)
and [notebook.log](book/results/ch03/validation/notebook.log).

Notebook runtime was Evcxr 0.15.1 with Rust 1.85.1. Its separate temporary Cargo
project needed one missing cached dependency; that repair did not change the
workspace lockfile or notebook dependencies. One top-level result binding was
changed from a borrow to an owned clone for Evcxr persistence. Earlier attempt
logs are retained, and the final saved notebook has no error outputs. Earlier
notebooks were preserved rather than rerun; their relevant source and recorded
result regressions are checked above.

### Current limits

- The new NEAT implementation is an explicit feedforward variant. Recurrent
  NEAT, the original non-Markov double-pole scoring/generalization protocol,
  and Box2D BipedalWalker reproduction are not implemented.
- `SpringWalker` is an original three-particle teaching environment. The digit
  experiment uses real UCI Optdigits with a declared educational split, not
  MNIST or UCI's original writer-independent split.
- DeepGa's compact seed lineage has growing replay cost. It does not include
  the paper's distributed noise table or extra elite-contender reevaluations.
- Rayon provides optional CPU evaluation parallelism. Burn/Bevy integration,
  explicit SIMD, GPU batching, distributed execution, and checkpointing remain
  outside this implementation. No parallel speedup is claimed by these tests.
- RNN/LSTM, convolution, and attention/Transformer components remain forward
  references; the implemented analytic backpropagation applies to MLPs.
- Tests ran on the recorded Linux toolchain. Windows/macOS execution and remote
  GitHub Actions results have not been established. Seeded replay is not a
  promise of bitwise equality across platforms or future numerical versions.
- The companion follows public book outlines and primary references; it is not
  an exhaustive port of currently unavailable full chapter text.

### Repeat the current gates

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
cargo test --workspace --all-features --locked
cargo test --workspace --release --all-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps --locked
```

## Historical Chapter 2 validation

The following section preserves the earlier Chapter 2 evidence and limitations
as recorded at that stage. Its feature limits and test counts are historical;
the current PDF-review status is stated above.

Verified on **2026-11-19**, Linux `x86_64-unknown-linux-gnu`.

### Executed workspace validation

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

### What the new tests establish

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

### Recorded search results

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

### Real notebook execution

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

### Remaining limits

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

### Repeat checks

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
