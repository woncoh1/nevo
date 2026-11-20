# Validation status

Verified on 2026-11-19 on Linux (`x86_64-unknown-linux-gnu`).

## Executed Rust validation

- `rustc 1.85.1 (4eb161250 2025-03-15)`
- `cargo 1.85.1 (d73d2caf9 2024-12-31)`
- Workspace `Cargo.lock` generated and accepted by Cargo.

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed. |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | Passed, no warnings. |
| `cargo test --workspace --locked --offline` | 50 tests and 2 documentation tests passed. |
| `cargo test --workspace --release --locked --offline` | The same 50 tests and 2 documentation tests passed. |
| `cargo doc --workspace --no-deps --locked --offline` with `RUSTDOCFLAGS=-D warnings` | Passed. |
| `cargo run -p nevo --example sphere --locked --offline` | All four reference probes passed. |
| `cargo run -p nevo --example ch01_introduction --locked --offline -- results/ch01` | Completed; actual data and output recorded. |

Tests cover GES identity/order/ownership, failure propagation, deterministic
seed replay, minimization and maximization, strict whole-batch validation,
incumbent/tie behavior, ID exhaustion, finite arithmetic, Sphere reference
values, and the target-reaching dynamics/reset semantics. Seeded Sphere smoke
checks demonstrate improvement on four fixed seeds without claiming a global
convergence rate.

The core and teaching algorithm have been compiled and executed. This replaces
the earlier starter's unverified compilation status.

## Recorded teaching run

The default Chapter 1 example ran with seed 42, 32 candidates per batch, and
100 batches including initialization. It performed 3,200 search evaluations,
12,800 episodes, and 1,024,000 training simulation steps.

| Measurement | Observed loss |
| --- | ---: |
| No-action controller on training starts | 0.625000000 |
| Best initial random controller | 0.062843333 |
| Best final controller | 0.036473747 |
| Final controller on held-out starts | 0.070859762 |

The actual command output and CSVs are in `book/results/ch01/`. Baseline scoring,
held-out scoring, and trajectory recording are extra diagnostic work outside
the search evaluation count. One successful run is a demonstration, not a
statistical benchmark or a claim about all random seeds.

## Notebook validation

The notebook was executed with a real Rust/Evcxr kernel from a fresh session:

- Evcxr Jupyter **0.15.1**, using Rust **1.85.1** and its `rust-src` component.
- `nbformat 5.11.1`, `nbclient 0.11.0`, `jupyter_client 8.10.0` for validation.
- Plotters **0.3.7**, with `evcxr` and `line_series` features.
- **All 7 code cells passed**, including the local library dependency, training,
  held-out evaluation, and two inline SVG charts.
- The notebook's selected genes and training loss agree with the command-line
  example. Its real execution outputs are included in the delivered notebook.

The notebook tools are separate from the Cargo library and are not required
for the command-line examples. Notebook package resolution belongs to Evcxr's
own generated Cargo project; the repository lockfile controls the workspace
build. A current Evcxr release may require a newer Rust toolchain, so the setup
guide also provides a `stable` toolchain route.

## Limits

- The prepared GitHub Actions workflow has not yet run on a remote repository.
- Local Windows/macOS execution was not tested here; CI is configured for Linux
  and Windows. Platform-dependent floating-point differences remain possible.
- Canonical GA/ES, Bevy, Burn, GPU/SIMD execution, and published research
  reproductions are outside this milestone.
- Source licensing remains a project decision before an open-source release.

## Repeat the checks

From the repository root with the selected Rust toolchain installed:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo test --workspace --release --locked
cargo doc --workspace --no-deps --locked
cargo run -p nevo --example sphere --locked
cargo run -p nevo --example ch01_introduction --locked
```

See `notebooks/README.md` for interactive or headless notebook execution.
