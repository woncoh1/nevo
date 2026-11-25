# Validation for the recurrent/generalization update

Executed on **2026-11-25**, Linux x86_64, using the pinned **Rust 1.100.0**
development toolchain and unchanged dependency lockfile. The library's declared
**1.85.1 MSRV** was separately tested. GitHub Actions is configured for Linux
and Windows; remote CI and local Windows/macOS execution were not performed.

## Rust gates

Exact commands, exit codes and elapsed times are in [checks.json](checks.json).
Each check has a corresponding `.log` file.

| Check | Observed result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed |
| Clippy, workspace/all targets/all features, `-D warnings` | Passed |
| Default tests | **228 unit/integration + 3 doctests = 231 passed** |
| All-feature tests | **234 unit/integration + 3 doctests = 237 passed** |
| Release, all features | **234 unit/integration + 3 doctests = 237 passed** |
| Controlled-driver protocol tests | **2 additional example tests passed** |
| API documentation, `RUSTDOCFLAGS=-D warnings` | Passed |
| Rust 1.85.1, all-feature tests | **237 passed**, including doctests |

The update adds 16 library/integration tests: ten for recurrent execution and
evolution, one feedforward regression using 12 pre-change batch fingerprints,
and five for trial aggregation/control adapters. The two driver tests verify
the fixed complete-batch budgets and deterministic disjoint bounded suites.

Recurrent tests cover current inputs versus delayed neuron sources, simultaneous
updates independent of historical node order, pulse memory, reset, failed-step
state preservation, self-links/output feedback, mode identity, innovations,
mutation, crossover and deterministic GES replay. Trial tests cover sample SD,
singleton behavior, invalid/error cases, trace-free/recorded consistency,
independent episode resets, observation shapes and objective direction.

The existing Clippy diagnostic in `es_modes.rs` is fixed with
`std::slice::from_ref(&incumbent)`. No lint allowance was added.

## Experiments and independent data review

All **60 declared full-protocol runs completed**. The execution log and binary,
source and lockfile hashes are in
[experiment_execution.json](experiment_execution.json); each seed has its own
captured output. Four independent seed processes ran concurrently. These elapsed
times are execution records, not isolated algorithm-speed benchmarks.

The reporting script validates every declared seed/condition, complete batch
budgets, shared disjoint starts, saved policy identity, raw trial statistics,
species counts/quotas and predeclared replay outcomes. Its validation report and
raw-source hashes are in [summary.json](../summary.json).
The separate [review.md](review.md) records an independent source and full-data
audit, including the limits of the recurrent state-reset interpretation.

Every original result file was preserved: **141 historical files unchanged**.
Chapter 1–2 notebook bytes and `Cargo.lock` are also unchanged; see
[preserved_artifacts.json](preserved_artifacts.json).

The historical seed-42 double-pole example was regenerated using the updated
code and Rust 1.100.0. **All eight artifacts match the old files byte for byte**,
including parameters/genes, curves, trials, trajectories, run summary and
manifest. See [doublepole_replay.json](doublepole_replay.json). This is separate
from the new fixed-budget study and its different initial-state distribution.

## Figures and notebook

Six PNG/SVG figure pairs were generated from the completed Rust data and visually
inspected. Header spacing was corrected where needed. The all-seed comparisons,
training curves, topology/species diagnostics, predeclared seed-0 topologies,
paired state-reset comparison and actual state/action traces are in the parent
directory. SD intervals are descriptive and may extend beyond bounded metrics.

The final notebook passed **22 code cells** in approximately **50.9 s**, with a
fresh **Evcxr 0.22.0 / Rust 1.100.0** kernel and zero error outputs. It produced
five live SVG figures and displays a saved topology SVG. The added live cells
use 384 teaching-training episodes, six diagnostic episodes and one saved-policy
replay outside the full-study budget. [notebook_checks.json](notebook_checks.json)
records counts and figure hashes.

The notebook execution record is [notebook_execution.json](notebook_execution.json).
It records the actual kernel/toolchain, fresh execution, elapsed time and exit
status; [notebook_ch03.log](notebook_ch03.log) contains the full captured output.
Early kernel setup attempts are retained separately and are not counted as
successful notebook validation. The finished notebook contains only successful
cell outputs. Earlier Chapter 1–2 notebooks retain their previous validation
history and saved outputs.

## Limits

These gates validate the specified implementation and recorded experiment.
They do not establish a historical NEAT benchmark reproduction, broad robustness
outside the fixed local test distribution, cross-platform bitwise equality,
or a general algorithm-performance ranking. Only the MLP implements analytic
backpropagation; recurrent NEAT is evolved rather than trained with BPTT.
Burn/Bevy, tensor/SIMD/GPU and distributed adapters remain outside this update.
