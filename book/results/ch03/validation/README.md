# Chapter 3 workspace validation

Executed on 2026-11-23 with Rust/Cargo **1.85.1**, Linux
`x86_64-unknown-linux-gnu`. Exact versions are in [toolchain.log](toolchain.log).
[checks.json](checks.json) records command arguments, environment overrides,
start/end times, elapsed wall time, actual exit codes, and parsed test totals.
These durations include build/cache effects and are not performance benchmarks.

| Gate | Result | Log |
| --- | --- | --- |
| `cargo fmt --all -- --check` | Passed | [Final format check](fmt_attempt2.log) |
| Strict Clippy, workspace/all targets/all features | Passed | [Final Clippy check](clippy_attempt2.log) |
| Workspace tests, default features | 181 unit/integration tests + 3 doctests passed | [Default tests](tests_default.log) |
| Workspace tests, all features | 187 unit/integration tests + 3 doctests passed | [All-feature tests](tests_all_features.log) |
| Workspace release tests, all features | 187 unit/integration tests + 3 doctests passed | [Release tests](tests_release_all_features.log) |
| Workspace docs, all features, warnings denied | Passed | [Documentation](docs.log) |

The first Clippy attempt found mechanical example issues: an unnecessary borrow
of a copyable factory and formatting parameter strings through an iterator.
The example owners corrected these; the final formatting and Clippy gates passed.
The original [failed attempt](clippy.log) is preserved, with its nonzero exit
code in the command record. The corrections do not change algorithm behavior.

## Regression evidence

- [Sphere](regression_sphere.log): all four exact reference values passed;
  program stdout matches the prior Chapter 2 validation output.
- [Chapter 1](regression_ch01.log): the fixed seed-42 run completed. Both
  regenerated CSVs in [regression_ch01](regression_ch01/) are byte-for-byte
  identical to the recorded Chapter 1 learning and trajectory files.
- [Chapter 2 neural example](regression_ch02_neural.log): program stdout matches
  the previous Chapter 2 validation output, excluding Cargo build messages.
- [SHA-256 comparisons](prior_chapters_comparison.json): all **31** prior Chapter
  1/2 result artifacts and **30** selected preexisting source/test files are
  unchanged. New module exports are intentionally excluded from the latter
  comparison.

Previous results were read without being overwritten. The six Chapter 3
experiments were not rerun as part of these validation gates; their manifests
and outputs are in the parent results directory. Notebook execution is recorded
separately after its fresh-kernel run.

## Executed Chapter 3 notebook

[notebook_checks.json](notebook_checks.json) records the actual Rust/Evcxr run:
17 code cells, consecutive counts 1–17, zero error outputs, four SVG figures,
57.25 seconds. [notebook.log](notebook.log) contains captured execution output.
Figure hashes correspond to visually inspected renders. Walker replay agrees
with frozen CSV output; serial/four-worker Deep GA results agree exactly.
The attributed classification data checksum is unchanged. Earlier setup and
ownership-error attempt logs are retained separately.
