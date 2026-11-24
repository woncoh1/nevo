# PDF-review validation record

Executed on **2026-11-24**, Linux x86_64, Rust/Cargo **1.85.1**.
The workspace lockfile and core protocol files are unchanged. All Cargo checks
used `--locked --offline`; dependencies were already cached. These wall times
include build/cache effects and are not speed benchmarks.

## Commands actually executed

| Check | Exit | Seconds | Passed tests |
| --- | ---: | ---: | ---: |
| [fmt](fmt.log) | 0 | 0.185 | — |
| [clippy](clippy.log) | 0 | 11.314 | — |
| [tests_default](tests_default.log) | 0 | 14.286 | 215 |
| [tests_all_features](tests_all_features.log) | 0 | 20.873 | 221 |
| [tests_release_all_features](tests_release_all_features.log) | 0 | 21.610 | 221 |
| [docs](docs.log) | 0 | 0.902 | — |
| [ch01_population](ch01_population.log) | 0 | 3.957 | — |
| [ch02_pdf_updates](ch02_pdf_updates.log) | 0 | 0.572 | — |
| [regression_sphere](regression_sphere.log) | 0 | 0.021 | — |
| [regression_ch01](regression_ch01.log) | 0 | 0.485 | — |
| [regression_ch02_neural](regression_ch02_neural.log) | 0 | 0.020 | — |
| [regression_ch02_search](regression_ch02_search.log) | 0 | 1.167 | — |

Exact argument arrays are in [checks.json](checks.json). All required gates
passed: formatting, strict Clippy for all targets/features, default tests,
all-feature tests, release all-feature tests and warning-free API documentation.
Tests include **31 new cases**: five GA parent-pool tests, nine ES-mode tests,
nine DE tests and eight neural-operation tests. The old invalid ES parent-count
case was adjusted to permit mu=lambda; `(1+1)` plus survival is tested explicitly.

Default totals are **212 unit/integration tests + 3 doctests = 215**.
All-feature and release totals are **218 + 3 = 221**. No tests failed or were
ignored in these runs. The six optional executor tests explain the feature-count
difference. Remote GitHub Actions and Windows/macOS execution were not performed.

## Regression evidence

[regressions.json](regressions.json) records SHA-256 values for all **109**
previous result files, all unchanged. The regenerated Chapter 1 learning and
trajectory CSVs, and four Chapter 2 search artifacts, match their committed
pre-audit counterparts exactly. The original algorithms' default trajectories
therefore remain intact in those exercises. This is stronger evidence than
merely comparing new configuration values with old defaults, but is not a
promise of bitwise replay across different platforms or toolchains.

Chapter 3 was not retuned or rerun. Its NEAT, Deep GA, physics and evaluator
modules are unchanged, and the previous experiment records are retained. Its
notebook's 17 full code-cell objects, including outputs/counts, are identical.
Only Markdown explanations changed there. The core traits, runner and Cargo.lock
also match the previous milestone exactly.

## Notebook execution

- `notebooks/ch01_introduction.ipynb`: 8 code cells, 15 total cells, 42.144 s, zero errors, execution counts 1–8, 2 inline SVG figures.
- `notebooks/ch02_foundations.ipynb`: 14 code cells, 26 total cells, 48.056 s, zero errors, execution counts 1–14, 3 inline SVG figures.

Both notebooks executed from fresh Rust/Evcxr kernels. Runtime was Evcxr 0.15.1
with Rust 1.85.1. [notebook_checks.json](notebook_checks.json) and notebook logs
retain execution evidence. New population and Sphere outputs agree with CLI
records to their displayed precision. Pooling, softmax and two-head outputs
match the hand calculations.

All five SVG figure strings exactly match their previously visually reviewed
counterparts; [notebook_figures.json](notebook_figures.json) records their hashes.
No new chart or changed chart layout was introduced, so that previous visual
review remains applicable. The notebook code additions are independently executed,
not copied saved output.

## Interpretation

These checks establish the specified scalar operations and example behavior.
They do not establish complete textbook coverage or research-benchmark
reproduction. See [the source audit](../../../source-audit-ch01-ch03.md) and
[coverage](../../../../coverage.md) for the recurrent-NEAT, articulated-walker,
Deep GA elite-trial and accelerator/distribution boundaries.
