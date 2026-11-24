# Results for the Chapters 1–3 PDF review

Recorded **2026-11-24**, Rust 1.85.1, Linux x86_64. These are original small
companion experiments with declared settings, not replications of the book's
published Rastrigin, BipedalWalkerHardcore or recurrent double-pole results.
No seed, stopping point or hyperparameter was selected after inspecting results.

## Chapter 1 population companion

`ch01_population` uses the original `TargetReach` task with Chapter 1's
100-candidate / 20-survivor / 80-child population structure. Only the top 20
current candidates can reproduce; all 100, including copied elites, receive
fresh evaluation jobs each batch.

Settings: seed 42; 32 batches including initialization; uniform initial genes
in [-1,1); uniform parent sampling; uniform crossover for every child; independent
per-gene Gaussian mutation with probability 1/3 and sigma 0.2. This mutation
operator is a stated companion choice, not an exact one-weight mutation rule.

| Measurement | Recorded value |
| --- | ---: |
| Search calls | 3,200 |
| Training episodes per call | 4 |
| Steps per episode | 80 |
| Training simulation steps | 1,024,000 |
| Final training loss | 0.036622373 |
| Held-out-start loss | 0.071014781 |

Selected genes `[error weight, velocity weight, bias]`:
`[5.848152730449426, -1.5602649988541044, 0.0077269158081610945]`.
Held-out starts [-1.25,-0.75,0.75,1.25] are consulted after training, not for
selection or stopping. Their four reporting episodes add 320 simulation steps
outside the search budget. The learning curve is in
[ch01_population/learning.csv](ch01_population/learning.csv); the executed
notebook reproduces the printed losses.

## DE and fixed-scale ES on Sphere

All 15 runs use 8 dimensions, seeds 0–4, 64 candidates per batch and 100 batches,
including initialization: **6,400 objective calls per run**, 96,000 in total.
At each seed, all three variants receive identical initial uniform jobs in
[-5,5). The objective is the unmodified sum of squared coordinates; lower is
better. There is no score shaping, hidden clipping, adaptive stopping or tuning.

- **DE/rand/1/bin:** F=0.8, CR=0.9, three distinct non-target donors, one forced
  mutant coordinate, synchronous generation and strict target/trial survival.
- **Fixed comma ES:** mu=16, lambda=64, isotropic sigma=0.2, offspring-only survival.
- **Fixed plus ES:** identical fixed-scale settings, with cached parents also eligible.

Cached scores in DE/plus ES are valid here because Sphere is deterministic and
stationary. Each displayed value is the best actually observed score over the
run; it is not the current population mean or a separate validation score.

| Variant | Median final loss | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| DE/rand/1/bin | 0.17170737 | 0.08933228 | 0.25060485 |
| Fixed comma ES | 0.05370741 | 0.03660894 | 0.06949522 |
| Fixed plus ES | 0.01926653 | 0.01220993 | 0.02675395 |

All 15 runs improved from their initial best score. These particular values do
not establish a general method ranking: the methods and scale choices differ,
there are only five seeds, and this is one smooth low-dimensional function.
The fixed Gaussian scale also limits final precision. The purpose is to make
the new algorithms and survivor choices executable and inspectable.

Full records:
[per-run scores](search_runs.csv), [per-batch curves](search_curves.csv),
[configuration](search_manifest.json), [computed summary](summary.json).
The Chapter 2 notebook independently reproduces seed 0's final scores:
DE 0.1717074, fixed comma 0.06026347 and fixed plus 0.02675395.

## Neural operations

The CLI and notebook exercise hand-checkable cases:

| Operation | Expected and recorded output |
| --- | --- |
| 2×2 max pooling, stride 2 | `[[6.0, 8.0]]` |
| Softmax of `[1000, 1000 + ln(3)]` | Approximately `[0.25, 0.75]` without positive exponent overflow. |
| Two independent attention heads | Probabilities `[0.75,0.25]` and `[0.25,0.75]`; head values 3 and 4. |
| Output projection of `[3,4]` | `[-1,10]` for the declared 2×2 matrix. |

Focused tests also cover negative pooling values, incomplete windows, per-head
rather than total-width scaling, rectangular projections, masking, invalid
shapes, nonfinite inputs and overflowing projection arithmetic.

## Repeat

From the repository root:

```sh
cargo run -p nevo --example ch01_population --release --locked -- book/results/pdf_review/ch01_population
cargo run -p nevo --example ch02_pdf_updates --release --locked -- book/results/pdf_review
```

Then run the Chapter 1 and Chapter 2 Rust notebooks from `notebooks/`.
[Validation records](validation/README.md) contain actual exit codes and counts.
The **109** preexisting Chapter 1–3 result files are byte-for-byte unchanged.
Regenerated Chapter 1 curves/trajectories and Chapter 2 search data match the
previous records exactly, preserving the original algorithm defaults.
