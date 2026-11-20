# Sphere: first reference optimization problem

**Identifier:** `sphere-v1`  
**Implementation:** `crates/nevo/src/benchmarks/sphere.rs`  
**Status:** evaluator and deterministic checks implemented and executed.
The teaching `ElitistSearch` is also tested on Sphere; the formal GA/ES comparison below remains future work.

Here "benchmark" means an optimization test problem. This milestone does not
supply a `cargo bench` timing suite or report performance measurements.

## Definition

For a positive dimension D:

```text
f(x) = sum(x[j]^2, j = 0 .. D-1)
objective: minimize
mathematical domain: R^D
known minimizer: [0, ..., 0]
known minimum: 0
```

Each square is nonnegative, so the mathematical minimum follows immediately.
The implementation uses finite `f64` inputs and rejects a nonfinite result.
Very small nonzero coordinates can square to floating-point zero: the numerical
implementation does not claim that only the origin can produce an exact zero.

Initialization bounds belong to an optimizer, not to this evaluator. Values
outside the initialization interval are valid; there is no clipping or repair.
A later bounded variant must use a distinct benchmark configuration.

## Implemented reference checks

The executable example uses D = 8 and seed 42. Sphere ignores the seed because
it has no stochastic behavior.

| Input | Exact expected value |
| --- | --- |
| Eight zeros | 0 |
| Eight ones | 8 |
| Eight values equal to -5 | 200 |
| `[1, -2, 3, -4, 0, 0.5, -0.5, 2]` | 34.5 |

The zero vector is included to test the evaluator. It must **not** be injected
into an optimizer's initial population to manufacture apparent convergence.

Additional tests cover dimension mismatch, NaN/infinities, square and sum
overflow, seed independence, input preservation, sign symmetry, scaling,
array/slice/vector inputs, and 64 deterministic rational-valued fixtures.
Those 64 fixtures are not random samples or a baseline search method.

## Frozen comparison protocol for the first GA and ES

This specifies the next experiment; it has not yet been executed.

| Setting | Value |
| --- | --- |
| Dimensions | 8 |
| Initial sampling | Independent uniform coordinates in `[-5.0, 5.0)` |
| Population / batch size | 64 |
| Budget | 6,400 objective evaluations per run, including initialization |
| Full batches | 100 total: initial 64 plus 99 further batches of 64 |
| Smoke-test run seed | 42 |
| Comparison run seeds | Integers 0 through 19 inclusive |
| Target | Best raw objective <= `1e-8` |
| Search domain after initialization | Unbounded; invalid numerics cause failure |
| Direction | Minimize |

Count actual objective calls, including repeated trials. Reusing a stored score
without evaluating again does not consume an evaluation. Comparisons use equal
budgets, not merely equal numbers of generations. Any pilot configuration that
changes these settings must be labeled separately.

Before stochastic runs, record the named RNG, its version, seed expansion,
stream allocation, and every GA/ES operator setting. Those algorithm-specific
settings are intentionally not invented before the algorithms are specified.

For fixed-budget comparisons, run the full budget and separately record the
first evaluation at which the target is reached. Report final best objective
for all seeds, median and interquartile range, target-hit rate, and evaluations
to first hit. A failed run is recorded as a failure, not silently dropped.
Record whether an initialization population was shared between algorithms.

No convergence threshold or success rate has been established yet. A passing
GES smoke test is not evidence that GA, ES, or neuroevolution is correct.

## Commands

From the workspace root:

```sh
cargo run -p nevo --example sphere --locked
cargo test -p nevo --test sphere --locked
cargo test -p nevo --test ges --locked
```

The expected numeric rows are also in `sphere.expected.csv`. Check
`../VALIDATION.md` for what was and was not executed when this starter was built.
