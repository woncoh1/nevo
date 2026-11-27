# Working on nevo-rs

Read [DESIGN.md](DESIGN.md) for the API contract and [STYLE.md](STYLE.md)
for coding conventions. The repository is `nevo-rs`; crate names remain
`nevo` and `nevo-core`.

## Keep changes simple

- Use **Generate → Evaluate → Select**. Select includes all search-state updates.
- Prefer small functions and concrete types. Add traits only for demonstrated reuse.
- Keep `nevo-core` dependency-free; keep optional execution and backend details out
  of its interfaces.
- Make objective direction, seeds, evaluation budgets and recurrent resets explicit.
- Test invariants and known results. Do not require stochastic searches to converge
  in every run or use timing thresholds as correctness tests.
- Preserve recorded experiments. Write new runs to `results/`; document their
  protocol before replacing published results.
- Keep examples and notebooks on the public API. Update `coverage.md` when adding
  an algorithm or changing its implemented scope.

## Check your work

Use the pinned toolchain and default rustfmt style. From the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo bench -p nevo --bench foundations --locked -- --test
```

For dependency changes, also test and compile benchmarks with Rust 1.85.1,
the declared MSRV. Commit `Cargo.lock` and Proptest regression seeds when a
failure produces them. Keep benchmark output under `target/`; compare timings
on the same machine and toolchain. See [README.md](README.md#property-tests-and-benchmarks)
for commands and benchmark scope.
