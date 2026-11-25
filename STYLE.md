# Writing nevo

Use the [Rust Style Guide](https://doc.rust-lang.org/style-guide/) for formatting
and the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) for
API design. The [Google Rust Style Guide](https://google.github.io/styleguide/rustguide.html)
provides additional implementation guidance; project requirements and clarity
determine which organizational preferences apply. Architecture belongs in
[DESIGN.md](DESIGN.md).

## Automated conventions

- Use the default `rustfmt` style from the pinned development toolchain.
- Treat enabled Clippy warnings as errors in CI. Do not enable every pedantic
  or restriction lint by default. Fix useful diagnostics; explain any narrowly
  scoped exception where mathematical clarity requires it.
- Keep the workspace's `unsafe_code = "forbid"` policy. Use stable language
  features and retain the declared minimum supported Rust version unless an
  intentional compatibility change is documented and validated.

Before committing, run from the workspace root:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
```

`rust-toolchain.toml` pins Rust **1.100.0** and includes rustfmt, Clippy,
rust-analyzer and standard-library sources. `Cargo.toml` separately declares
the library's minimum compiler version. The committed `.vscode` settings enable
Rust formatting and Clippy checks on save; extension recommendations do not
install software automatically. Keep personal themes, credentials and absolute
language-server paths in user settings.

## Clear scientific code

- Prefer borrowing to unnecessary cloning, and accept slices where the
  operation only needs a borrowed sequence. Copies required by an algorithm's
  ownership or validation contract should remain explicit.
- Use explicit imports in library modules. Small notebook setup cells may use
  glob imports to keep demonstrations readable.
- Document public behavior: shapes, objective direction, randomness, state
  reset, finite-value requirements, error conditions and evaluation costs.
- Keep mathematical names such as `sigma`, `mu`, `lambda` and local `x`, `m`,
  or `c` when their definitions and equations are nearby. Prefer descriptive
  names at public boundaries. Established algorithm names such as `CmaEs`,
  `Nsga2` and `Neat` remain appropriate.
- Choose indexed loops or iterator chains according to which makes the
  computation clearer. Do not obscure matrix equations merely to shorten code.
- Preserve consistent fallible `new(config) -> Result<...>` constructors;
  renaming them to `try_new` is not required by this project.
- Introduce traits and macros when a concrete reusable abstraction needs them.
  Keep the Generate–Evaluate–Select vocabulary consistent across algorithms.
- Test mathematical invariants, independent known values, state transitions and
  reproducibility. Measure performance before adding an optimization.

Toolchain upgrades receive their own validation record because new formatter
and lint versions can change diagnostics without changing the algorithm.
