# Changelog

## Chapter 1 companion — 2026-11-19

- Adopt Generate → Evaluate → Select as the public lifecycle.
- Replace `Algorithm::update` with `Algorithm::select` and `StepError::Update`
  with `StepError::Select`. Select includes all algorithm-specific state updates.
- Add a reusable uniform-mutation `ElitistSearch` with explicit minimization or
  maximization and independent ChaCha streams for candidates and evaluation seeds.
- Add a deterministic target-reaching environment with a three-gene tanh controller.
- Add a runnable Chapter 1 companion, original walkthrough, and Rust notebook.
- Preserve the Sphere reference problem and document remaining Chapter 2 coverage.
- Add a generated result trace, tests, and recorded validation on Rust 1.85.1.

This is an unpublished development API. The Chapter 1 demo is original companion
material, not an official book exercise or a canonical GA/ES reproduction.
