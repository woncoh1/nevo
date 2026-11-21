# Changelog

## Chapter 2 foundations — 2026-11-19

- Preserve the two-trait Generate → Evaluate → Select API and Chapter 1 behavior.
- Add real-valued GA, self-adaptive comma ES, classic full-covariance CMA-ES,
  mirrored OpenAI-style SGD ES, and a specified NSGA-II variant.
- Extract reusable tournament, crossover, mutation, Pareto and crowding functions.
- Add scalar MLP/backpropagation, RNN/LSTM cells, cross-correlation, attention,
  positions, normalization and a small single-head Transformer forward block.
- Add Rastrigin, ZDT1 and XOR evaluators with checked shapes/domains.
- Add original Chapter 2 prose, executed Rust notebook, reproducible Rust
  experiments, multi-seed CSV results, plots, and configuration manifests.
- Add numerical equation/gradient checks and regression coverage, including
  preservation of pre-truncation NSGA-II crowding distances.
- Add nalgebra 0.33.2 for symmetric covariance decomposition; the core still
  has no external dependencies. Burn/Bevy remain optional future integrations.

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
