# Changelog

## Full-PDF review of Chapters 1–3 · 2026-11-24

- Read the supplied front matter and full Chapters 1–3; replaced current
  outline-only source claims with printed-page mappings and an original audit.
- Added optional top-parent mating to `Ga` and a 100/20/80 population companion.
- Added fixed/self-adaptive ES scale and comma/plus survival, including `(1+1)`.
  Previous self-adaptive comma defaults and random draws remain unchanged.
- Added synchronous `De` / `DeConfig` with the explicit rand/1/bin variant.
- Added public stable softmax, valid max pooling and multi-head attention on
  caller-projected Q/K/V with output projection. The existing Transformer block
  remains single-head and forward only.
- Added focused semantic tests, declared Sphere experiments and notebook cells;
  retained existing experiment records rather than reinterpreting them as book
  reproductions.
- Clarified remaining recurrent NEAT, BipedalWalkerHardcore, Deep GA elite-trial,
  full neural-training and accelerator/distribution gaps. No Burn/Bevy dependency
  or new core trait was introduced.

## Chapter 3 neuroevolution — 2026-11-23

- Preserve the two core GES traits and prior chapter behavior.
- Add feedforward NEAT and reusable genes, innovation tracking, structural
  mutations, aligned crossover, compatibility, species sharing and quotas.
- Add mutation-only Deep GA with reproducible seed-history encoding.
- Add optional Rayon execution with fresh per-job evaluators and ordered results.
- Add validated classification data, stratified splits, MLP/NEAT classifiers,
  and the attributed genuine Optdigits subset.
- Add checked native CartPole/DoublePole physics and an original spring robot,
  closure rollouts, and fixed-topology MLP evaluator adapters.
- Add six reproducible CLI experiments, honest success/failure and held-out
  results, original chapter, executed Rust notebook and scientific plots.
- Record structural, numerical, physics, protocol, seed replay and parallel
  checks; preserve earlier chapter outputs and source behavior.

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
