# Coverage

This registry separates implemented components, tested variants, original
teaching exercises, and research reproductions. A chapter label never means
that all methods, extensions, questions, or experiments in its text are done.

| Book area | Current scope/status | Evidence |
| --- | --- | --- |
| Chapter 1: introduction | Original neural-controller companion; retained and regression checked. | `ch01_introduction`, chapter and executed notebook. |
| GES execution | Two generic traits, typed jobs/results, checked complete-batch serial runner. | `nevo-core`, protocol tests. |
| Chapter 2 §2.1: evolutionary ingredients | Real-vector representations, populations, variation, evaluation, replacement, caller-owned stopping. | Public operators, five algorithms, explicit budgets. |
| Chapter 2 §2.2: genetic algorithms | Tested real-valued variant: tournaments, uniform crossover, Gaussian mutation, reevaluated elites. | `Ga`, `ga_es` tests. |
| Evolution strategies | Tested self-adaptive single-sigma `(mu, lambda)` variant; no recombination, offspring-only replacement. | `Es`, sigma-association and replacement tests. |
| CMA-ES | Tested classic non-active full-covariance method; positive weights, paths, rank-one/rank-mu and sigma adaptation. | `CmaEs`, independent equation and covariance tests. |
| OpenAI ES | Tested mirrored perturbation estimator with raw or average-tie centered ranks and explicit local SGD. | `OpenAiEs`, analytic gradient-scaling and tie tests. |
| Multiobjective techniques | NSGA-II survival/mating, standalone dominance/front/crowding functions; uniform/Gaussian box-bounded variation. | `Nsga2`, hand-calculated fronts/crowding and truncation regression. |
| Further evolutionary techniques | No exhaustive coverage claim; specialized algorithms/constraints/restarts remain unimplemented. | Explicit variant documentation. |
| Chapter 2 §2.3: feedforward networks | Checked scalar MLP, flat weight encoding, identity/tanh/sigmoid/ReLU. | `nn::Mlp`, known-value tests. |
| Gradient descent | Analytic full-batch MSE backpropagation for MLP. | Central finite-difference comparisons and update tests. |
| RNN/LSTM | Checked forward cells, explicit hidden/cell state; no BPTT implementation. | `RnnCell`, `LstmCell`, known-step tests. |
| CNN | Single-channel valid 2D cross-correlation, unit stride; not a complete CNN training stack. | `conv2d_valid`, hand-calculated tests. |
| Transformers | Scaled attention, causal mask, sinusoidal positions, normalization, single-head post-norm block; forward only. | `nn::attention`, numerical/shape/residual tests. |
| Chapter 2 §2.4: integration | Original fixed-topology XOR weight evolution and MLP gradient-descent comparison. | `tasks::Xor`, `ch02_xor`, executed notebook. |
| Chapter 2 programming assignment themes | Original GA/ES black-box experiments plus additional named-method demonstrations. | Sphere/Rastrigin runs with 20 seeds and explicit manifests. |
| Chapter 2 review questions | Original extension prompts only; textbook questions/solutions are not reproduced. | Written chapter and notebook. |
| Numerical benchmarks | Sphere, unrotated Rastrigin, configurable ZDT1 with domain checks. | Known-value and domain tests. |
| Chapters 3–14 | No chapter reproductions implemented yet. | Future work. |
| Burn/Bevy, GPU/SIMD, parallelism | Planned optional adapters/executors; no implementation in this milestone. | Scalar reference baseline. |

## Source boundary

The [official site](https://neuroevolutionbook.com/) provides the chapter outline,
assignments, and links to teaching material. Checked on 2026-11-19, it schedules
the public HTML text's return for 2027-01-28. This companion therefore does not
claim a line-by-line implementation of unavailable chapter text. Exact variants
are specified in the source, [Chapter 2](book/ch02_foundations.md), and its primary
reference list.

NSGA-II is the concrete method chosen here for the outline's general
multiobjective topic; the outline alone does not establish that it is the
book's required implementation. CNN/Transformer code demonstrates constituent
forward computations, not a full training system. Toy numerical and XOR results
are not replications of the papers' robotics, Atari, or other experiments.

## Validation levels

1. **Implemented:** executable code exists.
2. **Tested:** specified behavior passes recorded checks.
3. **Reference-validated:** compared with a separately maintained implementation
   under a matched specification.
4. **Research-reproduced:** reproduces a published environment, budget, metrics,
   and multi-seed experimental protocol.

The delivered components reach levels 1 and 2. Some have independent equation
or finite-difference checks; these do not alone claim levels 3 or 4. See
[VALIDATION.md](VALIDATION.md) and the committed result records.
