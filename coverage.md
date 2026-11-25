# Coverage

Reviewed against the supplied full PDFs of Chapters 1–3 and the front matter on
**2026-11-24**, with recurrent/generalization coverage updated **2026-11-25**.
Page numbers below are the book's printed page numbers. This
registry separates implemented methods, declared variants, original teaching
tasks, and experiments still to reproduce. A chapter label does not mean every
method mentioned in that chapter has been implemented.

## Evolutionary algorithms and execution

| Book area | Current scope | Evidence / remaining boundary |
| --- | --- | --- |
| Chapter 1 §1.1, pp.3–5 | Original mutation-only controller plus a GA companion with 100 candidates, 20 elites, and a top-20 mating pool. | `ch01_introduction`, `ch01_population`; per-gene Gaussian mutation is a declared choice, not exactly one changed weight. |
| Chapter 2 §2.1, pp.16–22 | Representations, populations, evaluation, variation, replacement and caller-owned stopping. | Two core traits and the complete-batch Generate–Evaluate–Select protocol. |
| GA §2.2.1, pp.22–24 | Real-valued tournaments, uniform crossover, Gaussian mutation, optional restricted mating pool, reevaluated elites. | `Ga`, parent-pool/GA tests. Roulette, rank-proportional selection, single/two-point crossover are not public standalone operators yet. |
| ES §2.2.2, pp.24–25 | Comma or plus survival; self-adaptive or fixed isotropic Gaussian scale, including `(1+1)` plus. | `EsSelection`, `EsStepSize`; defaults remain self-adaptive comma. Uniform initialization and scalar scale differ from the illustrated origin-centered/coordinate-scale example. |
| CMA-ES §2.2.3, pp.25–28 | Classic non-active full covariance, weighted mean, evolution paths, rank-one/rank-mu and global scale adaptation. | `CmaEs`; a complete cumulative update, not a substitution of the introductory covariance-estimation box. No restarts or active covariance. |
| OpenAI ES §2.2.4, pp.27–30 | Mirrored perturbations, raw/centered-rank utilities, local SGD. | `OpenAiEs`; no Adam, shared noise table, observation normalization or distributed training system. |
| NSGA-II §2.2.5, pp.30–33 | Nondominated sorting, crowding, mating, combined-population survival. | `Nsga2`; explicitly named in the PDF. Uniform/Gaussian variation; no SBX, polynomial mutation, NSGA-III or historical archive. |
| DE §2.2.6, pp.34–35 | Synchronous `DE/rand/1/bin`, three distinct non-target donors, forced mutant coordinate and strict target/trial survival. | `De`; finite scalar outputs, complete-batch validation, both objective directions. Parent scores cached; no ongoing bounds or restarts. |
| Other §2.2.6 methods, pp.33–35 | GP, EP, CGP, PSO, ACO and EDA are catalogued but unimplemented. | Conceptual survey and later-chapter links are not a complete specification of every variant. |
| PGPE, listing 2.2 p.35 | Named in a commented solver choice; unimplemented. | Mention does not establish a complete algorithm specification. |
| §2.2.7, pp.35–36 | Original 8D Sphere/Rastrigin comparisons plus new declared Sphere checks. | The book's 100D Rastrigin comparison and OpenAI-ES sigma annealing are not reproduced. |
| Numerical tasks | Sphere, Rastrigin, configurable ZDT1. | Known-value/domain tests; these do not imply full textbook benchmark replication. |
| GES execution | Checked serial execution and optional ordered Rayon CPU jobs. | Core traits have no mandatory tensor/ECS type or Send/Sync bound. |

## Neural building blocks

| Book area | Current scope | Remaining boundary |
| --- | --- | --- |
| MLP/backpropagation §§2.3.1–2.3.2, pp.37–40 | Flat affine layers, identity/tanh/sigmoid/ReLU, analytic full-batch MSE gradient and update. | Mean-squared loss convention is explicit; the PDF's half-squared scalar loss uses a different normalization. |
| RNN/LSTM §§2.3.3–2.3.4, pp.40–43 | Checked forward cells with caller-owned hidden/cell state. | No backpropagation through time. |
| CNN §2.3.5, pp.43–45 | Single-channel valid cross-correlation; explicit-window/stride max pooling; public stable softmax. | No multi-channel CNN architecture, padding/dilation framework or CNN training stack. |
| Attention §2.3.6, pp.45–47 | Scaled attention, causal masking, positions, normalization; multi-head attention on caller-projected Q/K/V, concatenation and output matrix. | Supports separate query/key sequences, not a complete encoder–decoder model. |
| Transformer block §2.3.6 | Existing single-head post-norm block and feedforward layer remain. | New multi-head function is composable but is not silently substituted into this block; no tokenizer, embedding table, decoder stack or training. |
| Integration §2.4, pp.47–48 | Direct fixed-topology XOR weight evolution and MLP gradient comparison. | Original companion task, not a generalization benchmark. |

## Chapter 3: neuroevolution

| Book area | Current scope | Remaining boundary |
| --- | --- | --- |
| Taxonomy §3.1, pp.49–52 | Weight vectors, topology/weight graphs and replayable seed histories. | Seed history is not a developmental indirect encoding; developmental methods begin in Chapter 4. |
| Robot case study §3.2, pp.53–56 | Native CartPole and original three-particle SpringWalker, explicit trial resets and held-out reporting. | **BipedalWalkerHardcore is unimplemented.** Its Box2D body, 24 observations, 4 actions, two hidden layers, 16 random training rollouts and 100-trial evaluation are not reproduced. |
| Stochastic policies §3.2.5, p.56 | Current control policies are deterministic. | Reusable Gaussian action head and associated sampling policy are not implemented. |
| NEAT §§3.3.1–3.3.3, pp.57–62 | Feedforward and synchronous recurrent graphs, historical genes, structure mutation, aligned crossover, persistent species, sharing and exact quotas. | Fixed threshold, species-membership sharing, explicit tick/reset semantics and other variant choices are documented. |
| Double pole §3.3.4, pp.63–65 | Historical RK4 dynamics; full-state and positions-only 500-step tasks; recurrent control, 10-seed equal-rollout comparisons, 100 held-out starts and memory-reset ablation. | The original 100,000-step success criterion, oscillation fitness, 625-case generalization and published ablation protocols are not reproduced. |
| Trial aggregation | `EpisodeSummary`, `TrialSummary`, `evaluate_trials`, trace-free `run_episode`, MLP summaries and `NeatDoublePoleTask`. | Fixed explicit starts and independent resets. No universal experiment runner or stochastic Gaussian action head. |
| Deep neuroevolution §3.4.2, pp.67–69 | Mutation-only Deep GA, truncation, seed replay, ordinary elite reevaluation; existing OpenAI ES. | No Atari/humanoid reproduction, random-search algorithm, shared noise table, or top-10 contenders ×30 additional elite trials. |
| Compute §3.4.3, pp.69–70 | Optional CPU evaluation through Rayon; exact serial/worker replay checks. | No GPU/TPU, SIMD, tensor batching or multi-machine distribution. |
| Published assignment themes | Genuine 8×8 Optdigits classification and CartPole GA/ES experiments. | Original accessible tasks/splits, not MNIST, official solutions or paper benchmark replications. |
| Review questions | Original explanations and extension prompts. | Textbook questions/solutions are not redistributed. |
| Chapters 4–14 and Chapter 15 epilogue | Not implemented. | Front matter and §1.4 p.11 establish the full chapter range. |
| Burn/Bevy integration | Planned optional adapters. | No backend dependencies added in this source audit. |

## Source and interpretation boundary

The supplied PDFs now establish actual chapter contents; previous outline-only
access statements are historical. See [the source audit](book/source-audit-ch01-ch03.md)
for exact source identifiers, the additions, and places where mathematical
conventions require care. The [official site](https://neuroevolutionbook.com/)
and original method references remain useful context. The PDFs and extracted
text are not included in this repository.

In particular, the plus-selection prose and offspring-only maximization in
Eq.2.2 describe different survivor rules. Both are explicit ES choices here.
The mirrored OpenAI estimator keeps `1/(2K sigma)` for `2K` evaluations. The MLP
keeps the derivative of its declared full mean-square loss. NSGA-II's finite
population can discard nondominated points; it is not an unlimited archive.
CMA-ES retains its cumulative updates and the documented cubic eigensolve cost.

## Validation levels

1. **Implemented:** executable code exists.
2. **Tested:** specified behavior passes recorded checks.
3. **Reference-validated:** matched against a separately maintained implementation.
4. **Research-reproduced:** matches a published environment, budget, metrics and experimental protocol.

The delivered components reach levels 1 and 2. Independent equations and
finite differences strengthen those checks but do not alone establish levels
3 or 4. See [VALIDATION.md](VALIDATION.md) and committed results for actual work.
