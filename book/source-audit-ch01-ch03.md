# Source audit: supplied Chapters 1–3

**Reviewed 2026-11-24; Chapter 3 implementation status updated 2026-11-25.** The supplied PDFs of the front matter and Chapters 1–3
were read in full. Equations and decisive figures were also inspected visually.
This review updates the existing `nevo` milestone; it does not redistribute the
textbook or claim to reproduce every experiment it cites.

## Finding and decisions

No numerical defect was established in the existing implementations of GA,
self-adaptive comma ES, CMA-ES, mirrored OpenAI ES, NSGA-II, NEAT or Deep GA.
Their declared variants and numerical conventions remain valid. The PDFs did
reveal specific missing building blocks and made the previous outline-only
coverage claims obsolete.

The update therefore adds narrowly specified components, retains previous
algorithm defaults and experiment records, and records the remaining scope
by printed section/page. All additions use **Generate → Evaluate → Select**.
`Algorithm` and `Evaluator` remain the only required core traits.

## Chapter 1: an actual population recipe

Section 1.1, pp.3–5, describes a population of 100 weight encodings, retaining
20 and producing 80 offspring from those survivors. The original `TargetReach`
example uses an intentionally simpler mutation-only `ElitistSearch`: one
incumbent, no crossover, and a cached deterministic incumbent score. It is
useful, but it is not this population recipe.

`GaConfig::parent_pool_size` now restricts mating to the best current K
candidates independently of elite copying. `None` preserves the original
full-population tournaments. The new `ch01_population` example uses:

- Population 100, 20 copied elites and a top-20 mating pool.
- Tournament size 1, giving uniform parent draws within that pool.
- Uniform crossover for every offspring; all 100 candidates reevaluated.
- The existing original three-parameter `TargetReach` controller.

The chapter leaves the task, distributions, seed and stopping rule unspecified.
Our seed 42, 32 batches, initialization bound 1 and Gaussian sigma 0.2 are
explicit companion choices. Mutation is independent per coordinate with
probability 1/3. It changes one coordinate on average, but can select zero or
several; it is not an exact one-coordinate mutation rule. The example and
notebook state this distinction instead of claiming an exact experimental port.

## Chapter 2: explicit algorithm and neural additions

| PDF reference | Addition | Declared implementation boundary |
| --- | --- | --- |
| §2.2.2, pp.24–25 | `EsSelection::{Comma, Plus}` | Comma selects new offspring; Plus also considers cached parents. Exact ties preserve old parents in Plus. |
| §2.2.2, pp.24–25 | `EsStepSize::{SelfAdaptive, Fixed}` | Fixed uses one isotropic sigma and omits adaptation draws. Initialization remains uniform rather than the illustration's Gaussian around the origin. |
| §2.2.6, pp.34–35 | `De` and `DeConfig` | Synchronous `DE/rand/1/bin`; three distinct non-target donors, forced mutant coordinate, strict target/trial replacement. |
| §2.3.5, p.44, Eq.2.24 | `max_pool2d` | Explicit window/stride, complete windows, one channel, no padding or training. |
| §2.3.5, p.45, Eq.2.26 | `softmax` | Public finite-input probability conversion with stable maximum subtraction. |
| §2.3.6, p.46, Eq.2.29 | `multi_head_attention` | Caller-projected Q/K/V, per-head scaling, concatenation and an explicit output matrix; forward only. |

All four ES scale/survival combinations are available. Parent count can equal
population, allowing `(1+1)` fixed-plus ES. The old default remains
self-adaptive comma with the same random operations. Plus ES and DE reuse
cached scores: their comparison policy is suitable for stationary deterministic
objectives. No noisy-parent reevaluation is implied.

DE freezes its parent population for a whole trial batch, so an early successful
trial cannot affect later donors from that same generation. `parents()` retains
target-slot order. Its local defaults are 64 candidates, F=0.8, CR=0.9 and
uniform initial coordinates in [-5,5); search remains unbounded afterward.
The forced coordinate guarantees inheritance from the mutant, not necessarily
a numerical change when donor values coincide.

Multi-head attention does not alter the existing single-head `TransformerBlock`.
Queries can come from a different sequence than keys/values, making the new
function usable for cross-attention. A complete encoder–decoder, embeddings,
training stack and general automatic differentiation remain outside this API.
The stable softmax function does not change existing sigmoid/MSE classifiers.

### Interpret the formulas before changing code

1. **ES survivor ambiguity.** The p.25 prose calls its illustration plus/elitist,
   while Eq.2.2 selects the next mean from the offspring. These rules can choose
   different survivors. The API exposes both, rather than treating them as the
   same operation.
2. **CMA-ES detail.** The p.28 covariance box is an introductory estimate, not a
   complete replacement for cumulative path and covariance adaptation. `nevo`
   retains its explicit positive-weight algorithm. Its top-half default differs
   from the top-quarter illustration. The current per-batch eigendecomposition
   still has cubic cost even though covariance storage is quadratic.
3. **Mirrored ES scaling.** Eq.2.10 uses N evaluations. K positive/negative pairs
   are N=2K, so the existing denominator `2*K*sigma` is correct. SGD remains a
   declared optimizer choice; Eq.2.11 illustrates Adam. Centered-rank utilities
   are not the raw objective gradient.
4. **Loss normalization.** The p.39 scalar half-squared loss differs from this
   API's mean squared error across examples and outputs. The factor 2 in the
   implemented gradient belongs to its declared objective and must remain.
5. **Finite Pareto populations.** A capacity-limited NSGA-II generation can
   discard nondominated tradeoffs. Its current front is not an unlimited archive
   of every good historical point, despite loose retention language on p.32.

The PDF explicitly names NSGA-II on pp.31–32; the old suggestion that it was
merely our interpretation of a generic multiobjective outline has been removed.

## Chapter 3: precise distinctions that matter

The core NEAT mechanisms in §§3.3.2–3.3.3 match the declared implementation:
historical alignment, splitting an edge with weights 1 and the old weight,
compatibility distance, and species sharing. Inserting a nonlinear hidden node
does not preserve the old function exactly. Species-size sharing is the chosen
membership interpretation of the sharing discussion, not a separate all-pairs
sharing implementation. Small-genome normalization, fixed threshold, quotas
and stagnation conventions remain explicitly documented choices.

Three experiment boundaries remain significant:

| Book experiment | What `nevo` actually contains |
| --- | --- |
| §3.2, pp.53–56: Box2D BipedalWalkerHardcore; 24 observations, four actions, two hidden layers; 16 random training rollouts and 100-trial evaluation. | Original 13-observation/two-action SpringWalker and separate CartPole tasks, declared fixed starts and held-out reports. The articulated walker and its exact protocol remain unimplemented. |
| §3.3.4, pp.63–65: recurrent positions-only double pole, including the self-loop in Fig.3.7 and a 100,000-step success criterion. | Full-state and positions-only 500-step companions now support recurrent graphs, including self-loops. A 10-seed, fixed-rollout study compares training suites and includes a memory-reset ablation. The book's long-horizon historical protocol is still separate. |
| §3.4.2, p.68: Deep GA's ten elite contenders each receive 30 extra episodes. | Ordinary copied-elite reevaluation with compact seed histories. The additional elite-contender phase and Atari experiment are absent. |

The 2026-11-25 update implements a stateful recurrent phenotype, explicit
synchronous tick/reset semantics, mode-aware mutation/crossover and shared
historical tracking. Current inputs and bias feed the new tick; hidden/output
sources use their previous values. This convention follows the same broad
two-buffer approach as [NEAT-Python's recurrent implementation](https://neat-python.readthedocs.io/en/latest/_modules/nn/recurrent.html),
while retaining nevo's own genes, sigmoid, bias and evolutionary rules. It is not
an exact reproduction of the original NEAT software's activation loop.

The new controlled experiment is documented in
[its own result folder](results/ch03_generalization/README.md). It retains the
earlier single-start study unchanged and distinguishes trial count from fitness
calls and environment steps. A 500-step survival-return task with 100 held-out
starts does not reproduce the original paper's oscillation-penalized DPNV
objective, 100,000-step qualification or 625-start generalization test.
Clearing memory every action measures dependence on persistent neural state,
including delayed acyclic paths; it does not isolate graph cycles alone.

A reusable Gaussian action head, independent random-search baseline,
BipedalWalkerHardcore, accelerator batching and distributed workers remain
future work.
Rayon covers CPU job evaluation only; it is not the hardware/distribution stack
reviewed in §3.4.3, pp.69–70.

## Remaining named methods and experiment scope

Section 2.2.6 surveys GP, EP, CGP, PSO, ACO and EDAs, with later-chapter links.
They are explicitly unimplemented in the registry. The PGPE comment in listing
2.2 is also recorded without implying a full algorithm specification. Additional
selection/crossover variants, NSGA-III, full CNN/Transformer training and all
later chapters are not silently counted as covered.

The front matter and Chapter 1 §1.4 establish Chapters 1–15, including the
Epilogue. The coverage range has been corrected accordingly.

The saved Chapter 2 comparison uses 8D functions. Section 2.2.7's 100D Rastrigin
comparison, including OpenAI-ES sigma annealing, is a different experiment.
The new `ch02_pdf_updates` command records a separate five-seed, 8D Sphere
exercise for DE and fixed comma/plus ES. These results and the new population
example are in [the audit-run report](results/pdf_review/README.md). Earlier
result artifacts retain their original protocols and claims.

## Evidence and source identification

- [Coverage registry](../coverage.md): exact implemented and deferred scope.
- [Design](../DESIGN.md): the unchanged two-trait contract and additive choices.
- [Validation](../VALIDATION.md): commands actually run, regression comparisons
  and notebook execution records.
- Chapter companions: [1](ch01_introduction.md), [2](ch02_foundations.md),
  [3](ch03_neuroevolution.md).
- [Source checksums](results/pdf_review/source_files.json) identify the exact PDF bytes.
- Source editions: the user's supplied `ne_book-0(1).pdf` (front matter),
  `ne_book-1(1).pdf` (Introduction, printed pp.1–14), `ne_book-2(1).pdf`
  (The Basics, pp.15–48), and `ne_book-3(1).pdf` (The Fundamentals of
  Neuroevolution, pp.49–71 plus a blank final page). These are distinct from
  whatever edition the public website may serve at another date.

The project contains original code, explanatory mapping and source filenames;
it does not bundle the PDFs, extracted textbook text, figures or official
exercise solutions. The source-code license remains unselected and crate
publication remains disabled, as before this review.
