# nevo: design contract

**Status:** recurrent NEAT and controlled generalization update, 2026-11-25;
source mapping follows the full-PDF review of Chapters 1–3.
This is an evolving library design, not a stable 1.0 API.

## Purpose and vocabulary

`nevo` is a Rust library of composable neuroevolution building blocks. It aims
to support faithful implementations of published methods and experiments that
combine their components. Teaching examples use the public library API.

The lifecycle is **Generate → Evaluate → Select (GES)**. **Select means
incorporate evaluation results and choose the next search state.** Depending on
the method, that includes survivors, archive entries, distribution parameters,
species, or agent–environment assignments. Selection includes necessary state
updates; it does not require returning a subset of a population.

## Principles

1. Keep the common path readable; introduce abstractions when working code needs them.
2. Separate search state, problem evaluation, representation, and execution.
3. Use typed inputs and outputs so users can extend the library without editing it.
4. Make objective direction, randomness, budgets, and failure semantics explicit.
5. Establish scalar correctness before parallel, SIMD, or GPU execution.

## Minimal interfaces

Only two core traits are required:

```rust
pub trait Algorithm {
    type Input;
    type Output;
    type Error: std::error::Error + 'static;

    fn generate(&mut self) -> Result<Vec<Job<Self::Input>>, Self::Error>;

    fn select(
        &mut self,
        results: Vec<Evaluated<Self::Input, Self::Output>>,
    ) -> Result<(), Self::Error>;
}

pub trait Evaluator<T: ?Sized> {
    type Output;
    type Error: std::error::Error + 'static;

    fn evaluate(&mut self, input: &T, seed: u64) -> Result<Self::Output, Self::Error>;
}
```

| Item | Responsibility |
| --- | --- |
| `JobId` | Run-local identity of an evaluation, not an organism. |
| `Job<T>` | Owned input, job ID, and explicit evaluation seed. |
| `Evaluated<T, O>` | Original job plus typed evaluation output. |
| `Algorithm` | Generate jobs, then select its next search state. |
| `Evaluator<T>` | Evaluate borrowed inputs in its problem-specific context. |
| `step()` | Execute one serial GES cycle and return its evaluation count. |

`Input` can describe a vector, graph, or agent–environment task. `Output` can be
a score or a structured record of objectives, descriptors, and learned state.
The core imposes no `Clone`, `Send`, `Sync`, or mandatory `Genome` trait. Large
inputs may contain handles rather than copied worlds. Evaluation borrows them;
selection receives ownership back. Mutable evaluators can reuse buffers and
simulators, but must define how independent trials reset their state.

`select()` is the sole result-consumption method. The previous `update()` method
and `StepError::Update` variant are replaced by `select()` and `StepError::Select`;
there is no deprecated alias in this unpublished API.

## Batch and failure contract

The initial protocol permits one complete, nonempty outstanding batch:

1. `generate()` assigns run-unique IDs before execution.
2. Evaluation preserves each job's input, ID, seed, and submission order.
3. `select()` receives exactly the issued batch and validates it before committing.
4. After success, the caller may request another batch or stop.

The executor rejects empty batches and duplicate IDs within a batch. Algorithms
own cross-batch identity and pending-work validation. The optional parallel
executor preserves submission order before invoking this synchronous interface.

Errors are explicit. No partial batch reaches `select()` after evaluation fails.
Failures stop the run: there is no automatic retry, rollback, checkpoint restore,
or panic recovery. Generation and earlier evaluations may already have changed
state. Restore or reconstruct the run before restarting. Error-to-fitness
penalties, if needed scientifically, must be an explicit evaluator policy.

POET, interactive evolution, and asynchronous search may require richer
orchestration later. Typed jobs provide a useful boundary; they do not prove
that every advanced method is already supported.

## First working composition

`ElitistSearch` is an original teaching optimizer. It initializes a batch of real
vectors, retains the best, generates uniformly perturbed descendants, and
compares them with the incumbent. It has `(1 + lambda)` elitism after
initialization. It has no crossover, Gaussian mutation, or step-size adaptation;
it is not a claimed reproduction of a textbook GA or canonical ES.

Its `Objective::{Minimize, Maximize}` configuration keeps raw scores unchanged.
Exact ties retain the incumbent, or the first issued candidate in the initial
batch. It caches the incumbent's score and therefore targets deterministic
evaluators. Noisy fitness needs an explicit reevaluation policy.

The same optimizer works with `Sphere` and `TargetReach`. `TargetReach` evolves
three parameters of a fixed neural controller in a one-dimensional environment.
The environment and search do not depend on each other's concrete types.

## Randomness and numerical reproducibility

`nevo-core` remains standard-library-only. `nevo` uses `rand_chacha = 0.9.0`
for `ChaCha8Rng`, with versions locked in `Cargo.lock`. Both streams start with
`seed_from_u64(config.seed)`: stream 0 generates candidate coordinates and
stream 1 supplies evaluation seeds. Uniform samples use the high 53 bits of
`next_u64()` divided by `2^53`. Sampling arithmetic and endpoint handling are
documented and tested in the implementation. No randomness depends on worker
order, time, or unspecified hashes.

An evaluation seed is distinct from candidate-generation randomness. Repeated
seeds may be useful for matched trials. The included evaluators ignore seeds
because they are deterministic and reset state on every call.

Record configuration, seeds, lockfile, toolchain, target, backend, and objective
call count. Same-configuration replay is tested; bitwise equality across all
platforms is not promised. `f64` accumulation and `tanh` can vary across platforms
or future backends. Sphere remains the scalar numerical reference.

## Repository and teaching boundary

| Package or directory | Role |
| --- | --- |
| `nevo-core` | GES protocol and serial executor; no external dependencies. |
| `nevo` | Public re-exports, search methods, neural primitives, tasks, and benchmarks. |
| `book/` | Original explanations and recorded example results. |
| `notebooks/` | Rust/Evcxr examples using the public library. |

Bevy and Burn will be optional integrations. Bevy supplies external simulation;
learned internal world models belong to neural computation and learning. Neither
is necessary for these small scalar reference examples.

The Chapter 1 companions are original educational code. The supplied §1.1,
pp.3–5, also gives a concrete 100/20/80 population structure. `Ga` now permits
a restricted top-parent pool to enact it, while the original mutation-only
example remains. `coverage.md` distinguishes a teaching illustration from an
algorithm reproduction and research validation.

## Chapter 2 compositions

The core traits are unchanged. Six additional algorithms demonstrate distinct
meanings of `select`:

| Search | Selected state | Evaluation output |
| --- | --- | --- |
| `Ga` | Evaluated parents used by subsequent tournaments; copied elites are reevaluated. | `f64` |
| `Es` | Best `mu` eligible candidates (comma or plus) and their fixed/inherited scales. | `f64` |
| `De` | One target slot at a time, retaining a strictly better trial or its cached target. | `f64` |
| `CmaEs` | Mean, full covariance, evolution paths, and global step size. | `f64` |
| `OpenAiEs` | A mean updated by mirrored perturbation rewards and explicit SGD. | `f64` |
| `Nsga2` | Parents retained from parents plus offspring by rank and crowding. | `Vec<f64>` |

`select` remains the single public result-consumption/update operation. Operator
calls inside an algorithm may choose mating parents while constructing a batch;
users do not have to orchestrate separate public update methods. Distribution
means are not implicitly evaluated. `best()` on scalar searches means best
actually evaluated candidate; `pareto_front()` means the current selected
nondominated population, not a historical archive.

The new numerical searches share a private checked batch implementation.
`SearchError` reports configuration, protocol and numerical failures. This
internal reuse does not require a new public trait or force custom algorithms
to adopt `Vec<f64>`. Returned numerical genes and seeds must match issued jobs
bit-for-bit; objective values must be finite. Rejected selection does not commit
new state, but the public failure policy remains fail-stop rather than a general
retry protocol.

Reusable functions live in `operators` and `multiobjective`. The default GA/ES/DE
search domain is unbounded after initialization. NSGA-II explicitly clips to
configured box bounds and caches deterministic parent fitness. Alternative
variation, constraints, noisy evaluation, and archives are separate choices.

`GaConfig::parent_pool_size` restricts mating without changing elite copying.
`None` preserves the original full-population tournaments. ES defaults remain
self-adaptive comma; `EsSelection::Plus` includes cached parents, while
`EsStepSize::Fixed` skips the adaptation draw. Parent-plus-child ties retain the
parent. DE freezes the generation during trial construction and compares each
trial only with its own target; ties keep the target. Both plus ES and DE cache
fitness, so stochastic or changing objectives need an explicit reuse policy.
No new core trait, updater method or hidden evaluation phase is introduced.
The book companion specifies each variant rather than using its algorithm name
as a claim that every known extension is implemented.

`nn::Mlp` separates architecture from a flat parameter vector. Each output
neuron stores its input weights and then its bias. It supports analytic MSE
backpropagation; the RNN, LSTM, convolution and attention components provide
checked forward computations. Caller-owned recurrent state makes trial reset
explicit. `Xor` decodes an ordinary vector into the same MLP used by gradient
descent. Public softmax and valid max pooling expose further Chapter 2
operations. Multi-head attention takes already projected Q/K/V, splits feature
slices, concatenates head results and applies a row-major output matrix. It
does not silently change the existing single-head `TransformerBlock` or add a
training framework. This is the scalar reference boundary for future tensor adapters.

## Numerical and execution costs

Gaussian noise uses an uncached Box–Muller transform with two 64-bit RNG draws
per scalar. OpenAI-style ES deliberately shares an evaluation seed between
mirrored candidates. Example initial means use an independent ChaCha stream 2,
while algorithms use stream 0 for search and stream 1 for evaluation seeds.

The strict reference batch layer keeps an issued copy for identity validation;
this costs memory proportional to population times genome size. It is not a
large-model zero-copy executor. CMA-ES stores full covariance and performs an
eigendecomposition per selected batch; memory grows quadratically and that
factorization grows cubically with dimension. It has no restarts or automatic
stopping heuristics. These costs are intentional and documented, and motivate
future backend and representation choices.

Use caller-owned stopping rules and report actual objective calls. The search
comparison includes initialization in its fixed budget; the XOR example uses an
explicit loss target plus a budget cap. Gradient steps and evolutionary fitness
calls remain distinct work units.

## Chapter 3 compositions

The same two traits now support three concrete input representations:

| Representation | Search/evaluator composition | What `select` updates |
| --- | --- | --- |
| `Vec<f64>` | Existing GA/ES/CMA-ES with `MlpClassifier` and control tasks. | The chosen optimizer's parents or distribution. |
| `neat::Genome` | `Neat` with `NeatXor`, `NeatClassifier`, or `NeatDoublePoleTask`. | Population, species, representatives, stagnation, offspring quotas and best-observed record. |
| `encoding::SeedGenome` | `DeepGa` with a decoder composed inside an evaluator. | Top current parents and a separate best-observed record. |

`Genome` is a concrete NEAT graph, not a new mandatory trait. Public genes,
`InnovationTracker`, graph compilation, structural mutation, aligned crossover,
compatibility distance and offspring allocation are usable separately. The
strict structured-batch helper privately requires `Clone + PartialEq`; graph
equality compares weight bits. The core interfaces acquire no new bounds.

NEAT starts with input/bias-to-output links and no hidden nodes. Feedforward
DAGs remain the default. `NetworkMode::Recurrent` permits self-links, cycles and
output feedback while forbidding edges into inputs or bias. Disabled genes
remain historical records in both modes. Innovation and node IDs are run-global
and monotonic, while matching structural events reuse IDs within one generation.
Node splitting uses weights one and the old weight, but the inserted sigmoid
changes the function; in recurrent mode it also adds a tick of signal delay.
Feedforward crossover guards against enabled cycles; both modes enforce unique
endpoints and historical identities. Mixing execution modes is an error.

Execution mode belongs to genotype identity and its innovation tracker, not to
`ConnectionGene`. `Genome::new` and `compile` keep their feedforward semantics.
`Genome::new_recurrent` constructs a recurrent genotype; `compile_recurrent`
creates its stateful phenotype with `step` and `reset`. The genotype contains
no episode memory. Each recurrent tick
uses current external inputs and bias, but **previous-tick hidden/output values**
on every neuron-source edge; all computed neurons update synchronously. Even an
acyclic hidden-to-output path therefore has a delay. `reset` clears all neuron
state to zero. A failed step does not commit partial memory. This explicit
timing convention is not a claim to reproduce the original NEAT activation loop.

No extra core trait, scheduler, tensor type or required environment interface
is introduced. XOR and classification adapters remain feedforward tasks unless
the caller defines a meaningful sequence/reset protocol.

This NEAT variant maximizes finite nonnegative fitness. Loss evaluators
transform their objective explicitly (`1 - MSE` for bounded classification).
Compatibility uses all historical genes, including disabled connections.
Species sharing sums `fitness / species_size` before culling. A fixed threshold,
top-fraction parent pools, largest-remainder quotas, stagnation removal and
quota-based champion copying are explicit choices. The current champion's
species is exempt from stagnation removal; it is not guaranteed a nonzero
offspring quota. The original global-stagnation rescue rule is not implemented.

`DeepGa` is a mutation-only truncation variant inspired by Such et al. Each
lineage stores an initialization seed followed by ordered Gaussian mutation
seeds. Elites are copied within the population and evaluated with fresh job
IDs. The implementation uses uniform initialization and does not reproduce
the paper's extra elite-contender trials, shared noise table or large tasks.
Seed payload is `8 * (1 + mutation_count)` bytes; decoding costs
`O(dimensions * (1 + mutation_count))`. Metadata, architecture, settings and
issued-batch copies add costs. No prefix cache or lineage compaction is hidden.

### Evaluation and physical worlds

The classification module validates finite samples, class labels and unique
sample IDs, and provides reproducible disjoint stratified partitions. The
bundled Optdigits subset has its own attribution and checksum in `data/`.
Training loss alone chooses candidates; validation/test metrics are reported
after the declared search. The small experiments are not official MNIST or
writer-independent UCI benchmark results.

Headless `CartPole`, `DoublePole` and `SpringWalker` expose configuration,
physical state, observations, checked actions, reset, termination, truncation,
and closure-based recorded rollouts or trace-free `run_episode` summaries.
The internal implementation-sharing trait does not add a required public abstraction. `CartPoleTask`, `DoublePoleTask`
and `SpringWalkerTask` decode MLP parameters and minimize negative mean return
over explicit starts, resetting each episode. Training evaluation does not
allocate full trajectories; reporting rollouts do. Their existing `value`
accumulation remains unchanged for numerical compatibility; the new `summary`
method reports return statistics separately.

`EpisodeSummary` records return, applied actions, termination and truncation.
`evaluate_trials(starts, episode)` aggregates a nonempty explicit suite through
a fallible closure. The caller resets the world and policy memory before each
episode. `TrialSummary` reports arithmetic mean, sample standard deviation
(`None` for one trial), worst return, episode/action counts, and horizon hits.
A horizon hit requires truncation without simultaneous task termination.
These are diagnostics; the GES output can remain one `f64`.

`NeatDoublePoleTask` compiles the genome's declared mode, maps its single sigmoid
output to force with `2 * output - 1`, and maximizes positive mean return.
It resets both world and recurrent phenotype between independent starts. The
controlled example records its training summary during the original fitness
evaluation, so plots do not hide repeated diagnostic rollouts. Held-out trials,
memory-reset ablations and representative trajectory replays are counted
separately. A complete fixed training suite makes best-observed scores
comparable across generations; a changing suite would need a different policy.

CartPole follows classic discrete-force equations. DoublePole uses the
historical two-pole dynamics with RK4 and configurable full-state or
position-only observations. Position-only inputs do not grant a feedforward
network memory. SpringWalker is an original spring/contact model, not an
articulated Box2D BipedalWalker reproduction. Physics is independent of Bevy;
visualization reads recorded states. Learned internal world models remain a
different neural-computation concern.

### Optional CPU orchestration

Enable `nevo`'s `parallel` feature to use `execution::step_parallel`. Rayon
evaluates owned jobs and collects results in their original order. The
algorithm still generates and selects on the caller's thread. A factory
constructs a fresh evaluator inside each job; users can capture immutable
configuration or thread-safe shared data.

Only the parallel function adds transport bounds: inputs, outputs and evaluator
errors must be `Send`; the factory is `Send + Sync`. The algorithm, algorithm
error and locally constructed evaluator need not be `Send`. Inputs need not
be `Sync` or `Clone` for this executor. Wrapping the whole call in a caller's
`ThreadPool::install` can add bounds at that outer boundary.

All normal job outcomes are collected before selection or error return.
If multiple jobs fail, the lowest submission-index error is returned; other
jobs may already have executed. No partial selection, side-effect rollback,
retry or panic recovery is provided. Determinism requires evaluation to depend
on explicit inputs/seeds, not scheduling or mutable shared state. CPU job
parallelism does not imply SIMD, tensor batching, GPU or distributed support.

## Next implementation boundary

Extend the coverage registry for the next chapter before adding abstractions.
Keep reusable algorithms independent of Burn/Bevy and introduce adapters when a
working experiment needs them. Add an `Experiment` builder or richer execution
interface only when concrete use cases justify it. A universal operator graph,
asynchronous execution, checkpointing and distributed scheduling remain future
work.

See `VALIDATION.md` for actual checks and limitations. Package publication remains
disabled; the project source-code license has not yet been selected.

## References

- Supplied full PDFs, Chapters 1–3, reviewed 2026-11-24;
  [section mapping and interpretation notes](book/source-audit-ch01-ch03.md).
- [Book and official assignments](https://neuroevolutionbook.com/)
- [Rust traits](https://doc.rust-lang.org/book/ch10-02-traits.html)
- [Cargo workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [Pinned ChaCha generator](https://docs.rs/rand_chacha/0.9.0/rand_chacha/)

The GES terminology and API above are `nevo` design choices.
