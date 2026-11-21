# nevo: design contract

**Status:** accepted for the Chapter 2 foundations milestone, 2026-11-19.
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
own cross-batch identity and pending-work validation. A future parallel executor
must restore submission order before invoking this synchronous interface.

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

The Chapter 1 companion is original educational code. The official book's listed
programming assignments start in Chapter 2. `coverage.md` distinguishes a teaching
illustration from an algorithm reproduction and research validation.

## Chapter 2 compositions

The core traits are unchanged. Five additional algorithms demonstrate distinct
meanings of `select`:

| Search | Selected state | Evaluation output |
| --- | --- | --- |
| `Ga` | Evaluated parents used by subsequent tournaments; copied elites are reevaluated. | `f64` |
| `Es` | Best `mu` offspring and their inherited mutation scales. | `f64` |
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

Reusable functions live in `operators` and `multiobjective`. The default GA/ES
search domain is unbounded after initialization. NSGA-II explicitly clips to
configured box bounds and caches deterministic parent fitness. Alternative
variation, constraints, noisy evaluation, and archives are separate choices.
The book companion specifies each variant rather than using its algorithm name
as a claim that every known extension is implemented.

`nn::Mlp` separates architecture from a flat parameter vector. Each output
neuron stores its input weights and then its bias. It supports analytic MSE
backpropagation; the RNN, LSTM, convolution and attention components provide
checked forward computations. Caller-owned recurrent state makes trial reset
explicit. `Xor` decodes an ordinary vector into the same MLP used by gradient
descent. This is the scalar reference boundary for future tensor adapters.

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

- [Book and official assignment outline](https://neuroevolutionbook.com/)
- [Rust traits](https://doc.rust-lang.org/book/ch10-02-traits.html)
- [Cargo workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [Pinned ChaCha generator](https://docs.rs/rand_chacha/0.9.0/rand_chacha/)

The GES terminology and API above are `nevo` design choices.
