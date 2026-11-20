# nevo: design contract

**Status:** accepted for the Chapter 1 companion milestone, 2026-11-19.
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
| `nevo` | Public re-exports, teaching optimizer, tasks, and benchmarks. |
| `book/` | Original explanations and recorded example results. |
| `notebooks/` | Rust/Evcxr examples using the public library. |

Bevy and Burn will be optional integrations. Bevy supplies external simulation;
learned internal world models belong to neural computation and learning. Neither
is necessary for a three-parameter scalar teaching example.

The Chapter 1 companion is original educational code. The official book's listed
programming assignments start in Chapter 2. `coverage.md` distinguishes a teaching
illustration from an algorithm reproduction and research validation.

## Next implementation boundary

Implement and specify the Chapter 2 GA and ES, then compare them on common
objective-evaluation budgets. Extract shared selection/mutation components from
working implementations. Add an `Experiment` builder when real use cases justify
it. Avoid committing the API to a universal operator graph prematurely.

See `VALIDATION.md` for actual checks and limitations. Package publication remains
disabled; the project source-code license has not yet been selected.

## References

- [Book and official assignment outline](https://neuroevolutionbook.com/)
- [Rust traits](https://doc.rust-lang.org/book/ch10-02-traits.html)
- [Cargo workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html)
- [Pinned ChaCha generator](https://docs.rs/rand_chacha/0.9.0/rand_chacha/)

The GES terminology and API above are `nevo` design choices.
