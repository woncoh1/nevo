# Chapter 2 companion: evolutionary search and neural networks

This chapter connects reusable search algorithms to explicit neural-network
representations. Everything runs through **Generate → Evaluate → Select**.
The examples use scalar `f64` arithmetic so that the algorithms, parameter
layouts, and numerical assumptions remain visible.

The book's public outline calls Chapter 2 “The Basics”: evolutionary ingredients,
GA, ES, CMA-ES, OpenAI ES, multiobjective methods, neural architectures, and their
integration [1]. This is an original implementation companion informed by that
outline, author teaching materials, and the references below. The complete
online chapter was unavailable when checked on 2026-11-19. This document does
not claim to reproduce its text, exact exercises, every variant, or the open-ended
“Further Evolutionary Computation Techniques” section.

## 1. Separate the ingredients

An evolutionary algorithm needs more than a mutation function. It needs an
encoding, an evaluation rule, a way to allocate trials, and a rule for changing
its search state after observing results. Those decisions are independent enough
to deserve separate names, but not necessarily separate Rust traits.

| Ingredient | This implementation |
| --- | --- |
| Representation | A `Vec<f64>` for object variables; ES also retains a step size per parent. |
| Population | Evaluated candidates, or samples from a maintained distribution. |
| Fitness | A scalar with an explicit `Objective`, or a vector with one direction per objective. |
| Variation | Uniform crossover, Gaussian perturbation, or distribution sampling. |
| Selection and adaptation | `Algorithm::select` commits the next population or distribution. |
| Termination | The caller chooses an evaluation budget or another stopping rule. |

For minimization, `Objective::Minimize` means smaller raw outputs are better.
An evaluator should return its measured loss without independently reversing
its sign. None of these algorithms knows whether the numbers describe neural
weights, a controller, or a numerical test function.

## 2. The shared contract

The [core protocol](../crates/nevo-core/src/lib.rs) has two algorithm methods:
`generate` produces owned `Job<Input>` values, and `select` consumes
`Evaluated<Input, Output>` values. Evaluation belongs to a separate `Evaluator`.
The convenience function `step` runs those three phases serially:

```rust
use nevo::algorithms::{Ga, GaConfig, Objective};
use nevo::benchmarks::Sphere;
use nevo::step;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = GaConfig::new(8, 42);
    config.objective = Objective::Minimize;
    let mut search = Ga::new(config)?;
    let mut problem = Sphere::new(8)?;

    for _ in 0..100 {
        step(&mut search, &mut problem)?;
    }
    if let Some(best) = search.best() {
        println!("{} evaluations; best loss {}", search.evaluations(), best.output);
    }
    Ok(())
}
```

With the default population, this makes 6,400 objective calls, including the
initial population. Calling `generate` manually gives the same interface to a
future parallel executor. Such an executor must return the complete batch in
issue order, preserving every job's identity, input, and evaluation seed.

Only one batch may be outstanding. The implementations validate complete results
before committing population or distribution updates; missing jobs, reordered
jobs, modified inputs, and nonfinite scores are errors. The minimal protocol
treats an error as terminal for that run. It does not define retries or rollback.

`Select` means the complete transition informed by evaluations: survivor
selection, step-size adaptation, covariance updates, or archive changes. The
executor does not implement those decisions. The core also imposes no universal
`Clone`, `Send`, `Sync`, tensor, or ECS requirement on user-defined inputs.

## 3. Five explicit algorithm variants

### Genetic algorithm: recombine a population

[`Ga`](../crates/nevo/src/algorithms/ga.rs) is a real-valued, generational GA.
Its default population is 64, initialized uniformly in `[-5, 5)`. It copies the
best current candidate and fills the remaining places using tournament-selected
parents. Each tournament samples three candidates with replacement. Crossover
occurs with probability 0.9; when it occurs, each coordinate independently
inherits one parent's value with probability one half.

Each coordinate then receives a Gaussian perturbation with probability `1/d`:

$$x'_j = x_j + 0.1z_j, \qquad z_j \sim \mathcal N(0,1).$$

The crossover is discrete inheritance, not an arithmetic average. These choices
are explicit teaching defaults rather than a unique definition of “the GA” [2].
The functions `operators::tournament_selection`, `uniform_crossover`, and
`gaussian_mutation` can also be used independently.

**All 64 candidates are evaluated in every batch, including the copied elite.**
Copied coordinates receive fresh job IDs and evaluation seeds. `parents()`
therefore describes the latest evaluated generation, while `best()` records the
best observation across the run. For noisy objectives, historical best is an
observation, not an unbiased estimate of solution quality.

### Evolution strategy: inherit a mutation scale

[`Es`](../crates/nevo/src/algorithms/es.rs) implements a self-adaptive
`(mu, lambda)` strategy: 16 parents and 64 candidates by default. Every individual
carries an object vector and one isotropic step size. For each offspring, choose
a parent uniformly, mutate its step size first, then mutate its coordinates:

$$\sigma' = \operatorname{clip}
\left(\sigma e^{\tau z_0},\sigma_{\min},\sigma_{\max}\right),\qquad
x'_j = x_j + \sigma'z_j.$$

Here the independent deviates are standard normal and
`tau = 1 / sqrt(2d)`. The default initial step size is 0.5; bounds
`[1e-12, 1e3]` are explicit numerical safeguards. This is the single-step-size
self-adaptation mechanism discussed by Beyer and Schwefel [3], without
recombination or coordinate-specific scales.

The comma matters: the next 16 parents come only from the 64 new candidates.
An excellent old parent may disappear. `best()` preserves its record but never
reinserts it into the population. `EsParent` keeps each selected evaluation
paired with the step size that produced it.

GA and ES deliberately share the same initial sampling order. Equal seed,
dimension, population, and initial bound give identical initial jobs, making
subsequent differences easier to examine.

### CMA-ES: adapt the shape of exploration

[`CmaEs`](../crates/nevo/src/algorithms/cma_es.rs) maintains a mean `m`, a
global scale `sigma`, and a full covariance matrix `C`:

$$x_i = m + \sigma BDz_i,\qquad C=BD^2B^\top.$$

Selection updates the mean using positive logarithmic weights from the best
half of the batch. Evolution paths accumulate information across batches;
step-size adaptation and rank-one/rank-mu covariance updates then change the
next sampling distribution [4].

This implementation uses the **non-active, positive-weight** formulation. It
performs a symmetric eigendecomposition after each selected batch and has no
restart policy or automatic stopping criterion. Full covariance takes quadratic
storage and this eigensolve has cubic cost in dimension, which matters when
moving from a tiny controller to a large network.

Construct it with `CmaEsConfig::new(initial_mean, seed)`. The default population
depends on dimension; configure it explicitly when matching evaluation budgets.
`covariance()` returns `C`, excluding `sigma²`. `best()` is an actually evaluated
sample; the current `mean()` has not necessarily been evaluated.

### OpenAI ES: estimate a direction from mirrored trials

[`OpenAiEs`](../crates/nevo/src/algorithms/openai_es.rs) samples 32 normal
directions by default and evaluates both signs, producing 64 jobs:

$$x_i^\pm=m\pm\sigma\epsilon_i,\qquad
m' = m + \frac{\alpha}{2K\sigma}
\sum_{i=1}^{K}(u_i^+-u_i^-)\epsilon_i.$$

Utilities `u` are oriented so larger is better. `ScoreTransform::Raw` uses
signed raw scores; the default `CenteredRanks` maps scores onto centered rank
utilities. Ties receive average ranks, so a completely flat batch contributes
zero update. Rank transformation changes what the estimator optimizes; it
should not be described as the gradient of the original raw loss.

The paired jobs share an evaluation seed, allowing matched stochastic trials.
This is a mirrored, fixed-sigma estimator with **explicit SGD**, informed by
Salimans and colleagues [5]. It does not reproduce their complete distributed
training system, optimizer settings, or benchmark results. The implementation
has no Adam, shared noise table, or observation normalization. As with CMA-ES,
evaluate the final mean separately if that is the policy you intend to deploy.

### NSGA-II: retain several tradeoffs

[`Nsga2`](../crates/nevo/src/algorithms/nsga2.rs) accepts vector outputs and
one `Objective` per coordinate. Candidate A dominates B when A is no worse on
every objective and strictly better on at least one. Equal objective vectors
do not dominate each other.

The algorithm combines nondominated sorting with crowding distance [6]. Lower
rank and then larger crowding distance guide parent tournaments. After evaluating
new offspring, it merges them with cached parents, keeps complete fronts, and
uses crowding to fill the last partial front.

The variation here is uniform crossover plus Gaussian mutation followed by box
clipping. It is not the paper's usual real-valued SBX/polynomial-mutation setup.
Parent scores are cached, so the default behavior assumes stationary objectives.
`pareto_front()` returns the current population's nondominated members, not an
external archive of every historical tradeoff.

The reusable `multiobjective::{dominates, nondominated_fronts,
crowding_distances}` functions expose these pieces. Constant objective
coordinates contribute no crowding distance; an entirely identical front has
zero distances rather than arbitrary boundary advantages.

## 4. What a neural genotype means

The [`nn` module](../crates/nevo/src/nn/mod.rs) gives these vectors an explicit
interpretation. `Mlp` stores parameters in **layer, output neuron, input weights,
bias** order. A `2 → 2 → 1` network has nine parameters. Its first three numbers
are the first hidden neuron's two weights and bias; the next three belong to the
second hidden neuron; the final three define the output neuron.

```rust
use nevo::nn::{Activation, Mlp};

let network = Mlp::new(
    &[2, 1],
    &[Activation::Identity],
    vec![2.0, -1.0, 0.5],
)?;
assert_eq!(network.forward(&[3.0, 4.0])?, vec![2.5]);
```

`Mlp::parameter_count` computes the required dimension. `set_parameters` changes
weights while retaining the architecture. Supported activations are identity,
tanh, sigmoid, and ReLU. Mean squared error averages over examples and output
coordinates. `mse_gradient` returns an analytic gradient in the same flat layout;
`train_step` performs full-batch gradient descent and returns the **pre-step**
loss. Measure `mse` afterward when reporting the updated model.

The other primitives make the architectural differences visible [7, 8]:

| Public building block | Implemented scope |
| --- | --- |
| `RnnCell` | Tanh recurrence with caller-owned hidden state. |
| `LstmCell`, `LstmState` | Input, forget, candidate, and output gates with explicit hidden and cell state. |
| `conv2d_valid` | One-channel, stride-one valid cross-correlation, using the common neural-network convention. |
| `scaled_dot_product_attention` | Stable `softmax(QKᵀ / sqrt(dk))V`, with optional causal masking. |
| `sinusoidal_positions` | Explicit position vectors for the caller to add. |
| `TransformerBlock` | Single-head projections, attention, residuals, post-normalization, and a ReLU feedforward block. |

Only the MLP includes backpropagation. The remaining components are forward
reference implementations, not a general automatic differentiation system or
complete model-training framework. The Transformer has no dropout, learned
normalization parameters, tokenizer, or multihead split. Recurrent state must
be reset between independent trials to avoid mixing candidate histories.

## 5. Integrate search and a network with XOR

[`tasks::Xor`](../crates/nevo/src/tasks/xor.rs) evaluates all four rows of the
XOR truth table. It decodes a `2 → hidden → 1` MLP with tanh hidden units and a
sigmoid output, then returns mean squared error. With four hidden units, the
search dimension is 17. The optimizer simply receives these 17 numbers:

```rust
use nevo::algorithms::{Es, EsConfig};
use nevo::tasks::Xor;
use nevo::step;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut xor = Xor::new(4)?;
    let mut search = Es::new(EsConfig::new(xor.dimensions(), 42))?;
    for _ in 0..100 {
        step(&mut search, &mut xor)?;
    }
    if let Some(best) = search.best() {
        let network = xor.network(best.job.input.clone())?;
        for row in xor.data() {
            println!("{:?} -> {:?}", row.input, network.forward(&row.input)?);
        }
    }
    Ok(())
}
```

This teaches fixed-topology, direct weight encoding. Every training row is also
reported afterward, so the result is an optimization demonstration, not a
held-out generalization test. The short ES snippet above illustrates the API;
the recorded `ch02_xor` experiment compares CMA-ES and gradient descent on the
same encoding. It stops each method at MSE `<= 0.001` or its budget cap: 400
CMA-ES batches of 32 candidates, and 5,000 full-batch gradient steps. Actual work
may be smaller and is recorded separately. Counting one gradient update as one
evolutionary fitness call would hide different computational work; report each
budget separately.

## 6. Run and interpret the experiments

From the repository root:

```sh
cargo run -p nevo --example ch02_search --release -- book/results/ch02
cargo run -p nevo --example ch02_xor --release -- book/results/ch02
cargo run -p nevo --example ch02_neural --release
```

The [Rust notebook](../notebooks/ch02_foundations.ipynb) uses the library APIs
for interactive experiments. See [the recorded results](results/ch02/README.md)
for actual settings, seeds, budgets, and outputs.

Sphere tests progress toward a single smooth optimum. Rastrigin adds many local
optima. ZDT1 introduces a continuous tradeoff front with both objectives
minimized; the evaluator requires coordinates in `[0, 1]`. Its dimensionality
must be reported, since a smaller teaching instance differs from the original
30-variable benchmark [9].

Compare algorithms by objective calls as well as elapsed time. Matching batch
counts is insufficient when population sizes differ. Initialization, step
sizes, and evaluation reuse also affect comparisons. A best-so-far scalar curve
can only improve by definition; it does not reveal whether the current ES
parents, CMA mean, or distribution have improved. Several recorded seeds provide
more information than one favorable trace, but do not establish universal
algorithm superiority.

## 7. Extend one piece at a time

To add a task, implement `Evaluator<Vec<f64>>` or an evaluator accepting slices.
Keep decoding and simulation reset rules explicit. To create a search variant,
compose the public operators inside an `Algorithm`; preserve job identity and
the complete-batch contract. Custom outputs can include objectives, behavior
descriptors, or episode diagnostics because the protocol is typed independently
of these scalar examples.

Burn can later execute tensor models behind evaluators, and Bevy can execute
world simulations there. Neither is required by this reference layer. Backend
batching and parallel evaluation should preserve candidate ordering and seed
assignment, while acknowledging that different floating-point kernels need not
produce bitwise-identical results.

Useful experiments include disabling crossover, changing tournament pressure,
plotting ES step sizes, comparing raw and ranked utilities, and evolving the
same MLP with several optimizers. Change one factor and record the full
configuration so the result remains interpretable.

## References and implementation mapping

1. **Risi, Tang, Ha, and Miikkulainen, _Neuroevolution_.**
   <https://neuroevolutionbook.com/>
   Mapping: Chapter 2 topic coverage and scope; not a claim of full-text access.
2. **Miikkulainen, evolutionary-algorithm teaching slides.**
   <https://www.cs.utexas.edu/~risto/cs378ne/private/slides/2.1_Basics-4up.pdf>
   Mapping: populations, representation, GA operators, and ES context. Exact
   defaults above are this implementation's choices.
3. **Beyer and Schwefel (2002), _Evolution strategies: A comprehensive introduction_.**
   <https://doi.org/10.1023/A:1015059928466>
   Mapping: comma replacement, inherited strategy parameters, and log-normal
   self-adaptation; see the single-step-size discussion in Section 4.2.
4. **Hansen, _The CMA Evolution Strategy: A Tutorial_.**
   <https://arxiv.org/abs/1604.00772>
   Mapping: full covariance, evolution paths, and positive-weight adaptation;
   the implementation uses the non-active formulation.
5. **Salimans et al. (2017), _Evolution Strategies as a Scalable Alternative to Reinforcement Learning_.**
   <https://arxiv.org/abs/1703.03864>
   Mapping: Gaussian perturbation estimator, mirrored sampling, and rank
   utilities; the local implementation explicitly uses SGD.
6. **Deb et al. (2002), _A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II_.**
   <https://doi.org/10.1109/4235.996017>
   Mapping: nondominated sorting, crowding, and parent–offspring survival;
   variation operators here are explicitly different.
7. **Miikkulainen, neural-network teaching slides.**
   <https://www.cs.utexas.edu/~risto/cs378ne/private/slides/2.2_Basics-4up.pdf>
   Mapping: feedforward computation, backpropagation, and recurrent/spatial
   architecture context. The module's layout is defined by its Rust API.
8. **Vaswani et al. (2017), _Attention Is All You Need_.**
   <https://arxiv.org/abs/1706.03762>
   Mapping: scaled attention, sinusoidal positions, and residual attention/feedforward
   structure; this companion implements a reduced single-head forward block.
9. **Zitzler, Deb, and Thiele (2000), _Comparison of Multiobjective Evolutionary Algorithms: Empirical Results_.**
   <https://doi.org/10.1162/106365600568202>
   Mapping: ZDT1 objective definitions and its original benchmark context.

These references establish algorithmic context. The chapter prose and Rust
implementations are original; references do not imply an endorsement by the
book authors or reproduction of the papers' experiments.
