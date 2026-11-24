# Chapter 2 companion: evolutionary search and neural networks

This chapter connects reusable search algorithms to explicit neural-network
representations. Everything runs through **Generate → Evaluate → Select**.
The examples use scalar `f64` arithmetic so that the algorithms, parameter
layouts, and numerical assumptions remain visible.

The supplied Chapter 2 PDF, “The Basics,” was checked on 2026-11-24 [1]. It
covers evolutionary ingredients, GA, ES, CMA-ES, OpenAI ES, NSGA-II, additional
search methods, neural architectures, and their integration. The PDF review adds
differential evolution, explicit ES survival and step-size choices, and missing
neural forward operations to this original implementation companion. Exact
variants and remaining coverage gaps are stated below; implementing a named
method does not reproduce every experiment or extension in the chapter.

| Book section and printed pages | Companion implementation |
| --- | --- |
| §2.1, pp. 15–22 | Representations, evaluation, variation, population updates, and caller-owned stopping through GES. |
| §§2.2.1–2.2.4, pp. 22–30 | GA, ES, CMA-ES, and a mirrored OpenAI-style ES estimator with explicitly documented choices. |
| §2.2.5, pp. 30–33 | NSGA-II rank/crowding selection and reusable multiobjective operations. |
| §2.2.6, pp. 33–35 | Differential evolution; the other named families remain future work. |
| §2.2.7, pp. 35–36 | Interchangeable solvers; our recorded problem sizes and settings differ from the book's 100D Rastrigin experiment. |
| §2.3, pp. 37–47 | MLP/backpropagation and explicit recurrent, convolution, pooling, softmax, and attention forward building blocks. |
| §2.4, pp. 47–48 | Original fixed-topology XOR weight-evolution example. |

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
| Variation | Uniform crossover, Gaussian perturbation, differential mutation, or distribution sampling. |
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
This preserves the separation shown in Listing 2.1 on page 17. The public names
are deliberately **Generate → Evaluate → Select**: `Select` includes every
evaluation-informed population or distribution update. The convenience function
`step` runs those three phases serially:

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
initial population. The optional `parallel` feature also provides `step_parallel`.
An executor must return the complete batch in
issue order, preserving every job's identity, input, and evaluation seed.

Only one batch may be outstanding. The implementations validate complete results
before committing population or distribution updates; missing jobs, reordered
jobs, modified inputs, and nonfinite scores are errors. The minimal protocol
treats an error as terminal for that run. It does not define retries or rollback.

`Select` means the complete transition informed by evaluations: survivor
selection, step-size adaptation, covariance updates, or archive changes. The
executor does not implement those decisions. The core also imposes no universal
`Clone`, `Send`, `Sync`, tensor, or ECS requirement on user-defined inputs.

## 3. Six explicit algorithm variants

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

`GaConfig::parent_pool_size` optionally restricts mating to the best `k` current
candidates, implementing the truncation option described in §2.2.1, page 22.
`None` preserves the original full-population tournaments. With `Some(k)` and
`tournament_size = 1`, parents are sampled uniformly from the top `k` with
replacement. This setting is independent of `elite_count`, which controls
unchanged copies. The [Chapter 1 population example](ch01_introduction.md#the-books-population-structure-100-candidates-20-retained)
uses both settings to retain 20 of 100 candidates and create 80 offspring from
those 20.

**All 64 candidates are evaluated in every batch, including the copied elite.**
Copied coordinates receive fresh job IDs and evaluation seeds. `parents()`
therefore describes the latest evaluated generation, while `best()` records the
best observation across the run. For noisy objectives, historical best is an
observation, not an unbiased estimate of solution quality.

### Evolution strategy: choose mutation and survivor rules

[`Es`](../crates/nevo/src/algorithms/es.rs) defaults to the original self-adaptive
`(mu, lambda)` strategy: 16 parents and 64 candidates. Every individual carries
an object vector and one isotropic step size. For each offspring, choose a
parent uniformly, mutate its step size first, then mutate its coordinates:

$$\sigma' = \operatorname{clip}
\left(\sigma e^{\tau z_0},\sigma_{\min},\sigma_{\max}\right),\qquad
x'_j = x_j + \sigma'z_j.$$

Here the independent deviates are standard normal and
`tau = 1 / sqrt(2d)`. The default initial step size is 0.5; bounds
`[1e-12, 1e3]` are explicit numerical safeguards. This is the single-step-size
self-adaptation mechanism discussed by Beyer and Schwefel [3], without
recombination or coordinate-specific scales.

Two independent configuration choices now make the distinction in §2.2.2,
pages 24–25, explicit:

| Setting | Behavior |
| --- | --- |
| `EsSelection::Comma` (default) | Choose the next `mu` parents from the new `lambda` candidates only. An excellent old parent may disappear. |
| `EsSelection::Plus` | Choose from the previous `mu` parents and new `lambda` candidates together. Old parent scores are cached; ties favor existing parents. |
| `EsStepSize::SelfAdaptive` (default) | Inherit and mutate one scalar scale per individual, as above. |
| `EsStepSize::Fixed` | Use `initial_sigma` for every coordinate mutation; consume no adaptation-noise draw. `tau` and sigma bounds are unused. |

For example, `config.parents = 1`, `config.selection = EsSelection::Plus`, and
`config.step_size = EsStepSize::Fixed` choose a fixed-scale `(1 + lambda)` ES.
Setting `population = 1` as well gives `(1 + 1)` after initialization. With
comma selection and `mu = lambda`, every offspring survives, so there is no
survivor truncation. `best()` is always a separate reporting record, never an
additional parent. `EsParent` keeps each surviving evaluation paired with its
own scale. Plus selection assumes stationary fitness when comparing cached
parents with newly evaluated offspring.

**Source ambiguity:** the worked example on page 25 calls its method elitist
`(mu + lambda)` with `mu = 1`, but Eq. 2.2 writes the new mean as an `argmax`
over sampled offspring alone. Those descriptions imply different survivor
rules. `nevo` exposes both explicitly: choose `Plus` to retain an incumbent
when no child improves it, or `Comma` to choose only among children. It does
not silently treat these rules as equivalent.

GA and ES deliberately share the same initial sampling order. Equal seed,
dimension, population, and initial bound give identical initial jobs, making
subsequent differences easier to examine. ES evaluates a **uniform initial
batch**, then retains its best `mu` members. This differs from the page-25
example's first Gaussian batch centered on the origin. Its per-axis sigma vector
also permits diagonal scales; this implementation uses one scalar scale per
individual, with no coordinate-specific scales or recombination. The new modes
therefore cover the survival and mutation-strength choices without claiming an
exact reproduction of Figure 2.6. The existing self-adaptive comma defaults and
their recorded experiments remain unchanged.

### CMA-ES: adapt the shape of exploration

[`CmaEs`](../crates/nevo/src/algorithms/cma_es.rs) maintains a mean `m`, a
global scale `sigma`, and a full covariance matrix `C`:

$$x_i = m + \sigma BDz_i,\qquad C=BD^2B^\top.$$

Selection updates the mean using positive logarithmic weights from the best
half of the batch. Evolution paths accumulate information across batches;
step-size adaptation and rank-one/rank-mu covariance updates then change the
next sampling distribution [4].

The book's §2.2.3, pages 25–28, illustrates selection of the top 25% and explains
covariance estimation. Our full algorithm uses the tutorial's positive-weight
update with the top half. The illustration is not a complete specification of
the path, learning-rate, or step-size formulas, so those details come from [4].

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

In the supplied chapter, Eq. 2.10 on page 29 writes an unpaired sampling estimator
and Eq. 2.11 illustrates an Adam-based update. The paired estimator above counts
`2K` evaluations, and its local SGD update remains an explicit implementation
choice. No Adam update or natural-gradient preconditioning is implied by the
type name.

### NSGA-II: retain several tradeoffs

[`Nsga2`](../crates/nevo/src/algorithms/nsga2.rs) accepts vector outputs and
one `Objective` per coordinate. Candidate A dominates B when A is no worse on
every objective and strictly better on at least one. Equal objective vectors
do not dominate each other.

The book explicitly presents NSGA-II in §2.2.5, pages 31–32. The algorithm
combines nondominated sorting with crowding distance [6]. Lower
rank and then larger crowding distance guide parent tournaments. After evaluating
new offspring, it merges them with cached parents, keeps complete fronts, and
uses crowding to fill the last partial front.

The variation here is uniform crossover plus Gaussian mutation followed by box
clipping. It is not the paper's usual real-valued SBX/polynomial-mutation setup.
Parent scores are cached, so the default behavior assumes stationary objectives.
`pareto_front()` returns the current population's nondominated members, not an
external archive of every historical tradeoff. If even the first front exceeds
capacity, some nondominated points are removed by crowding. Finite-population
elitism therefore does not guarantee retention of every historical nondominated
point or monotonic improvement of every possible front-quality measure.

The reusable `multiobjective::{dominates, nondominated_fronts,
crowding_distances}` functions expose these pieces. Constant objective
coordinates contribute no crowding distance; an entirely identical front has
zero distances rather than arbitrary boundary advantages.

### Differential evolution: compare each trial with its target

[`De`](../crates/nevo/src/algorithms/differential_evolution.rs) implements the
`DE/rand/1/bin` variant of the mechanism in §2.2.6, pages 34–35. For each current
target vector `x_i`, draw three distinct donors, all different from the target:

$$v_i = x_{r_1} + F(x_{r_2} - x_{r_3}).$$

Binomial crossover independently takes a coordinate from `v_i` with probability
`CR`, otherwise from `x_i`. One uniformly chosen coordinate must come from the
mutant. This guarantees mutant inheritance, not a numerically different vector
when parents or differences coincide.

**Generate** creates one trial per target using the frozen current population.
**Evaluate** scores all trials. **Select** compares each trial with its own target
and replaces that slot only for a strict improvement; ties keep the target.
Updates take effect together after the complete batch is validated. `parents()`
preserves target-slot order rather than sorting by fitness; `best()` reports the
best evaluated candidate separately.

`DeConfig::new(dimensions, seed)` chooses population 64, uniform initialization
in `[-5, 5)`, `F = 0.8`, `CR = 0.9`, and minimization. These are local defaults.
At least four candidates are required, with `0 < F < 2` and `0 <= CR <= 1`.
There are no ongoing coordinate bounds, hidden clipping, restarts, or objective
calls inside the optimizer. The first batch evaluates the initial population;
later batches evaluate one new trial per target and reuse parent scores. Cached
fitness requires a stationary evaluator or an explicit acceptance of comparisons
against earlier noisy observations.

### Remaining Chapter 2 methods and alternatives

Six concrete algorithms do not exhaust the chapter. Section 2.2.6, pages 33–35,
also discusses **genetic programming (GP), evolutionary programming (EP),
Cartesian genetic programming (CGP), particle swarm optimization (PSO), ant
colony optimization (ACO), and estimation of distribution algorithms (EDAs)**.
They are not implemented here. Listing 2.2 on page 35 mentions **PGPE** as an
interchangeable solver without developing it in this chapter; it also remains
unimplemented.

Other explicit gaps include roulette-wheel and rank-based GA selection,
single-point and two-point crossover (§2.2.1, pp. 22–23), coordinate-specific ES
scales and recombination (§2.2.2, pp. 24–25), NSGA-III (§2.2.5, p. 33), and the
structured island/cellular populations of §2.1.2 (p. 19). The public tournament,
uniform-crossover, mutation, and GES components support extension without
claiming that these alternatives already exist.

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

Equation 2.13 on page 39 uses half the squared error for one scalar output, so
its derivative in Eq. 2.14 has no factor of two. `Mlp` defines the full mean
squared error across rows and outputs; its gradient includes the corresponding
factor of two and averaging. This is an explicit loss convention, not a missing
factor in either implementation.

The other primitives make the architectural differences visible [7, 8]:

| Public building block | Implemented scope |
| --- | --- |
| `RnnCell` | Tanh recurrence with caller-owned hidden state. |
| `LstmCell`, `LstmState` | Input, forget, candidate, and output gates with explicit hidden and cell state. |
| `conv2d_valid` | One-channel, stride-one valid cross-correlation, using the common neural-network convention. |
| `max_pool2d` | Maximum over complete windows with explicit window size and stride; no padding (Eq. 2.24, p. 44). |
| `softmax` | Stable conversion of a finite logit vector to probabilities by subtracting its maximum before exponentiation (Eq. 2.26, p. 45). |
| `scaled_dot_product_attention` | Stable `softmax(QKᵀ / sqrt(dk))V`, with optional causal masking. |
| `multi_head_attention` | Split preprojected Q/K/V features into heads, compute attention per head, concatenate, and apply an explicit output projection (Eq. 2.29, p. 46). |
| `sinusoidal_positions` | Explicit position vectors for the caller to add. |
| `TransformerBlock` | Single-head projections, attention, residuals, post-normalization, and a ReLU feedforward block. |

Only the MLP includes backpropagation. The remaining components are forward
reference implementations, not a general automatic differentiation system or
complete model-training framework. `TransformerBlock` remains the original
**single-head** forward block: it has no dropout, learned normalization
parameters, tokenizer, or multihead split. The separate `multi_head_attention`
operation does not silently change this type or its saved outputs.

For multihead attention, the caller supplies already projected Q, K, and V.
Each row contains equally sized contiguous slices for the heads; query/key
widths must agree, while value width may differ. Each head uses its own key
width in the scaling factor. The returned head outputs are concatenated and
multiplied by the caller's output matrix. Different query and key sequences
allow cross-attention; causal mode requires aligned sequence lengths and masks
future keys independently in every head. Q/K/V projection parameters,
embeddings, a complete encoder/decoder stack, and training remain caller-owned.
Recurrent state must be reset between independent trials to avoid mixing
candidate histories.

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
cargo run -p nevo --example ch02_search --release --locked -- book/results/ch02
cargo run -p nevo --example ch02_xor --release --locked -- book/results/ch02
cargo run -p nevo --example ch02_neural --release --locked
cargo run -p nevo --example ch02_pdf_updates --release --locked -- book/results/pdf_review
```

The [Rust notebook](../notebooks/ch02_foundations.ipynb) uses the library APIs
for interactive experiments. See [the recorded results](results/ch02/README.md)
for actual settings, seeds, budgets, and outputs.

The original scalar search records use **eight dimensions**, seeds 0–19, and
6,400 objective calls per run, as specified in
[search_manifest.json](results/ch02/search_manifest.json). Their OpenAI-style
ES uses fixed sigma 0.5. They do not reproduce §2.2.7's **100-dimensional
Rastrigin** comparison on pages 35–36, which uses a sigma-annealing schedule for
OpenAI ES. Those original results remain intact. The additional PDF-review
demonstrations are recorded separately in [results/pdf_review](results/pdf_review/).

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
   Mapping: supplied Chapter 2 PDF, “The Basics,” printed pp. 15–48, reviewed
   2026-11-24. The section/page table above identifies implemented scope and
   the text distinguishes local defaults, source ambiguities, and remaining gaps.
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
   Mapping: scaled and multihead attention, sinusoidal positions, and residual
   attention/feedforward structure. The complete block remains a reduced
   single-head implementation; multihead attention is a separate forward operation.
9. **Zitzler, Deb, and Thiele (2000), _Comparison of Multiobjective Evolutionary Algorithms: Empirical Results_.**
   <https://doi.org/10.1162/106365600568202>
   Mapping: ZDT1 objective definitions and its original benchmark context.

These references establish algorithmic context. The chapter prose and Rust
implementations are original; references do not imply an endorsement by the
book authors or reproduction of the papers' experiments.
