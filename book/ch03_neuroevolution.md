# Chapter 3: Neuroevolution — policies, topology, and scale

This chapter turns the search methods from Chapter 2 into neural agents and
classifiers. It adds three useful boundaries: a genotype is distinct from its
executable network, evaluating a policy includes an entire task, and scheduling
evaluations is distinct from changing evolutionary state.

The implementation was reviewed against the supplied Chapter 3 PDF on
2026-11-24: Risi, Tang, Ha, and Miikkulainen, *Neuroevolution: Harnessing
Creativity in AI Agent Design* (MIT Press, 2026), **“The Fundamentals of
Neuroevolution,” printed pp. 49–71**. The authors' teaching materials and the
primary references below provide additional implementation details. This is an
original Rust companion; its experiments have their own declared settings and
do not reproduce every experiment in the chapter.

| Book section and printed pages | Corresponding implementation and limits |
| --- | --- |
| §3.1, pp. 49–52: taxonomy | Flat weights, historical graphs, and seed lineages; developmental indirect encoding remains for Chapter 4. |
| §3.2, pp. 52–56: walking and generality | Episode aggregation and separate training/evaluation starts; the original spring robot is a teaching task, not the chapter's Box2D BipedalWalkerHardcore. |
| §3.3.2–3.3.3, pp. 58–62: NEAT mechanisms | Historical genes, structural mutation, aligned crossover, compatibility distance, and fitness sharing; an explicitly feedforward variant. |
| §3.3.4, pp. 63–65: double-pole balancing | Full-state 500-step experiment; recurrent NEAT and the book's 100,000-step success protocol remain unimplemented. |
| §3.4.2–3.4.3, pp. 67–70: deep neuroevolution and compute | Mutation-only Deep GA and ordered CPU evaluations; extra noisy elite-contender trials, accelerator batching, and distributed execution remain unimplemented. |

The classification and CartPole examples also follow the authors' separate
teaching materials. Their results are companion experiments, not additional
experiments reported in the supplied Chapter 3 text.

## 1. Choose what evolution can change

Fixed-topology neuroevolution chooses weights and biases for an architecture
specified by the experiment. A topology-and-weight method can also change the
network's graph. These are separate decisions from whether the policy has
recurrent state, and whether its encoding lists parameters explicitly or uses a
generative rule. The [original NEAT paper][neat-paper] discusses these
representation choices and the difficulty of recombining different networks.

The implemented representations make these choices concrete:

| Representation | Search object | Executable form | Current examples |
| --- | --- | --- | --- |
| Flat parameters | `Vec<f64>` | `nn::Mlp` with fixed widths and activations | GA/ES classifiers and control policies |
| Historical graph | `neat::Genome` | `FeedForwardNetwork` compiled from enabled edges | NEAT XOR, digit classification and double-pole control |
| Seed lineage | `encoding::SeedGenome` | `SeededEncoding::decode` reconstructs flat parameters | `DeepGa` and deterministic replay |

A seed lineage is a compact procedural representation of a fixed parameter
vector. It does not add developmental rules, spatial regularities, or a
HyperNEAT-style connectivity generator. Likewise, changing a graph is not the
same as giving it memory: this chapter's NEAT implementation only permits
feedforward directed acyclic graphs.

Keep the decoder in the evaluator. Then the optimizer can handle graphs or seed
histories without depending on how a neural network executes, or which world
supplies its observations.

## 2. The same Generate–Evaluate–Select contract

`Algorithm` still has only two evolutionary operations, `generate` and `select`;
an `Evaluator` supplies `evaluate`. **Select is the single result-consumption operation**, including updates to
parent pools, species, shared fitness, offspring allocation, and best-observed
records. Generate can sample mating parents internally while constructing
children; there is no second public update method to orchestrate.

Generate returns a complete batch of jobs. Each job has its input, a unique run
local ID, and an evaluation seed. Evaluate produces one output while preserving
the job. Select consumes every result in original submission order. A second
generation cannot begin while that batch is pending. Invalid identities, seeds,
inputs, order, missing results, extra results, and nonfinite scores are rejected
before selection commits.

The convenience function `step` executes that protocol. For example, this uses
the public NEAT and XOR building blocks directly:

```rust
use nevo::algorithms::{Neat, NeatConfig};
use nevo::tasks::NeatXor;
use nevo::step;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut search = Neat::new(NeatConfig::new(2, 1, 1))?;
    let mut task = NeatXor;
    let targets = [0.0, 1.0, 1.0, 0.0];

    for _ in 0..300 {
        step(&mut search, &mut task)?;
        let best = search.best().ok_or("no evaluated genome")?;
        let predictions = task.predictions(&best.job.input)?;
        if predictions.iter().zip(targets)
            .all(|(actual, target)| (actual - target).abs() <= 0.1)
        {
            break;
        }
    }
    println!("{} fitness evaluations", search.evaluations());
    Ok(())
}
```

The initial population counts as a batch. Copied champions receive fresh job IDs
and are evaluated again. A best-observed score records what happened at one
evaluation; under noise, it is not an unbiased estimate of the agent's quality.

## 3. NEAT: making structural changes comparable

NEAT combines historical alignment, speciation, and growth from minimal networks
[Stanley and Miikkulainen, 2002][neat-paper]. The reusable components live in
`nevo::neat`; `algorithms::Neat` composes them into GES.

**Historical genes.** A connection stores its innovation number, endpoints,
weight, and enable flag. Disabled connections remain available for alignment.
One `InnovationTracker` belongs to a run. Identical structural mutations reuse
identifiers within a reproduction generation; `begin_generation` clears those
lookup caches while retaining the global counters and identity records. Mixing
genomes from unrelated trackers is not a supported form of crossover.

**Structural mutation.** `add_connection` inserts an admissible missing edge.
`split_connection` disables an enabled edge of weight \(w\), introduces a hidden
node, and adds edges with weights \(1\) and \(w\). The new sigmoid changes the
function, so splitting is not exactly behavior preserving. This implementation
uses \(\sigma(4.9z)\), one constant bias, input sources, and output sinks.
Compilation follows a topological ordering, not increasing historical node IDs.

**Crossover.** Matching innovations choose a parent's gene at random. Unmatched
genes come from the fitter parent; equal-fitness parents contribute each
unmatched gene with probability one half. The default probability of disabling
a matching gene disabled in either parent is 0.75. Our feedforward guards retain
the earliest innovation when endpoint pairs collide and disable an inherited
edge that would close a cycle. They never rewrite historical identities.

**Species and sharing.** Compatibility uses

\[
\delta = c_E E/N + c_D D/N + c_W\overline{|\Delta w|}.
\]

Here \(E\) and \(D\) count excess and disjoint connection genes, including
disabled genes; the weight term averages matching genes. The default small
genome rule uses \(N=1\) below 20 genes. A genome joins the first representative
strictly within the fixed compatibility threshold.

For species \(s\), member fitness is divided by \(|s|\). Its reproductive weight
is their sum, equal to the species' mean raw fitness. Largest-remainder rounding
allocates an exact population budget; all-zero weights use equal species
weights. Parents are sampled uniformly from each species' highest-fitness
surviving fraction. Stagnant species lose allocation eligibility except for the
current champion's species. That exemption does not guarantee a positive quota.

The composed algorithm is approximately:

```text
Generate:
  initially: create minimal genomes with shared initial innovations
  later: clear generation innovation caches
         fill each species' quota with its champion, if eligible,
         and cloned/crossed parents followed by mutation
Evaluate:
  compile each genome and score its complete task
Select:
  validate the whole batch and nonnegative finite fitness
  assign species; update fitness histories and stagnation
  compute shared fitness, exact quotas, and surviving parent pools
  choose next representatives; commit population and reporting records
```

`Neat` maximizes nonnegative fitness. XOR uses
\((4-\sum_i |y_i-t_i|)^2\). Classifiers use \(1-\mathrm{MSE}\).
The fixed threshold, independent structural mutation events, quota-based
champion copying, and omitted global-stagnation rescue are explicit variant
choices. The compatibility and sharing calculations correspond to the book's
equations (3.1) and (3.2), p. 62. We divide fitness by assigned species size,
as described in its species-sharing explanation, rather than recomputing a
separate pairwise neighborhood for each individual. The small-genome rule and
generation-local innovation cache are details from the original NEAT paper.
Recurrent NEAT remains future work.

## 4. Fitness measures behavior under a trial distribution

A controller maps observations to actions, but its fitness is a property of the
resulting episodes. The control adapters minimize negative mean episode return:

\[
L(\theta)=-\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{T_k-1}r_{k,t}.
\]

Every candidate sees the same explicit training starts. The adapters reset the
environment for each start and use stateless tanh MLPs. Their job seed is unused
because this trial suite is fixed. A stochastic custom evaluator should instead
derive its randomness explicitly from the supplied seed. Evaluate held-out
starts after search, and report both episodes and actual environment steps:
one fitness call can contain many simulations.

This separation supports the robustness lesson in §3.2.5, pp. 55–56, and
[David Ha's walking experiments][ha-walking]: aggregating trials changes the
objective and the cost. The book's BipedalWalkerHardcore experiment averages
**16 random training rollouts per candidate** and then evaluates the final
agent over **100 random trials**, seeking an average return above 300. Our
fixed-start suites illustrate aggregation but do not implement that protocol
or establish broad generalization from a few successful trajectories.

The book also describes stochastic Gaussian action outputs on p. 56. The current
task adapters use deterministic policies; a Gaussian policy head is not yet a
reusable component here.

The headless environments have distinct, documented meanings:

| Environment | Observations and action | Integration and objective |
| --- | --- | --- |
| `CartPole` | Four state coordinates; one signed value selects left/right force | Explicit Euler, default 0.02 s; +1 per action, including the first failed transition; default 500-action horizon |
| `DoublePole` | Full state has six scaled coordinates; one continuous cart-force action | Two RK4 substeps per default 0.02 s action; count endpoints that remain within bounds; default 1,000-action horizon |
| `SpringWalker` | Thirteen observations; two bounded spring rest-length actions | Semi-implicit Euler with fixed substeps; displacement minus positive actuator-work cost and falling penalty |

CartPole follows the [Gymnasium equations and reward convention][cartpole], but
uses a different reset RNG, so matching integer seeds do not imply matching Gym
trajectories. DoublePole uses the [historical UT simulator's dynamics][ut-neat]
for two separate poles sharing a cart pivot. Its full-state teaching objective
does not reproduce the historical non-Markov oscillation scoring or
generalization protocol. `PositionsOnly` is available as an observation choice;
a feedforward controller still has no memory.

In §3.3.4, pp. 63–65, the book discusses both full-state and partially observed
double-pole tasks. Its compact positions-only solution has recurrent
connections, and its stated success criterion is **100,000 balanced steps**.
Our saved experiment uses full-state observations and a **500-step horizon**.
It cannot demonstrate that recurrent solution or claim the book's success
criterion. Stateful recurrent graph execution and its evolutionary operators
remain to be implemented.

SpringWalker is our original three-particle robot. Its body and feet interact
through springs and ground contact; there is no direct horizontal propulsion
force. It may hop or crawl as well as walk. Its morphology, contact model, and
reward are specific to this example. The book's §3.2 agent is instead an
articulated Box2D BipedalWalkerHardcore with **24 observations, four hip/knee
actions, and two hidden layers**. That environment and its terrain-generation
protocol are not implemented here; the spring robot's scores are not comparable
to its 300-point target.

## 5. A real classification experiment, with the right dataset name

The authors' public [classification exercise][exercise] calls its task MNIST,
but its code invokes `sklearn.datasets.load_digits`. That function loads the
small 8×8 digits dataset, not MNIST. Our bundled CSV comes from the real UCI
Optdigits `optdigits.tes` file: 1,797 samples, 64 integer features in 0–16, and ten
classes. Features are divided by 16. The original source, checksum, CC BY 4.0
attribution, and changes are recorded in [data/README.md](../data/README.md).
See the [UCI record][optdigits] and [scikit-learn dataset documentation][digits].

The experiment uses a declared stratified split: 60 training and 30 validation
examples per class, leaving 897 test examples. The split seed is 2026, and sample
IDs are written to `digits_split.csv`. These 600/300/897 partitions are disjoint;
they are a new educational split of the source test file, not UCI's original
writer-independent train/test protocol.

GA and ES optimize a literal **64 → 8 → 10** network: tanh hidden units,
independent sigmoid outputs, and 610 weights/biases. NEAT starts with 64 inputs,
one bias, ten sigmoid outputs, and 650 direct connections. All use mean squared
error against one-hot labels, averaged over samples and outputs; prediction is
the largest output, with the first class winning exact ties.

Each run uses 80 batches of 64 candidates: 5,120 training fitness evaluations,
each processing 600 examples. Three optimizer seeds are declared. Only training
fitness selects the reported model; validation and test metrics are calculated
after search. These modest budgets and different representations do not support
a tuned algorithm ranking or a comparison with gradient-trained digit models.

## 6. DeepGa and the cost of replaying seeds

`DeepGa` implements mutation-only truncation search inspired by
[Such et al.][deep-ga]. It retains the best `parents` candidates from the current
batch, samples parents uniformly, and mutates every decoded coordinate:

\[
\theta' = \theta + \sigma\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
\]

Defaults are 64 candidates, 16 retained parents, one reevaluated elite, uniform
initialization in \([-1,1)\), and fixed \(\sigma=0.1\). The objective direction
is configurable. There is no crossover or gradient estimation. Historical best
is a reporting record; only the current retained parents reproduce.

The genome contains an initialization seed and an ordered vector of mutation
seeds. `SeededEncoding` reconstructs the initial vector, then adds every
perturbation in lineage order. Each seed starts a fresh ChaCha8 stream; Gaussian
sampling uses uncached Box–Muller. Save dimensions, initialization bound,
mutation scale, and the implementation/version with a lineage.

For \(g\) mutations the seed payload is \(8(1+g)\) bytes, excluding allocation
metadata; decoding costs \(O(d(1+g))\) for \(d\) parameters. Long histories can be
larger than a small dense network and expensive to reconstruct. The reference
implementation has no shared noise table or checkpoint cache. Replay is exact
within the same numerical implementation/platform, not promised across future
RNG or transcendental-math changes. The book's §3.4.2, p. 68, describes the
same Gaussian mutation in equation (3.3), while also describing **30 additional
episodes for each of the top ten elite contenders** in the noisy Atari
experiments. Those extra trials and the large-scale experiments are outside
this implementation. Reevaluating our copied elite once in the next batch is
not that contender-selection procedure.

## 7. Parallel evaluation without a different algorithm API

Enable the optional `parallel` feature and use
`execution::step_parallel(&mut search, evaluator_factory)`. Generate and Select
stay on the caller's thread. [Rayon][rayon] distributes evaluations, and the
factory creates a fresh evaluator **per job**, including on a single worker.
Capture validated immutable settings or shared read-only data in the factory.

Inputs, outputs, and evaluator errors require `Send`; the factory requires
`Send + Sync`. Neither the algorithm nor the locally constructed evaluator
needs those bounds. The core traits remain unchanged. Results preserve
submission order, original inputs, IDs, and evaluation seeds. Tests compare
serial, one-worker, and four-worker runs exactly.

Empty or duplicate-ID batches fail before evaluators are constructed. Evaluation
does not short-circuit other jobs: after collecting outcomes, the executor
reports the error at the lowest submission index. Select receives no partial
batch. Side effects are not rolled back, panics propagate, and recovery is the
caller's responsibility. Scheduling-independent replay requires evaluators that
depend on explicit inputs and seeds, not shared mutable state.

This is CPU task parallelism. The accelerator and distributed systems discussed
in §3.4.3, pp. 69–70, are not reproduced by Rayon execution. Burn tensors, Bevy
integration, explicit SIMD, GPU batching, and distributed execution have not
been implemented here. Measure
end-to-end workloads before making speed claims; a fresh evaluator and lineage
decoding both have costs.

## Results and experiments

See the [recorded results and manifests](results/ch03/README.md) for exact
configurations, counts, curves, trajectories, and classification metrics.
The default NEAT XOR experiment solved 7 of the 10 declared seeds within its
150-candidate, 300-batch cap, using an error of at most 0.1 on every truth-table
row as the stop rule. Seed 1 is the short demonstration above; the other runs,
including failures, remain in the record. XOR has no held-out dataset.

Run examples from the repository root:

```sh
cargo run -p nevo --example ch03_neat --release -- book/results/ch03
cargo run -p nevo --example ch03_classification --release -- book/results/ch03
```

Useful extensions change one boundary at a time: hold the trial suite fixed
while changing the optimizer; keep the optimizer fixed while changing the
policy representation; then compare serial and parallel execution with identical
jobs. Record the added evaluations whenever a robustness check becomes part of
fitness. The public operators, environments, task adapters, and seed decoder are
available independently for these experiments.

## Primary references

- Risi, Tang, Ha, and Miikkulainen (2026), *Neuroevolution: Harnessing Creativity in AI Agent Design*, Chapter 3, pp. 49–71: supplied PDF reviewed on 2026-11-24; section mapping above. [Book website][book].
- [Stanley and Miikkulainen (2002)][neat-paper]: historical alignment, speciation, and incremental topology growth. [Accessible copy of the same paper][neat-copy].
- [UT NEAT software page][ut-neat]: historical double-pole simulator context; our equations and teaching objective are documented separately.
- [Ha (2017), Evolving Stable Strategies][ha-walking]: policy evaluation and averaging multiple locomotion trials.
- [Authors' Chapter 3 classification exercise][exercise]: instructional context and the actual `load_digits` call.
- [Alpaydin and Kaynak, UCI Optdigits][optdigits]: dataset provenance and attribution; [scikit-learn `load_digits`][digits] identifies the 1,797-sample subset.
- [Such et al., Deep Neuroevolution][deep-ga]: truncation-based mutation search and seed-history encoding.
- [Gymnasium CartPole documentation][cartpole]: benchmark state, action, reward, and termination conventions.
- [Rayon 1.10 indexed parallel iterators][rayon]: execution primitive; nevo's ordering and failure contract is specified above.

[book]: https://neuroevolutionbook.com/
[neat-paper]: https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
[neat-copy]: https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf
[ut-neat]: https://nn.cs.utexas.edu/soft-view.php?SoftID=4
[ha-walking]: https://blog.otoro.net/2017/11/12/evolving-stable-strategies/
[exercise]: https://www.cs.utexas.edu/~risto/cs378ne/private/hw2/Chapter3_Exercises_Problem1.ipynb
[optdigits]: https://doi.org/10.24432/C50P49
[digits]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
[deep-ga]: https://arxiv.org/abs/1712.06567
[cartpole]: https://gymnasium.farama.org/environments/classic_control/cart_pole/
[rayon]: https://docs.rs/rayon/1.10.0/rayon/iter/trait.IndexedParallelIterator.html
