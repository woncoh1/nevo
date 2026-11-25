# Independent implementation and protocol review

Source reviewed on 2026-11-25 before the full experiment results were available;
the full-data audit below was added after all 60 runs completed that day.
This record distinguishes read-only source inspection and independent output
checks from executing Rust test suites. No blocking correctness issue was
found in the reviewed implementation. Integrated test execution is recorded
separately; the final section audits the completed experiment records.

## Scope

Reviewed the recurrent phenotype and mode-aware NEAT operations, their new
regression tests, `control/trials.rs`, the trace-free episode runner,
`tasks/control.rs`, and `examples/ch03_generalization.rs`.

## Recurrent execution and state isolation

- Inputs and bias use current values. Every hidden/output source uses its
  previous-tick value, including on acyclic edges. All neuron updates commit
  simultaneously; node iteration order does not determine propagation time.
- Creation and reset initialize hidden/output activations to zero. Cloning a
  phenotype copies its state; compiling a genome creates fresh state. Episode
  state is not stored in the genotype.
- Shape, nonfinite-input, and affine-overflow errors do not commit partial
  neuron state. The next valid call overwrites the relevant scratch values.
- Execution mode is explicit in genomes and innovation trackers. Compilation,
  compatibility, crossover, and mutation check or preserve it. Feedforward mode
  retains its DAG rules; recurrent mode allows feedback and self-links while
  still forbidding destinations at input/bias nodes.
- The new regression tests cover direct-input timing, delayed hidden paths,
  self-loop memory, node-order independence, resets, recovery after partial
  scratch writes, structural history, mode isolation, and seeded replay. This
  review inspected those tests but did not execute them.

The [official NEAT-Python recurrent implementation][neat-python] provides a
precedent for this current-input/previous-neuron convention: it uses two
activation buffers and resets both. This does not establish equivalence with
the original NEAT software's activation loop or historical experiments.

## Trial statistics and episode accounting

- The reusable trial helper computes mean return, worst return, total applied
  actions, and horizon hits. Sample standard deviation uses Welford's
  recurrence and denominator `n - 1`; it is absent for a single trial.
- Horizon success requires `truncated && !terminated`. A failing final action
  counts as an action, even though double-pole assigns it zero reward.
- NEAT task evaluation resets both physical state and recurrent state before
  each start. Single-episode and trace methods create fresh policy state.
- Held-out summaries aggregate recorded episode results without rerunning the
  controller. The state-reset diagnostic has separate episode and step counts.
- Predeclared trace replays use seed 0, held-out case 0, after policy selection.
  Their extra episodes and actions have separate `trace_replay_*` counts and do
  not enter the training budget or held-out summary denominator.

The final smoke output was also inspected for CSV column alignment, episode
and action accounting, independent mean/sample-SD recalculation, agreement
between each trace replay and its original episode, and agreement between the
last training row and run summary. These checks found no discrepancy. This
checks reporting mechanics only; no algorithm-performance inference follows.

## Frozen comparison protocol

The reviewed full protocol has population 64, a 500-action horizon, seeds
0 through 9, and six conditions. Each condition receives exactly 32,768
training episodes, including initialization and without early stopping:

| Starts per candidate | Complete batches | Fitness calls | Training episodes |
| ---: | ---: | ---: | ---: |
| 1 | 512 | 32,768 | 32,768 |
| 16 | 32 | 2,048 | 32,768 |

The fixed 16-start training suite begins with the historical default start;
one-start conditions use that entry. A separately seeded 100-start held-out
suite is shared across conditions and optimizer seeds. The driver checks exact
train/test disjointness and held-out uniqueness and writes the protocol and
initial states before optimization.

Selection uses only the highest mean training return actually evaluated;
exact ties keep the earliest record. Fixed training suites make historical
scores comparable. The selected job is checked against the algorithm's
recorded best candidate and its controller is saved before held-out evaluation.
Held-out results do not affect search, stopping, or final-policy selection.

## Limits for interpreting the eventual results

1. Equal episode budgets do not imply equal action counts, runtime, or search
   updates. Architectures, activations, initializations, and optimizer settings
   also differ. Report episode counts, action counts, and measured time with
   their distinct meanings.
2. The driver's `seconds` field measures the whole condition, including setup,
   training, policy saving, held-out evaluation, any state-reset diagnostic,
   trace replays, and some output work. It is not training-only time. The
   execution record reports up to four concurrent seed processes, so these
   wall times are not isolated timing benchmarks.
3. Variation across ten optimizer seeds is conditional on the fixed training
   and held-out suites. Reusing 100 held-out starts does not create 1,000
   independent initial-state samples. Across-start SD and across-seed
   uncertainty describe different quantities.
4. Resetting the recurrent phenotype before every action suppresses all
   previous-neuron signals, including delayed acyclic hidden-to-output paths.
   This measures dependence on persistent synchronous state, not dependence
   exclusively on cycle edges or performance of an equivalent feedforward
   network. Its additional 100 episodes are diagnostic, not training.
5. The start distribution is local. Short-horizon results under this
   survival-return objective do not establish global control robustness or
   reproduce the original non-Markov benchmark. The [2002 NEAT paper][neat]
   uses different fitness and generalization criteria; the distinction must
   remain explicit in plots, notebooks, and conclusions.
6. Smoke runs check wiring only. Source inspection and smoke output alone do
   not establish algorithm superiority, convergence reliability, or an
   empirical benefit from recurrence or additional training starts.

## Full-data audit

The final audit independently read all ten seed directories and recalculated
statistics from their trial records, without rerunning experiments or Rust
tests. It found no discrepancy in:

- All 60 declared run records and 11,520 training-batch rows, including exact
  episode budgets, batch sequences, monotonic best-training scores, and final
  run/training-row agreement.
- Per-batch action accounting: total actions equal valid-endpoint rewards plus
  one final failing action for each unsuccessful episode.
- Every held-out and state-reset mean, sample SD, minimum, horizon count, and
  action count, recomputed from 7,000 episode records.
- Equality of all ten saved start-suite files; train/test disjointness and
  uniqueness of the 100 held-out starts.
- Saved-policy selection metadata, trace replay counts, and agreement between
  each replay's final return/flags and its original held-out episode.
- Numeric agreement between per-seed records, aggregate `runs.csv`, and every
  reported mean/sample SD in `summary.csv`.

The audited totals are **1,966,080 training episodes and 262,852,364 training
actions**, **7,000 held-out/ablation episodes and 2,243,733 actions**, and **seven
reporting replays with 2,595 actions**. Trace files have 2,602 state rows because
each replay also records its initial state. These work categories remain
separate in the data.

The results support bounded conclusions about this protocol:

- In full-state control, mean held-out horizon rates changed from 76.5% to
  63.8% for CMA-ES and from 90.5% to 94.9% for feedforward NEAT when moving from
  one to 16 training starts. Seed-level changes are mixed. The larger suite
  also reduces evaluated batches from 512 to 32, so this is a budget-allocation
  tradeoff, not an isolated measurement of training diversity.
- Positions-only recurrent NEAT had mean held-out return 290.555, with
  across-seed sample SD 196.207; feedforward NEAT had 73.622 with SD 10.948.
  Nevertheless, five recurrent runs had zero horizon hits. Their hit counts
  in seed order were `99, 99, 0, 0, 0, 0, 0, 100, 100, 49` out of 100.
- Resetting the same selected recurrent policies before each action reduced
  mean return in eight seeds and left two unchanged; every reset policy had
  zero horizon hits. This supports dependence on persistent state in the
  successful controllers under the declared ablation, without isolating
  cycle edges or establishing that all recurrent runs learn reliable control.

The earlier single-controller comparison did not establish an algorithm
ranking, and this broader experiment still does not establish a universal one.
The sample SDs describe variability; no confidence-interval or statistical-
significance claim follows from them. Current README, DESIGN, coverage, and
Chapter 3 text were also checked for stale claims that recurrence is absent;
none were found. Historical result sections are explicitly retained as such.

[neat-python]: https://neat-python.readthedocs.io/en/latest/_modules/nn/recurrent.html
[neat]: https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
