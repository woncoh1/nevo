# Double-pole controlled protocol: controlled_v1

This file and starts.csv were written before any optimization or held-out evaluation.

- Optimizer seed: 6; full study uses seeds 0 through 9.
- Population: 64; training rollout episodes per condition: 32768; horizon: 500; held-out starts: 100.
- One-start conditions use 512 complete batches; 16-start conditions use 32. Initialization is batch 1 and counts toward the budget. No early stopping.
- Fixed training suite: historical default first, then 15 pseudorandom starts. A single-start condition uses only the first entry.
- Suite RNG: ChaCha8; stream 4; training seed 5640009007068496201; independent held-out seed 5640009007067648852.
- Sampling draw order: position, velocity, long angle, short angle, long angular velocity, short angular velocity. Uniform half-open intervals [-b,b), b=[0.1,0.02,0.07,0.005,0.02,0.05]. Exact train/test disjointness and held-out uniqueness are checked.
- Selection: highest mean training return actually observed at the fixed budget; exact ties retain the earliest evaluated record. CMA minimizes its negative. Fixed suites permit historical comparison. Only selected final policies see held-out starts; no held-out tuning, selection or stopping.
- Reward: one per physically valid control endpoint, zero for a failing endpoint. Step counts include a failing action. Horizon success requires truncated=true AND terminated=false.
- Every episode resets environment and recurrent phenotype. Recurrent mode uses synchronous ticks: current inputs/bias, previous hidden/output activations. Initial hidden/output state is zero.
- Recurrent diagnostic: the same selected recurrent genome is additionally evaluated with state reset before every action, on the same held-out starts. This also suppresses delayed acyclic hidden-to-output paths; it is not an edge-deletion or equivalent feedforward baseline. These 100 extra episodes are reported separately and never used for selection.
- Predeclared traces: only optimizer seed 0 and held-out case 0, for all six selected policies plus the recurrent state-reset diagnostic. These reporting replays occur after selection and are counted separately in trace_replay_episodes/control_steps. The trace return column is cumulative; step 0 is the initial state with no action.
- Sample SD uses n-1, with a blank CSV value for n=1. Trial SD is variability across starts; optimizer-run uncertainty must be computed across independent seed rows, not pooled trials.
- CMA controller: 6 -> 4 -> 1 tanh/tanh MLP, 33 parameters. Initial mean uniform [-0.5,0.5) from optimizer seed, stream 2; sigma 0.5. NEAT: direct sigmoid output mapped to 2*y-1; defaults except population and mode.
- Fairness scope: equal rollout-episode budgets and common populations; fitness-call counts differ with suite size. Control steps and runtime are separately reported. Architectures, activations, search methods, initializations and adaptation horizons differ. This is not a benchmark of optimizers with an identical representation, and the initial-state distribution is deliberately local.
- Reported seconds cover the whole condition: setup, training, policy serialization, held-out/ablation evaluation and trace replays, with some I/O. It is not training-only timing. Across-seed uncertainty is conditional on this one fixed training and held-out suite.
- This is a modern controlled teaching experiment, not a reproduction of the original 2002 NEAT non-Markov oscillation/generalization protocol, not a reproduction of all settings in the book, and not evidence from a smoke run.

Sources: <https://neuroevolutionbook.com/>; <https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf>; <https://arxiv.org/abs/1604.00772>.

## full_cma_1

Training starts: 1; physics:

```rust
DoublePoleConfig {
    gravity: 9.8,
    cart_mass: 1.0,
    pole_masses: [
        0.1,
        0.01,
    ],
    half_lengths: [
        0.5,
        0.05,
    ],
    hinge_damping: 2e-6,
    force: 10.0,
    dt: 0.02,
    substeps: 2,
    position_limit: 2.4,
    angle_limit: 0.628329,
    max_steps: 500,
    observation: FullState,
}
```

Search configuration:

```rust
CmaEsConfig {
    initial_mean: [
        -0.21464177976937193,
        0.21762937008518068,
        -0.4186286041670282,
        -0.3524487967633251,
        0.17073170150409467,
        -0.49570758138786075,
        -0.1010592297147892,
        -0.2505879343834786,
        0.10699033593667262,
        -0.16241441018885416,
        -0.28185604815197884,
        0.194375477585919,
        -0.2346168615573765,
        -0.33737790060212725,
        -0.4966629864820318,
        -0.27832280539033794,
        -0.2748450934969813,
        -0.4316905178402347,
        -0.4448823464105318,
        -0.4980444005698479,
        0.39156274695412974,
        -0.04320014122320248,
        0.36341730349434664,
        -0.4870838814624251,
        0.17946624480953655,
        -0.04372844328325065,
        0.28402852291349046,
        0.0302547572844184,
        0.449757943643049,
        0.4297321831978236,
        0.38284779483628484,
        -0.04577926928358933,
        0.13692598582569748,
    ],
    sigma: 0.5,
    population: 64,
    objective: Minimize,
    seed: 6,
}
```

## full_cma_16

Training starts: 16; physics:

```rust
DoublePoleConfig {
    gravity: 9.8,
    cart_mass: 1.0,
    pole_masses: [
        0.1,
        0.01,
    ],
    half_lengths: [
        0.5,
        0.05,
    ],
    hinge_damping: 2e-6,
    force: 10.0,
    dt: 0.02,
    substeps: 2,
    position_limit: 2.4,
    angle_limit: 0.628329,
    max_steps: 500,
    observation: FullState,
}
```

Search configuration:

```rust
CmaEsConfig {
    initial_mean: [
        -0.21464177976937193,
        0.21762937008518068,
        -0.4186286041670282,
        -0.3524487967633251,
        0.17073170150409467,
        -0.49570758138786075,
        -0.1010592297147892,
        -0.2505879343834786,
        0.10699033593667262,
        -0.16241441018885416,
        -0.28185604815197884,
        0.194375477585919,
        -0.2346168615573765,
        -0.33737790060212725,
        -0.4966629864820318,
        -0.27832280539033794,
        -0.2748450934969813,
        -0.4316905178402347,
        -0.4448823464105318,
        -0.4980444005698479,
        0.39156274695412974,
        -0.04320014122320248,
        0.36341730349434664,
        -0.4870838814624251,
        0.17946624480953655,
        -0.04372844328325065,
        0.28402852291349046,
        0.0302547572844184,
        0.449757943643049,
        0.4297321831978236,
        0.38284779483628484,
        -0.04577926928358933,
        0.13692598582569748,
    ],
    sigma: 0.5,
    population: 64,
    objective: Minimize,
    seed: 6,
}
```

## full_ff_neat_1

Training starts: 1; physics:

```rust
DoublePoleConfig {
    gravity: 9.8,
    cart_mass: 1.0,
    pole_masses: [
        0.1,
        0.01,
    ],
    half_lengths: [
        0.5,
        0.05,
    ],
    hinge_damping: 2e-6,
    force: 10.0,
    dt: 0.02,
    substeps: 2,
    position_limit: 2.4,
    angle_limit: 0.628329,
    max_steps: 500,
    observation: FullState,
}
```

Search configuration:

```rust
NeatConfig {
    inputs: 6,
    outputs: 1,
    mode: FeedForward,
    population: 64,
    seed: 6,
    mutation: MutationConfig {
        weight_probability: 0.8,
        perturb_probability: 0.9,
        perturb_bound: 0.5,
        weight_bound: 1.0,
        add_node_probability: 0.03,
        add_connection_probability: 0.05,
    },
    compatibility: CompatibilityConfig {
        excess: 1.0,
        disjoint: 1.0,
        weight: 0.4,
        small_genome_threshold: 20,
    },
    compatibility_threshold: 3.0,
    crossover_probability: 0.75,
    disable_probability: 0.75,
    interspecies_mating_probability: 0.001,
    survival_fraction: 0.2,
    stagnation_generations: 15,
    champion_min_offspring: 5,
}
```

## full_ff_neat_16

Training starts: 16; physics:

```rust
DoublePoleConfig {
    gravity: 9.8,
    cart_mass: 1.0,
    pole_masses: [
        0.1,
        0.01,
    ],
    half_lengths: [
        0.5,
        0.05,
    ],
    hinge_damping: 2e-6,
    force: 10.0,
    dt: 0.02,
    substeps: 2,
    position_limit: 2.4,
    angle_limit: 0.628329,
    max_steps: 500,
    observation: FullState,
}
```

Search configuration:

```rust
NeatConfig {
    inputs: 6,
    outputs: 1,
    mode: FeedForward,
    population: 64,
    seed: 6,
    mutation: MutationConfig {
        weight_probability: 0.8,
        perturb_probability: 0.9,
        perturb_bound: 0.5,
        weight_bound: 1.0,
        add_node_probability: 0.03,
        add_connection_probability: 0.05,
    },
    compatibility: CompatibilityConfig {
        excess: 1.0,
        disjoint: 1.0,
        weight: 0.4,
        small_genome_threshold: 20,
    },
    compatibility_threshold: 3.0,
    crossover_probability: 0.75,
    disable_probability: 0.75,
    interspecies_mating_probability: 0.001,
    survival_fraction: 0.2,
    stagnation_generations: 15,
    champion_min_offspring: 5,
}
```

## partial_ff_neat_16

Training starts: 16; physics:

```rust
DoublePoleConfig {
    gravity: 9.8,
    cart_mass: 1.0,
    pole_masses: [
        0.1,
        0.01,
    ],
    half_lengths: [
        0.5,
        0.05,
    ],
    hinge_damping: 2e-6,
    force: 10.0,
    dt: 0.02,
    substeps: 2,
    position_limit: 2.4,
    angle_limit: 0.628329,
    max_steps: 500,
    observation: PositionsOnly,
}
```

Search configuration:

```rust
NeatConfig {
    inputs: 3,
    outputs: 1,
    mode: FeedForward,
    population: 64,
    seed: 6,
    mutation: MutationConfig {
        weight_probability: 0.8,
        perturb_probability: 0.9,
        perturb_bound: 0.5,
        weight_bound: 1.0,
        add_node_probability: 0.03,
        add_connection_probability: 0.05,
    },
    compatibility: CompatibilityConfig {
        excess: 1.0,
        disjoint: 1.0,
        weight: 0.4,
        small_genome_threshold: 20,
    },
    compatibility_threshold: 3.0,
    crossover_probability: 0.75,
    disable_probability: 0.75,
    interspecies_mating_probability: 0.001,
    survival_fraction: 0.2,
    stagnation_generations: 15,
    champion_min_offspring: 5,
}
```

## partial_recurrent_neat_16

Training starts: 16; physics:

```rust
DoublePoleConfig {
    gravity: 9.8,
    cart_mass: 1.0,
    pole_masses: [
        0.1,
        0.01,
    ],
    half_lengths: [
        0.5,
        0.05,
    ],
    hinge_damping: 2e-6,
    force: 10.0,
    dt: 0.02,
    substeps: 2,
    position_limit: 2.4,
    angle_limit: 0.628329,
    max_steps: 500,
    observation: PositionsOnly,
}
```

Search configuration:

```rust
NeatConfig {
    inputs: 3,
    outputs: 1,
    mode: Recurrent,
    population: 64,
    seed: 6,
    mutation: MutationConfig {
        weight_probability: 0.8,
        perturb_probability: 0.9,
        perturb_bound: 0.5,
        weight_bound: 1.0,
        add_node_probability: 0.03,
        add_connection_probability: 0.05,
    },
    compatibility: CompatibilityConfig {
        excess: 1.0,
        disjoint: 1.0,
        weight: 0.4,
        small_genome_threshold: 20,
    },
    compatibility_threshold: 3.0,
    crossover_probability: 0.75,
    disable_probability: 0.75,
    interspecies_mating_probability: 0.001,
    survival_fraction: 0.2,
    stagnation_generations: 15,
    champion_min_offspring: 5,
}
```

