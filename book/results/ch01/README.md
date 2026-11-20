# Recorded Chapter 1 run

These files come from the Rust example with seed 42 and the configuration in
`run.json`. This is a small demonstration, not a published benchmark reproduction.

- `run.txt`: actual command-line output.
- `learning.csv`: best retained loss after each of the 100 batches.
- `trajectories.csv`: no-action and evolved-controller traces at starts ±1.25.
- `fig_ch01.png`: summary chart rendered from those CSV values with Matplotlib.
- `run.json`: toolchain, seed, and experimental settings.

Regenerate the numerical files from the repository root:

```sh
cargo run -p nevo --example ch01_introduction --locked -- results/ch01
```

The Rust Jupyter notebook generates its own inline figures with Plotters from a
fresh run. The summary image does not participate in any optimization or tests.
