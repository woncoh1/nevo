# Rust notebooks

The notebooks use the `nevo` library in this repository and contain real saved
outputs from a Rust/Evcxr kernel. They are original companion materials inspired
by [Neuroevolution](https://neuroevolutionbook.com/), not official exercise
solutions or copies of the textbook.

| Notebook | Contents |
| --- | --- |
| [Chapter 1](ch01_introduction.ipynb) | Evolve a three-gene controller, inspect held-out trajectories, then run the 100/20/80 GA population companion. |
| [Chapter 2](ch02_foundations.ipynb) | GES, original four scalar searches, DE and ES modes, NSGA-II/ZDT1, MLP/XOR, recurrence, pooling/softmax and single/multi-head attention. |
| [Chapter 3](ch03_neuroevolution.ipynb) | NEAT structure/GES, genuine Optdigits classification, control/held-out trials, spring-robot replay, seed encoding and parallel Deep GA. |

## Run interactively

Install [Rust](https://www.rust-lang.org/tools/install) and Python, then:

```bash
python -m pip install jupyterlab
rustup toolchain install stable
rustup component add rust-src --toolchain stable
cargo +stable install --locked evcxr_jupyter
evcxr_jupyter --install
```

From the repository root:

```bash
cd notebooks
rustup run stable jupyter lab
```

Open a notebook, select the **Rust** kernel, and choose **Restart Kernel and Run
All Cells**. Its working directory must be `notebooks/`; the dependency cell
loads `../crates/nevo`. First execution downloads and compiles dependencies.
Start from a fresh kernel when reproducing a run: rerunning stateful training
cells can continue existing state or replace it according to that cell's code.

The library targets Rust 1.85.1 or newer. The current Evcxr release may require
a newer compiler, so the commands above use the `stable` toolchain for notebook
execution. The recorded validation used **Evcxr 0.15.1 and Rust 1.85.1**; see
[VALIDATION.md](../VALIDATION.md) for the tested versions.

All three notebooks pin Plotters 0.3.7 with `evcxr` and `line_series` features. The
Chapter 2 notebook also declares rand_chacha 0.9.0 to reproduce independent
initialization stream 2. Chapter 3 enables `nevo`'s `parallel` feature and
pins Rayon 1.10.0 for an explicitly sized four-worker pool. These notebook dependencies do not add plotting tools
to the library dependency graph. Notebook package resolution uses Evcxr's
separate generated Cargo project; the workspace lockfile controls Cargo builds
of the repository itself.

All search, neural and environment computations in the notebooks are Rust.
Python supplies Jupyter and execution tooling. Charts are inline SVG; no
simulation window is required.

## Chapter 2 data and work budgets

The delivered notebook reads the committed 20-seed experiment. To regenerate
that CSV before opening the notebook, run from the repository root:

```sh
cargo run -p nevo --example ch02_search --release --locked -- book/results/ch02
cargo run -p nevo --example ch02_xor --release --locked -- book/results/ch02
```

Live scalar runs use 6,400 objective calls each. GA/ES share a uniform initial
population; CMA/OpenAI use a Gaussian around a separately randomized center.
ZDT1 uses 30 genes and 12,800 evaluations. XOR stops at MSE at most 0.001 or its
budget cap; evolutionary evaluations and gradient steps are different work
units. The notebook states these distinctions alongside the plots.

Chapter 1's original demo uses 3,200 search evaluations, each with four 80-step episodes.
The appended population companion uses another 3,200 calls with the same task
and independently reports its held-out loss. Chapter 2's appended DE/fixed-ES
cells add three runs of 6,400 calls each, plus hand-computed neural operations.
Its baseline, held-out loss and trajectory recording are additional diagnostics.
Held-out starts are not used for selection. See the respective result folders
for exact configurations and interpretation.

## Chapter 3 data and work budgets

The notebook uses the bundled attributed Optdigits CSV and reads the committed
Chapter 3 experiment records. Its live computations include:

- Feedforward NEAT XOR, seed 1, population 150, at most 300 selected batches.
- GA classification, seed 0, 80 batches × 64 fitness calls × 600 training images.
- GA CartPole, seed 0, three training starts, stop at full training horizon or
  60 batches, then evaluate 20 held-out starts.
- Replay the saved spring-robot parameters at neutral and held-out low friction.
- Deep GA on a 609-parameter MLP, seed 42, serial and four-worker runs checked
  for identical outcomes, with at most 100 batches × 64 candidates.

The full CLI experiments additionally include ten XOR seeds, three classifier
seeds per method, GA/ES CartPole, full-state double-pole, and the spring-robot
search. Commands and work accounting are in
[the Chapter 3 results report](../book/results/ch03/README.md). Training calls,
reporting passes and held-out episodes are separate quantities. No reported
holdout score selected the saved policies or their stopping points.

All inline notebook charts use Rust Plotters. Optional static report figures
can be regenerated with the separate Python plotting script; this does not
replace any Rust search or evaluation.

## Headless execution

After installing the kernel and nbconvert:

```bash
python -m pip install nbconvert
cd notebooks
rustup run stable jupyter nbconvert --to notebook --execute --inplace ch03_neuroevolution.ipynb --ExecutePreprocessor.timeout=600
```

Replace the filename with `ch01_introduction.ipynb` or `ch02_foundations.ipynb`
to run an earlier chapter. The
command-line examples provide an alternative when Jupyter is unnecessary.

## References

- [Evcxr installation and usage](https://github.com/evcxr/evcxr/blob/main/evcxr_jupyter/README.md)
- [Plotters 0.3.7](https://docs.rs/plotters/0.3.7/plotters/)
- [Chapter 2 explanations and method references](../book/ch02_foundations.md)

- [Chapter 3 explanations and method references](../book/ch03_neuroevolution.md)
