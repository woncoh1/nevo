# Rust notebooks

The notebooks use the `nevo` library in this repository and contain real saved
outputs from a Rust/Evcxr kernel. They are original companion materials inspired
by [Neuroevolution](https://neuroevolutionbook.com/), not official exercise
solutions or copies of the textbook.

| Notebook | Contents |
| --- | --- |
| [Chapter 1](ch01_introduction.ipynb) | Evolve a three-gene controller, then inspect learning and held-out trajectories. |
| [Chapter 2](ch02_foundations.ipynb) | GES, four scalar searches, multi-seed results, NSGA-II/ZDT1, MLP gradients, XOR, recurrence, convolution and attention. |

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

Both notebooks pin Plotters 0.3.7 with `evcxr` and `line_series` features. The
Chapter 2 notebook also declares rand_chacha 0.9.0 to reproduce independent
initialization stream 2. These notebook dependencies do not add plotting tools
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

Chapter 1 still uses 3,200 search evaluations, each with four 80-step episodes.
Its baseline, held-out loss and trajectory recording are additional diagnostics.
Held-out starts are not used for selection. See the respective result folders
for exact configurations and interpretation.

## Headless execution

After installing the kernel and nbconvert:

```bash
python -m pip install nbconvert
cd notebooks
rustup run stable jupyter nbconvert --to notebook --execute --inplace ch02_foundations.ipynb --ExecutePreprocessor.timeout=600
```

Replace the filename with `ch01_introduction.ipynb` to run Chapter 1. The
command-line examples provide an alternative when Jupyter is unnecessary.

## References

- [Evcxr installation and usage](https://github.com/evcxr/evcxr/blob/main/evcxr_jupyter/README.md)
- [Plotters 0.3.7](https://docs.rs/plotters/0.3.7/plotters/)
- [Chapter 2 explanations and method references](../book/ch02_foundations.md)
