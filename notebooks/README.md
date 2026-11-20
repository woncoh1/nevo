# Rust notebooks

`ch01_introduction.ipynb` evolves a three-gene neural controller through the
public `nevo` API, then plots its learning curve and held-out trajectories.
The notebook is original companion material inspired by Chapter 1 of
[Neuroevolution: Harnessing Creativity in AI Agent Design](https://neuroevolutionbook.com/).
It is not an official book assignment or a reproduction of the chapter text.
The book's listed coding assignments start at Chapter 2.

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

Open `ch01_introduction.ipynb`, select the **Rust** kernel, and use **Restart
Kernel and Run All Cells**. The notebook expects its working directory to be
`notebooks/`: its first cell loads `../crates/nevo`. Adjust that path if you
move the notebook. The first execution downloads and compiles dependencies;
later cells reuse the same kernel state.

The library targets Rust 1.85.1 or newer. Installing the current Evcxr release
may require a newer compiler, which is why the commands use the `stable`
toolchain for notebook execution. This does not change the repository's
minimum supported library version.

The notebook uses Plotters 0.3.7 with only `evcxr` and `line_series` enabled.
Plotters is a notebook dependency; it is not added to the library's dependency
graph. Charts are inline SVG output, so no graphical simulation window is
needed. No Python optimizer or simulation code is involved; Python supplies
the Jupyter application.

## Reproducibility

Start with a fresh kernel when changing settings or repeating the experiment.
Rerunning only the training cell continues the existing search for another
100 batches. The default complete run uses seed 42 and 100 batches of 32
candidates, including random initialization. This means 3,200 search
evaluations, 12,800 training episodes, and 1,024,000 simulation steps.

The no-action baseline, held-out score, and plotted rollouts are extra
diagnostics outside that search budget. Held-out starts are not consulted
until training finishes. Their scores are useful diagnostics, not a proof
of generalization. Do not compare raw losses as if the training and held-out
start distributions were identical.

The delivered notebook includes real outputs from a successful seven-cell
Rust/Evcxr run, including both inline figures. See the root `VALIDATION.md` for
the exact tested toolchain and kernel versions.
Notebook cells call library code rather than carrying a second optimizer
implementation.

## Command-line alternatives

Run the equivalent library example without Jupyter, from the repository root:

```bash
cargo run -p nevo --example ch01_introduction --locked -- results/ch01
```

To execute the notebook and save its real outputs after installing nbconvert:

```bash
python -m pip install nbconvert
cd notebooks
rustup run stable jupyter nbconvert --to notebook --execute --inplace ch01_introduction.ipynb --ExecutePreprocessor.timeout=600
```

The first command-line example writes numerical CSV outputs; the notebook
produces inline figures. Both run the same `nevo` implementation.

## Documentation

- [Evcxr Jupyter installation and usage](https://github.com/evcxr/evcxr/blob/main/evcxr_jupyter/README.md)
- [Plotters 0.3.7](https://docs.rs/plotters/0.3.7/plotters/)
- [Book contents and exercise list](https://neuroevolutionbook.com/)
