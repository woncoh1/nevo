# nevo book companion

Original explanations and executable experiments using the library in this
same repository. These are companion materials inspired by
[Neuroevolution](https://neuroevolutionbook.com/), not a redistribution of the
textbook or official exercise solutions.

| Chapter | Walkthrough | Notebook | Recorded results |
| --- | --- | --- | --- |
| 1 · Introduction | [Neural controller](ch01_introduction.md) | [Rust notebook](../notebooks/ch01_introduction.ipynb) | [Run data](results/ch01/) |
| 2 · Foundations | [Evolutionary and neural building blocks](ch02_foundations.md) | [Rust notebook](../notebooks/ch02_foundations.ipynb) | [Experiments](results/ch02/README.md) |
| 3 · Neuroevolution | [Topology, control and scaling](ch03_neuroevolution.md) | [Rust notebook](../notebooks/ch03_neuroevolution.ipynb) | [Controlled generalization](results/ch03_generalization/README.md) · [historical experiments](results/ch03/README.md) |

The supplied full Chapters 1–3 were reviewed on 2026-11-24. See the
[source audit](source-audit-ch01-ch03.md) and [new example records](results/pdf_review/README.md).

Read [coverage](../coverage.md) for the exact implemented variants and remaining
scope. All algorithm implementations live in the library; the chapters and
notebooks demonstrate its public APIs.
