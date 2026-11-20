# Coverage

This registry separates runnable teaching examples, implemented methods, and
validated reproductions. A chapter entry does not imply that every method or
case study in that chapter is implemented.

| Book area | Current status | Evidence |
| --- | --- | --- |
| Chapter 1: introduction | Original companion demonstration implemented. | `book/ch01_introduction.md`, `examples/ch01_introduction.rs`, Rust notebook. |
| Chapter 1 official programming assignment | None listed on the official site. | Official assignment list begins with Chapter 2. |
| Foundational execution interface | GES protocol, typed jobs/results, serial runner implemented and tested. | `nevo-core`, protocol and integration tests. |
| Original uniform-mutation elitist search | Implemented for deterministic scalar objectives. | `algorithms::ElitistSearch`, minimization/maximization and protocol tests. |
| Sphere reference problem | Implemented; numerical/reference checks included. | `benchmarks/sphere.md`, evaluator and tests. |
| Chapter 2: specified GA and ES | Next planned algorithm implementations. The teaching search does not close this coverage item. | No claim of canonical reproduction yet. |
| Chapters 3–14 | Not implemented as chapter reproductions. | Future work. |
| Burn and Bevy | Planned optional integrations. | No dependency or adapter yet. |

## Source boundary

The official contents identify Chapter 1 as introductory, with the first
programming assignments in Chapter 2. The full HTML chapter was unavailable
when checked on 2026-11-19; the site announces return on 2027-01-28. This milestone
does not claim a line-by-line reading, reproduction of review questions, or
completion of all Chapter 1 material.

Source: [official book site](https://neuroevolutionbook.com/).

## Validation levels

1. **Implemented:** executable code exists.
2. **Tested:** specified behavior passes recorded automated checks.
3. **Reference-validated:** compared with the original method and a reference implementation.
4. **Research-reproduced:** environment, budget, metrics, and multiple-seed results match a published protocol.

The current companion reaches the first two levels. It makes no claims of
published research reproduction. See `VALIDATION.md` for the concrete commands.
