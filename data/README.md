# Optdigits teaching data

`optdigits.csv` is the **unchanged 1,797-row `optdigits.tes` file** from the
UCI Optical Recognition of Handwritten Digits dataset, renamed for the Rust
examples. It contains 64 pixel values in 0–16 followed by a digit label in 0–9.
The library parser divides pixels by 16; the stored file is unmodified.

**Attribution:** E. Alpaydin and C. Kaynak (1998), *Optical Recognition of
Handwritten Digits*, UCI Machine Learning Repository,
[DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49).

**Dataset license:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
as specified on the [UCI dataset page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
This license applies to the data, independently of the project's still-unselected
source-code license.

Downloaded from the [official archive](https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip)
on 2026-11-23. Stored file size: 264,712 bytes.

SHA-256 of the original stored bytes:

```text
6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8
```

The author’s public Chapter 3 exercise describes “MNIST” but actually calls
`sklearn.datasets.load_digits`, which uses this 8×8 Optdigits subset. We name
the data **Optdigits/Digits**; it is not 28×28 MNIST.

Our example makes a new, explicit teaching split: 60 training and 30 validation
rows per class, with all remaining rows held out for test (600/300/897 rows).
Each class is shuffled with ChaCha8 seed 2026. Row IDs allow verifying that the
partitions are disjoint. This repartitions an original benchmark test file; it
is not the UCI benchmark's original train/test protocol and should not be used
to claim comparable benchmark scores or independence by writer.

The experiments use only training rows for selection. Validation and test data
are evaluated after the run and are not used to select generations or settings.
