# Put this starter on GitHub

Extract the archive and open a terminal inside the `nevo` directory (the one
containing `Cargo.toml`). The archive contains the source files and a CI workflow,
not a preconfigured Git repository or credentials.

## Verify locally

The pinned development toolchain is Rust 1.100.0, including rustfmt, Clippy,
rust-analyzer and rust-src. Rustup installs it when you enter this workspace;
the library's separately tested minimum Rust version remains 1.85.1.
Shared `.vscode/settings.json` and extension recommendations are safe to commit:
they contain project settings and no machine-specific paths or credentials.
Read [STYLE.md](STYLE.md) for the conventions.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo run -p nevo --example ch03_generalization --release --locked -- results/ch03_smoke --smoke
```

## First push to a new, empty repository

Create an empty repository named `nevo` on GitHub. For this workflow, leave its
README, license, and `.gitignore` initialization unchecked so the remote has no
separate initial commit. Then run:

```sh
git init -b main
git add .
git commit -m "Add nevo foundations and Chapters 1-3 companions"
git remote add origin https://github.com/YOUR_USERNAME/nevo.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub account. Use your normal GitHub
authentication. If you already have a local repository or nonempty remote,
integrate the files with that history rather than overwriting it or force-pushing.

The included workflow runs Rust formatting, linting, tests, documentation, and
the Chapter 1–3 examples on Linux and Windows. The new controlled study uses a
small CI smoke protocol; its full 60-run dataset is a separate experiment. A
Linux job also tests the declared 1.85.1 MSRV. Review the Actions run after push.
Notebook execution is documented separately and is not currently a CI job.

## Source license

No source-code license has been chosen for this project. Choose and add the
license you want collaborators to use before describing it as a reusable
open-source release. The dependency crates retain their own licenses. The bundled Optdigits data
has its own CC BY 4.0 attribution in `data/README.md`; keep that notice with it.

`publish = false` disables Cargo publication; it does not prevent a GitHub push.
No remote has been created or modified while preparing this archive.
