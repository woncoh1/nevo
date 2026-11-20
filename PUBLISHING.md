# Put this starter on GitHub

Extract the archive and open a terminal inside the `nevo` directory (the one
containing `Cargo.toml`). The archive contains the source files and a CI workflow,
not a preconfigured Git repository or credentials.

## Verify locally

```sh
cargo test --workspace --locked
cargo run -p nevo --example ch01_introduction --locked
```

## First push to a new, empty repository

Create an empty repository named `nevo` on GitHub. For this workflow, leave its
README, license, and `.gitignore` initialization unchecked so the remote has no
separate initial commit. Then run:

```sh
git init -b main
git add .
git commit -m "Add GES core and Chapter 1 neuroevolution companion"
git remote add origin https://github.com/YOUR_USERNAME/nevo.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub account. Use your normal GitHub
authentication. If you already have a local repository or nonempty remote,
integrate the files with that history rather than overwriting it or force-pushing.

The included workflow runs Rust formatting, linting, tests, documentation, and
both examples on Linux and Windows. Review the resulting Actions run after push.
Notebook execution is documented separately and is not currently a CI job.

## Source license

No source-code license has been chosen for this project. Choose and add the
license you want collaborators to use before describing it as a reusable
open-source release. The dependency crates retain their own licenses.

`publish = false` disables Cargo publication; it does not prevent a GitHub push.
No remote has been created or modified while preparing this archive.
