#!/usr/bin/env python3
"""Validate and plot the recorded Rust double-pole generalization experiment.

From the repository root:
    python scripts/plot_ch03_generalization.py
    python scripts/plot_ch03_generalization.py --input results/my_run --output results/my_run

Requires Python 3.10+, Matplotlib, and Graphviz's `dot` executable. Optimizers,
physics, starts, and final evaluations all run in Rust. This script only reads
saved records, checks their accounting, and produces descriptive statistics.
It rejects an incomplete controlled study. Explicit --smoke accepts a small
smoke-only study and marks every figure accordingly; it never fills missing runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shlex
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "book/results/ch03_generalization"
CONDITIONS = (
    "full_cma_1", "full_cma_16", "full_ff_neat_1", "full_ff_neat_16",
    "partial_ff_neat_16", "partial_recurrent_neat_16",
)
RECURRENT = CONDITIONS[-1]
LABELS = {
    "full_cma_1": "CMA-ES · 1 start",
    "full_cma_16": "CMA-ES · 16 starts",
    "full_ff_neat_1": "NEAT feedforward · 1 start",
    "full_ff_neat_16": "NEAT feedforward · 16 starts",
    "partial_ff_neat_16": "NEAT feedforward · 16 starts",
    "partial_recurrent_neat_16": "NEAT recurrent · 16 starts",
}
COLORS = {
    "full_cma_1": "#3372AF", "full_cma_16": "#168B8D",
    "full_ff_neat_1": "#AC83B2", "full_ff_neat_16": "#654894",
    "partial_ff_neat_16": "#B87532", "partial_recurrent_neat_16": "#CE5363",
}
INK, MUTED = "#263548", "#596A7D"
STATE_COLUMNS = (
    "position", "velocity", "long_angle", "short_angle",
    "long_angular_velocity", "short_angular_velocity",
)
RUN_INTS = set("seed training_starts population horizon training_budget batches fitness_calls "
               "training_rollouts training_control_steps selected_job_id selected_batch "
               "train_horizon_hits train_episodes heldout_horizon_hits heldout_episodes "
               "heldout_control_steps ablation_horizon_hits ablation_episodes ablation_control_steps "
               "best_hidden_nodes best_enabled_connections best_total_connections final_species_count "
               "trace_replay_episodes trace_replay_control_steps".split())
TRAINING_INTS = set("seed batch fitness_calls training_rollouts training_control_steps "
                    "batch_horizon_hits batch_episodes best_train_horizon_hits selected_job_id "
                    "selected_batch best_hidden_nodes best_enabled_connections best_total_connections "
                    "species_count".split())
SPECIES_INTS = set("seed batch species_id size stagnant_generations offspring".split())
TEXT_COLUMNS = {"protocol", "condition", "suite", "kind"}
BOOL_COLUMNS = {"terminated", "truncated", "horizon_hit", "protected", "enabled"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"Nonfinite numeric record: {value}")
    return result


def convert(row: dict[str, str], integers: set[str]) -> dict:
    result = {}
    for key, value in row.items():
        if key in TEXT_COLUMNS:
            result[key] = value
        elif value == "":
            result[key] = None
        elif key in BOOL_COLUMNS:
            require(value in {"true", "false"}, f"Invalid boolean {key}={value}")
            result[key] = value == "true"
        elif key in integers:
            result[key] = int(value)
        else:
            result[key] = finite(value)
    return result


def describe(values) -> dict:
    values = list(values)
    require(bool(values), "Cannot summarize an empty group")
    return {
        "n": len(values), "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values), "minimum": min(values), "maximum": max(values),
    }


def groups(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def near(actual, expected, context: str) -> None:
    require(actual is not None and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
            f"Inconsistent {context}: {actual} != {expected}")


@dataclass
class Study:
    root: Path
    smoke: bool
    runs: list[dict] = field(default_factory=list)
    training: list[dict] = field(default_factory=list)
    trials: list[dict] = field(default_factory=list)
    species: list[dict] = field(default_factory=list)
    policies: dict = field(default_factory=dict)
    starts: list[dict] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)

    def source(self, path: Path) -> bytes:
        contents = path.read_bytes()
        self.sources[path.relative_to(self.root).as_posix()] = hashlib.sha256(contents).hexdigest()
        return contents

    def csv(self, path: Path, integers: set[str], allow_empty=False) -> list[dict]:
        text = self.source(path).decode("utf-8")
        records = list(csv.DictReader(text.splitlines()))
        require(bool(records) or allow_empty, f"Empty CSV: {path}")
        require(all(None not in row and all(v is not None for v in row.values()) for row in records),
                f"Malformed CSV: {path}")
        return [convert(row, integers) for row in records]

    @property
    def seeds(self) -> list[int]:
        return sorted({row["seed"] for row in self.runs})

    @property
    def horizon(self) -> int:
        return self.runs[0]["horizon"]

    @property
    def budget(self) -> int:
        return self.runs[0]["training_budget"]

    @property
    def population(self) -> int:
        return self.runs[0]["population"]

    def title(self, text: str) -> str:
        return f"SMOKE ONLY — {text}" if self.smoke else text


def validate_trials(rows: list[dict], run: dict, prefix: str) -> None:
    count = run[f"{prefix}_episodes"]
    require(len(rows) == count and sorted(r["trial"] for r in rows) == list(range(count)),
            f"Missing/duplicate {prefix} trials for {run['condition']} seed {run['seed']}")
    for row in rows:
        require(row["horizon_hit"] == (row["truncated"] and not row["terminated"]),
                "A recorded horizon success disagrees with episode termination")
        require(0 <= row["return"] <= run["horizon"], "Return is outside the declared horizon")
        require(row["control_steps"] == row["return"] + int(row["terminated"]),
                "Valid-endpoint reward and applied-action accounting disagree")
        require(row["terminated"] or row["truncated"], "An episode stopped without an end condition")
        require(row["control_steps"] <= run["horizon"], "Episode exceeds declared horizon")
        require(not row["truncated"] or row["control_steps"] == run["horizon"],
                "Time-limit truncation occurred before the horizon")
    returns = [row["return"] for row in rows]
    near(run[f"{prefix}_mean"], statistics.mean(returns), f"{prefix} mean")
    near(run[f"{prefix}_worst"], min(returns), f"{prefix} worst return")
    near(run[f"{prefix}_sample_sd"], statistics.stdev(returns), f"{prefix} sample SD")
    require(sum(row["horizon_hit"] for row in rows) == run[f"{prefix}_horizon_hits"],
            f"{prefix} horizon hits disagree with raw trials")
    require(sum(row["control_steps"] for row in rows) == run[f"{prefix}_control_steps"],
            f"{prefix} action count disagrees with raw trials")


def read_policy(study: Study, directory: Path, run: dict) -> dict:
    condition = run["condition"]
    prefix = directory / "policies" / condition
    metadata = dict(line.split("=", 1) for line in
                    study.source(prefix.with_suffix(".metadata.txt")).decode().splitlines() if line)
    require(int(metadata["selected_job_id"]) == run["selected_job_id"] and
            int(metadata["selected_batch"]) == run["selected_batch"], "Saved policy identity differs from run record")
    if "cma" in condition:
        require(metadata["mode"] == "FeedForward" and metadata["encoding"] == "Mlp", "Wrong CMA policy mode")
        weights = study.csv(prefix.with_suffix(".weights.csv"), {"index"})
        require([row["index"] for row in weights] == list(range(33)), "CMA policy must retain all 33 weights")
        return {"metadata": metadata, "weights": weights}
    expected_mode = "Recurrent" if condition == RECURRENT else "FeedForward"
    require(metadata["mode"] == expected_mode and metadata["encoding"] == "NeatGenome", "Wrong saved NEAT mode")
    nodes = study.csv(prefix.with_suffix(".nodes.csv"), {"id"})
    links = study.csv(prefix.with_suffix(".connections.csv"), {"innovation", "from", "to"}, allow_empty=True)
    require(len(nodes) == len({node["id"] for node in nodes}), "Duplicate saved node IDs")
    require(len(links) == len({link["innovation"] for link in links}), "Duplicate saved innovation IDs")
    require(len(links) == len({(link["from"], link["to"]) for link in links}), "Duplicate saved endpoint pair")
    roles = {node["id"]: node["kind"] for node in nodes}
    require(all(role in {"Input", "Bias", "Hidden", "Output"} for role in roles.values()), "Unknown saved node role")
    require(sum(role == "Input" for role in roles.values()) == (3 if condition.startswith("partial") else 6),
            "Saved input dimension differs from experiment condition")
    require(sum(role == "Output" for role in roles.values()) == 1 and
            sum(role == "Bias" for role in roles.values()) == 1, "Saved NEAT output/bias count differs")
    for link in links:
        require(link["from"] in roles and link["to"] in roles, "Saved connection endpoint missing")
        require(roles[link["to"]] in {"Hidden", "Output"}, "Saved edge targets an input/bias")
        if expected_mode == "FeedForward":
            require(roles[link["from"]] != "Output" and link["from"] != link["to"], "Saved feedforward graph has feedback")
    require(sum(node["kind"] == "Hidden" for node in nodes) == run["best_hidden_nodes"], "Saved hidden count differs")
    require(sum(link["enabled"] for link in links) == run["best_enabled_connections"], "Saved enabled count differs")
    require(len(links) == run["best_total_connections"], "Saved historical gene count differs")
    if expected_mode == "FeedForward":
        incoming = {node_id: 0 for node_id in roles}
        outgoing = defaultdict(list)
        for link in links:
            if link["enabled"]:
                incoming[link["to"]] += 1
                outgoing[link["from"]].append(link["to"])
        ready = [node_id for node_id, degree in incoming.items() if degree == 0]
        visited = 0
        while ready:
            node_id = ready.pop()
            visited += 1
            for destination in outgoing[node_id]:
                incoming[destination] -= 1
                if incoming[destination] == 0:
                    ready.append(destination)
        require(visited == len(nodes), "Saved feedforward genome contains an enabled cycle")
    return {"metadata": metadata, "nodes": nodes, "connections": links}


def load_study(directory: Path, smoke: bool) -> Study:
    study = Study(directory, smoke)
    seed_dirs = sorted(path for path in directory.glob("seed_*") if path.is_dir())
    require(bool(seed_dirs), f"No seed directories in {directory}")
    if not smoke:
        require([path.name for path in seed_dirs] == [f"seed_{seed:03}" for seed in range(10)],
                "Controlled results require exactly seed_000 through seed_009; use --smoke only for a smoke protocol")
    expected_protocol = "smoke_only" if smoke else "controlled_v1"
    expected = (8, 256, 40, 8) if smoke else (64, 32768, 500, 100)
    canonical_starts = None
    for directory in seed_dirs:
        seed = int(directory.name.removeprefix("seed_"))
        manifest = study.source(directory / "manifest.md").decode()
        require(expected_protocol in manifest, f"Wrong protocol in {directory / 'manifest.md'}")
        runs = study.csv(directory / "runs.csv", RUN_INTS)
        require(len(runs) == 6 and {run["condition"] for run in runs} == set(CONDITIONS),
                f"Expected one finished run for each of the six conditions in {directory}")
        training = study.csv(directory / "training.csv", TRAINING_INTS)
        trials = study.csv(directory / "trials.csv", {"seed", "trial", "control_steps"})
        species = study.csv(directory / "species.csv", SPECIES_INTS)
        traces = study.csv(directory / "traces.csv", {"seed", "trial", "step"}, allow_empty=True)
        for row in runs + training + trials + species:
            require(row["seed"] == seed and row["protocol"] == expected_protocol,
                    "A recorded seed or protocol disagrees with its directory")
            require(row["condition"] in CONDITIONS, "Unknown condition in raw records")
        starts = study.csv(directory / "starts.csv", {"trial"})
        require({row["suite"] for row in starts} == {"training", "heldout"}, "Unexpected initial-state split")
        suites = groups(starts, ["suite"])
        for suite_name, count in [("training", 16), ("heldout", expected[3])]:
            rows = suites[(suite_name,)]
            require([row["trial"] for row in rows] == list(range(count)), "Missing or duplicate saved initial states")
            states = [tuple(row[key] for key in STATE_COLUMNS) for row in rows]
            require(len(states) == len(set(states)), "Duplicate initial state within a suite")
        train_states = {tuple(row[key] for key in STATE_COLUMNS) for row in suites[("training",)]}
        test_states = {tuple(row[key] for key in STATE_COLUMNS) for row in suites[("heldout",)]}
        require(train_states.isdisjoint(test_states), "Training and held-out initial states overlap")
        if canonical_starts is None:
            canonical_starts = starts
        require(starts == canonical_starts, "The fixed suites differ between optimizer seeds")
        by_training = groups(training, ["condition"])
        by_trials = groups(trials, ["condition", "suite"])
        by_species = groups(species, ["condition", "batch"])
        require(len(trials) == 7 * expected[3], "Unexpected number of held-out/ablation episodes")
        require({row["suite"] for row in trials} == {"heldout", "memory_reset"}, "Unknown final trial suite")
        for run in runs:
            condition = run["condition"]
            context = f"{condition} seed {seed}"
            require((run["population"], run["training_budget"], run["horizon"], run["heldout_episodes"]) == expected,
                    f"Protocol settings differ for {context}")
            starts_count = 1 if condition.endswith("_1") else 16
            require(run["training_starts"] == starts_count and run["train_episodes"] == starts_count,
                    f"Training-suite accounting differs for {context}")
            require(run["training_rollouts"] == run["training_budget"] == run["fitness_calls"] * starts_count,
                    f"Training budget not exhausted exactly for {context}")
            require(run["batches"] * run["population"] == run["fitness_calls"], f"Incomplete batches for {context}")
            require(0 <= run["selected_job_id"] < run["fitness_calls"] and
                    run["selected_batch"] == run["selected_job_id"] // run["population"] + 1,
                    f"Invalid selected job identity for {context}")
            history = sorted(by_training[(condition,)], key=lambda row: row["batch"])
            require([row["batch"] for row in history] == list(range(1, run["batches"] + 1)), f"Incomplete learning curve for {context}")
            previous_best, previous_actions = -math.inf, 0
            for row in history:
                require(row["fitness_calls"] == row["batch"] * run["population"] and
                        row["training_rollouts"] == row["fitness_calls"] * starts_count and
                        row["batch_episodes"] == run["population"] * starts_count, "Learning-curve work accounting differs")
                require(row["best_train_mean"] >= previous_best and row["training_control_steps"] >= previous_actions,
                        "Best-seen training fitness or cumulative work decreases")
                previous_best, previous_actions = row["best_train_mean"], row["training_control_steps"]
                if "neat" in condition:
                    batch_species = by_species[(condition, row["batch"])]
                    require(len(batch_species) == row["species_count"] and len(batch_species) > 0,
                            "Species diagnostics missing for a selected batch")
                    require(len({r["species_id"] for r in batch_species}) == len(batch_species), "Duplicate species ID within batch")
                    require(sum(r["size"] for r in batch_species) == run["population"] and
                            sum(r["offspring"] for r in batch_species) == run["population"], "Species populations or quotas do not sum to population")
                else:
                    require(not by_species[(condition, row["batch"])], "CMA run unexpectedly contains NEAT species")
            last = history[-1]
            for key in ["fitness_calls", "training_rollouts", "training_control_steps", "selected_job_id", "selected_batch",
                        "best_hidden_nodes", "best_enabled_connections", "best_total_connections"]:
                require(last[key] == run[key], f"Last learning-curve record differs from run field {key}")
            near(last["best_train_mean"], run["train_mean"], "selected training mean")
            require(last["species_count"] == run["final_species_count"], "Final species count differs")
            validate_trials(by_trials[(condition, "heldout")], run, "heldout")
            if condition == RECURRENT:
                require(run["ablation_episodes"] == expected[3], "Missing recurrent memory ablation")
                validate_trials(by_trials[(condition, "memory_reset")], run, "ablation")
            else:
                require(run["ablation_episodes"] == 0 and not by_trials[(condition, "memory_reset")],
                        "Unexpected ablation for a nonrecurrent policy")
            study.policies[(seed, condition)] = read_policy(study, directory, run)
        by_trace = groups(traces, ["condition", "suite"])
        for run in runs:
            expected_trace_count = (2 if run["condition"] == RECURRENT else 1) if seed == 0 else 0
            require(run["trace_replay_episodes"] == expected_trace_count, "Wrong predeclared trace replay count")
            trace_actions = 0
            for suite_name in ["heldout", "memory_reset"]:
                rows = by_trace[(run["condition"], suite_name)]
                expected_here = seed == 0 and (suite_name == "heldout" or run["condition"] == RECURRENT)
                require(bool(rows) == expected_here, "A required representative trace is missing or an undeclared trace exists")
                if not rows:
                    continue
                require(all(row["seed"] == 0 and row["trial"] == 0 for row in rows), "Trace does not use the predeclared seed/case")
                require([row["step"] for row in rows] == list(range(len(rows))), "Trace step sequence is incomplete")
                require(rows[0]["action"] is None and rows[0]["return"] == 0, "Trace initial row is not the unevaluated start")
                initial = suites[("heldout",)][0]
                require(all(rows[0][key] == initial[key] for key in STATE_COLUMNS), "Trace initial state differs from held-out case 0")
                require(all(row["action"] is not None and -1 <= row["action"] <= 1 for row in rows[1:]),
                        "Recorded normalized control is missing or out of bounds")
                trial = next(row for row in by_trials[(run["condition"], suite_name)] if row["trial"] == 0)
                last = rows[-1]
                require(last["step"] == trial["control_steps"] and last["return"] == trial["return"] and
                        last["terminated"] == trial["terminated"] and last["truncated"] == trial["truncated"],
                        "Representative replay differs from its original held-out episode")
                trace_actions += last["step"]
            require(trace_actions == run["trace_replay_control_steps"], "Trace replay action accounting differs")
        require(all(row["condition"] in CONDITIONS and row["suite"] in {"heldout", "memory_reset"} for row in traces),
                "Unexpected trace condition or suite")
        study.runs.extend(runs)
        study.training.extend(training)
        study.trials.extend(trials)
        study.species.extend(species)
        study.traces.extend(traces)
    study.starts = canonical_starts
    study.runs.sort(key=lambda row: (row["seed"], CONDITIONS.index(row["condition"])))
    return study


def summarize(study: Study) -> dict:
    by_condition = groups(study.runs, ["condition"])
    conditions = {}
    for condition in CONDITIONS:
        runs = by_condition[(condition,)]
        conditions[condition] = {
            "label": LABELS[condition], "optimizer_seeds": [row["seed"] for row in runs],
            "training_starts": runs[0]["training_starts"],
            "train_mean": describe(row["train_mean"] for row in runs),
            "heldout_mean": describe(row["heldout_mean"] for row in runs),
            "heldout_worst": describe(row["heldout_worst"] for row in runs),
            "heldout_horizon_rate": describe(row["heldout_horizon_hits"] / row["heldout_episodes"] for row in runs),
            "training_control_steps": describe(row["training_control_steps"] for row in runs),
            "fitness_calls": describe(row["fitness_calls"] for row in runs),
            "training_rollouts": describe(row["training_rollouts"] for row in runs),
            "seconds": describe(row["seconds"] for row in runs),
            "best_hidden_nodes": describe(row["best_hidden_nodes"] for row in runs) if "neat" in condition else None,
            "best_enabled_connections": describe(row["best_enabled_connections"] for row in runs) if "neat" in condition else None,
            "final_species_count": describe(row["final_species_count"] for row in runs) if "neat" in condition else None,
        }
    recurrent = by_condition[(RECURRENT,)]
    return {
        "schema_version": 1, "generated_by": "scripts/plot_ch03_generalization.py",
        "protocol": "smoke_only" if study.smoke else "controlled_v1",
        "optimizer_seeds": study.seeds, "primary_runs": len(study.runs),
        "population": study.population, "training_rollouts_per_run": study.budget,
        "horizon": study.horizon, "heldout_starts": study.runs[0]["heldout_episodes"],
        "total_training_rollouts": sum(row["training_rollouts"] for row in study.runs),
        "total_primary_heldout_episodes": sum(row["heldout_episodes"] for row in study.runs),
        "additional_trace_replay_episodes": sum(row["trace_replay_episodes"] for row in study.runs),
        "additional_trace_replay_control_steps": sum(row["trace_replay_control_steps"] for row in study.runs),
        "plotting_runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "validation": {
            "complete_declared_seed_condition_matrix": True, "exact_training_rollout_budgets": True,
            "training_and_heldout_starts_disjoint": True, "shared_starts_equal_across_seeds": True,
            "saved_final_policy_identity_matches": True, "raw_trials_match_reported_statistics": True,
            "species_sizes_and_offspring_quotas_match": True,
            "predeclared_trace_replays_match_original_episodes": True,
        },
        "statistical_conventions": {
            "unit": f"One independently optimized policy per condition and optimizer seed; {study.runs[0]['heldout_episodes']} fixed held-out starts are repeated measurements of that policy",
            "conditioning": "Across-seed variation is conditional on the shared training and held-out suites; it does not estimate variability across freshly resampled suites",
            "sample_sd": "Across optimizer-run summaries, denominator n-1; null for a one-seed smoke run",
            "intervals": "Mean ± sample SD is descriptive, not a confidence interval or pooled-trial standard error",
            "pairing": "Conditions share optimizer seed labels and explicit starts; memory ablation compares the same saved recurrent policy on the same starts",
            "selection": "Best mean training return observed within the frozen budget; no held-out tuning, selection, or early stopping",
            "scope": "Equal rollout-episode budgets and populations; fitness calls, adaptation generations, architectures, activations, runtime, and applied control actions differ",
            "memory_reset": "Reset before every action also removes delayed acyclic neuron-source paths; it is not an independently evolved feedforward controller",
            "smoke": "Smoke-only records verify wiring and cannot support the scientific comparisons",
        },
        "conditions": conditions,
        "memory_ablation": {
            "condition": RECURRENT,
            "reset_heldout_mean": describe(row["ablation_mean"] for row in recurrent),
            "reset_heldout_horizon_rate": describe(row["ablation_horizon_hits"] / row["ablation_episodes"] for row in recurrent),
            "paired_return_difference_memory_minus_reset": describe(row["heldout_mean"] - row["ablation_mean"] for row in recurrent),
            "paired_horizon_rate_difference_memory_minus_reset": describe(
                row["heldout_horizon_hits"] / row["heldout_episodes"] - row["ablation_horizon_hits"] / row["ablation_episodes"]
                for row in recurrent),
            "additional_diagnostic_episodes": sum(row["ablation_episodes"] for row in recurrent),
        },
        "source_sha256": dict(sorted(study.sources.items())),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregates(study: Study, summary: dict, output: Path) -> None:
    write_csv(output / "runs.csv", study.runs)
    rows = []
    for condition, metrics in summary["conditions"].items():
        row = {"condition": condition, "evaluation": "heldout", "optimizer_runs": len(study.seeds),
               "training_starts": metrics["training_starts"], "training_rollouts": study.budget}
        for key in ["train_mean", "heldout_mean", "heldout_worst", "heldout_horizon_rate", "fitness_calls",
                    "training_control_steps", "seconds", "best_hidden_nodes", "best_enabled_connections", "final_species_count"]:
            stats = metrics[key]
            for name in ["mean", "sample_sd"]:
                row[f"{key}_{name}"] = stats[name] if stats is not None else None
        rows.append(row)
    reset = dict(rows[-1])
    reset["evaluation"] = "memory_reset"
    for destination, source in [("heldout_mean", "reset_heldout_mean"),
                                ("heldout_horizon_rate", "reset_heldout_horizon_rate")]:
        for name in ["mean", "sample_sd"]:
            reset[f"{destination}_{name}"] = summary["memory_ablation"][source][name]
    reset["heldout_worst_mean"] = statistics.mean(row["ablation_worst"] for row in study.runs if row["condition"] == RECURRENT)
    worst = [row["ablation_worst"] for row in study.runs if row["condition"] == RECURRENT]
    reset["heldout_worst_sample_sd"] = statistics.stdev(worst) if len(worst) > 1 else None
    rows.append(reset)
    write_csv(output / "summary.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 10.5,
        "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.titlepad": 12,
        "axes.labelcolor": INK, "axes.edgecolor": "#BAC5CF", "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#DCE3EA", "grid.alpha": 0.7, "grid.linewidth": 0.65,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.hashsalt": "nevo-ch03-generalization-v1", "legend.frameon": False,
    })


def save(fig, output: Path, name: str, dpi: int) -> None:
    fig.savefig(output / f"{name}.png", dpi=dpi)
    fig.savefig(output / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def header(fig, study: Study, title: str, subtitle: str) -> None:
    fig.suptitle(study.title(title), x=0.055, y=0.975, ha="left", fontsize=18, weight="bold")
    # Reserve the same physical title/subtitle spacing on short and tall figures.
    subtitle_y = 0.975 - 34 / (72 * fig.get_figheight())
    fig.text(0.055, subtitle_y, subtitle, fontsize=10.5, color=MUTED)


def run_points(ax, study: Study, conditions: tuple, metric: str) -> None:
    extent = [0.0, float(study.horizon if metric == "return" else 100)]
    for position, condition in enumerate(conditions):
        runs = [row for row in study.runs if row["condition"] == condition]
        values = [row["heldout_mean"] if metric == "return" else 100 * row["heldout_horizon_hits"] / row["heldout_episodes"]
                  for row in runs]
        stats = describe(values)
        jitter = [(index - (len(runs) - 1) / 2) * 0.045 for index in range(len(runs))]
        ax.scatter(values, [position + offset for offset in jitter], s=28, color=COLORS[condition],
                   alpha=0.85, edgecolors="white", linewidths=0.55, zorder=3)
        sd = stats["sample_sd"] or 0
        ax.errorbar(stats["mean"], position, xerr=sd, fmt="D", color=INK, markerfacecolor="white",
                    markersize=6, linewidth=1.8, capsize=5, zorder=5)
        extent.extend([stats["mean"] - sd, stats["mean"] + sd])
    low, high = min(extent), max(extent)
    ax.set_xlim(low - 0.035 * (high - low), high + 0.035 * (high - low))
    ax.set_yticks(range(len(conditions)), [LABELS[c] for c in conditions])
    ax.set_ylim(len(conditions) - 0.55, -0.55)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean valid control steps per held-out episode" if metric == "return" else "Held-out episodes reaching the horizon (%)")


def comparison_figure(study: Study, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={"height_ratios": [1.6, 1]})
    fig.subplots_adjust(left=0.225, right=0.97, bottom=0.17, top=0.845, hspace=0.7, wspace=0.63)
    header(fig, study, "Double-pole generalization after a fixed training budget",
           f"{study.budget:,} training episodes per run · population {study.population} · {study.horizon}-step horizon · "
           f"{study.runs[0]['heldout_episodes']} shared held-out starts · {len(study.seeds)} optimizer seeds")
    for row, conditions in enumerate([CONDITIONS[:4], CONDITIONS[4:]]):
        for column, metric in enumerate(["return", "horizon"]):
            run_points(axes[row, column], study, conditions, metric)
        axes[row, 0].set_title("Full state: positions and velocities" if row == 0 else "Partial state: positions only", loc="left")
        axes[row, 1].set_title("Full-horizon success", loc="left")
    fig.text(0.055, 0.095, "Points: optimizer runs. Diamonds/whiskers: mean ± sample SD, not confidence intervals. SD extents may exceed a metric's feasible range.", fontsize=10)
    fig.text(0.055, 0.065, "Final policies were selected using training returns only. Held-out trials were evaluated afterward on a disjoint, fixed local start distribution.", fontsize=10)
    fig.text(0.055, 0.035, "Episode budgets are equal; CMA uses a fixed 33-parameter MLP while NEAT evolves graph topology. Fitness calls, generations, and compute differ.", fontsize=10)
    save(fig, output, "generalization_comparison", dpi)


def curve(ax, study: Study, condition: str, metric: str, label: str | None = None) -> None:
    by_work = groups([row for row in study.training if row["condition"] == condition], ["training_rollouts"])
    points = sorted(by_work.items())
    for _, rows in points:
        require(len(rows) == len(study.seeds), "A curve point is missing an optimizer seed")
    x = [key[0] for key, _ in points]
    stats = [describe(row[metric] for row in rows) for _, rows in points]
    means = [value["mean"] for value in stats]
    lower = [value["mean"] - (value["sample_sd"] or 0) for value in stats]
    upper = [value["mean"] + (value["sample_sd"] or 0) for value in stats]
    line = "--" if condition.endswith("_1") else "-"
    ax.fill_between(x, lower, upper, color=COLORS[condition], alpha=0.11, linewidth=0)
    ax.plot(x, means, color=COLORS[condition], linestyle=line, linewidth=2, label=label or LABELS[condition])
    ax.set_xlim(0, study.budget)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0"))
    ax.set_xlabel("Cumulative training rollout episodes")


def learning_figure(study: Study, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.7), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.22, top=0.81, wspace=0.18)
    header(fig, study, "Training progress at equal rollout budgets",
           "Best mean training return observed so far · initialization is included · all runs consume their complete fixed budget")
    for ax, conditions, title in zip(axes, [CONDITIONS[:4], CONDITIONS[4:]], ["Full state", "Positions only"]):
        for condition in conditions:
            curve(ax, study, condition, "best_train_mean")
        ax.axhline(study.horizon, color=MUTED, linestyle=":", linewidth=1)
        ax.set_title(title, loc="left")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    axes[0].set_ylabel("Selected policy's mean training return")
    fig.text(0.055, 0.13, f"Lines: mean over {len(study.seeds)} optimizer seeds. Bands: ± sample SD across runs; descriptive SD may extend beyond 0–{study.horizon}.", fontsize=10)
    fig.text(0.055, 0.095, "One-start and 16-start fitness scores describe different suites. A larger suite consumes 16 episodes per candidate and receives fewer generations.", fontsize=10)
    fig.text(0.055, 0.06, "Reaching the training horizon on one start is not evidence of generalization. These curves do not use held-out outcomes.", fontsize=10)
    save(fig, output, "training_curves", dpi)


def diagnostics_figure(study: Study, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.7))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.80, hspace=0.45, wspace=0.23)
    header(fig, study, "What NEAT grew during search",
           "Recorded graph and species diagnostics · solid lines use 16 training starts; dashed line uses one full-state start")
    conditions = CONDITIONS[2:]
    labels = {condition: ("Full state · " if condition.startswith("full") else "Positions only · ") +
              ("recurrent" if condition == RECURRENT else "feedforward") +
              (" · 1 start" if condition.endswith("_1") else " · 16 starts") for condition in conditions}
    for ax, metric, title in zip(axes.flat,
                                ["best_hidden_nodes", "best_enabled_connections", "population_mean_hidden_nodes", "species_count"],
                                ["Selected policy: hidden nodes", "Selected policy: enabled connections", "Population mean: hidden nodes", "Species after selection"]):
        for condition in conditions:
            curve(ax, study, condition, metric, labels[condition])
        ax.set_title(title, loc="left")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.065, 0.895), ncol=2, fontsize=9.5)
    fig.text(0.055, 0.10, f"Lines and bands: mean ± sample SD across {len(study.seeds)} optimizer seeds. Descriptive SD can extend below zero; individual counts remain nonnegative.", fontsize=10)
    fig.text(0.055, 0.066, "Selected-policy counts may decrease when a different genome becomes best. Population means describe all evaluated genomes in the current batch.", fontsize=10)
    fig.text(0.055, 0.032, "Species count and network growth are diagnostics; neither alone proves useful memory or improved held-out control.", fontsize=10)
    save(fig, output, "search_diagnostics", dpi)


def graph_layout(policy: dict) -> tuple[dict, list, float, float]:
    nodes, links = policy["nodes"], [link for link in policy["connections"] if link["enabled"]]
    lines = ['digraph G {', 'graph [rankdir=LR, nodesep=0.3, ranksep=0.65, margin=0, splines=true];',
             'node [shape=circle, width=0.42, height=0.42, fixedsize=true, label=""];']
    for node in nodes:
        lines.append(f'n{node["id"]};')
    for role, rank in [({"Input", "Bias"}, "source"), ({"Output"}, "sink")]:
        lines.append('{ rank=' + rank + '; ' + '; '.join(f'n{node["id"]}' for node in nodes if node["kind"] in role) + '; }')
    for link in links:
        lines.append(f'n{link["from"]} -> n{link["to"]};')
    lines.append('}')
    result = subprocess.run(["dot", "-Tplain"], input="\n".join(lines), text=True,
                            capture_output=True, check=True)
    positions, paths, width, height = {}, [], 1.0, 1.0
    lookup = {(link["from"], link["to"]): link for link in links}
    for line in result.stdout.splitlines():
        parts = shlex.split(line)
        if parts[0] == "graph":
            width, height = float(parts[2]), float(parts[3])
        elif parts[0] == "node":
            positions[int(parts[1][1:])] = tuple(float(value) for value in parts[2:6])
        elif parts[0] == "edge":
            count = int(parts[3])
            coords = [float(value) for value in parts[4:4 + 2 * count]]
            points = list(zip(coords[::2], coords[1::2]))
            paths.append((lookup[(int(parts[1][1:]), int(parts[2][1:]))], points))
    require(len(positions) == len(nodes), "Graphviz omitted a saved node")
    return positions, paths, width, height


def draw_graph(ax, policy: dict) -> None:
    positions, paths, width, height = graph_layout(policy)
    roles = {node["id"]: node["kind"] for node in policy["nodes"]}
    for link, points in paths:
        color = "#3474A5" if link["weight"] >= 0 else "#C17639"
        delayed = policy["metadata"]["mode"] == "Recurrent" and roles[link["from"]] in {"Hidden", "Output"}
        require((len(points) - 1) % 3 == 0, "Unexpected Graphviz spline representation")
        path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.CURVE4] * (len(points) - 1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=9, color=color,
                                    linewidth=0.55 + min(abs(link["weight"]), 3) * 0.55,
                                    linestyle="--" if delayed else "-", alpha=0.72, zorder=1))
    inputs = sorted(node_id for node_id, role in roles.items() if role == "Input")
    labels = ["x", "v", "θ₁", "ω₁", "θ₂", "ω₂"] if len(inputs) == 6 else ["x", "θ₁", "θ₂"]
    inputs = dict(zip(inputs, labels))
    face = {"Input": "#EDF2F6", "Bias": "#F7E7BF", "Hidden": "#DCECF3", "Output": "#E6DDF0"}
    for node_id, (x, y, w, h) in positions.items():
        role = roles[node_id]
        ax.add_patch(Ellipse((x, y), w, h, facecolor=face[role], edgecolor="#6B7E8E", linewidth=1, zorder=3))
        label = inputs[node_id] if role == "Input" else ("1" if role == "Bias" else "y" if role == "Output" else str(node_id))
        ax.text(x, y, label, ha="center", va="center", fontsize=9 if len(label) < 3 else 7.5, zorder=4)
    ax.set_xlim(-0.15, width + 0.15)
    ax.set_ylim(-0.15, height + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")


def topology_figure(study: Study, output: Path, dpi: int) -> None:
    seed = 0 if 0 in study.seeds else study.seeds[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.8))
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.16, top=0.84, hspace=0.34, wspace=0.24)
    header(fig, study, f"Selected NEAT policies from optimizer seed {seed}",
           "A fixed representative seed, chosen before inspecting outcomes · enabled connections only; historical disabled genes remain saved")
    for ax, condition in zip(axes.flat, CONDITIONS[2:]):
        policy = study.policies[(seed, condition)]
        draw_graph(ax, policy)
        run = next(row for row in study.runs if row["seed"] == seed and row["condition"] == condition)
        location = "Full state" if condition.startswith("full") else "Positions only"
        mode = "recurrent" if condition == RECURRENT else "feedforward"
        ax.set_title(f"{location} · {mode} · {run['training_starts']} training start{'s' if run['training_starts'] != 1 else ''}\n"
                     f"{run['best_hidden_nodes']} hidden · {run['best_enabled_connections']} enabled / {run['best_total_connections']} historical links",
                     fontsize=11, loc="left")
    fig.text(0.055, 0.10, "Blue edges: nonnegative weights. Orange edges: negative weights. Width reflects |weight|, capped for readability; arrowheads show direction.", fontsize=10)
    fig.text(0.055, 0.066, "Dashed recurrent neuron-source edges read the previous tick, including hidden-to-output paths. Inputs/bias use current values; hidden node labels are IDs.", fontsize=10)
    fig.text(0.055, 0.032, "x = cart position, v = cart velocity, θ = pole angle, ω = angular velocity; input values use the task's documented scaling. These are illustrative policies.", fontsize=10)
    save(fig, output, "policy_topologies_seed0", dpi)


def memory_figure(study: Study, output: Path, dpi: int) -> None:
    runs = [row for row in study.runs if row["condition"] == RECURRENT]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8))
    fig.subplots_adjust(left=0.08, right=0.965, bottom=0.26, top=0.80, wspace=0.25)
    header(fig, study, "Effect of clearing recurrent state before every action",
           "The same selected recurrent genome, paired on the same held-out starts · additional diagnostic episodes are outside the training budget")
    for ax, metric in zip(axes, ["return", "horizon"]):
        memory, reset = [], []
        for index, run in enumerate(runs):
            normal = run["heldout_mean"] if metric == "return" else 100 * run["heldout_horizon_hits"] / run["heldout_episodes"]
            cleared = run["ablation_mean"] if metric == "return" else 100 * run["ablation_horizon_hits"] / run["ablation_episodes"]
            memory.append(normal)
            reset.append(cleared)
            jitter = (index - (len(runs) - 1) / 2) * 0.015
            ax.plot([jitter, 1 + jitter], [normal, cleared], color=COLORS[RECURRENT], alpha=0.35, linewidth=1)
            ax.scatter([jitter, 1 + jitter], [normal, cleared], s=32, color=COLORS[RECURRENT],
                       edgecolors="white", linewidth=0.6, zorder=3)
        for x, values in enumerate([memory, reset]):
            stats = describe(values)
            ax.errorbar(x, stats["mean"], yerr=stats["sample_sd"] or 0, fmt="D", color=INK,
                        markerfacecolor="white", markersize=7, capsize=5, linewidth=1.8, zorder=5)
        ax.set_xticks([0, 1], ["State carried\nbetween actions", "State reset\nbefore every action"])
        ax.set_xlim(-0.32, 1.32)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Held-out mean return" if metric == "return" else "Held-out horizon successes (%)")
        ax.set_title("Return" if metric == "return" else "Full-horizon success", loc="left")
    fig.text(0.055, 0.16, f"Lines pair each seed's results. Diamonds: mean ± sample SD across {len(runs)} policies; SD may extend beyond metric bounds. No ablation-based selection.", fontsize=10)
    fig.text(0.055, 0.12, "Resetting removes all carried activations and also suppresses delayed acyclic hidden-to-output signal paths in the synchronous network.", fontsize=10)
    fig.text(0.055, 0.08, "Consequently, this is a temporal-state ablation of the same policy; compare separately evolved feedforward policies in the main results.", fontsize=10)
    save(fig, output, "memory_ablation", dpi)


def trace_figure(study: Study, output: Path, dpi: int) -> None:
    if not study.traces:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.79, hspace=0.4, wspace=0.22)
    header(fig, study, "One predeclared recurrent-policy replay",
           "Optimizer seed 0 · held-out initial state 0 · the same selected genome with state carried or reset before every action")
    by_suite = groups([row for row in study.traces if row["condition"] == RECURRENT], ["suite"])
    for ax, metric, label in zip(axes.flat,
                               ["position", "long_angle", "short_angle", "action"],
                               ["Cart position (m)", "Long-pole angle (rad)", "Short-pole angle (rad)", "Normalized control action"]):
        for suite_name, color, linestyle, legend in [
            ("heldout", COLORS[RECURRENT], "-", "State carried between actions"),
            ("memory_reset", "#596A7D", "--", "State reset before every action"),
        ]:
            rows = [row for row in by_suite[(suite_name,)] if row[metric] is not None]
            ax.plot([row["step"] for row in rows], [row[metric] for row in rows],
                    color=color, linestyle=linestyle, linewidth=1.7, label=legend)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Applied control actions")
        if metric == "action":
            ax.set_ylim(-1.12, 1.12)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.065, 0.89), ncol=2)
    run = next(row for row in study.runs if row["seed"] == 0 and row["condition"] == RECURRENT)
    original = next(row for row in study.trials if row["seed"] == 0 and row["condition"] == RECURRENT and row["suite"] == "heldout" and row["trial"] == 0)
    reset = next(row for row in study.trials if row["seed"] == 0 and row["condition"] == RECURRENT and row["suite"] == "memory_reset" and row["trial"] == 0)
    fig.text(0.055, 0.10, f"This start's return: carried state {original['return']:g}/{run['horizon']}; reset state {reset['return']:g}/{run['horizon']}. Curves end at each actual episode boundary.", fontsize=10)
    fig.text(0.055, 0.066, "The seed and case were fixed before inspecting outcomes. These two trajectories illustrate behavior; the all-seed held-out summaries support comparisons.", fontsize=10)
    fig.text(0.055, 0.032, "State reset also suppresses delayed acyclic paths. Saved trace replays exactly match the original final-evaluation episode returns and termination flags.", fontsize=10)
    save(fig, output, "memory_trace_seed0", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA, help="directory containing seed_000 through seed_009")
    parser.add_argument("--output", type=Path, help="aggregate/figure destination; defaults to --input")
    parser.add_argument("--smoke", action="store_true", help="explicitly accept a smoke_only protocol; all figures are marked")
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution (default 160); SVG is also saved")
    args = parser.parse_args()
    require(args.dpi > 0, "DPI must be positive")
    directory = args.input.resolve()
    output = (args.output or directory).resolve()
    require(not output.name.startswith("seed_"), "Write aggregates to a study directory, not inside a raw seed directory")
    study = load_study(directory, args.smoke)
    output.mkdir(parents=True, exist_ok=True)
    style()
    comparison_figure(study, output, args.dpi)
    learning_figure(study, output, args.dpi)
    diagnostics_figure(study, output, args.dpi)
    topology_figure(study, output, args.dpi)
    memory_figure(study, output, args.dpi)
    trace_figure(study, output, args.dpi)
    summary = summarize(study)
    write_aggregates(study, summary, output)
    figures = 6 if study.traces else 5
    print(f"Validated {len(study.runs)} runs across {len(study.seeds)} optimizer seeds; wrote summaries and {figures} PNG/SVG figures to {output}")


if __name__ == "__main__":
    main()
