#!/usr/bin/env python3
"""Summarize recorded Rust experiments and regenerate the Chapter 3 figures.

Usage from the repository root:
    python scripts/plot_ch03.py

Requires Python 3.10+ and Matplotlib. It reads existing CSV/JSON records; it does
not run an optimizer, change a split, interpolate missing runs, or tune settings.
PNG and vector SVG figures plus summary.json are written to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"ga": "#087F8C", "es": "#2868AF", "neat": "#8556A4", "cma_es": "#C4652C"}
INK = "#273447"
MUTED = "#596575"
GREEN = "#167A62"
RED = "#B23A48"
SEED_MARKERS = ["o", "s", "^"]
SEED_LINES = ["-", "--", ":"]


def read_csv(directory: Path, name: str) -> list[dict[str, str]]:
    with (directory / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Empty or malformed CSV: {name}")
    return rows


def number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite recorded value: {value}")
    return result


def typed(row: dict[str, str], integers=(), floats=(), booleans=()) -> dict:
    result = dict(row)
    for key in integers:
        result[key] = int(row[key])
    for key in floats:
        result[key] = number(row[key])
    for key in booleans:
        if row[key] not in {"true", "false"}:
            raise ValueError(f"Invalid boolean in {key}: {row[key]}")
        result[key] = row[key] == "true"
    return result


def describe(values) -> dict:
    values = list(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def grouped(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def summarize(directory: Path) -> tuple[dict, dict]:
    names = [
        "neat_xor_runs.csv", "neat_xor_curves.csv", "neat_xor_predictions.csv",
        "digits_runs.csv", "digits_curves.csv", "digits_split.csv",
        "cartpole_runs.csv", "cartpole_trials.csv", "doublepole_runs.csv", "doublepole_trials.csv",
        "walker_runs.csv", "walker_trajectories.csv", "parallel_runs.csv", "deep_ga_curves.csv",
    ]
    records = {name: read_csv(directory, name) for name in names}
    manifests = {
        name: json.loads((directory / name).read_text(encoding="utf-8"))
        for name in ["neat_manifest.json", "digits_manifest.json", "cartpole_manifest.json",
                     "doublepole_manifest.json", "walker_manifest.json", "parallel_manifest.json"]
    }
    xor = [typed(r, ("seed", "evaluations", "hidden_nodes", "enabled_connections"), ("fitness", "mse"), ("solved",))
           for r in records["neat_xor_runs.csv"]]
    successful = [r for r in xor if r["solved"]]
    digits = [typed(r, ("seed", "evaluations", "samples"), ("mse", "accuracy")) for r in records["digits_runs.csv"]]
    classifier_stats = {}
    for algorithm in ["ga", "es", "neat"]:
        classifier_stats[algorithm] = {}
        for partition in ["training", "validation", "test"]:
            rows = [r for r in digits if r["algorithm"] == algorithm and r["partition"] == partition]
            classifier_stats[algorithm][partition] = {
                "accuracy_fraction": describe(r["accuracy"] for r in rows),
                "mse": describe(r["mse"] for r in rows),
            }
    cart = [typed(r, ("seed", "evaluations", "heldout_full_horizon"), ("training_mean", "heldout_mean"))
            for r in records["cartpole_runs.csv"]]
    cart_stats = {}
    for algorithm in ["ga", "es"]:
        rows = [r for r in cart if r["algorithm"] == algorithm]
        cart_stats[algorithm] = {
            "fitness_calls": describe(r["evaluations"] for r in rows),
            "heldout_return_mean_per_run": describe(r["heldout_mean"] for r in rows),
            "horizon_hits_across_policy_start_pairs": sum(r["heldout_full_horizon"] for r in rows),
            "policy_start_pairs": 20 * len(rows),
        }
    double = [typed(r, ("seed", "batches", "fitness_evaluations", "training_control_steps", "heldout_horizon_hits",
                        "heldout_trials", "hidden_nodes", "enabled_connections"),
                    ("training_valid_steps", "heldout_mean_valid_steps", "heldout_min_valid_steps"), ("training_solved",))
              for r in records["doublepole_runs.csv"]]
    walker = [typed(r, ("search_seed", "horizon", "steps"),
                    ("friction", "initial_torso_x", "initial_torso_vx", "return", "com_distance", "actuator_work"),
                    ("terminated", "truncated")) for r in records["walker_runs.csv"]]
    selected_heldout = [r for r in walker if r["policy"] == "selected" and r["split"] == "held_out"]
    contact_proxy = {}
    for case in ["neutral_500", "friction_06"]:
        rows = [r for r in records["walker_trajectories.csv"] if r["policy"] == "selected" and r["case"] == case]
        contact_proxy[case] = {
            "recorded_states_including_start": len(rows),
            "left_fraction_at_or_below_foot_radius": sum(number(r["left_y"]) <= 0.06 for r in rows) / len(rows),
            "right_fraction_at_or_below_foot_radius": sum(number(r["right_y"]) <= 0.06 for r in rows) / len(rows),
            "interpretation": "Geometric center-height proxy at recorded control endpoints, not contact force or physical time fraction",
        }
    parallel = [typed(r, ("evaluations", "batches", "mutations", "seed_payload_bytes", "dense_payload_bytes"),
                      ("best_mse", "wall_seconds")) for r in records["parallel_runs.csv"]]
    modes = grouped(records["deep_ga_curves.csv"], ["mode"])
    canonical = [{k: v for k, v in row.items() if k != "mode"} for row in modes[("serial",)]]
    equal_curves = all([{k: v for k, v in row.items() if k != "mode"} for row in rows] == canonical for rows in modes.values())
    if not equal_curves:
        raise ValueError("Recorded executor histories differ")
    source_names = names + list(manifests)
    summary = {
        "schema_version": 1,
        "generated_by": "scripts/plot_ch03.py",
        "plotting_runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in sorted(source_names)},
        "statistical_conventions": {
            "sample_sd": "Across declared search seeds, denominator n-1; null when n=1",
            "intervals": "Displayed standard deviations are descriptive, not confidence intervals",
            "xor_success_costs": "Conditional on observed successes; capped failures are retained separately and are right-censored",
            "comparison_scope": "Different task rewards, representations, and work budgets do not establish a general algorithm ranking",
        },
        "neat_xor": {
            "protocol": manifests["neat_manifest.json"], "runs": xor,
            "successes": len(successful), "declared_runs": len(xor),
            "successful_seeds": [r["seed"] for r in successful],
            "capped_failure_seeds": [r["seed"] for r in xor if not r["solved"]],
            "successful_evaluation_counts_only": describe(r["evaluations"] for r in successful),
        },
        "classification": {"protocol": manifests["digits_manifest.json"], "runs": digits, "statistics": classifier_stats},
        "cartpole": {"protocol": manifests["cartpole_manifest.json"], "runs": cart, "statistics": cart_stats},
        "doublepole": {"protocol": manifests["doublepole_manifest.json"], "runs": double},
        "walker": {
            "protocol": manifests["walker_manifest.json"], "runs": walker,
            "selected_heldout_falls": [r["case"] for r in selected_heldout if r["terminated"]],
            "selected_heldout_horizon_completions": sum(r["truncated"] and not r["terminated"] for r in selected_heldout),
            "heldout_case_count": len(selected_heldout), "foot_height_proxies": contact_proxy,
        },
        "deep_ga_execution": {
            "protocol": manifests["parallel_manifest.json"], "runs": parallel,
            "recorded_curves_identical_across_modes": equal_curves,
            "seed_to_dense_parameter_payload_ratio": parallel[0]["seed_payload_bytes"] / parallel[0]["dense_payload_bytes"],
            "payload_caveat": "Excludes Vec metadata, allocations, architecture/settings, issued copies, decoded weights, and other run state",
            "timing_caveat": "One local trial per mode, not a controlled throughput benchmark or an inferred speedup",
        },
    }
    return summary, records


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 13, "axes.titleweight": "semibold",
        "axes.labelcolor": INK, "axes.edgecolor": "#BCC4CE", "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#DCE2E8", "grid.alpha": 0.65, "grid.linewidth": 0.65,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.hashsalt": "nevo-ch03-recorded-results", "legend.frameon": False,
    })


def thousands(value, _):
    return "0" if value == 0 else f"{value / 1000:g}k"


def save(fig, output: Path, name: str, dpi: int):
    fig.savefig(output / f"{name}.png", dpi=dpi)
    fig.savefig(output / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def neat_figure(summary: dict, records: dict, output: Path, dpi: int):
    runs = summary["neat_xor"]["runs"]
    curves = grouped(records["neat_xor_curves.csv"], ["seed"])
    fig, axes = plt.subplots(3, 1, figsize=(11.4, 10.2), sharex=True)
    fig.subplots_adjust(left=0.095, right=0.965, top=0.81, bottom=0.105, hspace=0.29)
    fig.suptitle(f"NEAT on XOR: {summary['neat_xor']['successes']} of {len(runs)} declared seeds met the target",
                 x=0.065, y=0.976, ha="left", fontsize=18, weight="bold")
    fig.text(0.065, 0.94, "150 genomes per batch · every prediction must be within 0.1 of its target · cap 45,000 evaluations", fontsize=10.5)
    handles, labels = [], []
    palette = plt.get_cmap("tab10")
    for run in runs:
        seed = run["seed"]
        rows = sorted(curves[(str(seed),)], key=lambda row: int(row["batch"]))
        x = [int(row["evaluations"]) for row in rows]
        color = palette(seed)
        line = "-" if run["solved"] else "--"
        marker = "o" if run["solved"] else "x"
        for ax, column in zip(axes, ["best_fitness", "hidden_nodes", "enabled_connections"]):
            y = [number(row[column]) for row in rows]
            ax.plot(x, y, color=color, linestyle=line, linewidth=1.6, alpha=0.9,
                    drawstyle="default" if column == "best_fitness" else "steps-post")
            ax.plot(x[-1], y[-1], marker=marker, color=color, markersize=5, markeredgewidth=1.5)
        handles.append(Line2D([], [], color=color, linestyle=line, marker=marker, markersize=5, linewidth=1.6))
        labels.append(f"Seed {seed} · {'solved' if run['solved'] else 'capped'}")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.912), ncol=5, fontsize=10, columnspacing=1.6)
    axes[0].set_title("Training fitness, all ten runs", loc="left", pad=9)
    axes[0].set_ylabel("Best fitness")
    axes[0].set_ylim(6.5, 16.45)
    axes[0].set_yticks([7, 9, 12, 16])
    axes[0].axhline(16, color=MUTED, linewidth=0.9, linestyle=":", zorder=0)
    axes[0].text(44500, 15.7, "maximum = 16", color=MUTED, ha="right", va="top", fontsize=9)
    axes[1].set_title("Hidden nodes in the best-fitness genome", loc="left", pad=9)
    axes[1].set_ylabel("Hidden nodes")
    axes[1].set_ylim(-0.4, 12)
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].set_title("Expressed connections in the same selected genome", loc="left", pad=9)
    axes[2].set_ylabel("Enabled connections")
    axes[2].set_ylim(0, 23)
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].set_xlabel("Training fitness evaluations, including initialization")
    axes[2].set_xlim(0, 46000)
    axes[2].set_xticks([0, 10000, 20000, 30000, 40000, 45000])
    axes[2].xaxis.set_major_formatter(FuncFormatter(thousands))
    fig.text(0.065, 0.047, "Curves end at each run's actual stopping batch; capped seeds 0, 3 and 6 are retained. No held-out XOR data.", fontsize=9.5, color=MUTED)
    fig.text(0.065, 0.026, "Topology counts describe the best-fitness genome, not population-wide growth; they can decrease when a different genome becomes best.", fontsize=9.5, color=MUTED)
    save(fig, output, "neat_xor_overview", dpi)


def classification_figure(summary: dict, records: dict, output: Path, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.7))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.77, bottom=0.21, wspace=0.32)
    fig.suptitle("Optdigits: all three search seeds, with untouched held-out tests", x=0.055, y=0.975,
                 ha="left", fontsize=18, weight="bold")
    fig.text(0.055, 0.915, "600 training / 300 validation / 897 test rows · 5,120 fitness calls per run · 600 training examples per call", fontsize=10.5)
    seed_handles = [Line2D([], [], color=MUTED, marker=marker, linestyle=line, label=f"Seed {seed}")
                    for seed, (marker, line) in enumerate(zip(SEED_MARKERS, SEED_LINES))]
    fig.legend(handles=seed_handles, loc="upper left", bbox_to_anchor=(0.055, 0.89), ncol=3, fontsize=10.5)
    names = ["ga", "es", "neat"]
    test_rows = [r for r in summary["classification"]["runs"] if r["partition"] == "test"]
    for x, algorithm in enumerate(names):
        rows = sorted((r for r in test_rows if r["algorithm"] == algorithm), key=lambda row: row["seed"])
        scores = [100 * row["accuracy"] for row in rows]
        mean, sd = statistics.mean(scores), statistics.stdev(scores)
        axes[0].errorbar(x, mean, yerr=sd, fmt="_", color=INK, markersize=19, capsize=7, elinewidth=1.4, zorder=2)
        for row, score in zip(rows, scores):
            axes[0].scatter(x + (row["seed"] - 1) * 0.16, score, marker=SEED_MARKERS[row["seed"]],
                            s=57, color=COLORS[algorithm], edgecolor="white", linewidth=0.7, zorder=3)
        axes[0].text(x, max(max(scores), mean + sd) + 6, f"{mean:.1f}% ± {sd:.1f} pp", ha="center", fontsize=10, color=INK)
    axes[0].set_title("Final test accuracy", loc="left", pad=10)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_xlim(-0.55, 2.55)
    axes[0].set_xticks(range(3), ["GA\n64 → 8 → 10", "ES\n64 → 8 → 10", "NEAT\n64 → 10 initially"])
    axes[0].grid(axis="x", visible=False)
    curves = grouped(records["digits_curves.csv"], ["algorithm", "seed"])
    for algorithm in names:
        for seed in range(3):
            rows = sorted(curves[(algorithm, str(seed))], key=lambda row: int(row["batch"]))
            axes[1].plot([int(r["evaluations"]) for r in rows], [number(r["best_training_mse"]) for r in rows],
                         color=COLORS[algorithm], linestyle=SEED_LINES[seed], linewidth=1.6,
                         label=algorithm.upper() if seed == 0 else None)
    axes[1].set_title("Training objective, all nine runs", loc="left", pad=10)
    axes[1].set_ylabel("Best training MSE")
    axes[1].set_xlabel("Training fitness evaluations")
    axes[1].set_xlim(0, 5200)
    axes[1].set_ylim(bottom=0)
    axes[1].xaxis.set_major_formatter(FuncFormatter(thousands))
    axes[1].legend(loc="upper right", fontsize=10)
    fig.text(0.055, 0.1, "Points are individual seeds. Black intervals show mean ± sample SD (n=3), not confidence intervals; pp = percentage points.", fontsize=9.5, color=MUTED)
    fig.text(0.055, 0.075, "Selection uses training MSE only. Equal fitness-call counts do not equal identical networks, search settings, or compute costs.", fontsize=9.5, color=MUTED)
    fig.text(0.055, 0.05, "This is a new split of the 8×8 Optdigits test-file subset, not MNIST, the original UCI split, or a writer-independent benchmark.", fontsize=9.5, color=MUTED)
    save(fig, output, "digits_classification", dpi)


def control_figure(summary: dict, records: dict, output: Path, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2))
    fig.subplots_adjust(left=0.115, right=0.965, top=0.76, bottom=0.23, wspace=0.5)
    fig.suptitle("Control: training performance and final held-out starts", x=0.055, y=0.973,
                 ha="left", fontsize=18, weight="bold")
    fig.text(0.055, 0.915, "Training selects each final policy; held-out episodes are evaluated afterward and never select a replacement.", fontsize=10.5)
    legend = [
        Line2D([], [], color=INK, marker="o", markerfacecolor="white", linestyle="none", markersize=8, label="Training mean"),
        Line2D([], [], color=INK, marker="D", linestyle="none", markersize=6, label="Held-out mean"),
        Line2D([], [], color="#AAB4C0", marker="|", linestyle="none", markersize=9, label="One held-out start"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.055, 0.89), ncol=3, fontsize=10.5)
    cart = sorted(summary["cartpole"]["runs"], key=lambda row: (0 if row["algorithm"] == "ga" else 1, row["seed"]))
    trials = records["cartpole_trials.csv"]
    for y, run in enumerate(cart):
        algorithm, seed = run["algorithm"], run["seed"]
        held = [number(r["return"]) for r in trials if r["algorithm"] == algorithm and int(r["seed"]) == seed and r["partition"] == "heldout"]
        offsets = [(i - (len(held) - 1) / 2) * 0.014 for i in range(len(held))]
        axes[0].scatter(held, [y + offset for offset in offsets], marker="|", color="#B4BFCB", s=38, linewidth=0.8, zorder=1)
        axes[0].plot([run["training_mean"], run["heldout_mean"]], [y, y], color=COLORS[algorithm], linewidth=1.5)
        axes[0].plot(run["training_mean"], y, "o", color=INK, markerfacecolor="white", markersize=9, zorder=3)
        axes[0].plot(run["heldout_mean"], y, "D", color=COLORS[algorithm], markersize=6, zorder=4)
        axes[0].text(552, y, f"{run['heldout_full_horizon']}/20", ha="center", va="center", fontsize=10)
    axes[0].set_yticks(range(len(cart)), [f"{r['algorithm'].upper()} · seed {r['seed']}\n{r['evaluations']:,} calls" for r in cart], fontsize=9.5)
    axes[0].set_ylim(len(cart) - 0.5, -0.5)
    axes[0].set_title("Cart-pole · fixed 4 → 4 → 1 MLP", loc="left", pad=23)
    axes[0].set_xlabel("Return (reward includes a failing action)")
    double = summary["doublepole"]["runs"]
    for y, run in enumerate(double):
        algorithm = run["algorithm"]
        held = [number(r["valid_steps"]) for r in records["doublepole_trials.csv"] if r["algorithm"] == algorithm and r["partition"] == "heldout"]
        offsets = [(i - (len(held) - 1) / 2) * 0.018 for i in range(len(held))]
        axes[1].scatter(held, [y + offset for offset in offsets], marker="|", color="#AAB5C2", s=42, linewidth=1.0, zorder=1)
        axes[1].plot([run["training_valid_steps"], run["heldout_mean_valid_steps"]], [y, y], color=COLORS[algorithm], linewidth=1.6)
        axes[1].plot(run["training_valid_steps"], y, "o", color=INK, markerfacecolor="white", markersize=9, zorder=3)
        axes[1].plot(run["heldout_mean_valid_steps"], y, "D", color=COLORS[algorithm], markersize=7, zorder=4)
        axes[1].text(552, y, f"{run['heldout_horizon_hits']}/16", ha="center", va="center", fontsize=10)
        if algorithm == "neat":
            axes[1].annotate(f"mean {run['heldout_mean_valid_steps']:.1f}", (run["heldout_mean_valid_steps"], y),
                             xytext=(0, -25), textcoords="offset points", ha="center", fontsize=10, color=COLORS[algorithm])
    axes[1].set_yticks(range(len(double)), [f"{'CMA-ES' if r['algorithm'] == 'cma_es' else 'NEAT'} · seed 42\n{r['fitness_evaluations']:,} calls" for r in double], fontsize=10)
    axes[1].set_ylim(1.55, -0.55)
    axes[1].set_title("Double-pole · full state, different policies", loc="left", pad=23)
    axes[1].set_xlabel("Valid endpoints (failing action earns zero)")
    for ax in axes:
        ax.set_xlim(0, 590)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.grid(axis="y", visible=False)
        ax.axvline(522, color="#D6DDE5", linewidth=0.8)
        ax.text(552 / 590, 1.014, "Horizon\nhits", transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=MUTED)
    fig.text(0.055, 0.145, "Cart-pole: three training starts; the same 20 held-out starts per seed. Double-pole: one training start and 16 fixed perturbations.", fontsize=9.5, color=MUTED)
    fig.text(0.055, 0.116, "Double-pole CMA evolves 33 fixed MLP parameters; NEAT evolves a graph. Their budgets are 32×150 versus 150×150, with early stops.", fontsize=9.5, color=MUTED)
    fig.text(0.055, 0.087, "Separate tasks, reward definitions, representations, and budgets: these observations do not establish an equal-cost algorithm ranking.", fontsize=9.5, color=MUTED)
    save(fig, output, "control_generalization", dpi)


def walker_figure(summary: dict, records: dict, output: Path, dpi: int):
    rows = records["walker_trajectories.csv"]
    traces = {
        (policy, case): sorted((r for r in rows if r["policy"] == policy and r["case"] == case), key=lambda r: int(r["control_step"]))
        for policy, case in [("selected", "neutral_500"), ("selected", "friction_06"), ("zero_action", "neutral_500")]
    }
    neutral = traces[("selected", "neutral_500")]
    fallen = traces[("selected", "friction_06")]
    zero = traces[("zero_action", "neutral_500")]
    fig, axes = plt.subplots(2, 2, figsize=(12.7, 8.8))
    fig.subplots_adjust(left=0.08, right=0.965, top=0.79, bottom=0.145, wspace=0.25, hspace=0.43)
    fig.suptitle("Spring robot: forward motion and a retained low-friction failure", x=0.055, y=0.975,
                 ha="left", fontsize=18, weight="bold")
    fig.text(0.055, 0.935, "One training-selected CMA policy (seed 42) · 13 → 4 → 2 tanh MLP · original three-particle physics", fontsize=10.5)
    conditions = [(neutral, GREEN, "Selected · neutral, friction 0.8"),
                  (fallen, RED, "Selected · held-out friction 0.6"), (zero, MUTED, "Zero-action reference")]
    fig.legend(handles=[Line2D([], [], color=color, linewidth=2, label=label) for _, color, label in conditions],
               loc="upper left", bbox_to_anchor=(0.055, 0.89), ncol=3, fontsize=10)
    for trace, color, _ in conditions:
        steps = [int(r["control_step"]) for r in trace]
        axes[0, 0].plot(steps, [number(r["com_distance"]) for r in trace], color=color, linewidth=1.8)
        axes[0, 1].plot(steps, [number(r["torso_y"]) for r in trace], color=color, linewidth=1.6)
    axes[0, 0].set_title("Actual center-of-mass displacement", loc="left")
    axes[0, 0].set_ylabel("Horizontal displacement (m)")
    axes[0, 0].margins(y=0.15)
    axes[0, 0].annotate(f"{number(neutral[-1]['com_distance']):.2f} m", (500, number(neutral[-1]["com_distance"])),
                        xytext=(-4, 9), textcoords="offset points", ha="right", color=GREEN, fontsize=10)
    last_step = int(fallen[-1]["control_step"])
    axes[0, 0].plot(last_step, number(fallen[-1]["com_distance"]), "x", color=RED, markersize=7, markeredgewidth=2)
    axes[0, 0].annotate(f"fall at step {last_step}\n{number(fallen[-1]['com_distance']):.2f} m", (last_step, number(fallen[-1]["com_distance"])),
                        xytext=(12, 18), textcoords="offset points", color=RED, fontsize=10)
    axes[0, 1].set_title("Torso height and termination", loc="left")
    axes[0, 1].set_ylabel("Torso center height (m)")
    axes[0, 1].axhline(0.15, color=RED, linestyle=":", linewidth=1)
    axes[0, 1].text(493, 0.18, "fall threshold 0.15 m", ha="right", va="bottom", fontsize=9, color=RED)
    axes[0, 1].plot(last_step, number(fallen[-1]["torso_y"]), "x", color=RED, markersize=7, markeredgewidth=2)
    axes[0, 1].set_ylim(0, 1.65)
    foot_colors = {"left_y": "#2868AF", "right_y": "#CF7D27"}
    for ax, trace, case, title in [
        (axes[1, 0], neutral, "neutral_500", "Neutral: asymmetric foot motion"),
        (axes[1, 1], fallen, "friction_06", "Low friction: motion stops at the fall"),
    ]:
        steps = [int(r["control_step"]) for r in trace]
        for column, color in foot_colors.items():
            ax.plot(steps, [number(r[column]) for r in trace], color=color, linewidth=1.2,
                    label="Left foot" if column == "left_y" else "Right foot")
        ax.axhline(0.06, color=MUTED, linewidth=0.9, linestyle=":")
        ax.set_title(title, loc="left")
        ax.set_ylabel("Foot center height (m)")
        ax.set_ylim(-0.03, 1.65)
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        proxy = summary["walker"]["foot_height_proxies"][case]
        ax.text(0.02, 0.84,
                f"At/below contact height: left {proxy['left_fraction_at_or_below_foot_radius']:.0%}, right {proxy['right_fraction_at_or_below_foot_radius']:.0%}",
                transform=ax.transAxes, fontsize=8.5, color=MUTED)
    axes[1, 1].axvspan(last_step, 500, color="#F6E8EA", alpha=0.7)
    axes[1, 1].text((last_step + 500) / 2, 0.53, "No states after\ntermination", ha="center", va="center", color=RED, fontsize=10)
    for ax in axes.flat:
        ax.set_xlim(0, 510)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.set_xlabel("Control step")
    fig.text(0.055, 0.075, "Training used 250-step episodes; these final replays allow 500. Time-limit truncation is not a locomotion-success definition.", fontsize=9.5, color=MUTED)
    fig.text(0.055, 0.05, "The 0.06 m foot-height reference is a geometric proxy at recorded endpoints, not a contact-force or time-fraction measurement.", fontsize=9.5, color=MUTED)
    fig.text(0.055, 0.025, "These are actual trajectories of an original spring robot, not Gym BipedalWalker or a validated alternating biped gait.", fontsize=9.5, color=MUTED)
    save(fig, output, "walker_motion", dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "book/results/ch03", help="Directory containing the recorded Rust outputs")
    parser.add_argument("--output", type=Path, help="Plot/summary destination; defaults to --results")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution; SVG remains vector")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    output = args.output or args.results
    output.mkdir(parents=True, exist_ok=True)
    summary, records = summarize(args.results)
    style()
    neat_figure(summary, records, output, args.dpi)
    classification_figure(summary, records, output, args.dpi)
    control_figure(summary, records, output, args.dpi)
    walker_figure(summary, records, output, args.dpi)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote four PNG/SVG figure pairs and summary.json to {output}")


if __name__ == "__main__":
    main()
