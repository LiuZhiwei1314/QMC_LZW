#!/usr/bin/env python3
"""Analyze QMC_LZW TensorBoard outputs.

This script reads scalar summaries from a run directory, exports CSV files,
plots the main training curves, and computes summary statistics over the tail
of the run. It is intentionally independent of the training code so it can be
used after a run has finished.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - handled at runtime
    plt = None
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception as exc:  # pragma: no cover - handled at runtime
    event_accumulator = None
    _TENSORBOARD_ERROR = exc
else:
    _TENSORBOARD_ERROR = None


MAIN_TAGS = [
    "pretrain/loss",
    "train/loss",
    "train/variance",
    "train/pmove",
    "train/pmove_window",
    "train/mcmc_width",
]


@dataclass
class ScalarSeries:
    tag: str
    steps: np.ndarray
    values: np.ndarray
    wall_times: np.ndarray

    def tail(self, tail: int | None = None, burnin_step: int | None = None) -> "ScalarSeries":
        mask = np.ones_like(self.steps, dtype=bool)
        if burnin_step is not None:
            mask &= self.steps >= burnin_step
        steps = self.steps[mask]
        values = self.values[mask]
        wall_times = self.wall_times[mask]
        if tail is not None and tail > 0 and len(steps) > tail:
            steps = steps[-tail:]
            values = values[-tail:]
            wall_times = wall_times[-tail:]
        return ScalarSeries(self.tag, steps, values, wall_times)


def _safe_name(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _event_dir(run_dir: Path) -> Path:
    tb_dir = run_dir / "tensorboard"
    return tb_dir if tb_dir.exists() else run_dir


def _deduplicate_by_step(steps: list[int], values: list[float], wall_times: list[float]) -> ScalarSeries:
    # Keep the latest event for duplicated steps, which can happen when resuming
    # into the same log directory.
    latest: dict[int, tuple[float, float]] = {}
    for step, value, wall_time in zip(steps, values, wall_times):
        latest[int(step)] = (float(value), float(wall_time))
    sorted_steps = np.array(sorted(latest), dtype=np.int64)
    sorted_values = np.array([latest[int(step)][0] for step in sorted_steps], dtype=np.float64)
    sorted_wall_times = np.array([latest[int(step)][1] for step in sorted_steps], dtype=np.float64)
    return sorted_steps, sorted_values, sorted_wall_times


def load_scalars(run_dir: Path) -> dict[str, ScalarSeries]:
    if event_accumulator is None:
        raise RuntimeError(f"TensorBoard event reader is unavailable: {_TENSORBOARD_ERROR}")

    tb_dir = _event_dir(run_dir)
    event_files = sorted(tb_dir.glob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {tb_dir}")

    grouped: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for event_file in event_files:
        acc = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            steps, values, wall_times = grouped.setdefault(tag, ([], [], []))
            for event in acc.Scalars(tag):
                steps.append(int(event.step))
                values.append(float(event.value))
                wall_times.append(float(event.wall_time))

    series = {}
    for tag, (steps, values, wall_times) in grouped.items():
        s, v, w = _deduplicate_by_step(steps, values, wall_times)
        series[tag] = ScalarSeries(tag, s, v, w)
    return series


def write_csv(series: dict[str, ScalarSeries], out_dir: Path) -> None:
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for tag, item in series.items():
        data = np.column_stack([item.steps, item.values, item.wall_times])
        path = csv_dir / f"{_safe_name(tag)}.csv"
        np.savetxt(path, data, delimiter=",", header="step,value,wall_time", comments="")

    available = [tag for tag in MAIN_TAGS if tag in series]
    if not available:
        return
    all_steps = sorted(set().union(*(map(int, series[tag].steps) for tag in available)))
    table = np.full((len(all_steps), len(available) + 1), np.nan, dtype=np.float64)
    table[:, 0] = all_steps
    for col, tag in enumerate(available, start=1):
        value_by_step = {int(step): value for step, value in zip(series[tag].steps, series[tag].values)}
        table[:, col] = [value_by_step.get(int(step), np.nan) for step in all_steps]
    header = "step," + ",".join(_safe_name(tag) for tag in available)
    np.savetxt(out_dir / "scalars_combined.csv", table, delimiter=",", header=header, comments="")


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _plot_series(ax, item: ScalarSeries, title: str, rolling: int = 1) -> None:
    ax.plot(item.steps, item.values, alpha=0.35 if rolling > 1 else 1.0, linewidth=1.0, label="raw")
    if rolling > 1 and len(item.values) >= rolling:
        ax.plot(item.steps, _rolling_mean(item.values, rolling), linewidth=1.8, label=f"rolling {rolling}")
        ax.legend(frameon=False)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)


def make_plots(series: dict[str, ScalarSeries], out_dir: Path, tail: int | None, burnin_step: int | None) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib is unavailable: {_MATPLOTLIB_ERROR}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    plots = [
        ("train/loss", "train/loss", 100, (-50.0, 0.0)),
        ("train/variance", "train/variance", 100, (0.0, 10000.0)),
        ("train/pmove", "train/pmove", 100, None),
        ("train/mcmc_width", "train/mcmc_width", 1, None),
    ]
    for ax, (tag, title, rolling, ylim) in zip(axes.flat, plots):
        if tag in series:
            _plot_series(ax, series[tag], title, rolling=rolling)
            if ylim is not None:
                ax.set_ylim(*ylim)
        else:
            ax.set_axis_off()
            ax.set_title(f"missing: {tag}")
    fig.savefig(out_dir / "training_curves.png", dpi=180)
    plt.close(fig)

    if "train/loss" in series:
        tail_series = series["train/loss"].tail(tail=tail, burnin_step=burnin_step)
        if len(tail_series.values) > 0:
            mean = float(np.mean(tail_series.values))
            sem = float(np.std(tail_series.values, ddof=1) / math.sqrt(len(tail_series.values))) if len(tail_series.values) > 1 else math.nan
            fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
            ax.plot(tail_series.steps, tail_series.values, linewidth=1.0, alpha=0.65)
            ax.axhline(mean, color="tab:red", linewidth=1.5, label=f"mean = {mean:.8f} Eh")
            if math.isfinite(sem):
                ax.fill_between(
                    tail_series.steps,
                    mean - sem,
                    mean + sem,
                    color="tab:red",
                    alpha=0.18,
                    label=f"naive SEM = {sem:.2e} Eh",
                )
            ax.set_ylim(-50.0, 0.0)
            ax.set_title("train/loss tail statistics")
            ax.set_xlabel("step")
            ax.set_ylabel("energy / Eh")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            fig.savefig(out_dir / "loss_tail.png", dpi=180)
            plt.close(fig)


def summarize(series: dict[str, ScalarSeries], tail: int | None, burnin_step: int | None) -> dict[str, dict[str, float | int | None]]:
    summary = {}
    for tag in MAIN_TAGS:
        if tag not in series:
            continue
        item = series[tag].tail(tail=tail, burnin_step=burnin_step)
        values = item.values
        if len(values) == 0:
            continue
        std = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        sem = float(std / math.sqrt(len(values))) if len(values) > 1 else math.nan
        summary[tag] = {
            "count": int(len(values)),
            "first_step": int(item.steps[0]),
            "last_step": int(item.steps[-1]),
            "last": float(values[-1]),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": std,
            "naive_sem": sem,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def print_summary(summary: dict[str, dict[str, float | int | None]]) -> None:
    for tag, stats in summary.items():
        print(f"\n[{tag}]")
        print(f"  steps: {stats['first_step']} .. {stats['last_step']}  n={stats['count']}")
        print(f"  mean:  {stats['mean']:.10g}")
        print(f"  median:{stats['median']:.10g}")
        print(f"  last:  {stats['last']:.10g}")
        print(f"  std:   {stats['std']:.10g}")
        print(f"  sem*:  {stats['naive_sem']:.4g}  (*naive, ignores autocorrelation)")
        print(f"  min/max: {stats['min']:.10g} / {stats['max']:.10g}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Run directory, e.g. outputs/carbon_formal_001")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to <run>/analysis")
    parser.add_argument("--tail", type=int, default=5000, help="Use only the last N scalar points for summary/tail plot. Use 0 for all.")
    parser.add_argument("--burnin-step", type=int, default=None, help="Ignore scalar points before this step for summary/tail plot.")
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV and summary JSON.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = args.run.expanduser().resolve()
    out_dir = (args.out.expanduser().resolve() if args.out else run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    tail = None if args.tail == 0 else args.tail

    series = load_scalars(run_dir)
    write_csv(series, out_dir)
    summary = summarize(series, tail=tail, burnin_step=args.burnin_step)
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    if not args.no_plots:
        make_plots(series, out_dir, tail=tail, burnin_step=args.burnin_step)
    print(f"Loaded {len(series)} scalar tags from {run_dir}")
    print(f"Wrote analysis to {out_dir}")
    print_summary(summary)


if __name__ == "__main__":
    main()
