"""Regenerate every figure and table from recorded experiment artifacts.

Figures are produced from ``summary.json`` alone, never from a live model, so
``make figures`` reproduces exactly what a recorded run measured. Matplotlib runs on the
non-interactive Agg backend so the command works headlessly in CI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audire.config.paths import artifacts_dir
from audire.experiments.registry import load_runs


def _latest_completed() -> str | None:
    completed = [r for r in load_runs() if r.get("status") == "completed"]
    return completed[-1]["run_id"] if completed else None


def _load_summary(run_id: str) -> dict[str, Any] | None:
    path = artifacts_dir() / run_id / "summary.json"
    if not path.exists():
        return None
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def regenerate(*, run_id: str | None = None, all_runs: bool = False) -> list[Path]:
    """Regenerate figures and tables. Returns the paths written."""
    ids: list[str]
    if all_runs:
        ids = [r["run_id"] for r in load_runs() if r.get("status") == "completed"]
    else:
        chosen = run_id or _latest_completed()
        ids = [chosen] if chosen else []

    written: list[Path] = []
    for rid in ids:
        summary = _load_summary(rid)
        if summary is None:
            continue
        out = artifacts_dir() / rid / "figures"
        out.mkdir(parents=True, exist_ok=True)
        written += _ablation_table(summary, out)
        written += _contrast_table(summary, out)
        written += _caption_frontier_figure(summary, out)
        written += _calibration_table(summary, out)
        written += _threshold_table(summary, out)
    return written


# --------------------------------------------------------------------------- tables


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> list[Path]:
    if not rows:
        return []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return [path]


def _ablation_table(summary: dict[str, Any], out: Path) -> list[Path]:
    rows = [
        {
            "arm": r["arm"],
            "model": r["model"],
            "n_listeners": r["n_listeners"],
            "n_trials": r["n_trials"],
            "n_features": r["n_features"],
            "pr_auc_mean": round(r["pr_auc"]["mean"], 4),
            "pr_auc_sd": round(r["pr_auc"]["sd"], 4),
            "roc_auc_mean": round(r["roc_auc"]["mean"], 4),
            "brier_mean": round(r["brier"]["mean"], 4),
            "log_loss_mean": round(r["log_loss"]["mean"], 4),
            "ece_mean": round(r["ece"]["mean"], 4),
            "recall_mean": round(r["recall"]["mean"], 4),
            "precision_mean": round(r["precision"]["mean"], 4),
            "specificity_mean": round(r["specificity"]["mean"], 4),
            "f1_mean": round(r["f1"]["mean"], 4),
            "prevalence_floor_pr_auc": round(r["prevalence_floor_pr_auc"]["mean"], 4),
        }
        for r in summary["arms"]
    ]
    return _write_csv(out / "table_ablation.csv", rows)


def _contrast_table(summary: dict[str, Any], out: Path) -> list[Path]:
    rows = [
        {
            "metric": c["metric"],
            "arm": c["arm"],
            "reference": c["reference"],
            "model": c["model"],
            "difference_mean": round(c["difference"]["mean"], 5),
            "difference_sd": round(c["difference"]["sd"], 5),
            "difference_min": round(c["difference"]["min"], 5),
            "difference_max": round(c["difference"]["max"], 5),
            "n_seeds_excluding_zero": c["n_seeds_excluding_zero"],
            "n_seeds": c["n_seeds"],
            "consistent_direction": c["consistent_direction"],
        }
        for c in summary["contrasts"]
    ]
    return _write_csv(out / "table_contrasts.csv", rows)


def _calibration_table(summary: dict[str, Any], out: Path) -> list[Path]:
    rows = [
        {
            "arm": r["arm"],
            "model": r["model"],
            "brier": round(r["brier"]["mean"], 4),
            "log_loss": round(r["log_loss"]["mean"], 4),
            "ece": round(r["ece"]["mean"], 4),
            "pr_auc": round(r["pr_auc"]["mean"], 4),
        }
        for r in summary["arms"]
    ]
    return _write_csv(out / "table_calibration.csv", rows)


def _threshold_table(summary: dict[str, Any], out: Path) -> list[Path]:
    rows = [
        {
            "target": key,
            "global_recall": round(v["global_recall"]["mean"], 4),
            "personalized_recall": round(v["personalized_recall"]["mean"], 4),
            "global_worst_listener": round(v["global_worst_listener"]["mean"], 4),
            "personalized_worst_listener": round(v["personalized_worst_listener"]["mean"], 4),
        }
        for key, v in sorted(summary.get("threshold_comparison", {}).items())
    ]
    return _write_csv(out / "table_threshold_comparison.csv", rows)


# --------------------------------------------------------------------------- figures


def _caption_frontier_figure(summary: dict[str, Any], out: Path) -> list[Path]:
    frontier = summary.get("caption_frontier", [])
    if not frontier:
        return []

    written: list[Path] = []
    for mode in ("per_listener", "pooled"):
        rows = [r for r in frontier if r["budget_mode"] == mode]
        if not rows:
            continue
        by_strategy: dict[str, list[tuple[float, float, float]]] = {}
        for r in rows:
            by_strategy.setdefault(r["strategy"], []).append(
                (r["budget"], r["misheard_recall"]["mean"], r["worst_listener_recall"]["mean"])
            )

        fig, (ax_agg, ax_worst) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        for strategy, points in sorted(by_strategy.items()):
            points.sort()
            budgets = [p[0] for p in points]
            ax_agg.plot(budgets, [p[1] for p in points], marker="o", label=strategy)
            ax_worst.plot(budgets, [p[2] for p in points], marker="o", label=strategy)

        ax_agg.plot([0, 0.5], [0, 0.5], linestyle=":", color="grey", label="chance")
        ax_agg.set_xlabel("caption budget (fraction of words shown)")
        ax_agg.set_ylabel("misheard-word recall")
        ax_agg.set_title(f"Aggregate recall — {mode} budget")
        ax_worst.set_xlabel("caption budget (fraction of words shown)")
        ax_worst.set_ylabel("recall of the worst-served listener")
        ax_worst.set_title("Worst-listener recall (equity view)")
        for ax in (ax_agg, ax_worst):
            ax.grid(alpha=0.3)
        ax_agg.legend(fontsize=7, loc="upper left")
        fig.suptitle(
            f"{summary['experiment']} — synthetic data, {summary['n_seeds']} seeds "
            f"(not clinical evidence)",
            fontsize=10,
        )
        path = out / f"fig_caption_frontier_{mode}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written
