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
import numpy as np

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
        if "grid" in summary:
            written += _sensitivity_table(summary, out)
            written += _sensitivity_figure(summary, out)
        elif "table" in summary:
            written += _model_comparison_table(summary, out)
        elif "arms" in summary:
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


def _model_comparison_table(summary: dict[str, Any], out: Path) -> list[Path]:
    """E22 표.

    지는 후보를 걸러내지 않습니다. 정렬만 이득 순으로 하고 행은 전부 남기므로, 표를
    읽는 사람이 "무엇이 실패했는가" 를 사후 편집 없이 볼 수 있습니다. 교정 폴백 횟수와
    ``output_is_probability`` 를 함께 실어, 확률 지표를 읽어도 되는 행인지 판단할 수
    있게 합니다.
    """
    budget_keys = sorted(
        {k for r in summary["table"] for k in r if k.startswith("recall_at_")},
        key=lambda k: float(k.removeprefix("recall_at_")),
    )
    rows = [
        {
            "model": r["model"],
            "arm": r["arm"],
            "calibration_requested": r["calibration_requested"],
            "effective_methods": "|".join(r["effective_methods"]),
            "n_folds_fell_back": r["n_folds_fell_back"],
            "output_is_probability": r["output_is_probability"],
            "n_seeds": r["n_seeds"],
            "pr_auc_mean": round(r["pr_auc"]["mean"], 4),
            "pr_auc_sd": round(r["pr_auc"]["sd"], 4),
            "brier_mean": round(r["brier"]["mean"], 4),
            "ece_mean": round(r["ece"]["mean"], 4),
            **{k: round(r[k]["mean"], 4) for k in budget_keys},
            "recall_median_listener": round(r["recall_median_listener"]["mean"], 4),
            "recall_worst_listener": round(r["recall_worst_listener"]["mean"], 4),
            "gain_over_word_length_mean": round(r["gain_over_word_length"]["mean"], 5),
            "gain_over_word_length_sd": round(r["gain_over_word_length"]["sd"], 5),
            "n_seeds_beating_word_length": r["n_seeds_beating_word_length"],
        }
        for r in sorted(summary["table"], key=lambda r: -r["gain_over_word_length"]["mean"])
    ]
    return _write_csv(out / "table_model_comparison.csv", rows)


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


def _sensitivity_table(summary: dict[str, Any], out: Path) -> list[Path]:
    rows = [
        {
            "dirichlet_concentration": r["dirichlet_concentration"],
            "n_calibration_trials": r["n_calibration_trials"],
            "arm": r["arm"],
            "n_seeds": r["n_seeds"],
            "pr_auc_mean": round(r["pr_auc_mean"], 4),
            "misheard_recall_mean": round(r["misheard_recall_mean"], 4),
            "worst_listener_recall_mean": round(r["worst_listener_recall_mean"], 4),
            "recall_over_word_length_mean": round(r["recall_over_word_length_mean"], 4),
            "n_seeds_beating_word_length": r["n_seeds_beating_word_length"],
        }
        for r in summary["grid"]
    ]
    return _write_csv(out / "table_sensitivity.csv", rows)


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


def _sensitivity_figure(summary: dict[str, Any], out: Path) -> list[Path]:
    """Render the primary personalized arm over both preregistered sweep axes."""
    arm = "clinical_plus_confusion"
    rows = [r for r in summary["grid"] if r["arm"] == arm]
    if not rows:
        return []

    concentrations = sorted({float(r["dirichlet_concentration"]) for r in rows})
    calibrations = sorted({int(r["n_calibration_trials"]) for r in rows})
    row_index = {value: index for index, value in enumerate(concentrations)}
    column_index = {value: index for index, value in enumerate(calibrations)}
    shape = (len(concentrations), len(calibrations))
    gains = np.full(shape, np.nan)
    wins = np.full(shape, np.nan)
    seed_counts = np.full(shape, np.nan)
    for row in rows:
        index = (
            row_index[float(row["dirichlet_concentration"])],
            column_index[int(row["n_calibration_trials"])],
        )
        gains[index] = float(row["recall_over_word_length_mean"])
        wins[index] = float(row["n_seeds_beating_word_length"])
        seed_counts[index] = float(row["n_seeds"])

    gain_limit = max(
        (abs(float(value)) for value in gains.flat if np.isfinite(value)), default=0.001
    )
    gain_limit = max(gain_limit, 0.001)
    max_seeds = max((float(value) for value in seed_counts.flat if np.isfinite(value)), default=1.0)

    fig, (ax_gain, ax_wins) = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.15, top=0.80, wspace=0.38)
    gain_image = ax_gain.imshow(
        gains,
        cmap="coolwarm",
        vmin=-gain_limit,
        vmax=gain_limit,
        aspect="auto",
        origin="lower",
    )
    win_image = ax_wins.imshow(
        wins,
        cmap="viridis",
        vmin=0,
        vmax=max_seeds,
        aspect="auto",
        origin="lower",
    )

    for ax in (ax_gain, ax_wins):
        ax.set_xticks(range(len(calibrations)), labels=calibrations)
        ax.set_yticks(range(len(concentrations)), labels=[f"{value:g}" for value in concentrations])
        ax.set_xlabel("calibration trials")
        ax.set_ylabel("Dirichlet concentration (lower = more idiosyncratic)")
    ax_gain.set_title("Recall gain over word-length heuristic")
    ax_wins.set_title("Seeds beating word-length heuristic")

    for row_number, column_number in np.ndindex(shape):
        if np.isfinite(gains[row_number, column_number]):
            ax_gain.text(
                column_number,
                row_number,
                f"{gains[row_number, column_number]:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
            )
        if np.isfinite(wins[row_number, column_number]):
            wins_label = (
                f"{int(wins[row_number, column_number])}/"
                f"{int(seed_counts[row_number, column_number])}"
            )
            ax_wins.text(
                column_number,
                row_number,
                wins_label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if wins[row_number, column_number] > max_seeds / 2 else "black",
            )

    fig.colorbar(gain_image, ax=ax_gain, label="misheard-word recall difference")
    fig.colorbar(win_image, ax=ax_wins, label="number of seeds")
    fig.suptitle(
        f"{summary['experiment']}\nSynthetic sensitivity analysis — not clinical evidence",
        y=0.97,
        fontsize=10,
    )
    path = out / f"fig_sensitivity_{arm}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]
