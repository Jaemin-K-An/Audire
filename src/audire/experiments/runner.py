"""Config-driven experiment runners for RQ1, RQ2 and RQ3.

Each runner takes a YAML config, executes **every** declared seed, records provenance
into ``experiments/registry.yaml`` and writes metric artifacts. Nothing is selected after
the fact and no seed is dropped: results are aggregated across all of them with the seed
spread reported, which is what makes "do not cherry-pick the best seed" enforceable
rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from audire.config.logging import get_logger
from audire.eval.ablation import DEFAULT_CONTRASTS, ArmResult, evaluate_arm
from audire.eval.ablation import contrast as paired_contrast
from audire.eval.caption import compare_strategies, compare_thresholds, pareto_table
from audire.experiments.registry import RunRecord, finish_run, save_artifact, tracked_run
from audire.risk.calibration import CalibrationMethod
from audire.risk.features import ABLATION_ARMS
from audire.sim.cohort import build_cohort
from audire.sim.config import SimulationConfig

log = get_logger(__name__)


class ExperimentConfig(BaseModel):
    """A preregistered experiment specification."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    simulation: SimulationConfig
    arms: list[str] = Field(default_factory=lambda: list(ABLATION_ARMS))
    models: list[str] = Field(default_factory=lambda: ["logistic"])
    n_splits: int = Field(default=5, ge=2)
    calibration: CalibrationMethod = "none"
    n_bootstrap: int = Field(default=1000, ge=0)
    ece_bins: int = Field(default=10, ge=2)
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    caption_budgets: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    threshold_targets: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3])
    contrasts: list[tuple[str, str]] = Field(default_factory=lambda: list(DEFAULT_CONTRASTS))
    contrast_metrics: list[str] = Field(default_factory=lambda: ["pr_auc", "brier"])

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(slots=True)
class SeedResult:
    """Everything one seed produced."""

    seed: int
    arms: dict[tuple[str, str], ArmResult]
    contrasts: list[dict[str, Any]] = field(default_factory=list)
    caption_points: list[dict[str, Any]] = field(default_factory=list)
    threshold_points: list[dict[str, Any]] = field(default_factory=list)
    cohort_summary: dict[str, Any] = field(default_factory=dict)


def _aggregate(values: list[float]) -> dict[str, float]:
    """Mean, spread and range across seeds. Never a single favourable value."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n_seeds": 0,
        }
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": int(arr.size),
    }


def run_experiment(cfg: ExperimentConfig, *, record: RunRecord | None = None) -> dict[str, Any]:
    """Execute every seed of ``cfg`` and return the aggregated summary.

    The whole execution runs inside :func:`~audire.experiments.registry.tracked_run`, so
    the registry shows the run as ``running`` while it executes and as ``failed`` — with
    its config and traceback intact — if it raises. A failed run must never vanish.
    """
    if record is not None:
        return _execute(cfg, record)
    with tracked_run(
        cfg.name, cfg.model_dump(mode="json"), cfg.simulation.seeds, notes=cfg.description
    ) as rec:
        return _execute(cfg, rec)


def _execute(cfg: ExperimentConfig, rec: RunRecord) -> dict[str, Any]:
    log.info("experiment.start", name=cfg.name, seeds=cfg.simulation.seeds, run_id=rec.run_id)

    per_seed: list[SeedResult] = []
    for seed in cfg.simulation.seeds:
        cohort = build_cohort(cfg.simulation, seed)
        words = [t.word for r in cohort.records for t in r.word_trials]

        arms: dict[tuple[str, str], ArmResult] = {}
        for model_name in cfg.models:
            for arm in cfg.arms:
                if model_name == "phoneme_independence" and "confusion" not in ABLATION_ARMS[arm]:
                    continue
                arms[(arm, model_name)] = evaluate_arm(
                    cohort,
                    arm,
                    model_name,
                    seed=seed,
                    n_splits=cfg.n_splits,
                    calibration=cfg.calibration,
                    n_bootstrap=cfg.n_bootstrap,
                    ece_bins=cfg.ece_bins,
                    threshold=cfg.decision_threshold,
                )

        contrasts: list[dict[str, Any]] = []
        for model_name in cfg.models:
            for arm, reference in cfg.contrasts:
                a, b = (arm, model_name), (reference, model_name)
                if a in arms and b in arms:
                    for metric in cfg.contrast_metrics:
                        contrasts.append(
                            paired_contrast(
                                arms[a],
                                arms[b],
                                metric=metric,
                                n_bootstrap=cfg.n_bootstrap,
                                seed=seed,
                            ).to_dict()
                        )

        primary_model = cfg.models[0]
        caption_arms = {arm: r for (arm, m), r in arms.items() if m == primary_model}
        caption_points: list[dict[str, Any]] = []
        for per_listener in (True, False):
            caption_points += [
                {**row, "budget_mode": "per_listener" if per_listener else "pooled"}
                for row in pareto_table(
                    compare_strategies(
                        caption_arms,
                        words,
                        budgets=tuple(cfg.caption_budgets),
                        n_bootstrap=min(cfg.n_bootstrap, 200),
                        seed=seed,
                        per_listener=per_listener,
                    )
                )
            ]

        best_arm = caption_arms.get("clinical_plus_confusion") or next(iter(caption_arms.values()))
        threshold_points = [
            compare_thresholds(best_arm, t, n_bootstrap=0, seed=seed).to_dict()
            for t in cfg.threshold_targets
        ]

        per_seed.append(
            SeedResult(
                seed=seed,
                arms=arms,
                contrasts=contrasts,
                caption_points=caption_points,
                threshold_points=threshold_points,
                cohort_summary=cohort.summary(),
            )
        )
        log.info("experiment.seed_done", name=cfg.name, seed=seed, n_arms=len(arms))

    summary = _summarise(cfg, per_seed)
    save_artifact(
        rec,
        "arm_metrics.json",
        [{"seed": s.seed, **r.to_dict()} for s in per_seed for r in s.arms.values()],
    )
    save_artifact(rec, "contrasts.json", [c for s in per_seed for c in s.contrasts])
    save_artifact(
        rec,
        "caption_frontier.json",
        [{"seed": s.seed, **row} for s in per_seed for row in s.caption_points],
    )
    save_artifact(
        rec,
        "threshold_comparison.json",
        [{"seed": s.seed, **row} for s in per_seed for row in s.threshold_points],
    )
    save_artifact(rec, "cohort_summaries.json", [s.cohort_summary for s in per_seed])
    save_artifact(rec, "summary.json", summary)
    finish_run(rec, summary["headline"])
    log.info("experiment.done", name=cfg.name, run_id=rec.run_id, artifacts=len(rec.artifacts))
    return {"run_id": rec.run_id, "summary": summary, "artifacts": rec.artifacts}


def _summarise(cfg: ExperimentConfig, per_seed: list[SeedResult]) -> dict[str, Any]:
    """Aggregate across seeds. Reports spread, never a single favourable seed."""
    arm_rows: list[dict[str, Any]] = []
    keys = sorted({k for s in per_seed for k in s.arms})
    for arm, model in keys:
        results = [s.arms[(arm, model)] for s in per_seed if (arm, model) in s.arms]
        arm_rows.append(
            {
                "arm": arm,
                "model": model,
                "n_seeds": len(results),
                "n_listeners": results[0].n_listeners,
                "n_trials": results[0].n_trials,
                "n_features": results[0].n_features,
                "pr_auc": _aggregate([r.metrics.pr_auc for r in results]),
                "roc_auc": _aggregate([r.metrics.roc_auc for r in results]),
                "brier": _aggregate([r.metrics.brier for r in results]),
                "log_loss": _aggregate([r.metrics.log_loss for r in results]),
                "ece": _aggregate([r.metrics.ece for r in results]),
                "recall": _aggregate([r.metrics.recall for r in results]),
                "precision": _aggregate([r.metrics.precision for r in results]),
                "specificity": _aggregate([r.metrics.specificity for r in results]),
                "f1": _aggregate([r.metrics.f1 for r in results]),
                "prevalence_floor_pr_auc": _aggregate([r.prevalence_floor.pr_auc for r in results]),
            }
        )

    contrast_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for seed_result in per_seed:
        for c in seed_result.contrasts:
            grouped.setdefault((c["metric"], c["arm"], c["reference"], c["model"]), []).append(c)
    for (metric, arm, reference, model), items in sorted(grouped.items()):
        points = [i["interval"]["point"] for i in items]
        contrast_rows.append(
            {
                "metric": metric,
                "arm": arm,
                "reference": reference,
                "model": model,
                "difference": _aggregate(points),
                "n_seeds_excluding_zero": sum(1 for i in items if i["excludes_zero"]),
                "n_seeds": len(items),
                # The per-seed CIs are all in contrasts.json; this is the seed-level view.
                "consistent_direction": len({p > 0 for p in points if p != 0}) <= 1,
            }
        )

    caption_rows: dict[tuple[str, float, str], list[float]] = {}
    caption_extra: dict[tuple[str, float, str], list[float]] = {}
    for seed_result in per_seed:
        for row in seed_result.caption_points:
            ckey: tuple[str, float, str] = (
                str(row["strategy"]),
                float(row["budget"]),
                str(row["budget_mode"]),
            )
            caption_rows.setdefault(ckey, []).append(row["misheard_recall"])
            caption_extra.setdefault(ckey, []).append(row["recall_min"])
    caption_summary = [
        {
            "strategy": strategy,
            "budget": budget,
            "budget_mode": mode,
            "misheard_recall": _aggregate(values),
            "worst_listener_recall": _aggregate(caption_extra[(strategy, budget, mode)]),
        }
        for (strategy, budget, mode), values in sorted(caption_rows.items())
    ]

    threshold_summary: dict[str, Any] = {}
    for seed_result in per_seed:
        for row in seed_result.threshold_points:
            key = f"target_{row['target_ratio']:.2f}"
            slot = threshold_summary.setdefault(
                key,
                {
                    "global_recall": [],
                    "personalized_recall": [],
                    "global_worst_listener": [],
                    "personalized_worst_listener": [],
                },
            )
            slot["global_recall"].append(row["global"]["misheard_recall"])
            slot["personalized_recall"].append(row["personalized"]["misheard_recall"])
            slot["global_worst_listener"].append(row["global"]["recall_min"])
            slot["personalized_worst_listener"].append(row["personalized"]["recall_min"])
    threshold_summary = {
        k: {name: _aggregate(vals) for name, vals in v.items()}
        for k, v in threshold_summary.items()
    }

    best = max(
        (r for r in arm_rows if r["model"] == cfg.models[0]),
        key=lambda r: r["pr_auc"]["mean"],
        default=None,
    )
    return {
        "experiment": cfg.name,
        "description": cfg.description,
        "n_seeds": len(per_seed),
        "seeds": [s.seed for s in per_seed],
        "is_synthetic": True,
        "evidence": cfg.simulation.evidence_report(),
        "cohort": {
            "n_listeners": cfg.simulation.n_listeners,
            "n_calibration_trials": cfg.simulation.n_calibration_trials,
            "n_word_trials": cfg.simulation.n_word_trials,
            "mishear_rate": _aggregate([s.cohort_summary["mishear_rate"] for s in per_seed]),
        },
        "arms": arm_rows,
        "contrasts": contrast_rows,
        "caption_frontier": caption_summary,
        "threshold_comparison": threshold_summary,
        "headline": {
            "best_arm": best["arm"] if best else None,
            "best_pr_auc_mean": best["pr_auc"]["mean"] if best else None,
            "n_seeds": len(per_seed),
            "caveat": (
                "Synthetic data. These are engineering and design-sensitivity results, "
                "not clinical evidence about human listeners."
            ),
        },
    }
