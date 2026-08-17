"""E5/E25 — 사전등록 검증 격자: 교정 길이 × SNR × 화자 × 하위군 × 예산.

이 하니스가 답하려는 질문들
---------------------------
B1 개인화가 쓸모 있어지려면 교정이 얼마나 필요한가?
B2 개인화 이득이 음향 조건에 따라 달라지는가?
B3 화자 조건에 대해 표현이 견고한가?
B4 총합 평균 뒤에 붕괴하는 하위군이 있는가?
B5 예산에 따른 이득이 왜 단조롭지 않은가?

설계 결정: 조건을 코호트 **안에서** 변화시킨다
-----------------------------------------------
``WordTrial`` 은 시행마다 ``snr_db`` 와 ``speaker`` 를 갖습니다. 따라서 SNR 조건마다 별도
코호트를 만드는 대신, 한 코호트가 모든 조건을 담고 사후에 조건별로 잘라냅니다. 두 가지
이유에서 그렇게 합니다.

* **짝지어집니다.** 같은 청취자가 모든 SNR 을 경험하므로 조건 간 차이에서 청취자 변동이
  상쇄됩니다. 조건별 코호트를 쓰면 청취자가 달라져 그 변동이 그대로 남습니다.
* **배치 현실과 일치합니다.** 실제 모델은 여러 음향 조건이 섞인 데이터로 학습되고
  임의의 조건에서 예측합니다.

교정 길이만은 코호트 속성이므로 격자 축으로 남습니다.

빠진 칸은 치명적입니다
----------------------
선언한 격자 칸이 산출물에 하나라도 없으면 :func:`run_validation_sweep` 이 실패합니다.
조건이 조용히 사라지면 "모든 조건에서 확인했다" 는 진술이 거짓이 되기 때문입니다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from audire.config.logging import get_logger
from audire.eval.ablation import evaluate_arm
from audire.eval.caption import recall_by_listener, select_budget
from audire.eval.metrics import expected_calibration_error
from audire.experiments.registry import RunRecord, finish_run, save_artifact, tracked_run
from audire.risk.features import ABLATION_ARMS
from audire.risk.models import known_models
from audire.sim.cohort import Cohort, build_cohort
from audire.sim.config import SimulationConfig

log = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: 미션이 지정한 예산. 20% 가 주 판정 지점입니다.
DEFAULT_BUDGETS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)
#: B1 이 요구하는 교정 길이.
DEFAULT_CALIBRATION_LENGTHS: tuple[int, ...] = (10, 25, 50, 100, 200, 400)
#: B2 가 요구하는 음향 조건.
DEFAULT_SNR_CONDITIONS: tuple[float, ...] = (20.0, 10.0, 5.0, 0.0, -5.0)


class ValidationSweepConfig(BaseModel):
    """사전등록 검증 격자."""

    model_config = ConfigDict(extra="forbid")

    name: str = "e25_validation_sweep"
    description: str = ""
    base_simulation: SimulationConfig
    calibration_lengths: list[int] = Field(
        default_factory=lambda: list(DEFAULT_CALIBRATION_LENGTHS), min_length=1
    )
    snr_conditions_db: list[float] = Field(
        default_factory=lambda: list(DEFAULT_SNR_CONDITIONS), min_length=1
    )
    #: 개인화 arm 과 그 비개인화 대조군.
    arm: str = "clinical_plus_confusion"
    reference_arm: str = "clinical"
    model: str = "logistic"
    n_splits: int = Field(default=5, ge=2)
    n_bootstrap: int = Field(default=0, ge=0)
    ece_bins: int = Field(default=10, ge=2)
    budgets: list[float] = Field(default_factory=lambda: list(DEFAULT_BUDGETS), min_length=1)
    primary_budget: float = 0.2

    def model_post_init(self, _: Any) -> None:
        for arm in (self.arm, self.reference_arm):
            if arm not in ABLATION_ARMS:
                raise ValueError(f"알 수 없는 arm: {arm!r}")
        if self.model not in known_models():
            raise ValueError(f"알 수 없는 모델: {self.model!r}")
        if self.primary_budget not in self.budgets:
            raise ValueError(f"주 판정 예산 {self.primary_budget} 이 budgets 안에 없습니다")

    @property
    def n_cells(self) -> int:
        """선언된 격자 칸 수: 교정 길이 × 시드."""
        return len(self.calibration_lengths) * len(self.base_simulation.seeds)

    @classmethod
    def load(cls, path: Path) -> ValidationSweepConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- 하위군 정의


def _band(value: float | None, edges: Sequence[float], labels: Sequence[str]) -> str:
    """값을 사전 선언된 구간 이름으로 바꿉니다. ``None`` 은 숨기지 않고 그대로 표시합니다."""
    if value is None:
        return "unknown"
    for edge, label in zip(edges, labels, strict=False):
        if value < edge:
            return label
    return labels[-1]


def listener_subgroups(cohort: Cohort) -> dict[str, dict[str, str]]:
    """청취자 → {축 이름: 구간 이름}.

    구간 경계는 결과를 보기 전에 고정됩니다. 사후에 경계를 움직이면 원하는 하위군 결과를
    만들어낼 수 있기 때문입니다.
    """
    out: dict[str, dict[str, str]] = {}
    for record in cohort.records:
        hearing = record.hearing
        confusion = record.estimated_confusion
        wrs = hearing.mean_wrs() if hasattr(hearing, "mean_wrs") else None
        if wrs is None:
            values = [
                e.speech.wrs_percent for e in hearing.ears if e.speech.wrs_percent is not None
            ]
            wrs = float(np.mean(values)) if values else None

        out[record.listener_id] = {
            "severity": record.listener.stratum,
            "wrs_band": _band(wrs, (60.0, 80.0), ("wrs_low", "wrs_mid", "wrs_high")),
            # coverage 는 위치별 사전 관측 비율의 사전이므로 평균을 씁니다.
            "evidence_band": _band(
                float(np.mean(list(confusion.coverage.values()))),
                (0.25, 0.55),
                ("cov_low", "cov_mid", "cov_high"),
            ),
            # 증거가 전혀 없으면 None 이며, _band 가 "unknown" 으로 남깁니다.
            "idiosyncrasy_band": _band(
                confusion.overall_accuracy(), (0.55, 0.8), ("acc_low", "acc_mid", "acc_high")
            ),
        }
    return out


# --------------------------------------------------------------------------- 지표


def _safe(fn: Callable[..., float], y: IntArray, p: FloatArray) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    return float(fn(y, p))


def slice_metrics(
    y: IntArray,
    p: FloatArray,
    groups: npt.NDArray[np.str_],
    words: list[str],
    budgets: Sequence[float],
    *,
    ece_bins: int,
    seed: int,
) -> dict[str, Any]:
    """한 조각(전체 / SNR별 / 화자별 / 하위군별)의 전체 지표 묶음."""
    n_listeners = int(np.unique(groups).size)
    out: dict[str, Any] = {
        "n_rows": int(y.size),
        "n_listeners": n_listeners,
        "prevalence": float(y.mean()) if y.size else float("nan"),
        "pr_auc": _safe(average_precision_score, y, p),
        "roc_auc": _safe(roc_auc_score, y, p),
        "brier": float(brier_score_loss(y, p)) if y.size else float("nan"),
    }
    # (ECE, MCE) 를 함께 보고합니다. 미션이 둘 다 요구하고, MCE 는 최악 구간의 괴리라
    # 평균만으로는 보이지 않는 국소적 미교정을 드러냅니다.
    if y.size and np.unique(y).size >= 1:
        ece, mce = expected_calibration_error(y, p, n_bins=ece_bins)
        out["ece"], out["mce"] = float(ece), float(mce)
    else:
        out["ece"] = out["mce"] = float("nan")
    length = np.asarray([float(len(w)) for w in words], dtype=np.float64)
    for budget in budgets:
        key = f"{budget:g}"
        for label, scores in (("model", p), ("word_length", length)):
            chosen = select_budget(scores, groups, budget, per_listener=True, tie_seed=seed)
            hits = int((chosen & (y == 1)).sum())
            n_pos = int(y.sum())
            per = np.array(
                [v for v in recall_by_listener(y, groups, chosen).values() if np.isfinite(v)],
                dtype=np.float64,
            )
            prefix = f"recall@{key}" if label == "model" else f"word_length_recall@{key}"
            out[prefix] = hits / n_pos if n_pos else float("nan")
            if label == "model":
                out[f"recall_median@{key}"] = float(np.median(per)) if per.size else float("nan")
                out[f"recall_worst@{key}"] = float(per.min()) if per.size else float("nan")
                out[f"recall_q25@{key}"] = (
                    float(np.quantile(per, 0.25)) if per.size else float("nan")
                )
                # 접근성 지표: 사실상 자막을 못 받는 청취자 비율.
                out[f"frac_listeners_near_zero@{key}"] = (
                    float(np.mean(per < 0.05)) if per.size else float("nan")
                )
        out[f"gain@{key}"] = out[f"recall@{key}"] - out[f"word_length_recall@{key}"]
    return out


def _ndcg_and_map(y: IntArray, p: FloatArray, groups: npt.NDArray[np.str_]) -> dict[str, float]:
    """청취자 내 순위 품질. 선택 자막은 청취자 안에서의 순서만 소비합니다."""
    ndcgs, aps = [], []
    for listener in np.unique(groups):
        mask = groups == listener
        yi, pi = y[mask], p[mask]
        if yi.sum() == 0:
            continue
        order = np.argsort(-pi, kind="stable")
        gains = yi[order]
        discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
        dcg = float((gains * discounts).sum())
        ideal = float((np.sort(yi)[::-1] * discounts).sum())
        if ideal > 0:
            ndcgs.append(dcg / ideal)
        aps.append(float(average_precision_score(yi, pi)))
    return {
        "ndcg_mean": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "map_mean": float(np.mean(aps)) if aps else float("nan"),
        "n_listeners_scored": len(aps),
    }


# --------------------------------------------------------------------------- 실행


def run_validation_sweep(
    cfg: ValidationSweepConfig, *, record: RunRecord | None = None
) -> dict[str, Any]:
    if record is not None:
        return _execute(cfg, record)
    with tracked_run(
        cfg.name, cfg.model_dump(mode="json"), cfg.base_simulation.seeds, notes=cfg.description
    ) as rec:
        return _execute(cfg, rec)


def _execute(cfg: ValidationSweepConfig, rec: RunRecord) -> dict[str, Any]:
    log.info("e25.start", name=cfg.name, n_cells=cfg.n_cells, run_id=rec.run_id)

    rows: list[dict[str, Any]] = []
    declared: set[tuple[int, int]] = {
        (length, seed) for length in cfg.calibration_lengths for seed in cfg.base_simulation.seeds
    }
    produced: set[tuple[int, int]] = set()

    for length in cfg.calibration_lengths:
        sim = cfg.base_simulation.model_copy(
            update={
                "n_calibration_trials": length,
                "snr_conditions_db": list(cfg.snr_conditions_db),
            }
        )
        # 설정을 다시 검증합니다. model_copy 는 검증을 건너뛰므로, 그대로 두면 잘못된
        # 조합이 조용히 실행됩니다.
        sim = SimulationConfig.model_validate(sim.model_dump())

        for seed in cfg.base_simulation.seeds:
            cohort = build_cohort(sim, seed)
            subgroups = listener_subgroups(cohort)
            trials = [t for r in cohort.records for t in r.word_trials]
            words = [t.word for t in trials]
            snr = np.asarray([t.snr_db for t in trials], dtype=np.float64)
            speaker = np.asarray([t.speaker for t in trials])

            for arm in (cfg.arm, cfg.reference_arm):
                result = evaluate_arm(
                    cohort,
                    arm,
                    cfg.model,
                    seed=seed,
                    n_splits=cfg.n_splits,
                    n_bootstrap=cfg.n_bootstrap,
                    ece_bins=cfg.ece_bins,
                )
                y, p, groups = result.y_true, result.y_prob, result.groups
                if not (y.size == len(words) == snr.size):
                    raise RuntimeError("행 정렬이 깨졌습니다: 시행 목록과 평가 행 수가 다릅니다")

                base = {
                    "calibration_length": length,
                    "seed": seed,
                    "arm": arm,
                    "model": cfg.model,
                    "n_calibration_trials": length,
                }

                rows.append(
                    {
                        **base,
                        "slice_axis": "overall",
                        "slice_value": "all",
                        **slice_metrics(
                            y, p, groups, words, cfg.budgets, ece_bins=cfg.ece_bins, seed=seed
                        ),
                        **_ndcg_and_map(y, p, groups),
                        "mean_coverage": float(
                            np.mean(
                                [
                                    np.mean(list(r.estimated_confusion.coverage.values()))
                                    for r in cohort.records
                                ]
                            )
                        ),
                    }
                )
                for value in sorted(set(snr.tolist())):
                    m = snr == value
                    rows.append(
                        {
                            **base,
                            "slice_axis": "snr_db",
                            "slice_value": f"{value:g}",
                            **slice_metrics(
                                y[m],
                                p[m],
                                groups[m],
                                [w for w, k in zip(words, m, strict=True) if k],
                                cfg.budgets,
                                ece_bins=cfg.ece_bins,
                                seed=seed,
                            ),
                        }
                    )
                for value in sorted(set(speaker.tolist())):
                    m = speaker == value
                    rows.append(
                        {
                            **base,
                            "slice_axis": "speaker",
                            "slice_value": str(value),
                            **slice_metrics(
                                y[m],
                                p[m],
                                groups[m],
                                [w for w, k in zip(words, m, strict=True) if k],
                                cfg.budgets,
                                ece_bins=cfg.ece_bins,
                                seed=seed,
                            ),
                        }
                    )
                for axis in ("severity", "wrs_band", "evidence_band", "idiosyncrasy_band"):
                    labels = np.asarray([subgroups[str(g)][axis] for g in groups])
                    for value in sorted(set(labels.tolist())):
                        m = labels == value
                        rows.append(
                            {
                                **base,
                                "slice_axis": axis,
                                "slice_value": str(value),
                                **slice_metrics(
                                    y[m],
                                    p[m],
                                    groups[m],
                                    [w for w, k in zip(words, m, strict=True) if k],
                                    cfg.budgets,
                                    ece_bins=cfg.ece_bins,
                                    seed=seed,
                                ),
                            }
                        )
            produced.add((length, seed))
            log.info("e25.cell_done", calibration_length=length, seed=seed, n_rows=len(rows))

    missing = declared - produced
    if missing:
        raise RuntimeError(
            f"선언된 격자 칸 {len(missing)}개가 산출되지 않았습니다: {sorted(missing)[:5]}"
        )

    summary = _summarise(cfg, rows)
    save_artifact(rec, "validation_sweep_rows.json", rows)
    save_artifact(rec, "summary.json", summary)
    finish_run(rec, summary["headline"])
    log.info("e25.done", run_id=rec.run_id, n_rows=len(rows))
    return {"run_id": rec.run_id, "summary": summary, "rows": rows, "artifacts": rec.artifacts}


def _agg(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _summarise(cfg: ValidationSweepConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = f"{cfg.primary_budget:g}"
    gain_key, recall_key = f"gain@{primary}", f"recall@{primary}"

    def collect(axis: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        subset = [r for r in rows if r["slice_axis"] == axis and r["arm"] == cfg.arm]
        keys = sorted({(r["calibration_length"], r["slice_value"]) for r in subset})
        for length, value in keys:
            group = [
                r for r in subset if r["calibration_length"] == length and r["slice_value"] == value
            ]
            gains = [r[gain_key] for r in group]
            entry = {
                "calibration_length": length,
                "slice_axis": axis,
                "slice_value": value,
                "n_seeds": len(group),
                "pr_auc": _agg([r["pr_auc"] for r in group]),
                "brier": _agg([r["brier"] for r in group]),
                "ece": _agg([r["ece"] for r in group]),
                recall_key: _agg([r[recall_key] for r in group]),
                "recall_median": _agg([r[f"recall_median@{primary}"] for r in group]),
                "recall_worst": _agg([r[f"recall_worst@{primary}"] for r in group]),
                "frac_near_zero": _agg([r[f"frac_listeners_near_zero@{primary}"] for r in group]),
                "gain_over_word_length": _agg(gains),
                "n_seeds_beating_word_length": int(sum(1 for g in gains if g > 0)),
            }
            for budget in cfg.budgets:
                b = f"{budget:g}"
                entry[f"recall@{b}"] = _agg([r[f"recall@{b}"] for r in group])
                entry[f"gain@{b}"] = _agg([r[f"gain@{b}"] for r in group])
            out.append(entry)
        return out

    tables = {
        axis: collect(axis)
        for axis in (
            "overall",
            "snr_db",
            "speaker",
            "severity",
            "wrs_band",
            "evidence_band",
            "idiosyncrasy_band",
        )
    }

    overall = tables["overall"]
    # B1 의 핵심 질문: 개인화가 모든 시드에서 단어길이를 이기기 시작하는 최소 교정 길이.
    qualifying = [
        e["calibration_length"]
        for e in overall
        if e["n_seeds_beating_word_length"] == e["n_seeds"] and e["n_seeds"] > 0
    ]
    # B5: 예산에 따른 이득이 단조로운가.
    budgets_sorted = sorted(cfg.budgets)
    monotone_rows = []
    for e in overall:
        seq = [e[f"gain@{b:g}"]["mean"] for b in budgets_sorted]
        monotone_rows.append(
            {
                "calibration_length": e["calibration_length"],
                "gains_by_budget": dict(zip([f"{b:g}" for b in budgets_sorted], seq, strict=True)),
                "is_monotone_increasing": all(b >= a - 1e-12 for a, b in pairwise(seq)),
            }
        )

    headline: dict[str, Any] = {
        "primary_budget": cfg.primary_budget,
        "n_declared_cells": cfg.n_cells,
        "calibration_lengths": cfg.calibration_lengths,
        "snr_conditions_db": cfg.snr_conditions_db,
        "min_calibration_beating_word_length_on_every_seed": min(qualifying, default=None),
        "budget_monotonicity": monotone_rows,
        "n_budget_conditions_non_monotone": sum(
            1 for r in monotone_rows if not r["is_monotone_increasing"]
        ),
        "caveat": (
            "합성 청취자 코호트에서 얻은 결과이며 임상적 근거가 아닙니다. SNR 과 화자는 "
            "코호트 안에서 변하므로 조건 간 비교는 같은 청취자에 대해 짝지어져 있습니다."
        ),
    }
    return {"seeds": cfg.base_simulation.seeds, "tables": tables, "headline": headline}
