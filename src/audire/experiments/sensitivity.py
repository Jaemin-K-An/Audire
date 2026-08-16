"""E11 — 개인 특이성 × 교정 길이 민감도 스윕.

RQ2 는 청취자별 자막 예산에서 개인 혼동 프로파일이 단순 단어길이 휴리스틱을 이기지
못한다고 답했습니다. 그 결과가 **방법의 성질**인지, 아니면 시뮬레이터에서 선택된 두 값의
성질인지는 그 두 값을 직접 움직여 봐야 알 수 있습니다.

축 1 — ``dirichlet_concentration``
    청취자가 자신의 전역 능력이 예측하는 것을 넘어 얼마나 특이한가. 낮을수록 특이합니다.
    특이성이 없다면 WRS 가 이미 충분하고 혼동 프로파일이 더할 것이 없습니다.

축 2 — ``n_calibration_trials``
    그 특이성을 얼마나 잘 추정할 수 있는가. 특이성이 크더라도 추정이 나쁘면 쓸 수 없습니다.

두 축이 모두 유리할 때에만 혼동 프로파일이 도움이 될 수 있다는 것이 가설이며, 이 스윕이
그 경계를 찾습니다. 격자와 시드는 결과를 보기 전에 설정 파일에 고정됩니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from audire.config.logging import get_logger
from audire.eval.ablation import evaluate_arm
from audire.eval.caption import compare_strategies
from audire.experiments.registry import RunRecord, finish_run, save_artifact, tracked_run
from audire.sim.cohort import build_cohort
from audire.sim.config import SimulationConfig

log = get_logger(__name__)


class SensitivityConfig(BaseModel):
    """사전등록된 2차원 민감도 스윕 명세."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    base_simulation: SimulationConfig
    #: 축 1. 낮을수록 청취자가 더 특이하다. 0 이하는 분포가 아니다.
    dirichlet_concentration: list[Annotated[float, Field(gt=0.0)]] = Field(min_length=1)
    #: 축 2. 교정 시행 수는 양수여야 한다.
    calibration_lengths: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)
    arms: list[str] = Field(
        default_factory=lambda: ["word_context_only", "clinical", "clinical_plus_confusion"],
        min_length=1,
    )
    #: 선언된 모델 계열을 **전부** 평가한다. 예전에는 models[0] 만 쓰면서 복수형 이름이
    #: 모든 계열을 평가한다는 인상을 주었다.
    models: list[str] = Field(default_factory=lambda: ["logistic"], min_length=1)
    n_splits: int = Field(default=5, ge=2)
    #: 실제로 사용된다. 0 이면 신뢰구간을 만들지 않는다.
    n_bootstrap: int = Field(default=200, ge=0)
    caption_budgets: list[Annotated[float, Field(gt=0.0, le=1.0)]] = Field(
        default_factory=lambda: [0.20], min_length=1
    )

    @classmethod
    def load(cls, path: Path) -> SensitivityConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    @property
    def n_grid_rows(self) -> int:
        """Cells x arms x models — the number of rows the summary must contain."""
        return (
            len(self.dirichlet_concentration)
            * len(self.calibration_lengths)
            * len(self.arms)
            * len(self.models)
            * len(self.caption_budgets)
        )

    @property
    def n_cells(self) -> int:
        return (
            len(self.dirichlet_concentration)
            * len(self.calibration_lengths)
            * len(self.base_simulation.seeds)
        )


def build_cell_simulation(
    cfg: SensitivityConfig, *, concentration: float, n_calibration: int
) -> SimulationConfig:
    """Build one grid cell's simulation config, **revalidated**.

    ``model_copy(update=...)`` bypasses pydantic validation, so a sweep could previously
    generate and run an invalid configuration (a non-positive Dirichlet concentration, a
    zero-length calibration) without complaint. Round-tripping through
    ``model_validate`` restores the guarantees the config classes are supposed to give.
    """
    payload = cfg.base_simulation.model_dump()
    payload["n_calibration_trials"] = n_calibration
    payload["confusion"]["dirichlet_concentration"] = concentration
    return SimulationConfig.model_validate(payload)


def run_sensitivity(cfg: SensitivityConfig, *, record: RunRecord | None = None) -> dict[str, Any]:
    """격자의 모든 칸을 모든 시드로 실행합니다. 어떤 칸도 건너뛰지 않습니다.

    전체 실행이 :func:`~audire.experiments.registry.tracked_run` 안에서 돌기 때문에,
    도중에 예외가 나도 레지스트리에 설정과 트레이스백을 갖춘 ``failed`` 항목이 남습니다.
    """
    if record is not None:
        return _execute(cfg, record)
    with tracked_run(
        cfg.name,
        cfg.model_dump(mode="json"),
        cfg.base_simulation.seeds,
        notes=cfg.description,
    ) as rec:
        return _execute(cfg, rec)


def _execute(cfg: SensitivityConfig, rec: RunRecord) -> dict[str, Any]:
    log.info("sensitivity.start", name=cfg.name, n_cells=cfg.n_cells, run_id=rec.run_id)

    rows: list[dict[str, Any]] = []

    for conc in cfg.dirichlet_concentration:
        for n_cal in cfg.calibration_lengths:
            sim = build_cell_simulation(cfg, concentration=conc, n_calibration=n_cal)
            for seed in cfg.base_simulation.seeds:
                cohort = build_cohort(sim, seed)
                words = [t.word for r in cohort.records for t in r.word_trials]

                for model_name in cfg.models:
                    arms = {
                        arm: evaluate_arm(
                            cohort,
                            arm,
                            model_name,
                            seed=seed,
                            n_splits=cfg.n_splits,
                            n_bootstrap=0,  # arm 지표의 CI 는 이 스윕의 관심사가 아니다
                        )
                        for arm in cfg.arms
                    }
                    points = compare_strategies(
                        arms,
                        words,
                        budgets=tuple(cfg.caption_budgets),
                        # 설정값을 실제로 사용한다. 예전에는 0 이 하드코딩돼 있었다.
                        n_bootstrap=cfg.n_bootstrap,
                        seed=seed,
                        per_listener=True,
                    )
                    # (strategy, budget) 로 키를 잡는다. strategy 만으로 키를 잡으면
                    # 예산이 여러 개일 때 마지막 것만 살아남고 나머지가 사라진다.
                    by_key = {(p.strategy, round(p.budget, 6)): p for p in points}

                    for budget in cfg.caption_budgets:
                        b = round(budget, 6)
                        baseline = by_key[("word_length", b)]
                        for arm in arms:
                            pt = by_key[(f"model:{arm}", b)]
                            rows.append(
                                {
                                    "dirichlet_concentration": conc,
                                    "n_calibration_trials": n_cal,
                                    "seed": seed,
                                    "arm": arm,
                                    "model": model_name,
                                    "budget": b,
                                    "pr_auc": arms[arm].metrics.pr_auc,
                                    "brier": arms[arm].metrics.brier,
                                    "ece": arms[arm].metrics.ece,
                                    "misheard_recall": pt.misheard_recall,
                                    "worst_listener_recall": pt.recall_min,
                                    "median_listener_recall": pt.recall_median,
                                    "recall_ci_lo": (
                                        pt.recall_ci.lo if pt.recall_ci is not None else None
                                    ),
                                    "recall_ci_hi": (
                                        pt.recall_ci.hi if pt.recall_ci is not None else None
                                    ),
                                    # RQ2 의 핵심 질문: 비개인화 휴리스틱을 이기는가?
                                    "recall_over_word_length": pt.misheard_recall
                                    - baseline.misheard_recall,
                                    "beats_word_length": pt.misheard_recall
                                    > baseline.misheard_recall,
                                }
                            )
                log.info(
                    "sensitivity.cell_done",
                    concentration=conc,
                    n_calibration=n_cal,
                    seed=seed,
                )

    summary = _summarise(cfg, rows)
    save_artifact(rec, "sensitivity_cells.json", rows)
    save_artifact(rec, "summary.json", summary)
    finish_run(rec, summary["headline"])
    log.info("sensitivity.done", name=cfg.name, run_id=rec.run_id, n_rows=len(rows))
    return {"run_id": rec.run_id, "summary": summary, "artifacts": rec.artifacts}


def _summarise(cfg: SensitivityConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """격자 칸별로 시드를 집계합니다. 유리한 칸만 고르지 않습니다.

    키는 (집중도, 교정길이, arm, model, budget) 다섯 축 전부입니다. budget 을 키에서
    빼면 예산 조건이 조용히 사라집니다.
    """
    cells: dict[tuple[float, int, str, str, float], list[dict[str, Any]]] = {}
    for r in rows:
        key = (
            r["dirichlet_concentration"],
            r["n_calibration_trials"],
            r["arm"],
            r["model"],
            r["budget"],
        )
        cells.setdefault(key, []).append(r)

    def _mean_opt(values: list[Any]) -> float | None:
        present = [v for v in values if v is not None]
        return float(np.mean(present)) if present else None

    grid: list[dict[str, Any]] = []
    for (conc, n_cal, arm, model, budget), items in sorted(cells.items()):
        gains = [i["recall_over_word_length"] for i in items]
        grid.append(
            {
                "dirichlet_concentration": conc,
                "n_calibration_trials": n_cal,
                "arm": arm,
                "model": model,
                "budget": budget,
                "n_seeds": len(items),
                "pr_auc_mean": float(np.mean([i["pr_auc"] for i in items])),
                "brier_mean": float(np.mean([i["brier"] for i in items])),
                "ece_mean": float(np.mean([i["ece"] for i in items])),
                "misheard_recall_mean": float(np.mean([i["misheard_recall"] for i in items])),
                "median_listener_recall_mean": float(
                    np.mean([i["median_listener_recall"] for i in items])
                ),
                "worst_listener_recall_mean": float(
                    np.mean([i["worst_listener_recall"] for i in items])
                ),
                "recall_ci_lo": _mean_opt([i["recall_ci_lo"] for i in items]),
                "recall_ci_hi": _mean_opt([i["recall_ci_hi"] for i in items]),
                "recall_over_word_length_mean": float(np.mean(gains)),
                "n_seeds_beating_word_length": sum(1 for i in items if i["beats_word_length"]),
            }
        )

    combined = [g for g in grid if g["arm"] == "clinical_plus_confusion"]
    winning = [
        g for g in combined if g["n_seeds_beating_word_length"] == g["n_seeds"] and g["n_seeds"] > 0
    ]
    return {
        "experiment": cfg.name,
        "description": cfg.description,
        "is_synthetic": True,
        "n_cells": cfg.n_cells,
        "grid": grid,
        "headline": {
            "n_cells_where_personalization_always_beats_word_length": len(winning),
            "n_combined_cells": len(combined),
            "winning_cells": [
                {
                    "dirichlet_concentration": g["dirichlet_concentration"],
                    "n_calibration_trials": g["n_calibration_trials"],
                    "budget": g["budget"],
                    "model": g["model"],
                    "gain": round(g["recall_over_word_length_mean"], 4),
                }
                for g in winning
            ],
            "caveat": (
                "합성 데이터입니다. 개인화가 이득을 내는 영역이 존재하는지에 대한 설계 "
                "민감도 결과이며, 사람 청취자에 대한 근거가 아닙니다."
            ),
        },
    }
