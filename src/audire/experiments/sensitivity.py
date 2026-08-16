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
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from audire.config.logging import get_logger
from audire.eval.ablation import evaluate_arm
from audire.eval.caption import compare_strategies
from audire.experiments.registry import RunRecord, finish_run, new_run, save_artifact
from audire.sim.cohort import build_cohort
from audire.sim.config import SimulationConfig

log = get_logger(__name__)


class SensitivityConfig(BaseModel):
    """사전등록된 2차원 민감도 스윕 명세."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    base_simulation: SimulationConfig
    dirichlet_concentration: list[float] = Field(min_length=1)
    calibration_lengths: list[int] = Field(min_length=1)
    arms: list[str] = Field(
        default_factory=lambda: ["word_context_only", "clinical", "clinical_plus_confusion"]
    )
    models: list[str] = Field(default_factory=lambda: ["logistic"])
    n_splits: int = Field(default=5, ge=2)
    n_bootstrap: int = Field(default=200, ge=0)
    caption_budgets: list[float] = Field(default_factory=lambda: [0.20])

    @classmethod
    def load(cls, path: Path) -> SensitivityConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    @property
    def n_cells(self) -> int:
        return (
            len(self.dirichlet_concentration)
            * len(self.calibration_lengths)
            * len(self.base_simulation.seeds)
        )


def run_sensitivity(cfg: SensitivityConfig, *, record: RunRecord | None = None) -> dict[str, Any]:
    """격자의 모든 칸을 모든 시드로 실행합니다. 어떤 칸도 건너뛰지 않습니다."""
    rec = record or new_run(
        cfg.name,
        cfg.model_dump(mode="json"),
        cfg.base_simulation.seeds,
        notes=cfg.description,
    )
    log.info("sensitivity.start", name=cfg.name, n_cells=cfg.n_cells, run_id=rec.run_id)

    rows: list[dict[str, Any]] = []
    primary = cfg.models[0]

    for conc in cfg.dirichlet_concentration:
        for n_cal in cfg.calibration_lengths:
            sim = cfg.base_simulation.model_copy(
                update={
                    "n_calibration_trials": n_cal,
                    "confusion": cfg.base_simulation.confusion.model_copy(
                        update={"dirichlet_concentration": conc}
                    ),
                }
            )
            for seed in cfg.base_simulation.seeds:
                cohort = build_cohort(sim, seed)
                words = [t.word for r in cohort.records for t in r.word_trials]

                arms = {
                    arm: evaluate_arm(
                        cohort,
                        arm,
                        primary,
                        seed=seed,
                        n_splits=cfg.n_splits,
                        n_bootstrap=0,
                    )
                    for arm in cfg.arms
                }
                points = compare_strategies(
                    arms,
                    words,
                    budgets=tuple(cfg.caption_budgets),
                    n_bootstrap=0,
                    seed=seed,
                    per_listener=True,
                )
                by_strategy = {p.strategy: p for p in points}

                for arm, result in arms.items():
                    strategy = f"model:{arm}"
                    pt = by_strategy[strategy]
                    baseline = by_strategy["word_length"]
                    rows.append(
                        {
                            "dirichlet_concentration": conc,
                            "n_calibration_trials": n_cal,
                            "seed": seed,
                            "arm": arm,
                            "pr_auc": result.metrics.pr_auc,
                            "brier": result.metrics.brier,
                            "ece": result.metrics.ece,
                            "misheard_recall": pt.misheard_recall,
                            "worst_listener_recall": pt.recall_min,
                            # RQ2 의 핵심 질문: 비개인화 휴리스틱을 이기는가?
                            "recall_over_word_length": pt.misheard_recall
                            - baseline.misheard_recall,
                            "beats_word_length": pt.misheard_recall > baseline.misheard_recall,
                            "budget": pt.budget,
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
    """격자 칸별로 시드를 집계합니다. 유리한 칸만 고르지 않습니다."""
    cells: dict[tuple[float, int, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["dirichlet_concentration"], r["n_calibration_trials"], r["arm"])
        cells.setdefault(key, []).append(r)

    grid: list[dict[str, Any]] = []
    for (conc, n_cal, arm), items in sorted(cells.items()):
        gains = [i["recall_over_word_length"] for i in items]
        grid.append(
            {
                "dirichlet_concentration": conc,
                "n_calibration_trials": n_cal,
                "arm": arm,
                "n_seeds": len(items),
                "pr_auc_mean": float(np.mean([i["pr_auc"] for i in items])),
                "misheard_recall_mean": float(np.mean([i["misheard_recall"] for i in items])),
                "worst_listener_recall_mean": float(
                    np.mean([i["worst_listener_recall"] for i in items])
                ),
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
