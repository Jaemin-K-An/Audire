"""E22 — 후보 모델 계열의 청취자 수준 비교, 그리고 교정 방법 비교.

이 하니스가 답하려는 질문은 하나입니다.

    고정된 **청취자별** 자막 예산에서 (특히 20%), 개인화 모델이 단순한 단어길이
    휴리스틱보다 오청 단어를 더 많이 잡아내는가?

그 질문에 답하는 방식에 대해 몇 가지 규칙이 강제됩니다.

여러 시드에서 이겨야 한다
    한 시드에서만 유리한 결과는 결과가 아닙니다. 모든 시드가 집계되고, 유리한 시드를
    골라낼 수 없도록 "몇 개의 시드에서 이겼는가"가 함께 보고됩니다.

지는 조건도 보고한다
    후보가 참조 기저선보다 나쁘면 그대로 표에 남습니다. 실패한 후보를 표에서 빼는 것은
    선택 편향이고, 이 파일은 그것을 구조적으로 불가능하게 합니다.

교정은 요청과 실제를 구분해 기록한다
    교정기가 폴드에서 적합에 실패해 폴백하는 일이 실제로 일어납니다. 요청한 방법,
    실제 수행된 방법, 폴백 여부와 사유, 교정에 쓰인 청취자 수가 **폴드마다** 기록되어,
    "isotonic 을 평가했다" 는 주장이 실제로 isotonic 이 돌았다는 뜻인지 확인할 수 있습니다.

랭킹 점수는 확률이 아니다
    ``lambdamart`` 의 출력은 순위 점수입니다. 예산 기반 지표(Recall@k)는 순위만 쓰므로
    유효하지만, Brier/ECE 는 교정 없이는 의미가 없습니다. 그래서 확률 지표에는
    ``output_is_probability`` 플래그가 함께 실립니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from audire.config.logging import get_logger
from audire.eval.ablation import ArmResult, evaluate_arm
from audire.eval.caption import recall_by_listener, select_budget
from audire.experiments.registry import RunRecord, finish_run, save_artifact, tracked_run
from audire.risk.calibration import CalibrationMethod
from audire.risk.features import ABLATION_ARMS
from audire.risk.models import known_models
from audire.sim.cohort import build_cohort
from audire.sim.config import SimulationConfig

log = get_logger(__name__)

#: 보고되는 예산. 미션이 지정한 10/20/30/50% 이며 20% 가 주 판정 지점입니다.
DEFAULT_BUDGETS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)

#: 개인화가 반드시 이겨야 하는 비개인화 기준선. "긴 단어일수록 위험하다" 는 자막 연구에서
#: 흔한 무료 휴리스틱이고, 이것을 이기지 못하면 개인화는 비용을 정당화하지 못합니다.
WORD_LENGTH = "word_length"


class ModelComparisonConfig(BaseModel):
    """E22 비교의 사전 등록 명세."""

    model_config = ConfigDict(extra="forbid")

    name: str = "e22_model_comparison"
    description: str = ""
    simulation: SimulationConfig
    #: 비교 대상 모델. 로지스틱은 ADR-0012 의 참조 기저선이므로 반드시 포함됩니다.
    models: list[str] = Field(
        default_factory=lambda: [
            "logistic",
            "gradient_boosting",
            "phoneme_independence",
            "spline_gam",
            "residual",
            "lambdamart",
        ]
    )
    arm: str = "clinical_plus_confusion_rich"
    #: 음소 독립 기저선은 rich 블록을 쓰지 않으므로 별도 arm 이 필요합니다.
    fallback_arm: str = "clinical_plus_confusion"
    calibrations: list[CalibrationMethod] = Field(
        default_factory=lambda: cast("list[CalibrationMethod]", ["none", "platt"])
    )
    group_shrinkage: bool = False
    n_splits: int = Field(default=5, ge=2)
    n_bootstrap: int = Field(default=200, ge=0)
    ece_bins: int = Field(default=10, ge=2)
    budgets: list[float] = Field(default_factory=lambda: list(DEFAULT_BUDGETS))
    #: 주 판정 예산. 이 값에서의 단어길이 대비 이득이 headline 이 됩니다.
    primary_budget: float = 0.2

    @classmethod
    def load(cls, path: Path) -> ModelComparisonConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def model_post_init(self, _: Any) -> None:
        unknown = sorted(set(self.models) - set(known_models()))
        if unknown:
            raise ValueError(f"알 수 없는 모델: {unknown}; 사용 가능: {known_models()}")
        for arm in (self.arm, self.fallback_arm):
            if arm not in ABLATION_ARMS:
                raise ValueError(f"알 수 없는 arm: {arm!r}")
        if self.primary_budget not in self.budgets:
            raise ValueError(
                f"주 판정 예산 {self.primary_budget} 이 budgets {self.budgets} 안에 없습니다"
            )


def _word_length_scores(words: list[str]) -> np.ndarray:
    """음절 수를 위험 점수로 쓰는 비개인화 기준선."""
    return np.asarray([float(len(w)) for w in words], dtype=np.float64)


def budget_metrics(
    y: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    budgets: tuple[float, ...],
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    """청취자별 예산에서의 재현율과 청취자 간 분포.

    총합 재현율만 보면 위험한 청취자 몇 명을 잘 챙기고 나머지를 방치하는 정책이 좋아
    보입니다. 중앙값과 최하위 청취자를 함께 보고해야 그 절충이 드러납니다.
    """
    out: dict[str, dict[str, float]] = {}
    for budget in budgets:
        selected = select_budget(scores, groups, budget, per_listener=True, tie_seed=seed)
        hits = int((selected & (y == 1)).sum())
        n_misheard = int(y.sum())
        per = np.array(
            [v for v in recall_by_listener(y, groups, selected).values() if np.isfinite(v)],
            dtype=np.float64,
        )
        out[f"{budget:g}"] = {
            "budget": budget,
            "recall": hits / n_misheard if n_misheard else float("nan"),
            "recall_median_listener": float(np.median(per)) if per.size else float("nan"),
            "recall_worst_listener": float(per.min()) if per.size else float("nan"),
            "achieved_ratio": float(selected.sum() / y.size) if y.size else 0.0,
        }
    return out


def _row(result: ArmResult, cfg: ModelComparisonConfig, words: list[str]) -> dict[str, Any]:
    described = result.model_description
    base = described.get("base", described)
    budgets = tuple(cfg.budgets)

    model_budgets = budget_metrics(
        result.y_true, result.groups, result.y_prob, budgets, seed=result.seed
    )
    length_budgets = budget_metrics(
        result.y_true, result.groups, _word_length_scores(words), budgets, seed=result.seed
    )

    return {
        "seed": result.seed,
        "model": result.model,
        "arm": result.arm,
        "calibration_requested": result.calibration,
        "n_listeners": result.n_listeners,
        "n_trials": result.n_trials,
        "n_features": result.n_features,
        # 확률 지표. 랭킹 모델은 교정 전에는 이 값들이 의미 없으므로 플래그를 함께 싣습니다.
        "pr_auc": result.metrics.pr_auc,
        "roc_auc": result.metrics.roc_auc,
        "brier": result.metrics.brier,
        "ece": result.metrics.ece,
        "prevalence_pr_auc": result.prevalence_floor.pr_auc,
        "output_is_probability": bool(base.get("output_is_probability", True)),
        # 예산 지표. 순위만 쓰므로 교정 여부와 무관하게 유효합니다.
        "budgets": model_budgets,
        "word_length_budgets": length_budgets,
        "gain_over_word_length": {
            k: model_budgets[k]["recall"] - length_budgets[k]["recall"] for k in model_budgets
        },
        # 교정 기록. 요청과 실제를 구분하고, 폴드별로 남깁니다.
        "calibration_per_fold": described.get("calibration_per_fold", []),
        "n_folds_fell_back": described.get("n_folds_fell_back", 0),
        "group_shrinkage": described.get("group_shrinkage", []),
        "residual_strength": base.get("residual_strength"),
    }


def run_model_comparison(
    cfg: ModelComparisonConfig, *, record: RunRecord | None = None
) -> dict[str, Any]:
    """모든 시드 x 모델 x 교정 조합을 실행하고 집계 요약을 반환합니다."""
    if record is not None:
        return _execute(cfg, record)
    with tracked_run(
        cfg.name, cfg.model_dump(mode="json"), cfg.simulation.seeds, notes=cfg.description
    ) as rec:
        return _execute(cfg, rec)


def _execute(cfg: ModelComparisonConfig, rec: RunRecord) -> dict[str, Any]:
    log.info("e22.start", name=cfg.name, seeds=cfg.simulation.seeds, run_id=rec.run_id)

    rows: list[dict[str, Any]] = []
    for seed in cfg.simulation.seeds:
        cohort = build_cohort(cfg.simulation, seed)
        words = [t.word for r in cohort.records for t in r.word_trials]

        for model_name in cfg.models:
            # 음소 독립 기저선은 rich 블록을 소비하지 않습니다. 조용히 건너뛰지 않고
            # 정의된 arm 으로 평가해 표에 남깁니다.
            arm = cfg.fallback_arm if model_name == "phoneme_independence" else cfg.arm
            for calibration in cfg.calibrations:
                result = evaluate_arm(
                    cohort,
                    arm,
                    model_name,
                    seed=seed,
                    n_splits=cfg.n_splits,
                    calibration=calibration,
                    n_bootstrap=cfg.n_bootstrap,
                    ece_bins=cfg.ece_bins,
                    group_shrinkage=cfg.group_shrinkage,
                )
                rows.append(_row(result, cfg, words))
        log.info("e22.seed_done", seed=seed, n_rows=len(rows))

    summary = _summarise(cfg, rows)
    save_artifact(rec, "model_comparison_rows.json", rows)
    save_artifact(rec, "summary.json", summary)
    finish_run(rec, summary["headline"])
    log.info("e22.done", run_id=rec.run_id, n_rows=len(rows))
    return {"run_id": rec.run_id, "summary": summary, "rows": rows, "artifacts": rec.artifacts}


def _agg(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "min": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "n": int(arr.size),
    }


def _summarise(cfg: ModelComparisonConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """시드에 걸쳐 집계합니다. 유리한 한 시드를 보고할 수 없는 구조입니다."""
    primary = f"{cfg.primary_budget:g}"
    keys = sorted({(r["model"], r["calibration_requested"]) for r in rows})

    table: list[dict[str, Any]] = []
    for model_name, calibration in keys:
        group = [
            r
            for r in rows
            if r["model"] == model_name and r["calibration_requested"] == calibration
        ]
        gains = [r["gain_over_word_length"][primary] for r in group]
        entry: dict[str, Any] = {
            "model": model_name,
            "arm": group[0]["arm"],
            "calibration_requested": calibration,
            "n_seeds": len(group),
            "output_is_probability": group[0]["output_is_probability"],
            "pr_auc": _agg([r["pr_auc"] for r in group]),
            "brier": _agg([r["brier"] for r in group]),
            "ece": _agg([r["ece"] for r in group]),
            f"recall_at_{primary}": _agg([r["budgets"][primary]["recall"] for r in group]),
            "recall_median_listener": _agg(
                [r["budgets"][primary]["recall_median_listener"] for r in group]
            ),
            "recall_worst_listener": _agg(
                [r["budgets"][primary]["recall_worst_listener"] for r in group]
            ),
            "gain_over_word_length": _agg(gains),
            # 핵심 판정. 모든 시드에서 이겨야 주장할 수 있습니다.
            "n_seeds_beating_word_length": int(sum(1 for g in gains if g > 0)),
            # 교정이 실제로 돌았는가.
            "n_folds_fell_back": sum(r["n_folds_fell_back"] for r in group),
            # 교정을 요청하지 않은 행에는 폴드별 기록이 없습니다. 빈칸으로 두면 "기록이
            # 없다" 와 "교정이 없었다" 가 구분되지 않으므로 요청값을 명시합니다.
            "effective_methods": sorted(
                {e["effective_method"] for r in group for e in r["calibration_per_fold"]}
            )
            or [calibration],
            "fallback_reasons": sorted(
                {
                    e["fallback_reason"]
                    for r in group
                    for e in r["calibration_per_fold"]
                    if e["fallback_reason"]
                }
            ),
        }
        for budget in cfg.budgets:
            b = f"{budget:g}"
            entry[f"recall_at_{b}"] = _agg([r["budgets"][b]["recall"] for r in group])
        table.append(entry)

    # 모든 시드에서 단어길이를 이긴 조합만 후보로 인정합니다.
    winners = [e for e in table if e["n_seeds_beating_word_length"] == e["n_seeds"]]
    best = max(winners, key=lambda e: e["gain_over_word_length"]["mean"], default=None)
    reference = next(
        (e for e in table if e["model"] == "logistic" and e["calibration_requested"] == "none"),
        None,
    )

    headline: dict[str, Any] = {
        "primary_budget": cfg.primary_budget,
        "n_candidates": len(table),
        "n_candidates_beating_word_length_on_every_seed": len(winners),
        "best_candidate": None
        if best is None
        else {
            "model": best["model"],
            "calibration_requested": best["calibration_requested"],
            "gain_over_word_length": best["gain_over_word_length"]["mean"],
            f"recall_at_{primary}": best[f"recall_at_{primary}"]["mean"],
        },
        "reference_logistic": None
        if reference is None
        else {
            "gain_over_word_length": reference["gain_over_word_length"]["mean"],
            f"recall_at_{primary}": reference[f"recall_at_{primary}"]["mean"],
            "n_seeds_beating_word_length": reference["n_seeds_beating_word_length"],
        },
        # 새 계열이 참조 기저선을 실제로 이겼는가. 이기지 못했다면 그대로 적습니다.
        "best_beats_reference_logistic": None
        if best is None or reference is None
        else bool(
            best["gain_over_word_length"]["mean"] > reference["gain_over_word_length"]["mean"]
        ),
        "caveat": (
            "합성 청취자 코호트에서 얻은 결과이며 임상적 근거가 아닙니다. 로지스틱 회귀는 "
            "ADR-0012 에 따라 참조 기저선으로 유지되며, 여기서의 우위만으로 교체되지 않습니다."
        ),
    }
    return {"seeds": cfg.simulation.seeds, "table": table, "headline": headline}
