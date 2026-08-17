"""E30 — 라이브 자막 입력 계약 절제.

묻는 것
-------
    음향 맥락이 없을 때, 청취자별 임상·혼동 정보가 단어 정보만으로 얻는 것 위에
    홀드아웃 가치를 더하는가?

즉 ``LIVE-2 − LIVE-0`` 을 **동일 자막률**과 확률 품질 양쪽에서 봅니다.

왜 임계값 정책인가
------------------
연구용 정확 예산 정책은 후보 단어 집합을 미리 알아야 합니다. 라이브 자막은 큐 단위로
도착하므로 그 전제가 성립하지 않고, 문장마다 상위 20% 를 강제하면 위험이 낮은 다섯 단어
문장에서도 반드시 한 단어를 보여주게 됩니다. 따라서 라이브 기본값은 임계값입니다.

동일 자막률 비교가 왜 필요한가
------------------------------
자막을 더 많이 보여주면 재현율은 당연히 올라갑니다. 그래서 arm 을 비교할 때는 **훈련
청취자만으로** 임계값을 골라 달성 자막률을 맞춘 뒤 홀드아웃에서 재현율을 비교합니다.
바깥 홀드아웃 청취자로 임계값을 고르면 그 순간 비교가 무의미해집니다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from audire.config.logging import get_logger
from audire.eval.ablation import cohort_matrix
from audire.eval.caption import recall_by_listener, select_budget
from audire.eval.metrics import expected_calibration_error
from audire.eval.splits import assert_no_listener_leakage, listener_folds
from audire.experiments.registry import RunRecord, finish_run, save_artifact, tracked_run
from audire.live.contract import LIVE_CAPTION_V1, ContractViolation
from audire.risk.features import ABLATION_ARMS, FeatureMatrix, FeatureSpec
from audire.risk.models import make_model
from audire.sim.cohort import Cohort, build_cohort
from audire.sim.config import SimulationConfig

log = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: 사전등록 임계값 격자. 라이브는 큐 단위로 도착하므로 예산이 아니라 임계값으로 고릅니다.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70)
#: 역사적 비교를 위해 유지하는 예산 지점.
DEFAULT_BUDGETS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)

#: 세 라이브 arm. 각 arm 이 앞의 것을 포함하므로 차이가 곧 그 블록의 몫입니다.
LIVE_ARMS: tuple[str, ...] = (
    "live_word_context",
    "live_word_context_clinical",
    "live_word_context_clinical_confusion",
)
#: 진단 전용 대조. 기존 배포 arm 을 음향 맥락이 결측인 채로 돌립니다.
#: **제품 모델이 아닙니다** — 계약 불일치·분포 밖 사용임을 기록에 남깁니다.
DIAGNOSTIC_ARM = "clinical_plus_confusion"


class LiveAblationConfig(BaseModel):
    """E30 의 사전등록 명세."""

    model_config = ConfigDict(extra="forbid")

    name: str = "e30_live_contract"
    description: str = ""
    simulation: SimulationConfig
    live_arms: list[str] = Field(default_factory=lambda: list(LIVE_ARMS), min_length=1)
    #: 진단 대조를 포함할 것인가. 포함해도 제품 후보로 취급하지 않습니다.
    include_legacy_diagnostic: bool = True
    model: str = "logistic"
    n_splits: int = Field(default=5, ge=2)
    ece_bins: int = Field(default=10, ge=2)
    thresholds: list[float] = Field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    budgets: list[float] = Field(default_factory=lambda: list(DEFAULT_BUDGETS))
    #: 동일 자막률 비교의 목표. 훈련 청취자에서만 임계값을 찾습니다.
    matched_caption_rate: float = Field(default=0.20, gt=0.0, lt=1.0)
    #: 목표 자막률 허용 오차. 이 안에 들어오지 않으면 비교가 무효입니다.
    caption_rate_tolerance: float = Field(default=0.02, gt=0.0)

    def model_post_init(self, _: Any) -> None:
        for arm in self.live_arms:
            if arm not in ABLATION_ARMS:
                raise ValueError(f"알 수 없는 arm: {arm!r}")
            # 계약 위반 arm 이 라이브 실험에 들어오는 것을 여기서 막습니다.
            LIVE_CAPTION_V1.validate_blocks(ABLATION_ARMS[arm])

    @classmethod
    def load(cls, path: Path) -> LiveAblationConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _subset(matrix: FeatureMatrix, idx: IntArray) -> FeatureMatrix:
    return FeatureMatrix(
        X=matrix.X[idx],
        feature_names=matrix.feature_names,
        groups=matrix.groups[idx],
        y=None if matrix.y is None else matrix.y[idx],
        meta=matrix.meta,
    )


@dataclass(frozen=True, slots=True)
class LiveThreshold:
    """임계값과, 그 임계값에 정확히 걸린 동점 중 얼마를 통과시킬지.

    왜 동점 비율이 필요한가
    -----------------------
    단어 특징만 쓰는 모델은 같은 단어에 같은 점수를 줍니다. 실측하면 16,000행에서 고유
    점수가 33개뿐이고 훈련 행의 17.5% 가 임계값과 정확히 같습니다. 순수한 ``>= tau`` 는
    그 덩어리를 통째로 넣거나 빼므로 어떤 임계값으로도 목표 자막률을 낼 수 없습니다.

    이것은 비교만의 문제가 아닙니다. 사용자가 임계값 슬라이더를 움직여도 자막량이 계단식
    으로만 변한다는 뜻이고, 라이브 제품에서 그대로 드러납니다.

    큐 단위로 구현 가능한가
    -----------------------
    가능합니다. 경계에 걸린 항목은 ``(청취자, 단어)`` 의 결정적 해시로 통과 여부를 정하므로,
    전체 후보 집합을 몰라도 한 큐 안에서 판정할 수 있습니다. 같은 청취자·같은 단어는 항상
    같은 판정을 받아 화면이 깜빡이지 않습니다.
    """

    tau: float
    #: 경계 동점 중 통과시킬 비율. 동점이 없으면 의미가 없습니다.
    tie_pass_fraction: float


def _tie_key(listener: str, index: int) -> float:
    """경계 동점을 가르는 결정적 값 ``[0, 1)``.

    ``hash()`` 는 프로세스마다 salt 가 달라 재현성을 깨므로 쓰지 않습니다.
    """
    digest = hashlib.blake2b(f"{listener}\x00{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def select_with_threshold(
    scores: FloatArray, groups: npt.NDArray[np.str_], threshold: LiveThreshold
) -> npt.NDArray[np.bool_]:
    """임계값 정책. 경계 동점은 결정적으로 분할합니다."""
    chosen = scores > threshold.tau
    boundary = np.isclose(scores, threshold.tau, rtol=0.0, atol=1e-12)
    if boundary.any() and threshold.tie_pass_fraction > 0.0:
        keys = np.array(
            [_tie_key(str(groups[i]), int(i)) for i in np.flatnonzero(boundary)],
            dtype=np.float64,
        )
        chosen[np.flatnonzero(boundary)] = keys < threshold.tie_pass_fraction
    return chosen


def threshold_for_caption_rate(scores: FloatArray, target_rate: float) -> LiveThreshold:
    """목표 자막률을 내는 임계값과 경계 동점 통과 비율.

    **훈련 청취자의 점수만** 넘겨야 합니다. 홀드아웃 점수로 임계값을 고르면 그 순간
    비교가 무의미해집니다 — 호출자가 그 책임을 집니다.
    """
    if scores.size == 0:
        raise ValueError("빈 점수에서 임계값을 정할 수 없습니다")
    tau = float(np.quantile(scores, 1.0 - target_rate))
    above = float(np.mean(scores > tau))
    boundary = float(np.mean(np.isclose(scores, tau, rtol=0.0, atol=1e-12)))
    if boundary <= 0.0:
        return LiveThreshold(tau=tau, tie_pass_fraction=0.0)
    # 엄격히 위쪽만으로는 부족한 만큼을 경계 덩어리에서 채웁니다.
    needed = max(0.0, target_rate - above)
    return LiveThreshold(tau=tau, tie_pass_fraction=min(1.0, needed / boundary))


def threshold_metrics(
    y: IntArray, scores: FloatArray, groups: npt.NDArray[np.str_], threshold: float
) -> dict[str, float]:
    """한 임계값에서의 자막률·재현율·정밀도와 청취자 간 분포.

    사전등록 격자용이므로 동점 분할 없이 순수한 ``>= tau`` 를 씁니다. 동일 자막률 비교는
    :func:`threshold_for_caption_rate` 가 만든 :class:`LiveThreshold` 를 씁니다.
    """
    selected = scores >= threshold
    n_positive = int(y.sum())
    hits = int((selected & (y == 1)).sum())
    n_shown = int(selected.sum())

    per = np.array(
        [v for v in recall_by_listener(y, groups, selected).values() if np.isfinite(v)],
        dtype=np.float64,
    )
    rates = np.array(
        [float(selected[groups == g].mean()) for g in np.unique(groups)], dtype=np.float64
    )
    precision = hits / n_shown if n_shown else float("nan")
    recall = hits / n_positive if n_positive else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    return {
        "threshold": threshold,
        "caption_rate": float(selected.mean()) if y.size else float("nan"),
        "misheard_recall": recall,
        "precision": precision,
        "f1": f1,
        "recall_mean_listener": float(per.mean()) if per.size else float("nan"),
        "recall_median_listener": float(np.median(per)) if per.size else float("nan"),
        "recall_q25_listener": float(np.quantile(per, 0.25)) if per.size else float("nan"),
        "recall_worst_listener": float(per.min()) if per.size else float("nan"),
        "caption_rate_mean_listener": float(rates.mean()) if rates.size else float("nan"),
        "caption_rate_median_listener": float(np.median(rates)) if rates.size else float("nan"),
        "caption_rate_sd_listener": float(rates.std(ddof=1)) if rates.size > 1 else 0.0,
        # 접근성 지표: 사실상 자막을 못 받는 청취자 비율.
        "frac_listeners_near_zero_captions": float(np.mean(rates < 0.02))
        if rates.size
        else float("nan"),
    }


def _probability_metrics(y: IntArray, p: FloatArray, ece_bins: int) -> dict[str, float]:
    two_class = np.unique(y).size >= 2
    ece, mce = expected_calibration_error(y, p, n_bins=ece_bins)
    return {
        "pr_auc": float(average_precision_score(y, p)) if two_class else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if two_class else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
        if two_class
        else float("nan"),
        "ece": float(ece),
        "mce": float(mce),
        "prevalence": float(y.mean()),
    }


def _budget_metrics(
    y: IntArray, p: FloatArray, groups: npt.NDArray[np.str_], budgets: Sequence[float], seed: int
) -> dict[str, float]:
    """역사적 비교를 위해 유지하는 예산 지표."""
    out: dict[str, float] = {}
    for budget in budgets:
        chosen = select_budget(p, groups, budget, per_listener=True, tie_seed=seed)
        n_positive = int(y.sum())
        out[f"recall@{budget:g}"] = (
            int((chosen & (y == 1)).sum()) / n_positive if n_positive else float("nan")
        )
    return out


def evaluate_live_arm(
    cohort: Cohort,
    arm: str,
    cfg: LiveAblationConfig,
    seed: int,
) -> dict[str, Any]:
    """한 arm 을 청취자 수준 교차검증으로 평가합니다.

    임계값은 **폴드마다 훈련 청취자의 점수에서만** 고릅니다. 그래야 동일 자막률 비교가
    홀드아웃 정보를 쓰지 않습니다.
    """
    spec = FeatureSpec.arm(arm, speakers=tuple(sorted({*cohort.config.speakers, "unknown"})))
    matrix = cohort_matrix(cohort, spec)
    assert matrix.y is not None

    is_live = arm in cfg.live_arms
    if is_live:
        # 라이브 arm 은 학습·추론 양쪽에서 음향 열이 없어야 합니다.
        LIVE_CAPTION_V1.validate_columns(matrix.feature_names)

    folds = listener_folds(matrix.groups, matrix.y, n_splits=cfg.n_splits, stratify=True, seed=seed)
    oof = np.full(matrix.y.shape, np.nan, dtype=np.float64)
    matched_threshold_by_fold: list[dict[str, float]] = []
    matched_selected = np.zeros(matrix.y.shape, dtype=bool)

    for fold in folds:
        assert_no_listener_leakage(matrix.groups, fold.train_idx, fold.test_idx)
        model = make_model(cfg.model)
        train = _subset(matrix, fold.train_idx)
        model.fit(train)
        oof[fold.test_idx] = model.predict_proba(_subset(matrix, fold.test_idx))

        # 동일 자막률용 임계값: 훈련 청취자에 대한 **훈련된 모델의 예측**에서 찾습니다.
        train_scores = model.predict_proba(train)
        live_tau = threshold_for_caption_rate(train_scores, cfg.matched_caption_rate)
        matched_threshold_by_fold.append(
            {"tau": live_tau.tau, "tie_pass_fraction": live_tau.tie_pass_fraction}
        )
        matched_selected[fold.test_idx] = select_with_threshold(
            oof[fold.test_idx], matrix.groups[fold.test_idx], live_tau
        )

    if np.any(~np.isfinite(oof)):
        raise RuntimeError("일부 행이 폴드 밖 예측을 받지 못했습니다")

    y, groups = matrix.y, matrix.groups
    row: dict[str, Any] = {
        "seed": seed,
        "arm": arm,
        "model": cfg.model,
        "role": "live" if is_live else "diagnostic",
        "input_contract": LIVE_CAPTION_V1.version if is_live else "media-pipeline-v1",
        "n_features": len(matrix.feature_names),
        "n_listeners": int(np.unique(groups).size),
        "n_rows": int(y.size),
        **_probability_metrics(y, oof, cfg.ece_bins),
        **_budget_metrics(y, oof, groups, cfg.budgets, seed),
        "thresholds": [threshold_metrics(y, oof, groups, t) for t in cfg.thresholds],
    }

    # 동일 자막률 비교. 훈련에서 고른 임계값을 홀드아웃에 적용한 결과입니다.
    n_positive = int(y.sum())
    hits = int((matched_selected & (y == 1)).sum())
    per = np.array(
        [v for v in recall_by_listener(y, groups, matched_selected).values() if np.isfinite(v)],
        dtype=np.float64,
    )
    row["matched"] = {
        "target_caption_rate": cfg.matched_caption_rate,
        "achieved_caption_rate": float(matched_selected.mean()),
        "thresholds_by_fold": matched_threshold_by_fold,
        "misheard_recall": hits / n_positive if n_positive else float("nan"),
        "recall_median_listener": float(np.median(per)) if per.size else float("nan"),
        "recall_worst_listener": float(per.min()) if per.size else float("nan"),
    }
    return row


def run_live_ablation(
    cfg: LiveAblationConfig, *, record: RunRecord | None = None
) -> dict[str, Any]:
    if record is not None:
        return _execute(cfg, record)
    with tracked_run(
        cfg.name, cfg.model_dump(mode="json"), cfg.simulation.seeds, notes=cfg.description
    ) as rec:
        return _execute(cfg, rec)


def _execute(cfg: LiveAblationConfig, rec: RunRecord) -> dict[str, Any]:
    log.info("e30.start", name=cfg.name, seeds=cfg.simulation.seeds, run_id=rec.run_id)
    arms = list(cfg.live_arms)
    if cfg.include_legacy_diagnostic:
        arms.append(DIAGNOSTIC_ARM)

    rows: list[dict[str, Any]] = []
    for seed in cfg.simulation.seeds:
        cohort = build_cohort(cfg.simulation, seed)
        for arm in arms:
            rows.append(evaluate_live_arm(cohort, arm, cfg, seed))
        log.info("e30.seed_done", seed=seed, n_rows=len(rows))

    declared = {(arm, seed) for arm in arms for seed in cfg.simulation.seeds}
    produced = {(r["arm"], r["seed"]) for r in rows}
    if declared != produced:
        raise RuntimeError(f"선언된 칸이 빠졌습니다: {sorted(declared - produced)}")

    summary = _summarise(cfg, rows, arms)
    save_artifact(rec, "live_ablation_rows.json", rows)
    save_artifact(rec, "summary.json", summary)
    finish_run(rec, summary["headline"])
    log.info("e30.done", run_id=rec.run_id, n_rows=len(rows))
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


def _paired(rows: list[dict[str, Any]], a: str, b: str, key: str) -> dict[str, Any]:
    """``a − b`` 의 시드별 짝지은 차이. 같은 시드는 같은 코호트를 뜻합니다."""

    def pick(arm: str) -> dict[int, float]:
        return {r["seed"]: float(r["matched"][key]) for r in rows if r["arm"] == arm}

    x, z = pick(a), pick(b)
    seeds = sorted(set(x) & set(z))
    d = np.asarray([x[s] - z[s] for s in seeds], dtype=np.float64)
    return {
        "contrast": f"{a} - {b}",
        "metric": key,
        "n_seeds": len(seeds),
        "mean": float(d.mean()) if d.size else float("nan"),
        "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "n_seeds_positive": int((d > 0).sum()),
        "per_seed": {str(s): float(x[s] - z[s]) for s in seeds},
    }


def _summarise(
    cfg: LiveAblationConfig, rows: list[dict[str, Any]], arms: Sequence[str]
) -> dict[str, Any]:
    table = []
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        entry: dict[str, Any] = {
            "arm": arm,
            "role": group[0]["role"],
            "input_contract": group[0]["input_contract"],
            "n_features": group[0]["n_features"],
            "n_seeds": len(group),
        }
        for key in ("pr_auc", "brier", "ece", "mce", "log_loss"):
            entry[key] = _agg([r[key] for r in group])
        for budget in cfg.budgets:
            entry[f"recall@{budget:g}"] = _agg([r[f"recall@{budget:g}"] for r in group])
        entry["matched_recall"] = _agg([r["matched"]["misheard_recall"] for r in group])
        entry["matched_caption_rate"] = _agg([r["matched"]["achieved_caption_rate"] for r in group])
        entry["matched_recall_worst"] = _agg([r["matched"]["recall_worst_listener"] for r in group])
        entry["matched_recall_median"] = _agg(
            [r["matched"]["recall_median_listener"] for r in group]
        )
        table.append(entry)

    live = list(cfg.live_arms)
    contrasts = []
    if len(live) >= 3:
        contrasts = [
            _paired(rows, live[2], live[0], "misheard_recall"),
            _paired(rows, live[2], live[1], "misheard_recall"),
            _paired(rows, live[1], live[0], "misheard_recall"),
        ]

    # 동일 자막률이 실제로 맞았는지 확인합니다. 맞지 않으면 비교 자체가 무효입니다.
    rate_ok = all(
        abs(e["matched_caption_rate"]["mean"] - cfg.matched_caption_rate)
        <= cfg.caption_rate_tolerance
        for e in table
        if e["role"] == "live"
    )

    headline = {
        "matched_caption_rate_target": cfg.matched_caption_rate,
        "caption_rate_matched_within_tolerance": rate_ok,
        "caption_rate_tolerance": cfg.caption_rate_tolerance,
        "contrasts": contrasts,
        "caveat": (
            "합성 청취자 코호트에서 얻은 공학적 검증이며 사람 청취 이득의 근거가 아닙니다. "
            "진단 arm 은 계약 불일치 상태의 분포 밖 사용이므로 제품 후보가 아닙니다."
        ),
    }
    return {"seeds": cfg.simulation.seeds, "table": table, "headline": headline}


__all__ = [
    "DEFAULT_BUDGETS",
    "DEFAULT_THRESHOLDS",
    "DIAGNOSTIC_ARM",
    "LIVE_ARMS",
    "ContractViolation",
    "LiveAblationConfig",
    "LiveThreshold",
    "evaluate_live_arm",
    "run_live_ablation",
    "select_with_threshold",
    "threshold_for_caption_rate",
    "threshold_metrics",
]
