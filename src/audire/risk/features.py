"""Feature extraction for word-level mishearing risk.

Feature *blocks* are the unit of ablation. RQ1 asks whether the individual confusion
profile adds information beyond the clinical measures, so the arms must differ **only**
in the listener representation:

======================  ==================================================
block                   contents
======================  ==================================================
``word``                length, phoneme composition, syllable structure
``context``             SNR, speaker
``pta``                 audiogram-derived listener features
``clinical``            ``pta`` plus SRT, WRS, PBmax, MCL
``confusion``           features derived from the individual ``C_u``
======================  ==================================================

Every arm in the ablation includes ``word`` and ``context``; the arms differ in which of
``pta`` / ``clinical`` / ``confusion`` they add. A ``word+context`` arm with no listener
information at all is included as the non-personalized floor.

Missing clinical values are emitted as ``NaN`` together with an explicit missing-indicator
column, and are imputed **inside** the model pipeline so that imputation statistics are
fitted on training folds only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from audire.confusion.grouping import neutralise_coda
from audire.confusion.profile import POSITIONS, ConfusionProfile
from audire.hangul.inventory import NO_CODA, Position, categories_for
from audire.hangul.syllable import Syllable, decompose_syllable, is_hangul_syllable
from audire.profile.schema import Ear, HearingProfile

FloatArray = npt.NDArray[np.float64]

FeatureBlock = Literal[
    "word",
    "context",
    "pta",
    "clinical",
    "confusion",
    "confusion_rich",
    # Phase D. 음소를 개별적으로 지목하는 블록. 부류 평균이 지우는 정보를 되살립니다.
    "exact_target",
    "exact_target_offdiag",
]

#: The preregistered ablation arms (docs/EXPERIMENT_PLAN.md E4). Each is a set of blocks.
ABLATION_ARMS: dict[str, tuple[FeatureBlock, ...]] = {
    "word_context_only": ("word", "context"),
    "pta_only": ("word", "context", "pta"),
    "clinical": ("word", "context", "clinical"),
    "confusion_only": ("word", "context", "confusion"),
    "clinical_plus_confusion": ("word", "context", "clinical", "confusion"),
    # Phase D17. Additive to the original block so the old arm stays comparable.
    "confusion_rich_only": ("word", "context", "confusion", "confusion_rich"),
    "clinical_plus_confusion_rich": (
        "word",
        "context",
        "clinical",
        "confusion",
        "confusion_rich",
    ),
    # Phase D. 기존 arm 을 지우지 않고 위에 쌓습니다. 절제가 가능해야 어느 블록이 무엇을
    # 하는지 말할 수 있습니다.
    "exact_target": (
        "word",
        "context",
        "clinical",
        "confusion",
        "confusion_rich",
        "exact_target",
    ),
    "exact_target_offdiag": (
        "word",
        "context",
        "clinical",
        "confusion",
        "confusion_rich",
        "exact_target",
        "exact_target_offdiag",
    ),
    # ------------------------------------------------------------------ 라이브 자막 arm
    #
    # 브라우저 DOM 자막 모드에는 음향 맥락이 없습니다. 그래서 이 arm 들은 "context" 블록을
    # **아예 포함하지 않습니다** — 전체 행렬을 만든 뒤 c_snr_db 를 지우는 방식이 아닙니다.
    # 그렇게 하면 전처리 통계와 열 순서가 지워진 열의 존재에 의존하게 되고, 학습과 추론의
    # 스키마가 조용히 어긋날 수 있습니다.
    #
    # 라이브 모델은 추론 시 갖게 될 것과 같은 정보 제약 아래에서 학습됩니다
    # (audire.live.contract, ADR-0021).
    "live_word_context": ("word",),
    "live_word_context_clinical": ("word", "clinical"),
    "live_word_context_clinical_confusion": ("word", "clinical", "confusion"),
}

#: Floor value for a diagonal probability when taking logs.
_EPS = 1e-6

#: Column-name prefixes per block. The residual architecture needs to split a matrix into
#: "general difficulty" and "this listener's particular weakness" columns, and doing that
#: by prefix keeps the split declarative rather than a hand-maintained list.
BLOCK_PREFIXES: dict[str, tuple[str, ...]] = {
    "word": ("w_",),
    "context": ("c_",),
    "pta": ("h_",),
    "clinical": ("h_",),
    "confusion": ("x_",),
    "confusion_rich": ("ix_", "w2_", "x2_"),
    "exact_target": ("et_",),
    "exact_target_offdiag": ("eo_",),
}


@dataclass(frozen=True, slots=True)
class WordContext:
    """The acoustic context a word was presented in."""

    snr_db: float = 20.0
    speaker: str = "unknown"


def word_syllables(word: str) -> list[Syllable]:
    """Decompose the Hangul syllables of ``word``, ignoring other characters."""
    return [decompose_syllable(ch) for ch in word if is_hangul_syllable(ch)]


# --------------------------------------------------------------------------- word block


_STRUCTURES = ("V", "VC", "CV", "CVC")


def word_features(word: str) -> dict[str, float]:
    """Composition features that do not depend on the listener."""
    syls = word_syllables(word)
    n = len(syls)
    if n == 0:
        return {
            "w_n_syllables": 0.0,
            "w_n_phonemes": 0.0,
            "w_n_codas": 0.0,
            "w_coda_ratio": 0.0,
            "w_n_distinct_onsets": 0.0,
            "w_has_cluster_coda": 0.0,
            **{f"w_struct_{s}": 0.0 for s in _STRUCTURES},
        }
    n_codas = sum(1 for s in syls if s.has_coda)
    structures = [s.structure for s in syls]
    return {
        "w_n_syllables": float(n),
        # Every syllable contributes three positions, including an explicit "no coda".
        "w_n_phonemes": float(3 * n),
        "w_n_codas": float(n_codas),
        "w_coda_ratio": n_codas / n,
        "w_n_distinct_onsets": float(len({s.onset for s in syls})),
        "w_has_cluster_coda": float(
            any(s.has_coda and neutralise_coda(s.coda) != s.coda for s in syls)
        ),
        **{f"w_struct_{s}": float(structures.count(s)) / n for s in _STRUCTURES},
    }


# --------------------------------------------------------------------------- context block


def context_features(context: WordContext, speakers: tuple[str, ...]) -> dict[str, float]:
    """Acoustic-context features. ``speakers`` fixes the one-hot column order."""
    out: dict[str, float] = {"c_snr_db": float(context.snr_db)}
    for s in speakers:
        out[f"c_speaker_{s}"] = float(context.speaker == s)
    return out


# --------------------------------------------------------------------------- listener blocks


def pta_features(profile: HearingProfile) -> dict[str, float]:
    """Audiogram-derived features. Missing values are ``NaN`` plus an indicator."""
    better = profile.better_ear_pta()
    worse = profile.worse_ear_pta()
    slopes = [s for s in (e.audiogram.slope_db_per_octave() for e in profile.ears) if s is not None]
    slope = float(np.mean(slopes)) if slopes else None
    high = [
        p.db_hl
        for e in profile.ears
        for f, p in e.audiogram.thresholds.items()
        if f in (4000, 6000, 8000) and p.is_measured and p.db_hl is not None
    ]
    return _with_indicators(
        {
            "h_pta_better": better,
            "h_pta_worse": worse,
            "h_pta_asymmetry": (None if better is None or worse is None else worse - better),
            "h_audiogram_slope": slope,
            "h_high_freq_mean": float(np.mean(high)) if high else None,
        }
    )


def clinical_features(profile: HearingProfile) -> dict[str, float]:
    """``pta`` block plus the speech-audiometry measures."""
    out = dict(pta_features(profile))
    srt = profile.best_srt()
    better = profile.better_ear_pta()
    out.update(
        _with_indicators(
            {
                "h_srt": srt,
                # SRT-PTA agreement is clinically meaningful in its own right.
                "h_srt_minus_pta": (None if srt is None or better is None else srt - better),
                "h_wrs": profile.best_wrs(),
                "h_pbmax": profile.pbmax(),
                "h_mcl": profile.mcl(),
            }
        )
    )
    return out


def _with_indicators(values: dict[str, float | None]) -> dict[str, float]:
    """Emit ``NaN`` for missing values plus an explicit ``*_missing`` indicator column."""
    out: dict[str, float] = {}
    for key, value in values.items():
        out[key] = float("nan") if value is None else float(value)
        out[f"{key}_missing"] = float(value is None)
    return out


# --------------------------------------------------------------------------- confusion block


@dataclass(frozen=True, slots=True)
class PhonemeRisk:
    """The per-phoneme evidence behind a word's risk, retained for explanation."""

    position: Position
    target: str
    #: Smoothed probability that the phoneme is perceived correctly.
    p_correct: float
    #: Raw number of calibration trials supporting the estimate. Never discarded.
    n_observations: int
    #: Whether a back-off was used because the exact category had no evidence.
    backed_off: bool
    #: The most likely incorrect responses, for the explanation text.
    top_confusions: tuple[tuple[str, float, int], ...] = ()


def _lookup_p_correct(
    profile: ConfusionProfile, position: Position, target: str
) -> tuple[float, int, bool]:
    """Return ``(p_correct, n_observations, backed_off)`` for one target category.

    Back-off rule: an orthographic coda cluster with no evidence of its own falls back to
    its neutralised surface coda, because Korean coda neutralisation makes them the same
    perceptual event. No other back-off is applied — an unobserved onset stays unobserved
    and its probability is the prior, which ``n_observations == 0`` makes visible.
    """
    n = profile.evidence(position, target)
    if n > 0 or position is not Position.CODA or target == NO_CODA:
        return profile.p_correct(position, target), n, False

    surface = neutralise_coda(target)
    if surface != target and profile.evidence(position, surface) > 0:
        return (
            profile.p_correct(position, surface),
            profile.evidence(position, surface),
            True,
        )
    return profile.p_correct(position, target), n, False


def phoneme_risks(word: str, profile: ConfusionProfile, *, top_k: int = 2) -> list[PhonemeRisk]:
    """Per-phoneme correct-recognition evidence for every position in ``word``."""
    out: list[PhonemeRisk] = []
    for syl in word_syllables(word):
        for position in POSITIONS:
            target = syl.get(position)
            p, n, backed = _lookup_p_correct(profile, position, target)
            out.append(
                PhonemeRisk(
                    position=position,
                    target=target,
                    p_correct=p,
                    n_observations=n,
                    backed_off=backed,
                    top_confusions=tuple(profile.matrix(position).top_confusions(target, top_k)),
                )
            )
    return out


def phoneme_independence_risk(word: str, profile: ConfusionProfile) -> float:
    """``R_phon(w, u) = 1 - prod_k C_u(phi_k, phi_k)`` — the deterministic baseline.

    Interpretable and deliberately simple. Phoneme independence is **not** assumed to be
    scientifically correct; it is a comparator (docs/RESEARCH_PLAN.md).
    """
    risks = phoneme_risks(word, profile)
    if not risks:
        return 0.0
    log_correct = sum(float(np.log(max(r.p_correct, _EPS))) for r in risks)
    return float(1.0 - np.exp(log_correct))


def confusion_features(word: str, profile: ConfusionProfile) -> dict[str, float]:
    """Features derived from the listener's individual confusion profile."""
    risks = phoneme_risks(word, profile, top_k=0)
    if not risks:
        return {
            "x_r_phon": 0.0,
            "x_neg_log_correct": 0.0,
            "x_min_p_correct": 1.0,
            "x_mean_p_correct": 1.0,
            "x_max_error": 0.0,
            **{f"x_mean_p_{p.value}": 1.0 for p in POSITIONS},
            **{f"x_min_p_{p.value}": 1.0 for p in POSITIONS},
            "x_mean_row_entropy": 0.0,
            "x_min_evidence": 0.0,
            "x_mean_evidence": 0.0,
            "x_frac_unobserved": 1.0,
            "x_frac_backed_off": 0.0,
        }

    p = np.array([r.p_correct for r in risks], dtype=np.float64)
    neg_log = float(-np.sum(np.log(np.maximum(p, _EPS))))
    evidence = np.array([r.n_observations for r in risks], dtype=np.float64)

    per_position: dict[str, float] = {}
    for position in POSITIONS:
        vals = [r.p_correct for r in risks if r.position is position]
        per_position[f"x_mean_p_{position.value}"] = float(np.mean(vals)) if vals else 1.0
        per_position[f"x_min_p_{position.value}"] = float(np.min(vals)) if vals else 1.0

    entropies: list[float] = []
    for position in POSITIONS:
        labels = categories_for(position, axis="target")
        rows = profile.matrix(position).row_entropy()
        for r in risks:
            if r.position is position:
                entropies.append(float(rows[labels.index(r.target)]))

    return {
        # The independence baseline as a single feature, so a learned model can improve
        # on it rather than merely rediscover it.
        "x_r_phon": float(1.0 - np.exp(-neg_log)),
        "x_neg_log_correct": neg_log,
        "x_min_p_correct": float(p.min()),
        "x_mean_p_correct": float(p.mean()),
        "x_max_error": float(1.0 - p.min()),
        **per_position,
        "x_mean_row_entropy": float(np.mean(entropies)) if entropies else 0.0,
        # Evidence features let a model discount a confident-looking estimate that rests
        # on almost no calibration data.
        "x_min_evidence": float(evidence.min()),
        "x_mean_evidence": float(evidence.mean()),
        "x_frac_unobserved": float(np.mean(evidence == 0)),
        "x_frac_backed_off": float(np.mean([r.backed_off for r in risks])),
    }


# --------------------------------------------------------------------------- assembly


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A named set of feature blocks: one ablation arm."""

    name: str
    blocks: tuple[FeatureBlock, ...]
    speakers: tuple[str, ...] = ("male", "female", "unknown")

    @classmethod
    def arm(
        cls, name: str, speakers: tuple[str, ...] = ("male", "female", "unknown")
    ) -> FeatureSpec:
        try:
            blocks = ABLATION_ARMS[name]
        except KeyError as exc:
            raise KeyError(f"unknown arm {name!r}; known: {sorted(ABLATION_ARMS)}") from exc
        return cls(name=name, blocks=blocks, speakers=speakers)

    @property
    def uses_confusion(self) -> bool:
        return "confusion" in self.blocks

    def row(
        self,
        word: str,
        context: WordContext,
        hearing: HearingProfile | None,
        confusion: ConfusionProfile | None,
    ) -> dict[str, float]:
        """Build one feature row.

        Raises
        ------
        ValueError
            If a block is requested but its input is absent. Silently emitting zeros
            would make an ablation arm quietly degenerate into a different arm.
        """
        out: dict[str, float] = {}
        if "word" in self.blocks:
            out.update(word_features(word))
        if "context" in self.blocks:
            out.update(context_features(context, self.speakers))
        if "pta" in self.blocks or "clinical" in self.blocks:
            if hearing is None:
                raise ValueError(f"arm {self.name!r} needs a HearingProfile but none was given")
            out.update(
                clinical_features(hearing) if "clinical" in self.blocks else pta_features(hearing)
            )
        if "confusion" in self.blocks:
            if confusion is None:
                raise ValueError(f"arm {self.name!r} needs a ConfusionProfile but none was given")
            out.update(confusion_features(word, confusion))
        if "confusion_rich" in self.blocks:
            if confusion is None:
                raise ValueError(f"arm {self.name!r} needs a ConfusionProfile but none was given")
            from audire.risk.confusion_features import confusion_rich_features

            out.update(confusion_rich_features(word, confusion))
        if "exact_target" in self.blocks:
            if confusion is None:
                raise ValueError(f"arm {self.name!r} needs a ConfusionProfile but none was given")
            from audire.risk.exact_target import exact_target_features

            out.update(exact_target_features(word, confusion))
        if "exact_target_offdiag" in self.blocks:
            if confusion is None:
                raise ValueError(f"arm {self.name!r} needs a ConfusionProfile but none was given")
            from audire.risk.exact_target import exact_target_offdiag_features

            out.update(exact_target_offdiag_features(word, confusion))
        return out


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """A design matrix with its column names, listener groups and provenance."""

    X: FloatArray
    feature_names: tuple[str, ...]
    groups: npt.NDArray[np.str_]
    y: npt.NDArray[np.int64] | None = None
    meta: dict[str, Any] | None = None

    def __len__(self) -> int:
        return int(self.X.shape[0])


def build_matrix(
    spec: FeatureSpec,
    rows: Sequence[tuple[str, str, WordContext, HearingProfile | None, ConfusionProfile | None]],
    labels: Sequence[int] | None = None,
) -> FeatureMatrix:
    """Assemble a design matrix.

    ``rows`` items are ``(listener_id, word, context, hearing_profile, confusion_profile)``.
    Column order is taken from the first row and enforced for the rest, so a missing
    feature is an error rather than a silently shifted column.
    """
    if not rows:
        raise ValueError("cannot build a feature matrix from zero rows")
    if labels is not None and len(labels) != len(rows):
        raise ValueError(f"got {len(rows)} rows but {len(labels)} labels")

    first = spec.row(rows[0][1], rows[0][2], rows[0][3], rows[0][4])
    names = tuple(first)
    data = np.empty((len(rows), len(names)), dtype=np.float64)
    data[0] = [first[n] for n in names]

    for i, (_, word, ctx, hearing, confusion) in enumerate(rows[1:], start=1):
        r = spec.row(word, ctx, hearing, confusion)
        if tuple(r) != names:
            missing = set(names) - set(r)
            extra = set(r) - set(names)
            raise ValueError(
                f"inconsistent feature columns at row {i}: missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        data[i] = [r[n] for n in names]

    return FeatureMatrix(
        X=data,
        feature_names=names,
        groups=np.array([r[0] for r in rows], dtype=np.str_),
        y=None if labels is None else np.asarray(labels, dtype=np.int64),
        meta={"arm": spec.name, "blocks": list(spec.blocks), "n_features": len(names)},
    )


def ear_summary(profile: HearingProfile) -> dict[str, float | None]:
    """Compact per-ear PTA summary used by the UI explanation panel."""
    return {
        "pta_left": profile.pta(Ear.LEFT),
        "pta_right": profile.pta(Ear.RIGHT),
        "better_ear_pta": profile.better_ear_pta(),
    }
