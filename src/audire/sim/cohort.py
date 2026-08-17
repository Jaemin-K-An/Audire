"""Assembly of a complete synthetic cohort: listeners, calibration, and word trials.

A :class:`Cohort` is the unit that risk-model experiments consume. It bundles, for one
``(config, seed)`` pair:

* the synthetic listeners (with their true confusion structure, for recovery analysis);
* the calibration trials actually "administered" to each listener;
* the :class:`~audire.confusion.profile.ConfusionProfile` *estimated* from those trials;
* the word-level outcomes to be predicted.

The estimated profile — not the true structure — is what any model may use. That is
enforced by :meth:`Cohort.model_inputs`, which never exposes ``true_confusion``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from audire.confusion.matrix import SmoothingSpec
from audire.confusion.profile import CalibrationTrial, ConfusionProfile
from audire.data.stimuli import StimulusCatalog, build_balanced_catalog
from audire.profile.schema import HearingProfile
from audire.sim.config import SimulationConfig
from audire.sim.lexicon import build_lexicon
from audire.sim.listener import SyntheticListener, generate_cohort
from audire.sim.trials import (
    Vocabulary,
    WordTrial,
    build_vocabulary,
    simulate_calibration,
    simulate_word_trials,
)


@dataclass(frozen=True, slots=True)
class ListenerRecord:
    """Everything generated for one synthetic listener."""

    listener: SyntheticListener
    calibration: list[CalibrationTrial]
    estimated_confusion: ConfusionProfile
    word_trials: list[WordTrial]

    @property
    def listener_id(self) -> str:
        return self.listener.listener_id

    @property
    def hearing(self) -> HearingProfile:
        return self.listener.hearing


@dataclass(frozen=True, slots=True)
class Cohort:
    """A reproducible synthetic cohort."""

    config: SimulationConfig
    seed: int
    records: tuple[ListenerRecord, ...]
    vocabulary: Vocabulary
    calibration_catalog: StimulusCatalog
    is_synthetic: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def listener_ids(self) -> list[str]:
        return [r.listener_id for r in self.records]

    def record(self, listener_id: str) -> ListenerRecord:
        for r in self.records:
            if r.listener_id == listener_id:
                return r
        raise KeyError(f"no listener {listener_id!r} in this cohort")

    def model_inputs(self) -> list[tuple[HearingProfile, ConfusionProfile, list[WordTrial]]]:
        """What a model is allowed to see.

        Deliberately returns the *estimated* confusion profile only. The true generative
        structure is never included, so a model cannot accidentally consume it.
        """
        return [(r.hearing, r.estimated_confusion, r.word_trials) for r in self.records]

    def word_trial_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in self.records:
            for t in r.word_trials:
                rows.append(t.as_row())
        return rows

    def mishear_rate(self) -> float:
        rows = self.word_trial_rows()
        return sum(1 for r in rows if r["misheard"]) / len(rows) if rows else 0.0

    def stratum_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.listener.stratum] = out.get(r.listener.stratum, 0) + 1
        return out

    def mishear_rate_by_stratum(self) -> dict[str, float]:
        """Word-level mishearing base rate per severity stratum.

        Reported in every cohort summary because it is a *configured* property of the
        simulation (see ``TrialModel``), not an observed fact, and because PR-AUC is
        base-rate dependent.
        """
        agg: dict[str, list[int]] = {}
        for r in self.records:
            slot = agg.setdefault(r.listener.stratum, [0, 0])
            for t in r.word_trials:
                slot[0] += int(t.misheard)
                slot[1] += 1
        return {k: (v[0] / v[1] if v[1] else 0.0) for k, v in sorted(agg.items())}

    def syllable_accuracy_by_stratum(self) -> dict[str, float]:
        """Mean *true* whole-monosyllable accuracy per stratum.

        Used to check that the generator reproduces the literature-anchored accuracies it
        was configured with (E3).
        """
        agg: dict[str, list[float]] = {}
        for r in self.records:
            agg.setdefault(r.listener.stratum, []).append(r.listener.true_accuracy)
        return {k: float(sum(v) / len(v)) for k, v in sorted(agg.items())}

    def lexical_trap_rate(self) -> float:
        """오류가 난 시행 중 지각형이 어휘에 실재한 비율.

        V2 의 어휘 함정 항이 실제로 발동하는지 보여줍니다. 이 어휘에서는 0 에 가까우며,
        그 사실을 산출물에 남겨 두어야 "있으나 마나 한 항" 을 효과가 있는 것처럼 읽지
        않습니다.
        """
        from audire.sim.lexicon import build_lexicon

        lexicon = build_lexicon(self.vocabulary.words)
        errored = [t for r in self.records for t in r.word_trials if t.n_segment_errors > 0]
        if not errored:
            return float("nan")
        trapped = sum(1 for t in errored if t.perceived_word and lexicon.is_word(t.perceived_word))
        return trapped / len(errored)

    def summary(self) -> dict[str, Any]:
        return {
            "config_name": self.config.name,
            "seed": self.seed,
            "is_synthetic": True,
            # 어느 생성 과정이 이 결과를 만들었는가. V1 과 V2 는 결과 구조가 다르므로
            # 버전 없이 기록된 수치는 해석할 수 없습니다.
            "simulator_version": self.config.simulator_version,
            "lexical_trap_rate": self.lexical_trap_rate(),
            "n_listeners": len(self.records),
            "n_calibration_trials_per_listener": self.config.n_calibration_trials,
            "n_word_trials_total": sum(len(r.word_trials) for r in self.records),
            "mishear_rate": self.mishear_rate(),
            "mishear_rate_by_stratum": self.mishear_rate_by_stratum(),
            "true_syllable_accuracy_by_stratum": self.syllable_accuracy_by_stratum(),
            "stratum_counts": self.stratum_counts(),
            "vocabulary": self.vocabulary.provenance,
            "calibration_catalog": self.calibration_catalog.provenance,
            "evidence": self.config.evidence_report(),
        }


def build_cohort(
    cfg: SimulationConfig,
    seed: int,
    *,
    calibration_catalog: StimulusCatalog | None = None,
    smoothing: SmoothingSpec | None = None,
) -> Cohort:
    """Generate a complete cohort deterministically from ``(cfg, seed)``.

    Parameters
    ----------
    calibration_catalog:
        Stimuli to administer. Defaults to the balanced built-in design truncated to
        ``cfg.n_calibration_trials``.
    smoothing:
        Smoothing used when *estimating* each listener's confusion profile from their
        calibration trials.
    """
    catalog = calibration_catalog or build_balanced_catalog(cfg.n_calibration_trials)
    listeners = generate_cohort(cfg, seed)
    vocabulary = build_vocabulary(cfg, seed)
    # 어휘 구조는 청취자와 무관하므로 코호트당 한 번만 만듭니다. V1 에서는 쓰이지 않지만
    # 만드는 비용이 작고, 산출물에 어휘 규모를 남겨 두면 V1/V2 비교가 같은 어휘 위에서
    # 이루어졌음을 확인할 수 있습니다.
    lexicon = build_lexicon(vocabulary.words, vocabulary.provenance)

    records: list[ListenerRecord] = []
    for i, listener in enumerate(listeners):
        # A distinct, deterministic substream per listener so that adding a listener does
        # not perturb the trials of the listeners before them.
        cal_rng = np.random.default_rng([seed, i, 1])
        word_rng = np.random.default_rng([seed, i, 2])

        calibration = simulate_calibration(listener, catalog, cal_rng, cfg)
        estimated = ConfusionProfile.from_trials(
            listener.listener_id,
            calibration,
            is_synthetic=True,
            smoothing=smoothing,
            provenance={
                "estimated_from": "simulated calibration trials",
                "n_stimuli": len(catalog),
                "catalog": catalog.provenance,
                "config_name": cfg.name,
                "seed": seed,
            },
        )
        records.append(
            ListenerRecord(
                listener=listener,
                calibration=calibration,
                estimated_confusion=estimated,
                word_trials=simulate_word_trials(listener, vocabulary, word_rng, cfg, lexicon),
            )
        )

    return Cohort(
        config=cfg,
        seed=seed,
        records=tuple(records),
        vocabulary=vocabulary,
        calibration_catalog=catalog,
        provenance={
            "generator": "audire.sim.cohort.build_cohort",
            "config_name": cfg.name,
            "seed": seed,
            "is_synthetic": True,
            "simulator_version": cfg.simulator_version,
        },
    )
