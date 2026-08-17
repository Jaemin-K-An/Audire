"""Generative parameters for the synthetic cohort, with per-parameter evidence labels.

Every numeric assumption in the simulator lives here, and every block carries an
``evidence`` label that says what backs it:

``literature``
    Traceable to a registered publication in ``data/sources.yaml``. The
    ``evidence_source`` field names the registry id.
``clinical_convention``
    Standard audiological practice, not a specific numeric finding.
``assumption``
    **Declared with no numeric evidence.** Chosen to make the simulator produce a
    non-degenerate, interestingly-structured world. Results that depend on these values
    are engineering/design-sensitivity results, not empirical claims — see
    docs/RISK_REGISTER.md S1 and S2.

Nothing in the simulator may read a magic constant that is not defined here.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audire.profile.schema import PTAMethod, SeverityScheme

Evidence = Literal["literature", "clinical_convention", "assumption"]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


class Evidenced(BaseModel):
    """Base class forcing every parameter block to declare what backs it."""

    model_config = ConfigDict(extra="forbid")

    evidence: Evidence
    evidence_source: str | None = None
    rationale: str = ""

    @model_validator(mode="after")
    def _literature_needs_a_source(self) -> Evidenced:
        if self.evidence == "literature" and not self.evidence_source:
            raise ValueError(
                "evidence='literature' requires evidence_source naming a registry id in "
                "data/sources.yaml"
            )
        return self


class SeverityMix(Evidenced):
    """How synthetic listeners are distributed across severity strata."""

    evidence: Evidence = "literature"
    evidence_source: str | None = "joo2026_error_rates"
    rationale: str = (
        "Group sizes mirror the reference study's design (20 normal / 20 mild / "
        "20 moderate / 12 severe, n=72) so that the simulated cohort has a comparable "
        "shape. This fixes the *proportions*, not any listener's behaviour."
    )
    scheme: SeverityScheme = SeverityScheme.KOREAN_STUDY_4GROUP
    weights: dict[str, float] = Field(
        default_factory=lambda: {"normal": 20.0, "mild": 20.0, "moderate": 20.0, "severe": 12.0}
    )

    @model_validator(mode="after")
    def _positive(self) -> SeverityMix:
        if not self.weights or any(w < 0 for w in self.weights.values()):
            raise ValueError("severity weights must be non-empty and non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("severity weights must sum to a positive number")
        return self

    def normalised(self) -> dict[str, float]:
        total = sum(self.weights.values())
        return {k: v / total for k, v in self.weights.items()}


class AudiogramModel(Evidenced):
    """How an audiogram is drawn once a severity stratum has been chosen."""

    evidence: Evidence = "clinical_convention"
    rationale: str = (
        "Age-related hearing loss in older adults characteristically slopes downward "
        "toward high frequencies. The stratum PTA windows follow the named severity "
        "scheme; the slope and noise magnitudes are chosen to produce realistic-looking "
        "but not evidence-derived audiogram shapes."
    )
    #: Target better-ear PTA window per stratum, in dB HL. Sampled uniformly within.
    pta_window_db: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "normal": (5.0, 19.0),
            "mild": (20.0, 34.0),
            "moderate": (35.0, 49.0),
            "moderately_severe": (50.0, 64.0),
            "severe": (65.0, 85.0),
            "profound": (86.0, 100.0),
        }
    )
    #: Mean high-frequency slope in dB per octave (500 Hz -> 4 kHz), and its spread.
    slope_db_per_octave_mean: float = 6.0
    slope_db_per_octave_sd: float = Field(default=3.0, gt=0.0)
    #: Independent per-frequency measurement noise, dB.
    threshold_noise_db_sd: float = Field(default=4.0, gt=0.0)
    #: Interaural asymmetry, dB.
    interaural_asymmetry_db_sd: float = 5.0
    pta_method: PTAMethod = PTAMethod.PTA4


class SpeechScoreModel(Evidenced):
    """How SRT, WRS and PBmax are drawn given the audiogram and latent ability."""

    evidence: Evidence = "literature"
    evidence_source: str | None = "joo2026_error_rates"
    rationale: str = (
        "Monosyllable correct-response rates per hearing group are anchored to the "
        "reference study's reported total error rates (normal 18.3 %, mild 27.8 %, "
        "moderate 48.4 %, severe 80.4 % errors at MCL over 726 monosyllables), i.e. "
        "correct rates of 81.7 / 72.2 / 51.6 / 19.6 %. SRT-PTA agreement is a clinical "
        "convention rather than a finding. The within-group spread is an assumption."
    )
    #: Mean monosyllable *correct* rate per stratum, from the reference error rates.
    accuracy_by_stratum: dict[str, float] = Field(
        default_factory=lambda: {
            "normal": 0.817,
            "mild": 0.722,
            "moderate": 0.516,
            "moderately_severe": 0.400,
            "severe": 0.196,
            "profound": 0.100,
        }
    )
    #: Between-listener spread of latent ability within a stratum (logit scale).
    ability_sd_logit: float = Field(default=0.55, gt=0.0)
    #: SRT is sampled around the PTA; agreement within about +-6..10 dB is the clinical
    #: expectation for a reliable test.
    srt_minus_pta_mean_db: float = 0.0
    srt_minus_pta_sd_db: float = Field(default=5.0, gt=0.0)
    #: A single-level WRS is a noisy binomial estimate over `wrs_n_words` items.
    wrs_n_words: int = 50
    #: PBmax is at least the single-level WRS; this is the mean increment.
    pbmax_increment_mean: float = 6.0
    pbmax_increment_sd: float = 4.0
    #: Probability that a listener has a recorded PI function / MCL-UCL at all.
    p_has_pi_function: UnitInterval = 0.5
    p_has_loudness: UnitInterval = 0.6
    #: MCL relative to PTA and the dynamic range to UCL.
    mcl_above_pta_db: float = 30.0
    mcl_sd_db: float = 6.0
    dynamic_range_db_mean: float = 30.0
    dynamic_range_db_sd: float = 8.0


class ConfusionModel(Evidenced):
    """How an individual confusion matrix is generated.

    The row for target *i* is built as

        diagonal mass  d_i  = position-specific function of the listener's ability
        off-diagonal   ∝ exp(beta * similarity(i, j))  scaled to  1 - d_i

    and then perturbed with a Dirichlet draw so that listeners with the same ability
    still differ in *which* confusions they make.
    """

    evidence: Evidence = "literature"
    evidence_source: str | None = "ma2026_similarity"
    rationale: str = (
        "The reference study reports that perceptual distance between phonemes shrinks "
        "as hearing loss increases (onset 5.43 -> 3.05, nucleus 5.51 -> 3.53, coda "
        "5.78 -> 3.66 from normal to severe) and that confusions concentrate within "
        "phonetic classes. This motivates (a) within-class concentration via a "
        "similarity kernel and (b) a concentration parameter that rises with severity. "
        "The specific beta values and the Dirichlet concentration are assumptions."
    )
    #: Relative difficulty multiplier per position applied to the error mass. Larger
    #: means more errors at that position.
    position_difficulty: dict[str, float] = Field(
        default_factory=lambda: {"onset": 1.15, "nucleus": 0.70, "coda": 1.15}
    )
    #: Within-class concentration of confusions at normal and at severe hearing level;
    #: interpolated linearly in stratum index between them.
    similarity_beta_normal: float = 1.5
    similarity_beta_severe: float = 4.0
    #: Dirichlet concentration: higher means listeners resemble the structural prior more
    #: closely; lower means more between-listener heterogeneity.
    #: Higher means listeners resemble the structural prior more closely. Must be
    #: positive: a Dirichlet concentration of zero or less is not a distribution.
    dirichlet_concentration: float = Field(default=40.0, gt=0.0)
    #: Probability mass reserved for "no usable response", scaled by error mass.
    no_response_share: UnitInterval = 0.05


class TrialModel(Evidenced):
    """How word-level mishearing outcomes are produced from sampled phonemes.

    Deliberately **not** the phoneme-independence rule that ``R_phon`` assumes, so that
    the deterministic baseline is a misspecified model of this generator rather than a
    restatement of it (docs/RISK_REGISTER.md S2).
    """

    evidence: Evidence = "assumption"
    rationale: str = (
        "Lexical repair, word-length effects and SNR effects are declared assumptions "
        "with no numeric source. Their purpose is twofold. First, they make the "
        "generative process differ structurally from the phoneme-independence scoring "
        "model, so that comparing the two is informative rather than circular. Second, "
        "they bridge the gap between the context-free monosyllable task the literature "
        "measures and the connected speech AUDIRE captions: a listener who misses one "
        "segment of a word in running speech often still identifies it. The resulting "
        "word-level mishearing base rate is therefore a CONFIGURED property of the "
        "simulation, is reported in every cohort summary, and is swept explicitly in the "
        "sensitivity study rather than treated as a fixed fact."
    )
    #: Probability that a word with exactly one segmental error is nonetheless identified
    #: correctly through lexical/contextual repair. Longer words repair more easily
    #: because they carry more redundant evidence.
    lexical_repair_base: UnitInterval = 0.55
    lexical_repair_per_extra_syllable: float = 0.12
    #: Cap on the repair probability regardless of length.
    lexical_repair_max: UnitInterval = 0.90
    #: Words with two or more segmental errors repair much less often.
    lexical_repair_multi_error: UnitInterval = 0.15
    #: Additive shift (logit scale) applied to every diagonal probability per dB of SNR
    #: relative to the reference condition. Negative SNR therefore increases errors.
    snr_logit_per_db: float = 0.06
    reference_snr_db: float = 20.0
    #: Per-speaker additive shift (logit scale) on the diagonal.
    speaker_effects: dict[str, float] = Field(
        default_factory=lambda: {"male": -0.10, "female": 0.10, "synthetic_tts": 0.0}
    )


class WordSourceModel(Evidenced):
    """Where the words used for word-level trials come from."""

    evidence: Evidence = "assumption"
    rationale: str = (
        "The synthetic word generator samples syllable counts from a distribution that "
        "resembles Korean eojeol length in running text but is not measured from a "
        "corpus. Use `source='zeroth'` to draw words from the CC BY-licensed "
        "Zeroth-Korean transcripts instead when a real distribution matters."
    )
    source: Literal["synthetic", "zeroth"] = "synthetic"
    #: Relative weights for 1..4-syllable words.
    syllable_count_weights: dict[int, float] = Field(
        default_factory=lambda: {1: 0.20, 2: 0.45, 3: 0.25, 4: 0.10}
    )
    #: Size of the sampled vocabulary. Words repeat across listeners so that word identity
    #: is a shared factor, as it is in real captioning.
    vocabulary_size: int = Field(default=400, ge=10)


class LexicalRepairV2(Evidenced):
    """Simulator V2 의 복구 모형: **무엇을 어떻게 잘못 들었는가**에 조건화합니다.

    V1 의 한계
    ----------
    V1 은 복구 확률을 오류 **개수**와 음절 수만으로 정합니다. 그래서 "각→닥" 과 "각→삭" 의
    결과 분포가 정확히 같고, ``(오류 수, 음절 수)`` 만 아는 오라클이 PR-AUC 0.8307 에
    도달하는 반면 현재 모델이 이미 0.7884 로 그 95% 에 와 있습니다
    (``audire.sim.diagnostics``, docs/RESULTS.md §14). 단어 수준 신호가 남아 있지 않습니다.

    처음 시도한 기전과 그것이 실패한 이유
    -------------------------------------
    처음에는 **어휘 함정**으로 설계했습니다: 지각형이 실재 단어이면 청자가 오류를 알아채지
    못해 복구가 드물고, 비단어이면 알아채되 경쟁 이웃이 많을수록 어렵다는 것입니다.
    측정해 보니 오류가 난 4,878 시행 중 지각형이 어휘에 걸린 경우가 **13건(0.3%)** 뿐이었고,
    어휘 400단어 중 한 자리 치환 이웃을 가진 단어가 67개에 불과했습니다. 기본 어휘가
    무작위 합성 음절열이라 "실재 단어" 라는 개념이 성립하지 않는 것이 근본 원인입니다.
    경쟁자가 거의 항상 0 이므로 이웃 감쇠 항도 상수가 되고, 복구 확률이 다시 오류 개수만의
    함수로 돌아갑니다. 즉 그 설계로는 V1 의 한계를 그대로 재현합니다.

    채택한 기전
    -----------
    대신 **오류 위치와 음운적 거리**에 조건화합니다. 이것은 어휘 밀도와 무관하게 작동하고
    한국어 음운론에 근거가 있습니다.

    * **종성 오류는 잘 복구됩니다.** 한국어 종성은 7종으로 중화되고 연음·후행 자음에 따라
      표면형이 달라지므로, 청자는 종성을 문맥에서 재구성하는 데 익숙합니다.
    * **중성 오류는 잘 복구되지 않습니다.** 모음은 어휘 변별 부담이 크고 중화되지 않습니다.
    * **같은 부류 안의 치환은 더 잘 복구됩니다.** 평음↔격음(ㄱ↔ㅋ)처럼 조음 위치가 같은
      혼동은 후보가 좁혀지지만, 부류를 건너뛰면 그렇지 않습니다.

    이 기전이 단어별·청취자별 구조를 만듭니다. 단어마다 종성 비율과 음소 구성이 다르고,
    청취자마다 어느 위치·어느 부류에서 틀리는지가 다르기 때문입니다.

    순환이 아닌 이유
    ----------------
    예측 모델은 음소별 ``p_correct`` 의 곱과 그 집계를 계산합니다. "어느 위치가 틀렸는가"
    에 가중치를 주거나 "치환이 같은 부류 안이었는가" 를 보는 항이 없습니다. 따라서 이
    기전은 채점 수식의 재진술이 아니라 모델이 추정해야 할 대상입니다.
    """

    evidence: Evidence = "assumption"
    evidence_source: str | None = None
    rationale: str = (
        "기전의 **방향**(종성 중화로 인한 높은 복구 가능성, 모음의 큰 어휘 변별 부담, 같은 "
        "조음 부류 내 혼동의 상대적 회복 용이성)은 한국어 음운론에서 널리 기술되지만, 여기 "
        "쓰인 **수치**에는 한국어 연결발화에 대한 출처가 없습니다. 따라서 전부 명시적 "
        "시뮬레이션 가정입니다. 목적은 실제 복구율을 재현하는 것이 아니라, 단어 수준 개인화 "
        "신호가 **존재할 때** 추정기가 그것을 회복하는지 시험하는 것입니다. V1 은 그대로 "
        "실행 가능하며 두 생성 과정을 같은 시드로 비교합니다."
    )

    #: 오류가 하나 있을 때의 기저 복구 확률. 위치·거리 가중치가 여기에 곱해집니다.
    base_repair: UnitInterval = 0.55
    #: 위치별 복구 용이도 배수. 종성 > 초성 > 중성.
    coda_recovery: float = Field(default=1.45, gt=0.0)
    onset_recovery: float = Field(default=1.00, gt=0.0)
    nucleus_recovery: float = Field(default=0.55, gt=0.0)
    #: 치환이 같은 음운 부류(초성 조음방법 / 중성 활음형 / 종성 중화형) 안에서 일어났을 때의
    #: 배수. 후보가 좁혀지므로 복구가 쉬워집니다.
    same_class_bonus: float = Field(default=1.35, gt=0.0)
    #: 오류가 늘어날수록 지각형이 목표에서 멀어집니다. 추가 오류마다 곱해집니다.
    repair_per_extra_error_decay: float = Field(default=0.55, gt=0.0, le=1.0)
    #: 긴 단어는 잉여 정보가 많아 복구가 쉽습니다 (V1 과 같은 방향).
    repair_per_extra_syllable: float = Field(default=0.10, ge=0.0)

    #: 어휘 항목이 지각형과 일치할 때의 복구 배수(어휘 함정). 이 어휘에서는 거의 발동하지
    #: 않지만, 실제 한국어 어휘를 붙이면 의미를 갖도록 남겨 둡니다. 발동 빈도는 코호트
    #: 요약에 기록되므로 "있으나 마나 한 항" 인지 확인할 수 있습니다.
    lexical_trap_multiplier: float = Field(default=0.25, gt=0.0, le=1.0)

    repair_floor: UnitInterval = 0.02
    repair_ceiling: UnitInterval = 0.95


class SimulationConfig(BaseModel):
    """A complete, reproducible synthetic-cohort specification."""

    model_config = ConfigDict(extra="forbid")

    name: str
    #: Every random draw derives from these. A run executes *all* of them; reporting a
    #: single favourable seed is a protocol violation (docs/RISK_REGISTER.md X2).
    seeds: list[int] = Field(default_factory=lambda: [11, 22, 33, 44, 55], min_length=1)
    n_listeners: int = Field(default=80, ge=2)
    #: Calibration stimuli presented to each synthetic listener.
    n_calibration_trials: int = Field(default=100, ge=1)
    #: Word-level trials per listener, used for risk-model training and evaluation.
    n_word_trials: int = Field(default=300, ge=1)
    #: SNR conditions to generate word trials under.
    snr_conditions_db: list[float] = Field(default_factory=lambda: [20.0], min_length=1)
    speakers: list[str] = Field(default_factory=lambda: ["male", "female"], min_length=1)

    severity: SeverityMix = Field(default_factory=SeverityMix)
    audiogram: AudiogramModel = Field(default_factory=AudiogramModel)
    speech: SpeechScoreModel = Field(default_factory=SpeechScoreModel)
    confusion: ConfusionModel = Field(default_factory=ConfusionModel)
    trials: TrialModel = Field(default_factory=TrialModel)
    words: WordSourceModel = Field(default_factory=WordSourceModel)
    #: 어느 생성 과정을 쓸 것인가. **기본값은 v1** 이므로 기존 설정과 기록된 실행의 의미론이
    #: 바뀌지 않습니다. v2 는 복구를 지각형에 조건화합니다 (:class:`LexicalRepairV2`).
    simulator_version: Literal["v1", "v2"] = "v1"
    lexical_repair_v2: LexicalRepairV2 = Field(default_factory=LexicalRepairV2)

    @model_validator(mode="after")
    def _strata_are_known(self) -> SimulationConfig:
        strata = set(self.severity.weights)
        for field_name, table in (
            ("audiogram.pta_window_db", self.audiogram.pta_window_db),
            ("speech.accuracy_by_stratum", self.speech.accuracy_by_stratum),
        ):
            missing = strata - set(table)
            if missing:
                raise ValueError(f"{field_name} has no entry for strata {sorted(missing)}")
        unknown_speakers = set(self.speakers) - set(self.trials.speaker_effects)
        if unknown_speakers:
            raise ValueError(
                f"trials.speaker_effects has no entry for speakers {sorted(unknown_speakers)}"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        return self

    def evidence_report(self) -> list[dict[str, Any]]:
        """Enumerate every parameter block with its evidence label.

        Emitted into every result artifact so that a reader can see, without reading the
        code, which parts of the generative process are backed by literature and which
        are declared assumptions.
        """
        blocks: list[tuple[str, Evidenced]] = [
            ("severity", self.severity),
            ("audiogram", self.audiogram),
            ("speech", self.speech),
            ("confusion", self.confusion),
            ("trials", self.trials),
            ("words", self.words),
        ]
        return [
            {
                "block": name,
                "evidence": blk.evidence,
                "evidence_source": blk.evidence_source,
                "rationale": blk.rationale,
            }
            for name, blk in blocks
        ]
