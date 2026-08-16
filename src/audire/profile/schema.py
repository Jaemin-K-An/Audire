"""``HearingProfile`` — the clinical half of a listener's speech-perception profile.

Design constraints
------------------
* **Missingness is explicit and first-class.** Every optional measurement is ``None`` when
  absent, never zero and never imputed at construction time. :meth:`HearingProfile.missing`
  enumerates what is absent so that the missing-variable experiment scenarios are a
  configuration, not a code change.
* **Derived values name their method.** PTA is not "the PTA": it depends on which
  frequencies are averaged. Every derived value records the method and the frequencies
  actually used.
* **Nothing here is a diagnosis.** Severity strata are descriptive labels with a cited
  boundary scheme, used for stratified sampling and subgroup reporting only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audire.identity import ListenerId

#: Audiometric test frequencies in Hz, in ascending order.
STANDARD_FREQUENCIES: tuple[int, ...] = (125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000)

#: Plausible bounds for an air-conduction threshold in dB HL. Values outside this range are
#: rejected as data-entry errors rather than silently clipped.
MIN_DB_HL = -10.0
MAX_DB_HL = 130.0

DbHL = Annotated[float, Field(ge=MIN_DB_HL, le=MAX_DB_HL)]
Percent = Annotated[float, Field(ge=0.0, le=100.0)]


class Ear(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class ProfileSource(StrEnum):
    """Where a profile's numbers came from. Required — never inferred."""

    MANUAL = "manual"
    CLINICAL_EXPORT = "clinical_export"
    SYNTHETIC = "synthetic"


class HearingAidState(StrEnum):
    NONE = "none"
    UNAIDED = "unaided"
    AIDED = "aided"
    UNKNOWN = "unknown"


class PTAMethod(StrEnum):
    """Named pure-tone-average definitions.

    Several conventions coexist in audiology and they do not agree; AUDIRE therefore
    requires the method to be named wherever a PTA is used or reported.
    """

    #: Classic three-frequency average.
    PTA3 = "pta3_500_1000_2000"
    #: Four-frequency average including 4 kHz. AUDIRE's default.
    PTA4 = "pta4_500_1000_2000_4000"
    #: Four-frequency average including 3 kHz instead of 4 kHz.
    PTA4_3K = "pta4_500_1000_2000_3000"
    #: Six-frequency average.
    PTA6 = "pta6_500_1000_2000_3000_4000_6000"
    #: High-frequency average.
    HFA = "hfa_1000_2000_4000"


PTA_FREQUENCIES: dict[PTAMethod, tuple[int, ...]] = {
    PTAMethod.PTA3: (500, 1000, 2000),
    PTAMethod.PTA4: (500, 1000, 2000, 4000),
    PTAMethod.PTA4_3K: (500, 1000, 2000, 3000),
    PTAMethod.PTA6: (500, 1000, 2000, 3000, 4000, 6000),
    PTAMethod.HFA: (1000, 2000, 4000),
}

DEFAULT_PTA_METHOD = PTAMethod.PTA4

#: Version tag for the PTA computation itself, recorded alongside every derived value so
#: that a change in how PTA is computed invalidates cached results rather than silently
#: shifting them.
PTA_CALC_VERSION = "1.0.0"


class SeverityScheme(StrEnum):
    """Named severity-stratum boundary schemes."""

    #: WHO grades (better-ear average dB HL): <20 / 20-34 / 35-49 / 50-64 / 65-79 /
    #: 80-94 / >=95. Source: WHO, "Deafness and hearing loss" (accessed 2026-08-16);
    #: registered in data/sources.yaml as `who2021_hearing_grades`.
    WHO2021 = "who2021"
    #: The four-group split used by the reference Korean error-analysis studies
    #: (normal / mild / moderate / severe). Boundaries follow WHO up to 65 dB HL and
    #: collapse everything above into "severe".
    KOREAN_STUDY_4GROUP = "korean_study_4group"


_WHO_BOUNDS: tuple[tuple[float, str], ...] = (
    (20.0, "normal"),
    (35.0, "mild"),
    (50.0, "moderate"),
    (65.0, "moderately_severe"),
    (80.0, "severe"),
    (95.0, "profound"),
    (float("inf"), "complete"),
)

_KOREAN4_BOUNDS: tuple[tuple[float, str], ...] = (
    (20.0, "normal"),
    (35.0, "mild"),
    (50.0, "moderate"),
    (float("inf"), "severe"),
)

_SCHEME_BOUNDS = {
    SeverityScheme.WHO2021: _WHO_BOUNDS,
    SeverityScheme.KOREAN_STUDY_4GROUP: _KOREAN4_BOUNDS,
}


def severity_stratum(pta_db_hl: float, scheme: SeverityScheme = SeverityScheme.WHO2021) -> str:
    """Return the descriptive severity stratum for ``pta_db_hl`` under ``scheme``.

    This is a stratification label for sampling and subgroup reporting. It is **not** a
    diagnosis, a degree-of-disability determination, or a clinical assessment.
    """
    for upper, label in _SCHEME_BOUNDS[scheme]:
        if pta_db_hl < upper:
            return label
    raise AssertionError("severity bounds must end with infinity")  # pragma: no cover


# --------------------------------------------------------------------------- models


class AudiogramPoint(BaseModel):
    """One air-conduction threshold.

    ``db_hl`` is ``None`` when the frequency was not tested. ``no_response`` marks
    "no response at the audiometer's maximum output", which is *not* the same as
    "not tested" and is *not* the same as a threshold equal to that output level.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    db_hl: DbHL | None = None
    no_response: bool = False
    masked: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.no_response and self.db_hl is None:
            raise ValueError(
                "no_response=True requires db_hl to record the level at which no response "
                "was obtained (the audiometer's output limit)"
            )
        return self

    @property
    def is_measured(self) -> bool:
        """Whether this point carries a usable threshold (a response was obtained)."""
        return self.db_hl is not None and not self.no_response


class Audiogram(BaseModel):
    """Air-conduction thresholds for one ear, keyed by frequency in Hz."""

    model_config = ConfigDict(extra="forbid")

    ear: Ear
    thresholds: dict[int, AudiogramPoint] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_frequencies(self) -> Self:
        unknown = sorted(set(self.thresholds) - set(STANDARD_FREQUENCIES))
        if unknown:
            raise ValueError(
                f"non-standard audiometric frequencies {unknown}; "
                f"supported: {list(STANDARD_FREQUENCIES)}"
            )
        return self

    def measured_frequencies(self) -> tuple[int, ...]:
        return tuple(sorted(f for f, p in self.thresholds.items() if p.is_measured))

    def pta(self, method: PTAMethod = DEFAULT_PTA_METHOD) -> float | None:
        """Pure-tone average under ``method``.

        Returns ``None`` — never a partial average — if any required frequency is missing
        or produced no response, because substituting or dropping a frequency would make
        the value incomparable with the same value computed elsewhere.
        """
        required = PTA_FREQUENCIES[method]
        values: list[float] = []
        for f in required:
            point = self.thresholds.get(f)
            if point is None or not point.is_measured or point.db_hl is None:
                return None
            values.append(point.db_hl)
        return sum(values) / len(values)

    def pta_detail(self, method: PTAMethod = DEFAULT_PTA_METHOD) -> dict[str, Any]:
        """PTA plus the provenance of the computation."""
        required = PTA_FREQUENCIES[method]
        missing = [
            f for f in required if (p := self.thresholds.get(f)) is None or not p.is_measured
        ]
        return {
            "ear": self.ear.value,
            "method": method.value,
            "calc_version": PTA_CALC_VERSION,
            "frequencies_required": list(required),
            "frequencies_missing": missing,
            "value_db_hl": self.pta(method),
        }

    def slope_db_per_octave(self) -> float | None:
        """Mean change in threshold per octave from 500 Hz to 4000 Hz.

        Positive means a sloping (high-frequency) configuration. ``None`` when either
        anchor frequency is unmeasured.
        """
        low = self.thresholds.get(500)
        high = self.thresholds.get(4000)
        if low is None or high is None or not low.is_measured or not high.is_measured:
            return None
        assert low.db_hl is not None and high.db_hl is not None  # narrowed by is_measured
        return (high.db_hl - low.db_hl) / 3.0  # 500 -> 4000 Hz is three octaves


class SpeechScores(BaseModel):
    """Speech audiometry for one ear.

    ``wrs_percent`` is a word recognition score obtained with a named word list at a named
    presentation level. AUDIRE's own calibration accuracy is a different quantity and is
    never written into this field.
    """

    model_config = ConfigDict(extra="forbid")

    ear: Ear
    #: Speech recognition threshold: the level giving 50 % correct responses.
    srt_db_hl: DbHL | None = None
    wrs_percent: Percent | None = None
    wrs_presentation_level_db_hl: DbHL | None = None
    #: e.g. "KS-MWL-A list 1". Free text; recorded so a score is never level- or
    #: list-ambiguous.
    wrs_word_list: str | None = None
    wrs_n_words: Annotated[int, Field(ge=1, le=200)] | None = None
    #: Whether the WRS was obtained in noise, and at what SNR.
    wrs_snr_db: float | None = None

    @model_validator(mode="after")
    def _wrs_needs_a_level(self) -> Self:
        if self.wrs_percent is not None and self.wrs_presentation_level_db_hl is None:
            raise ValueError(
                "wrs_percent requires wrs_presentation_level_db_hl: a word recognition "
                "score is uninterpretable without the level at which it was obtained"
            )
        return self


class PIPoint(BaseModel):
    """One point of a performance-intensity function."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level_db_hl: DbHL
    score_percent: Percent


class PIFunction(BaseModel):
    """A performance-intensity function for one ear.

    ``pbmax`` is the maximum score attained across the measured levels. ``rollover_index``
    is reported as a descriptive summary of the curve's shape; AUDIRE attaches **no**
    clinical interpretation to it.
    """

    model_config = ConfigDict(extra="forbid")

    ear: Ear
    points: list[PIPoint] = Field(min_length=1)
    word_list: str | None = None

    @model_validator(mode="after")
    def _unique_levels(self) -> Self:
        levels = [p.level_db_hl for p in self.points]
        if len(set(levels)) != len(levels):
            raise ValueError("performance-intensity function has duplicate presentation levels")
        return self

    @property
    def sorted_points(self) -> list[PIPoint]:
        return sorted(self.points, key=lambda p: p.level_db_hl)

    @property
    def pbmax_percent(self) -> float:
        return max(p.score_percent for p in self.points)

    @property
    def pbmax_level_db_hl(self) -> float:
        best = max(self.points, key=lambda p: (p.score_percent, -p.level_db_hl))
        return best.level_db_hl

    @property
    def rollover_index(self) -> float | None:
        """``(PBmax - min score above the PBmax level) / PBmax``.

        ``None`` when no level above the PBmax level was measured, or when PBmax is 0.
        Descriptive only.
        """
        pts = self.sorted_points
        peak_level = self.pbmax_level_db_hl
        above = [p.score_percent for p in pts if p.level_db_hl > peak_level]
        if not above or self.pbmax_percent == 0:
            return None
        return (self.pbmax_percent - min(above)) / self.pbmax_percent


class LoudnessLevels(BaseModel):
    """Most-comfortable and uncomfortable loudness levels for one ear."""

    model_config = ConfigDict(extra="forbid")

    ear: Ear
    mcl_db_hl: DbHL | None = None
    ucl_db_hl: DbHL | None = None

    @model_validator(mode="after")
    def _ordering(self) -> Self:
        if (
            self.mcl_db_hl is not None
            and self.ucl_db_hl is not None
            and self.ucl_db_hl < self.mcl_db_hl
        ):
            raise ValueError(
                f"UCL ({self.ucl_db_hl} dB HL) cannot be below MCL ({self.mcl_db_hl} dB HL)"
            )
        return self

    @property
    def dynamic_range_db(self) -> float | None:
        if self.mcl_db_hl is None or self.ucl_db_hl is None:
            return None
        return self.ucl_db_hl - self.mcl_db_hl


class EarProfile(BaseModel):
    """Everything measured for one ear."""

    model_config = ConfigDict(extra="forbid")

    ear: Ear
    audiogram: Audiogram
    speech: SpeechScores
    pi_function: PIFunction | None = None
    loudness: LoudnessLevels | None = None

    @model_validator(mode="after")
    def _ears_agree(self) -> Self:
        parts = [("audiogram", self.audiogram.ear), ("speech", self.speech.ear)]
        if self.pi_function is not None:
            parts.append(("pi_function", self.pi_function.ear))
        if self.loudness is not None:
            parts.append(("loudness", self.loudness.ear))
        mismatched = [name for name, ear in parts if ear is not self.ear]
        if mismatched:
            raise ValueError(f"ear mismatch in {mismatched}: expected {self.ear.value}")
        return self


class HearingProfile(BaseModel):
    """A listener's clinical speech-perception profile.

    Kept structurally separate from :class:`~audire.confusion.profile.ConfusionProfile`
    (ADR-0008) so that "does the confusion matrix add information beyond the clinical
    measures?" remains an answerable question.
    """

    model_config = ConfigDict(extra="forbid")

    #: Opaque identifier. Must never be a name, initials, or any direct identifier.
    #: Validated by the single shared rule in :mod:`audire.identity`.
    listener_id: ListenerId
    source: ProfileSource
    is_synthetic: bool
    left: EarProfile | None = None
    right: EarProfile | None = None
    hearing_aid_state: HearingAidState = HearingAidState.UNKNOWN
    age_band: Literal["<20", "20-39", "40-59", "60-69", "70-79", "80+"] | None = None
    pta_method: PTAMethod = DEFAULT_PTA_METHOD
    severity_scheme: SeverityScheme = SeverityScheme.WHO2021
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if self.left is None and self.right is None:
            raise ValueError("a hearing profile must contain at least one ear")
        if self.left is not None and self.left.ear is not Ear.LEFT:
            raise ValueError("the `left` field must hold a left-ear profile")
        if self.right is not None and self.right.ear is not Ear.RIGHT:
            raise ValueError("the `right` field must hold a right-ear profile")
        if (self.source is ProfileSource.SYNTHETIC) != self.is_synthetic:
            raise ValueError(
                "source=synthetic and is_synthetic must agree; synthetic provenance "
                "cannot be partially declared"
            )
        return self

    # ------------------------------------------------------------------ accessors

    def ear(self, ear: Ear) -> EarProfile | None:
        return self.left if ear is Ear.LEFT else self.right

    @property
    def ears(self) -> list[EarProfile]:
        return [e for e in (self.right, self.left) if e is not None]

    # ------------------------------------------------------------------ derived

    def pta(self, ear: Ear, method: PTAMethod | None = None) -> float | None:
        e = self.ear(ear)
        return None if e is None else e.audiogram.pta(method or self.pta_method)

    def better_ear_pta(self, method: PTAMethod | None = None) -> float | None:
        """Lower (better) of the two ear PTAs. ``None`` if neither ear yields a PTA."""
        values = [
            v for v in (self.pta(Ear.LEFT, method), self.pta(Ear.RIGHT, method)) if v is not None
        ]
        return min(values) if values else None

    def worse_ear_pta(self, method: PTAMethod | None = None) -> float | None:
        values = [
            v for v in (self.pta(Ear.LEFT, method), self.pta(Ear.RIGHT, method)) if v is not None
        ]
        return max(values) if values else None

    def severity(self, scheme: SeverityScheme | None = None) -> str | None:
        """Descriptive severity stratum from the better-ear PTA. Not a diagnosis."""
        pta = self.better_ear_pta()
        if pta is None:
            return None
        return severity_stratum(pta, scheme or self.severity_scheme)

    def best_wrs(self) -> float | None:
        """Highest single-level WRS across ears, or ``None`` when no WRS was recorded."""
        values = [e.speech.wrs_percent for e in self.ears if e.speech.wrs_percent is not None]
        return max(values) if values else None

    def best_srt(self) -> float | None:
        """Lowest (best) SRT across ears."""
        values = [e.speech.srt_db_hl for e in self.ears if e.speech.srt_db_hl is not None]
        return min(values) if values else None

    def pbmax(self) -> float | None:
        """Highest PBmax across ears with a performance-intensity function."""
        values = [e.pi_function.pbmax_percent for e in self.ears if e.pi_function is not None]
        return max(values) if values else None

    def mcl(self) -> float | None:
        values = [
            e.loudness.mcl_db_hl
            for e in self.ears
            if e.loudness is not None and e.loudness.mcl_db_hl is not None
        ]
        return min(values) if values else None

    # ------------------------------------------------------------------ missingness

    #: Fields the risk models may consume. Order is stable so that "which variables were
    #: available?" is a reproducible description.
    CLINICAL_FIELDS: tuple[str, ...] = (
        "better_ear_pta",
        "worse_ear_pta",
        "srt",
        "wrs",
        "pbmax",
        "mcl",
        "audiogram_slope",
    )

    def available(self) -> dict[str, bool]:
        """Which clinical variables this profile actually carries."""
        slopes = [
            e.audiogram.slope_db_per_octave()
            for e in self.ears
            if e.audiogram.slope_db_per_octave() is not None
        ]
        return {
            "better_ear_pta": self.better_ear_pta() is not None,
            "worse_ear_pta": self.worse_ear_pta() is not None,
            "srt": self.best_srt() is not None,
            "wrs": self.best_wrs() is not None,
            "pbmax": self.pbmax() is not None,
            "mcl": self.mcl() is not None,
            "audiogram_slope": bool(slopes),
        }

    def missing(self) -> tuple[str, ...]:
        """Clinical variables that are absent. Never silently imputed."""
        return tuple(name for name, present in self.available().items() if not present)

    def completeness(self) -> float:
        """Fraction of the clinical field set that is present."""
        avail = self.available()
        return sum(avail.values()) / len(avail)

    # ------------------------------------------------------------------ reporting

    def summary(self) -> dict[str, Any]:
        """A flat, JSON-safe summary for the UI and for result provenance."""
        return {
            "listener_id": self.listener_id,
            "source": self.source.value,
            "is_synthetic": self.is_synthetic,
            "hearing_aid_state": self.hearing_aid_state.value,
            "age_band": self.age_band,
            "pta_method": self.pta_method.value,
            "pta_calc_version": PTA_CALC_VERSION,
            "pta_left": self.pta(Ear.LEFT),
            "pta_right": self.pta(Ear.RIGHT),
            "better_ear_pta": self.better_ear_pta(),
            "severity_scheme": self.severity_scheme.value,
            "severity_stratum": self.severity(),
            "srt": self.best_srt(),
            "wrs": self.best_wrs(),
            "pbmax": self.pbmax(),
            "mcl": self.mcl(),
            "missing": list(self.missing()),
            "completeness": self.completeness(),
            "disclaimer": (
                "Descriptive research summary. AUDIRE is not a medical device and this "
                "is not a diagnosis."
            ),
        }
