"""Calibration stimulus catalogues.

AUDIRE supports two stimulus sources and must remain fully functional with either:

``builtin``
    A deterministic, phoneme-balanced Korean monosyllable design generated from the
    Hangul inventory itself. It carries no third-party licence, is reproducible from a
    seed-free algorithm, and is what makes the phoneme-balanced arm of the
    calibration-length study (E5) possible. Audio is not bundled; the web client speaks
    the syllable with the browser's Korean speech synthesiser, which is **not** level-
    calibrated and is therefore unsuitable for clinical measurement.

``kmsp``
    The Korean Monosyllabic Speech Perception Test Dataset (CC BY-NC-ND 4.0), which
    supplies recorded stimuli from two speakers plus acoustic measurements. It must be
    fetched locally after the dataset card's human notification step, and neither its
    audio nor its metadata may be committed or redistributed.

The balanced design space
-------------------------
19 onsets x 21 nuclei x 8 coda categories = 3,192 combinations. Because 19, 21 and 8 are
pairwise coprime, the sequence ``(i mod 19, i mod 21, i mod 8)`` visits each of the
3,192 combinations exactly once as ``i`` runs over ``range(3192)``. Taking the first
``n`` items therefore yields the most marginally balanced ``n``-item subset available,
with no random selection involved. (The dataset card for the primary corpus likewise
describes a universe of 3,192 possible Korean consonant-vowel combinations.)

Nonsense syllables are expected and are standard in Korean monosyllabic perception
testing; the primary corpus explicitly mixes meaningful (유의미) and nonsense (무의미)
items.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from audire.config.paths import raw_dir
from audire.data.manifest import require_verified
from audire.data.sources import Source, registry
from audire.hangul.inventory import NO_CODA, NUCLEUS_JAMO, ONSET_JAMO
from audire.hangul.syllable import Syllable, compose_syllable, decompose_syllable

StimulusSource = Literal["builtin", "kmsp"]

#: Coda categories used by the built-in design: "no coda" plus the seven surface codas
#: of Korean coda neutralisation. Orthographic clusters are excluded because they are a
#: spelling phenomenon, not a distinct perceptual target.
BUILTIN_CODAS: tuple[str, ...] = (NO_CODA, "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ")

#: Total size of the balanced design space (19 x 21 x 8).
DESIGN_SPACE_SIZE: int = len(ONSET_JAMO) * len(NUCLEUS_JAMO) * len(BUILTIN_CODAS)


@dataclass(frozen=True, slots=True)
class Stimulus:
    """One calibration stimulus."""

    stimulus_id: str
    syllable: str
    source: StimulusSource
    #: ``"male"`` / ``"female"`` for recorded corpora, ``"synthetic_tts"`` for built-in.
    speaker: str
    structure: str
    #: Recorded audio, relative to the dataset root. ``None`` for the built-in design.
    audio_path: str | None = None
    #: Acoustic measurements supplied by the corpus, if any.
    acoustics: dict[str, float] | None = None

    @property
    def decomposition(self) -> Syllable:
        return decompose_syllable(self.syllable)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StimulusCatalog:
    """An ordered, immutable list of stimuli plus its provenance."""

    source: StimulusSource
    stimuli: tuple[Stimulus, ...]
    provenance: dict[str, Any]

    def __len__(self) -> int:
        return len(self.stimuli)

    def __iter__(self) -> Iterator[Stimulus]:
        return iter(self.stimuli)

    def __getitem__(self, index: int) -> Stimulus:
        return self.stimuli[index]

    def head(self, n: int) -> StimulusCatalog:
        """First ``n`` stimuli, preserving provenance.

        For the built-in catalogue this is exactly the phoneme-balanced ``n``-item subset.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        return StimulusCatalog(
            source=self.source,
            stimuli=self.stimuli[:n],
            provenance={**self.provenance, "subset": f"head({n})", "n": min(n, len(self))},
        )

    def coverage(self) -> dict[str, dict[str, int]]:
        """Count how many stimuli exercise each onset / nucleus / coda category."""
        onset: dict[str, int] = {}
        nucleus: dict[str, int] = {}
        coda: dict[str, int] = {}
        for s in self.stimuli:
            d = s.decomposition
            onset[d.onset] = onset.get(d.onset, 0) + 1
            nucleus[d.nucleus] = nucleus.get(d.nucleus, 0) + 1
            coda[d.coda] = coda.get(d.coda, 0) + 1
        return {"onset": onset, "nucleus": nucleus, "coda": coda}

    def balance_score(self) -> dict[str, float]:
        """Per-position balance in [0, 1]: 1.0 means every category appears equally often.

        Defined as ``min(count) / max(count)`` over the categories actually used, times
        the fraction of the position's design inventory that is used at all. This is the
        quantity the phoneme-balanced calibration arm maximises.
        """
        cov = self.coverage()
        inventories = {
            "onset": len(ONSET_JAMO),
            "nucleus": len(NUCLEUS_JAMO),
            "coda": len(BUILTIN_CODAS),
        }
        out: dict[str, float] = {}
        for pos, counts in cov.items():
            if not counts:
                out[pos] = 0.0
                continue
            used = len(counts)
            evenness = min(counts.values()) / max(counts.values())
            out[pos] = evenness * (used / inventories[pos])
        return out


def build_balanced_catalog(n: int = DESIGN_SPACE_SIZE) -> StimulusCatalog:
    """Return the deterministic phoneme-balanced built-in catalogue of ``n`` stimuli.

    The order is fixed and contains no randomness, so ``head(k)`` of the result is the
    canonical balanced ``k``-item calibration list for every ``k <= n``.

    Raises
    ------
    ValueError
        If ``n`` is outside ``1..3192``.
    """
    if not 1 <= n <= DESIGN_SPACE_SIZE:
        raise ValueError(f"n must be in 1..{DESIGN_SPACE_SIZE}, got {n}")
    items: list[Stimulus] = []
    for i in range(n):
        onset = ONSET_JAMO[i % len(ONSET_JAMO)]
        nucleus = NUCLEUS_JAMO[i % len(NUCLEUS_JAMO)]
        coda = BUILTIN_CODAS[i % len(BUILTIN_CODAS)]
        syllable = compose_syllable(onset, nucleus, coda)
        items.append(
            Stimulus(
                stimulus_id=f"builtin-{i:04d}",
                syllable=syllable,
                source="builtin",
                speaker="synthetic_tts",
                structure=decompose_syllable(syllable).structure,
            )
        )
    return StimulusCatalog(
        source="builtin",
        stimuli=tuple(items),
        provenance={
            "design": "coprime cyclic balanced design over 19 onsets x 21 nuclei x 8 codas",
            "design_space_size": DESIGN_SPACE_SIZE,
            "n": n,
            "deterministic": True,
            "license": "none (generated from the Hangul inventory)",
            "audio": "browser speech synthesis; NOT level-calibrated, not clinical",
        },
    )


# --------------------------------------------------------------------------- KMSP corpus

#: e.g. "남자_가" -> ("남자", "가")
_SPEAKER_ANSWER = re.compile(r"^\s*(?P<gender>[^_]+)_(?P<syllable>.+?)\s*$")
_GENDER_MAP = {"남자": "male", "여자": "female"}


def kmsp_root() -> Path:
    """Local directory of the fetched primary corpus."""
    return raw_dir() / "korean_monosyllabic_speech"


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_kmsp_catalog(root: Path | None = None, source: Source | None = None) -> StimulusCatalog:
    """Load the primary corpus catalogue from a locally fetched copy.

    Reads only the metadata table; audio is referenced by path and never copied. The
    caller must already have satisfied the corpus's human notification requirement --
    :func:`~audire.data.fetch.fetch_source` enforces that at download time and this
    function re-checks it before reading.

    Raises
    ------
    FileNotFoundError
        If the corpus has not been fetched.
    ValueError
        If the row count does not match the registry's expected 726.
    """
    src = source or registry().get("korean_monosyllabic_speech")
    src.require_acknowledgement()
    if root is None:
        # Verify the manifest at consumption time. Skipped only when the caller supplies
        # an explicit root, which is the test/inspection path rather than a research run.
        require_verified(src.id)
    base = (root or kmsp_root()).resolve()
    meta_name = str(src.expected.get("metadata_file", "metadata.csv"))
    meta = base / meta_name
    if not meta.exists():
        raise FileNotFoundError(
            f"primary corpus metadata not found at {meta}. "
            f"Run `python scripts/fetch_data.py korean_monosyllabic_speech` first."
        )

    stimuli: list[Stimulus] = []
    with meta.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Header keys carry stray spaces in the published file; normalise for lookup.
        for raw_row in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in raw_row.items()}
            label = row.get("Speaker Gender_Pronounced Answer", "")
            m = _SPEAKER_ANSWER.match(label)
            if not m:
                raise ValueError(f"unparseable stimulus label {label!r} in {meta}")
            syllable = m.group("syllable")
            gender = _GENDER_MAP.get(m.group("gender"), m.group("gender"))
            acoustics = {
                key: val
                for key, val in (
                    ("duration_s", _float_or_none(row.get("Total Duration", ""))),
                    ("mean_pitch_hz", _float_or_none(row.get("Mean Pitch", ""))),
                    ("f1_hz", _float_or_none(row.get("F1", ""))),
                    ("f2_hz", _float_or_none(row.get("F2 (Hz)", ""))),
                    ("f3_hz", _float_or_none(row.get("F3 (Hz)", ""))),
                )
                if val is not None
            }
            stimuli.append(
                Stimulus(
                    stimulus_id=f"kmsp-{row.get('Audio Sample ID', len(stimuli))}",
                    syllable=syllable,
                    source="kmsp",
                    speaker=gender,
                    structure=row.get("Syllable", "") or decompose_syllable(syllable).structure,
                    audio_path=row.get("file_name") or None,
                    acoustics=acoustics or None,
                )
            )

    expected_rows = int(src.expected.get("n_utterances", 726))
    if len(stimuli) != expected_rows:
        raise ValueError(
            f"primary corpus row count mismatch: expected {expected_rows}, read {len(stimuli)}. "
            f"The pinned revision may have changed; re-verify data/sources.yaml."
        )
    return StimulusCatalog(
        source="kmsp",
        stimuli=tuple(stimuli),
        provenance={
            "source_id": src.id,
            "license": src.license,
            "revision": src.revision,
            "local_path": str(base),
            "redistribution_allowed": src.redistribution_allowed,
            "n": len(stimuli),
        },
    )


def phoneme_balanced_subset(catalog: StimulusCatalog, n: int) -> StimulusCatalog:
    """Greedily choose ``n`` stimuli from ``catalog`` maximising phoneme coverage.

    Deterministic: ties break on the stimulus's position in ``catalog``. Used by the
    calibration-length study (E5) as the "phoneme-balanced" selection arm for corpora
    whose native order is not already balanced.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    remaining = list(catalog.stimuli)
    chosen: list[Stimulus] = []
    seen: dict[str, dict[str, int]] = {"onset": {}, "nucleus": {}, "coda": {}}

    def gain(s: Stimulus) -> tuple[int, int, int]:
        d = s.decomposition
        # Prefer the stimulus whose categories are currently least represented.
        return (
            -seen["onset"].get(d.onset, 0),
            -seen["nucleus"].get(d.nucleus, 0),
            -seen["coda"].get(d.coda, 0),
        )

    while remaining and len(chosen) < n:
        best_idx = max(range(len(remaining)), key=lambda i: (gain(remaining[i]), -i))
        s = remaining.pop(best_idx)
        chosen.append(s)
        d = s.decomposition
        seen["onset"][d.onset] = seen["onset"].get(d.onset, 0) + 1
        seen["nucleus"][d.nucleus] = seen["nucleus"].get(d.nucleus, 0) + 1
        seen["coda"][d.coda] = seen["coda"].get(d.coda, 0) + 1

    return StimulusCatalog(
        source=catalog.source,
        stimuli=tuple(chosen),
        provenance={**catalog.provenance, "subset": f"phoneme_balanced({n})", "n": len(chosen)},
    )


def random_subset(catalog: StimulusCatalog, n: int, seed: int) -> StimulusCatalog:
    """Choose ``n`` stimuli uniformly at random with an explicit seed (E5 control arm)."""
    import numpy as np

    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(catalog))[: min(n, len(catalog))]
    return StimulusCatalog(
        source=catalog.source,
        stimuli=tuple(catalog.stimuli[int(i)] for i in sorted(idx)),
        provenance={**catalog.provenance, "subset": f"random({n}, seed={seed})", "n": len(idx)},
    )


def catalog_from_syllables(
    syllables: Sequence[str], *, source: StimulusSource = "builtin", speaker: str = "synthetic_tts"
) -> StimulusCatalog:
    """Wrap an explicit syllable list (e.g. a clinician-supplied list) as a catalogue."""
    return StimulusCatalog(
        source=source,
        stimuli=tuple(
            Stimulus(
                stimulus_id=f"custom-{i:04d}",
                syllable=s,
                source=source,
                speaker=speaker,
                structure=decompose_syllable(s).structure,
            )
            for i, s in enumerate(syllables)
        ),
        provenance={"design": "explicit syllable list", "n": len(syllables)},
    )
