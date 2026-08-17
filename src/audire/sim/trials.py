"""Synthetic calibration and word-level trial generation.

Two trial types are produced:

**Calibration trials** present a single monosyllable and record what the listener
"reported". These are exactly what a real calibration session produces, so the estimated
:class:`~audire.confusion.profile.ConfusionProfile` is built by the same code path for
synthetic and real listeners.

**Word trials** present a multi-syllable word in a stated acoustic context and record
whether the listener identified it. The word-level outcome is deliberately **not** the
phoneme-independence product that ``R_phon`` assumes (docs/RISK_REGISTER.md S2):

1. perceived jamo are sampled independently per position from the listener's true matrices;
2. the perceived word form is reconstructed;
3. if the form matches the target, the word is identified;
4. if it does not, the listener may still identify the word through *lexical repair*,
   with a probability that depends on how many segments were wrong and on word length.

Step 4 introduces a dependence on word length and on error count that no per-phoneme
independence model can express, so comparing the two is informative rather than circular.
Acoustic context (SNR, speaker) shifts the diagonal probabilities on the logit scale
before sampling.

Every emitted record carries ``is_synthetic=True``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from audire.confusion.grouping import NUCLEUS_SHAPE, ONSET_MANNER, neutralise_coda
from audire.confusion.profile import POSITIONS, CalibrationTrial
from audire.data.stimuli import StimulusCatalog, build_balanced_catalog
from audire.hangul.inventory import NO_CODA, NO_RESPONSE, Position, categories_for
from audire.hangul.syllable import compose_syllable, decompose_syllable, is_hangul_syllable
from audire.sim.config import SimulationConfig
from audire.sim.lexicon import Lexicon, build_lexicon
from audire.sim.listener import SyntheticListener, TrueConfusion

FloatArray = npt.NDArray[np.float64]


# --------------------------------------------------------------------------- context


def _context_shifted(
    true_confusion: TrueConfusion,
    position: Position,
    target_index: int,
    logit_shift: float,
) -> FloatArray:
    """Return a row with its diagonal shifted by ``logit_shift`` on the logit scale.

    The remaining mass is rescaled proportionally, so the row stays stochastic and the
    *relative* structure of the confusions is preserved while the overall error rate moves.
    """
    row: FloatArray = true_confusion.matrices[position][target_index].copy()
    if logit_shift == 0.0:
        return row
    diag = float(row[target_index])
    diag = min(max(diag, 1e-6), 1 - 1e-6)
    new_diag = float(1.0 / (1.0 + np.exp(-(np.log(diag / (1 - diag)) + logit_shift))))
    remaining = 1.0 - new_diag
    others = row.copy()
    others[target_index] = 0.0
    total = float(others.sum())
    out: FloatArray = np.zeros_like(row)
    if total > 0:
        out = others * (remaining / total)
    out[target_index] = new_diag
    return out


def context_logit_shift(cfg: SimulationConfig, snr_db: float, speaker: str) -> float:
    """Total logit-scale shift applied to diagonals for one acoustic context."""
    t = cfg.trials
    snr_term = t.snr_logit_per_db * (snr_db - t.reference_snr_db)
    speaker_term = t.speaker_effects.get(speaker, 0.0)
    return float(snr_term + speaker_term)


# --------------------------------------------------------------------------- calibration


def simulate_calibration(
    listener: SyntheticListener,
    catalog: StimulusCatalog,
    rng: np.random.Generator,
    cfg: SimulationConfig,
    *,
    snr_db: float | None = None,
) -> list[CalibrationTrial]:
    """Present ``catalog`` to ``listener`` and return the raw responses.

    The returned trials are the *only* thing the estimator is allowed to see; the true
    confusion structure stays with the listener object.
    """
    trials: list[CalibrationTrial] = []
    for stim in catalog:
        shift = context_logit_shift(
            cfg, cfg.trials.reference_snr_db if snr_db is None else snr_db, stim.speaker
        )
        syl = decompose_syllable(stim.syllable)
        perceived: dict[Position, str] = {}
        for position in POSITIONS:
            targets = categories_for(position, axis="target")
            perceived_labels = categories_for(position, axis="perceived")
            idx = targets.index(syl.get(position))
            probs = _context_shifted(listener.true_confusion, position, idx, shift)
            perceived[position] = perceived_labels[int(rng.choice(len(probs), p=probs))]

        response = _render_response(perceived)
        trials.append(
            CalibrationTrial(
                stimulus_id=stim.stimulus_id,
                target=stim.syllable,
                response=response,
                condition=f"snr={snr_db if snr_db is not None else cfg.trials.reference_snr_db}"
                f";speaker={stim.speaker}",
            )
        )
    return trials


def _render_response(perceived: dict[Position, str]) -> str:
    """Turn sampled per-position categories into the string a listener would have typed.

    A ``NO_RESPONSE`` at any position makes the whole answer unusable, which is what
    happens when a listener cannot report a syllable at all.
    """
    if any(perceived[p] == NO_RESPONSE for p in POSITIONS):
        return ""
    return compose_syllable(
        perceived[Position.ONSET], perceived[Position.NUCLEUS], perceived[Position.CODA]
    )


# --------------------------------------------------------------------------- words


@dataclass(frozen=True, slots=True)
class WordTrial:
    """One word-level trial: the ground truth the risk models are evaluated against."""

    listener_id: str
    word: str
    n_syllables: int
    snr_db: float
    speaker: str
    #: The form the listener perceived. Empty when nothing usable was perceived.
    perceived_word: str
    #: Number of positions whose perceived category differed from the target.
    n_segment_errors: int
    #: Whether lexical repair rescued a word that had segmental errors.
    repaired: bool
    #: The outcome the models predict.
    misheard: bool
    is_synthetic: bool = True

    def as_row(self) -> dict[str, Any]:
        return {
            "listener_id": self.listener_id,
            "word": self.word,
            "n_syllables": self.n_syllables,
            "snr_db": self.snr_db,
            "speaker": self.speaker,
            "perceived_word": self.perceived_word,
            "n_segment_errors": self.n_segment_errors,
            "repaired": self.repaired,
            "misheard": self.misheard,
            "is_synthetic": self.is_synthetic,
        }


@dataclass(slots=True)
class Vocabulary:
    """A shared word list. Words repeat across listeners, as they do in real captioning."""

    words: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.words)


def build_vocabulary(cfg: SimulationConfig, seed: int) -> Vocabulary:
    """Build the shared vocabulary for a run.

    ``source="synthetic"`` composes words from the balanced monosyllable design space.
    ``source="zeroth"`` draws real Korean eojeol from the locally fetched CC BY-licensed
    Zeroth-Korean test transcripts, and raises if they have not been fetched.
    """
    if cfg.words.source == "zeroth":
        return _zeroth_vocabulary(cfg, seed)

    rng = np.random.default_rng(seed ^ 0x5EED)
    syllables = [s.syllable for s in build_balanced_catalog()]
    counts = sorted(cfg.words.syllable_count_weights)
    weights = np.array([cfg.words.syllable_count_weights[c] for c in counts], dtype=np.float64)
    weights = weights / weights.sum()

    seen: set[str] = set()
    words: list[str] = []
    while len(words) < cfg.words.vocabulary_size:
        n = int(rng.choice(counts, p=weights))
        w = "".join(syllables[int(i)] for i in rng.integers(0, len(syllables), size=n))
        if w not in seen:
            seen.add(w)
            words.append(w)
    return Vocabulary(
        words=tuple(words),
        provenance={
            "source": "synthetic",
            "design_space": "audire.data.stimuli.build_balanced_catalog",
            "seed": seed,
            "size": len(words),
            "license": "none (generated)",
        },
    )


def _zeroth_vocabulary(cfg: SimulationConfig, seed: int) -> Vocabulary:
    from audire.data.zeroth import load_zeroth_transcripts

    transcripts = load_zeroth_transcripts()
    rng = np.random.default_rng(seed ^ 0x5EED)
    tokens: list[str] = []
    seen: set[str] = set()
    for text in transcripts:
        for tok in text.split():
            word = "".join(ch for ch in tok if is_hangul_syllable(ch))
            if word and word not in seen:
                seen.add(word)
                tokens.append(word)
    if not tokens:
        raise ValueError("no Hangul word tokens found in the Zeroth-Korean transcripts")
    idx = rng.permutation(len(tokens))[: cfg.words.vocabulary_size]
    return Vocabulary(
        words=tuple(tokens[int(i)] for i in sorted(idx)),
        provenance={
            "source": "zeroth_korean_test",
            "license": "CC-BY-4.0",
            "seed": seed,
            "size": int(min(len(tokens), cfg.words.vocabulary_size)),
            "n_transcripts": len(transcripts),
        },
    )


def _repair_probability(cfg: SimulationConfig, n_syllables: int, n_errors: int) -> float:
    t = cfg.trials
    if n_errors == 0:
        return 0.0
    if n_errors >= 2:
        return t.lexical_repair_multi_error
    p = t.lexical_repair_base + t.lexical_repair_per_extra_syllable * max(0, n_syllables - 1)
    return float(min(p, t.lexical_repair_max))


@dataclass(frozen=True, slots=True)
class SegmentError:
    """한 위치에서 일어난 하나의 치환. V2 복구 모형의 입력입니다."""

    position: Position
    target: str
    perceived: str


def _same_phonological_class(err: SegmentError) -> bool:
    """치환이 같은 음운 부류 안에서 일어났는가.

    초성은 조음방법, 중성은 활음형, 종성은 7종 중화형을 기준으로 봅니다. 부류가 같으면
    청자가 후보를 좁히기 쉬우므로 복구가 상대적으로 수월합니다.
    """
    if err.perceived == NO_RESPONSE:
        return False
    if err.position is Position.ONSET:
        return ONSET_MANNER.get(err.target) is ONSET_MANNER.get(err.perceived)
    if err.position is Position.NUCLEUS:
        return NUCLEUS_SHAPE.get(err.target) is NUCLEUS_SHAPE.get(err.perceived)
    if NO_CODA in (err.target, err.perceived):
        return err.target == err.perceived
    return neutralise_coda(err.target) == neutralise_coda(err.perceived)


def repair_probability_v2(
    cfg: SimulationConfig,
    target: str,
    perceived: str,
    errors: Sequence[SegmentError],
    n_syllables: int,
    lexicon: Lexicon,
) -> float:
    """Simulator V2 의 복구 확률: 오류 **위치와 음운적 거리**에 조건화합니다.

    V1 의 :func:`_repair_probability` 는 오류 개수와 음절 수만 봅니다. 그 결과 "각→닥" 과
    "각→삭" 이 구분되지 않고 단어 수준 신호가 남지 않습니다(docs/RESULTS.md §14).

    여기서는 오류가 난 위치별 복구 용이도를 평균하고, 같은 부류 안의 치환에는 가산점을
    줍니다. 종성 중화가 있는 한국어에서 종성 오류는 문맥에서 재구성되기 쉽고, 모음 오류는
    어휘 변별 부담이 커 그렇지 않습니다. 단어마다 위치 구성이 다르고 청취자마다 틀리는
    위치가 다르므로, 이 항이 단어×청취자 구조를 만듭니다.
    """
    v2 = cfg.lexical_repair_v2
    if not errors:
        return 0.0

    weights = {
        Position.ONSET: float(v2.onset_recovery),
        Position.NUCLEUS: float(v2.nucleus_recovery),
        Position.CODA: float(v2.coda_recovery),
    }
    per_error = [
        weights[e.position] * (float(v2.same_class_bonus) if _same_phonological_class(e) else 1.0)
        for e in errors
    ]
    # 평균을 씁니다. 합을 쓰면 오류가 많을수록 복구가 쉬워지는 역전이 생깁니다.
    p = float(v2.base_repair) * float(np.mean(per_error))
    p *= float(v2.repair_per_extra_error_decay ** (len(errors) - 1))
    p += float(v2.repair_per_extra_syllable) * max(0, n_syllables - 1)

    # 어휘 함정. 이 어휘에서는 거의 발동하지 않으나(측정 0.3%), 실제 한국어 어휘를 붙이면
    # 의미를 갖도록 남겨 둡니다. 발동 여부는 시행에 기록됩니다.
    if lexicon.is_word(perceived) and perceived != target:
        p *= float(v2.lexical_trap_multiplier)

    return float(min(max(p, v2.repair_floor), v2.repair_ceiling))


def simulate_word_trial(
    listener: SyntheticListener,
    word: str,
    rng: np.random.Generator,
    cfg: SimulationConfig,
    *,
    snr_db: float,
    speaker: str,
    lexicon: Lexicon | None = None,
) -> WordTrial:
    """Simulate one word-level perception outcome."""
    shift = context_logit_shift(cfg, snr_db, speaker)
    perceived_syllables: list[str] = []
    n_errors = 0
    usable = True

    syllables = [ch for ch in word if is_hangul_syllable(ch)]
    # 위치별 오류를 그대로 모읍니다. V1 은 개수만 쓰지만 V2 는 어느 위치에서 무엇으로
    # 바뀌었는지를 씁니다.
    segment_errors: list[SegmentError] = []
    for ch in syllables:
        syl = decompose_syllable(ch)
        got: dict[Position, str] = {}
        for position in POSITIONS:
            targets = categories_for(position, axis="target")
            labels = categories_for(position, axis="perceived")
            idx = targets.index(syl.get(position))
            probs = _context_shifted(listener.true_confusion, position, idx, shift)
            choice = labels[int(rng.choice(len(probs), p=probs))]
            got[position] = choice
            if choice != syl.get(position):
                n_errors += 1
                segment_errors.append(
                    SegmentError(position=position, target=syl.get(position), perceived=choice)
                )
        if any(got[p] == NO_RESPONSE for p in POSITIONS):
            usable = False
            break
        perceived_syllables.append(
            compose_syllable(got[Position.ONSET], got[Position.NUCLEUS], got[Position.CODA])
        )

    if not usable:
        return WordTrial(
            listener_id=listener.listener_id,
            word=word,
            n_syllables=len(syllables),
            snr_db=snr_db,
            speaker=speaker,
            perceived_word="",
            n_segment_errors=n_errors,
            repaired=False,
            misheard=True,
        )

    perceived_word = "".join(perceived_syllables)
    exact = perceived_word == word
    repaired = False
    if not exact:
        if cfg.simulator_version == "v2":
            if lexicon is None:
                raise ValueError(
                    "simulator_version='v2' 는 어휘 구조를 필요로 합니다; lexicon 이 없습니다"
                )
            p_repair = repair_probability_v2(
                cfg, word, perceived_word, segment_errors, len(syllables), lexicon
            )
        else:
            p_repair = _repair_probability(cfg, len(syllables), n_errors)
        repaired = bool(rng.random() < p_repair)

    return WordTrial(
        listener_id=listener.listener_id,
        word=word,
        n_syllables=len(syllables),
        snr_db=snr_db,
        speaker=speaker,
        perceived_word=perceived_word,
        n_segment_errors=n_errors,
        repaired=repaired,
        misheard=not (exact or repaired),
    )


def simulate_word_trials(
    listener: SyntheticListener,
    vocabulary: Vocabulary,
    rng: np.random.Generator,
    cfg: SimulationConfig,
    lexicon: Lexicon | None = None,
) -> list[WordTrial]:
    """Simulate ``cfg.n_word_trials`` word trials for one listener across the conditions.

    ``lexicon`` 은 V2 에서만 쓰입니다. 어휘 구조는 청취자와 무관하므로 코호트 단위로 한 번
    만들어 넘겨받습니다 — 청취자마다 다시 만들면 같은 결과에 시간만 듭니다.
    """
    if cfg.simulator_version == "v2" and lexicon is None:
        lexicon = build_lexicon(vocabulary.words, vocabulary.provenance)
    trials: list[WordTrial] = []
    n = cfg.n_word_trials
    word_idx = rng.integers(0, len(vocabulary), size=n)
    snr_idx = rng.integers(0, len(cfg.snr_conditions_db), size=n)
    spk_idx = rng.integers(0, len(cfg.speakers), size=n)
    for k in range(n):
        trials.append(
            simulate_word_trial(
                listener,
                vocabulary.words[int(word_idx[k])],
                rng,
                cfg,
                snr_db=cfg.snr_conditions_db[int(snr_idx[k])],
                speaker=cfg.speakers[int(spk_idx[k])],
                lexicon=lexicon,
            )
        )
    return trials
