# ruff: noqa: E501 - the ASS Format/Style header lines are protocol-mandated single lines
"""Caption exporters: SRT, ASS and JSON.

All three are produced from the same :class:`~audire.caption.word.WordRisk` list, so what
a viewer sees, what a subtitle player renders and what a researcher analyses cannot drift
apart.

* **SRT** is the portable format. Only displayed words appear; hidden words leave gaps.
* **ASS** carries the visual emphasis: risk is mapped to a colour ramp so a viewer can see
  *how* uncertain each captioned word is.
* **JSON** carries everything, including hidden words, per-word explanations, the policy,
  the model version and the ASR confidence — this is the research artifact.

Cue timing is validated: cues are sorted, non-overlapping and have positive duration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from audire.caption.word import WordRisk

#: Minimum on-screen duration for a cue, in seconds. Very short ASR words are extended so
#: that a caption is readable; extension never overlaps the following cue.
MIN_CUE_DURATION_S = 0.4

#: Gap below which consecutive shown words are merged into one cue.
MERGE_GAP_S = 0.35

#: Maximum number of words in a merged cue.
MAX_WORDS_PER_CUE = 8


@dataclass(frozen=True, slots=True)
class Cue:
    """One subtitle cue."""

    index: int
    start_s: float
    end_s: float
    words: tuple[WordRisk, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def max_risk(self) -> float:
        return max((w.listener_risk for w in self.words), default=0.0)


def build_cues(
    words: list[WordRisk],
    *,
    merge_gap_s: float = MERGE_GAP_S,
    max_words: int = MAX_WORDS_PER_CUE,
    min_duration_s: float = MIN_CUE_DURATION_S,
) -> list[Cue]:
    """Group displayed words into non-overlapping, time-ordered cues."""
    shown = sorted((w for w in words if w.is_shown), key=lambda w: (w.start_s, w.end_s))
    if not shown:
        return []

    groups: list[list[WordRisk]] = [[shown[0]]]
    for w in shown[1:]:
        prev = groups[-1][-1]
        if w.start_s - prev.end_s <= merge_gap_s and len(groups[-1]) < max_words:
            groups[-1].append(w)
        else:
            groups.append([w])

    cues: list[Cue] = []
    for i, group in enumerate(groups, start=1):
        start = group[0].start_s
        end = max(group[-1].end_s, start + min_duration_s)
        cues.append(Cue(index=i, start_s=start, end_s=end, words=tuple(group)))

    # Trim any overlap introduced by the minimum-duration extension.
    for i in range(len(cues) - 1):
        if cues[i].end_s > cues[i + 1].start_s:
            cues[i] = Cue(
                index=cues[i].index,
                start_s=cues[i].start_s,
                end_s=max(cues[i].start_s, cues[i + 1].start_s - 0.001),
                words=cues[i].words,
            )
    validate_cues(cues)
    return cues


def validate_cues(cues: list[Cue]) -> None:
    """Raise if cues are out of order, overlapping, or have non-positive duration."""
    for i, cue in enumerate(cues):
        if cue.end_s <= cue.start_s:
            raise ValueError(f"cue {cue.index} has non-positive duration")
        if i and cue.start_s < cues[i - 1].end_s:
            raise ValueError(
                f"cue {cue.index} starts at {cue.start_s:.3f} before cue "
                f"{cues[i - 1].index} ends at {cues[i - 1].end_s:.3f}"
            )


# --------------------------------------------------------------------------- SRT


def _srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("timestamps cannot be negative")
    ms_total = round(seconds * 1000)
    ms = ms_total % 1000
    s = (ms_total // 1000) % 60
    m = (ms_total // 60_000) % 60
    h = ms_total // 3_600_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(words: list[WordRisk], **cue_kwargs: Any) -> str:
    """Render displayed words as SubRip text. Hidden words are simply absent.

    Each cue block is terminated by a blank line, including the last, which is what
    subtitle players expect.
    """
    blocks = [
        f"{cue.index}\n"
        f"{_srt_timestamp(cue.start_s)} --> {_srt_timestamp(cue.end_s)}\n"
        f"{cue.text}\n\n"
        for cue in build_cues(words, **cue_kwargs)
    ]
    return "".join(blocks)


# --------------------------------------------------------------------------- ASS


#: Risk bands mapped to ASS colours (``&HAABBGGRR``, alpha first, then BGR).
#: Chosen for luminance separation so the ramp survives greyscale rendering.
_RISK_COLOURS: tuple[tuple[float, str, str], ...] = (
    (0.35, "&H00FFFFFF", "low"),  # white
    (0.60, "&H0080D4FF", "moderate"),  # amber
    (1.01, "&H004040FF", "high"),  # red
)


def _risk_colour(risk: float) -> str:
    for upper, colour, _ in _RISK_COLOURS:
        if risk < upper:
            return colour
    return _RISK_COLOURS[-1][1]  # pragma: no cover - the last band ends above 1.0


def _ass_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("timestamps cannot be negative")
    cs_total = round(seconds * 100)
    cs = cs_total % 100
    s = (cs_total // 100) % 60
    m = (cs_total // 6000) % 60
    h = cs_total // 360_000
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


ASS_HEADER = """[Script Info]
Title: AUDIRE selective captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Audire,Noto Sans CJK KR,64,&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,0,0,0,0,100,100,0,0,3,3,1,2,80,80,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def to_ass(words: list[WordRisk], **cue_kwargs: Any) -> str:
    """Render displayed words as Advanced SubStation Alpha with per-word risk colouring."""
    lines = [ASS_HEADER]
    for cue in build_cues(words, **cue_kwargs):
        coloured = "".join(
            f"{{\\c{_risk_colour(w.listener_risk)}}}{_ass_escape(w.text)} " for w in cue.words
        ).rstrip()
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(cue.start_s)},{_ass_timestamp(cue.end_s)},"
            f"Audire,,0,0,0,,{coloured}"
        )
    return "\n".join(lines) + "\n"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


# --------------------------------------------------------------------------- JSON


def to_json(
    words: list[WordRisk],
    *,
    listener_id: str,
    policy: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    indent: int = 2,
) -> str:
    """Full research export: every word, shown or hidden, with its explanation.

    This is the artifact that makes a caption result auditable — it records the policy,
    the model version and arm, the per-word risk *and* the ASR confidence separately, and
    the caption ratio actually achieved.
    """
    from audire.caption.policy import caption_ratio, caption_reduction_ratio

    payload = {
        "schema": "audire.caption.v1",
        "listener_id": listener_id,
        "policy": policy,
        "n_words": len(words),
        "n_shown": sum(w.is_shown for w in words),
        "caption_ratio": caption_ratio(words),
        "caption_reduction_ratio": caption_reduction_ratio(words),
        "provenance": provenance or {},
        "disclaimer": (
            "AUDIRE is research and accessibility software, not a medical device. "
            "Listener risk and ASR confidence are distinct quantities."
        ),
        "words": [w.to_dict() for w in words],
    }
    return json.dumps(payload, ensure_ascii=False, indent=indent)
