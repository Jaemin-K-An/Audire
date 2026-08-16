"""Phonological similarity kernels used to shape simulated confusion structure.

The kernel answers "how confusable are these two categories a priori?" purely from shared
phonological features. It carries **no** numeric finding: it is a structural device that
makes simulated confusions concentrate within phonetic classes, which is the qualitative
pattern reported by Ma et al. (2026). The strength of that concentration is a configurable
parameter (``ConfusionModel.similarity_beta_*``), not a constant hidden in this file.

The same kernel is used by the risk models as a *feature*, so it must not encode anything
listener-specific.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt

from audire.confusion.grouping import (
    NUCLEUS_SHAPE,
    ONSET_MANNER,
    ONSET_PHONATION,
    ONSET_PLACE,
    ROUNDED_NUCLEI,
    neutralise_coda,
)
from audire.hangul.inventory import NO_CODA, NO_RESPONSE, Position, categories_for

FloatArray = npt.NDArray[np.float64]


def onset_similarity(a: str, b: str) -> float:
    """Similarity of two onset jamo in [0, 1]: the fraction of shared feature values."""
    if a == b:
        return 1.0
    shared = sum(
        (
            ONSET_PLACE[a] is ONSET_PLACE[b],
            ONSET_MANNER[a] is ONSET_MANNER[b],
            ONSET_PHONATION[a] is ONSET_PHONATION[b],
        )
    )
    return shared / 3.0


def nucleus_similarity(a: str, b: str) -> float:
    """Similarity of two nucleus jamo in [0, 1]: shared onglide shape and rounding."""
    if a == b:
        return 1.0
    shared = sum(
        (
            NUCLEUS_SHAPE[a] is NUCLEUS_SHAPE[b],
            (a in ROUNDED_NUCLEI) == (b in ROUNDED_NUCLEI),
        )
    )
    return shared / 2.0


def coda_similarity(a: str, b: str) -> float:
    """Similarity of two coda categories in [0, 1].

    ``NO_CODA`` is maximally dissimilar from every consonant: adding or dropping a coda is
    a categorically different event from swapping one coda for another.
    """
    if a == b:
        return 1.0
    if a == NO_CODA or b == NO_CODA:
        return 0.0
    return 1.0 if neutralise_coda(a) == neutralise_coda(b) else 0.25


_KERNELS = {
    Position.ONSET: onset_similarity,
    Position.NUCLEUS: nucleus_similarity,
    Position.CODA: coda_similarity,
}


def similarity(position: Position, a: str, b: str) -> float:
    """Similarity of two categories at ``position``."""
    if NO_RESPONSE in (a, b):
        return 0.0
    return _KERNELS[position](a, b)


@lru_cache(maxsize=8)
def similarity_matrix(position: Position) -> FloatArray:
    """Full ``(n_target, n_perceived)`` similarity matrix for ``position``.

    Rows follow the target alphabet, columns the perceived alphabet (which includes
    :data:`~audire.hangul.inventory.NO_RESPONSE`, whose column is all zeros).
    """
    targets = categories_for(position, axis="target")
    perceived = categories_for(position, axis="perceived")
    out = np.zeros((len(targets), len(perceived)), dtype=np.float64)
    for i, t in enumerate(targets):
        for j, p in enumerate(perceived):
            out[i, j] = similarity(position, t, p)
    return out
