"""Reproducible synthetic listener and trial generation.

Everything produced by this package carries ``is_synthetic=True``. Synthetic data is a
test instrument for pipeline validation, parameter recovery and design sensitivity. It is
**never** clinical evidence — see docs/RISK_REGISTER.md S1 and S2.
"""

from audire.sim.cohort import Cohort, ListenerRecord, build_cohort
from audire.sim.config import (
    AudiogramModel,
    ConfusionModel,
    SeverityMix,
    SimulationConfig,
    SpeechScoreModel,
    TrialModel,
    WordSourceModel,
)
from audire.sim.listener import (
    STRATUM_ORDER,
    SyntheticListener,
    TrueConfusion,
    generate_cohort,
    generate_listener,
)
from audire.sim.similarity import similarity, similarity_matrix
from audire.sim.trials import (
    Vocabulary,
    WordTrial,
    build_vocabulary,
    context_logit_shift,
    simulate_calibration,
    simulate_word_trial,
    simulate_word_trials,
)

__all__ = [
    "STRATUM_ORDER",
    "AudiogramModel",
    "Cohort",
    "ConfusionModel",
    "ListenerRecord",
    "SeverityMix",
    "SimulationConfig",
    "SpeechScoreModel",
    "SyntheticListener",
    "TrialModel",
    "TrueConfusion",
    "Vocabulary",
    "WordSourceModel",
    "WordTrial",
    "build_cohort",
    "build_vocabulary",
    "context_logit_shift",
    "generate_cohort",
    "generate_listener",
    "similarity",
    "similarity_matrix",
    "simulate_calibration",
    "simulate_word_trial",
    "simulate_word_trials",
]
