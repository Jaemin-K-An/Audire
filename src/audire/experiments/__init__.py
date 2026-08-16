"""Config-driven experiment execution, provenance registry and figure regeneration."""

from audire.experiments.registry import (
    RunRecord,
    append_run,
    fail_run,
    finish_run,
    git_sha,
    load_runs,
    new_run,
    save_artifact,
    tracked_run,
)
from audire.experiments.runner import ExperimentConfig, run_experiment
from audire.experiments.sensitivity import SensitivityConfig, run_sensitivity

__all__ = [
    "ExperimentConfig",
    "RunRecord",
    "SensitivityConfig",
    "append_run",
    "fail_run",
    "finish_run",
    "git_sha",
    "load_runs",
    "new_run",
    "run_experiment",
    "run_sensitivity",
    "save_artifact",
    "tracked_run",
]
