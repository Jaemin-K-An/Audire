"""Configuration, filesystem layout and structured logging."""

from audire.config.logging import configure_logging, get_logger
from audire.config.paths import (
    artifacts_dir,
    data_dir,
    ensure_runtime_dirs,
    experiment_configs_dir,
    experiments_dir,
    literature_dir,
    manifests_dir,
    models_dir,
    private_dir,
    processed_dir,
    raw_dir,
    repo_root,
    sources_file,
)

__all__ = [
    "artifacts_dir",
    "configure_logging",
    "data_dir",
    "ensure_runtime_dirs",
    "experiment_configs_dir",
    "experiments_dir",
    "get_logger",
    "literature_dir",
    "manifests_dir",
    "models_dir",
    "private_dir",
    "processed_dir",
    "raw_dir",
    "repo_root",
    "sources_file",
]
