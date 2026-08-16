"""Metrics, listener-level splits, bootstrap intervals and ablation runners."""

from audire.eval.ablation import (
    DEFAULT_CONTRASTS,
    ArmResult,
    ContrastResult,
    cohort_matrix,
    contrast,
    evaluate_arm,
    run_ablation,
)
from audire.eval.bootstrap import (
    DEFAULT_N_BOOTSTRAP,
    Interval,
    bootstrap_metric,
    metric_statistic,
    paired_bootstrap_difference,
)
from audire.eval.caption import (
    BudgetPoint,
    ThresholdComparison,
    budget_frontier,
    compare_strategies,
    compare_thresholds,
    pareto_table,
)
from audire.eval.metrics import (
    DEFAULT_ECE_BINS,
    ClassificationMetrics,
    compute_metrics,
    expected_calibration_error,
    prevalence_baseline_metrics,
    reliability_curve,
)
from audire.eval.splits import (
    Fold,
    LeakageError,
    LeakySplitter,
    assert_no_listener_leakage,
    leave_one_listener_out,
    listener_folds,
)

__all__ = [
    "DEFAULT_CONTRASTS",
    "DEFAULT_ECE_BINS",
    "DEFAULT_N_BOOTSTRAP",
    "ArmResult",
    "BudgetPoint",
    "ClassificationMetrics",
    "ContrastResult",
    "Fold",
    "Interval",
    "LeakageError",
    "LeakySplitter",
    "ThresholdComparison",
    "assert_no_listener_leakage",
    "bootstrap_metric",
    "budget_frontier",
    "cohort_matrix",
    "compare_strategies",
    "compare_thresholds",
    "compute_metrics",
    "contrast",
    "evaluate_arm",
    "expected_calibration_error",
    "leave_one_listener_out",
    "listener_folds",
    "metric_statistic",
    "paired_bootstrap_difference",
    "pareto_table",
    "prevalence_baseline_metrics",
    "reliability_curve",
    "run_ablation",
]
