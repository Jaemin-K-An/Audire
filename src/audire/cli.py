"""AUDIRE command-line interface.

Every research entrypoint in the Makefile routes through here, so the commands a fresh
evaluator runs are the same ones the results were produced with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from audire.config.logging import configure_logging, get_logger
from audire.config.paths import artifacts_dir, ensure_runtime_dirs

app = typer.Typer(
    name="audire",
    help="Personalized prediction of Korean word misrecognition for selective captioning.",
    no_args_is_help=True,
    add_completion=False,
)
log = get_logger(__name__)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)]


@app.callback()
def _main() -> None:
    configure_logging()
    ensure_runtime_dirs()


@app.command()
def simulate(
    config: ConfigOpt,
    out: Annotated[
        Path | None, typer.Option("--out", help="Where to write the cohort summary")
    ] = None,
) -> None:
    """Generate the synthetic cohorts declared in an experiment config."""
    from audire.experiments.runner import ExperimentConfig
    from audire.sim.cohort import build_cohort

    cfg = ExperimentConfig.load(config)
    summaries = []
    for seed in cfg.simulation.seeds:
        cohort = build_cohort(cfg.simulation, seed)
        summaries.append(cohort.summary())
        typer.echo(
            f"seed {seed}: {len(cohort)} listeners, "
            f"{sum(len(r.word_trials) for r in cohort.records)} word trials, "
            f"mishear rate {cohort.mishear_rate():.3f}"
        )
    target = out or (artifacts_dir() / f"{cfg.name}_cohorts.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"wrote {target}")


@app.command()
def evaluate(config: ConfigOpt) -> None:
    """Run the preregistered ablation, caption and threshold studies (RQ1-RQ3)."""
    from audire.experiments.runner import ExperimentConfig, run_experiment

    cfg = ExperimentConfig.load(config)
    result = run_experiment(cfg)
    summary = result["summary"]

    typer.echo(f"\nrun_id: {result['run_id']}   seeds: {summary['seeds']}\n")
    typer.echo(f"{'arm':26s} {'model':22s} {'PR-AUC':>16s} {'Brier':>9s} {'ECE':>8s}")
    typer.echo("-" * 86)
    for row in sorted(summary["arms"], key=lambda r: (r["model"], -r["pr_auc"]["mean"])):
        pr, br, ec = row["pr_auc"], row["brier"], row["ece"]
        typer.echo(
            f"{row['arm']:26s} {row['model']:22s} "
            f"{pr['mean']:6.4f}+-{pr['sd']:.4f} {br['mean']:9.4f} {ec['mean']:8.4f}"
        )

    typer.echo("\npaired listener-level contrasts (mean across seeds):")
    for c in summary["contrasts"]:
        d = c["difference"]
        typer.echo(
            f"  {c['metric']:7s} {c['arm']:24s} - {c['reference']:22s} = "
            f"{d['mean']:+.4f} +-{d['sd']:.4f}  "
            f"({c['n_seeds_excluding_zero']}/{c['n_seeds']} seeds exclude 0)"
        )
    typer.echo(f"\nartifacts: {len(result['artifacts'])} files under experiments/artifacts/")
    typer.echo(f"caveat: {summary['headline']['caveat']}")


@app.command("caption-eval")
def caption_eval(config: ConfigOpt) -> None:
    """Print the caption Pareto frontier from a completed or fresh run."""
    from audire.experiments.runner import ExperimentConfig, run_experiment

    cfg = ExperimentConfig.load(config)
    summary = run_experiment(cfg)["summary"]
    for mode in ("per_listener", "pooled"):
        typer.echo(f"\n--- {mode} budget ---")
        typer.echo(f"{'strategy':32s} {'budget':>7s} {'recall':>16s} {'worst listener':>16s}")
        typer.echo("-" * 78)
        for row in summary["caption_frontier"]:
            if row["budget_mode"] != mode:
                continue
            r, w = row["misheard_recall"], row["worst_listener_recall"]
            typer.echo(
                f"{row['strategy']:32s} {row['budget']:7.2f} "
                f"{r['mean']:6.4f}+-{r['sd']:.4f} {w['mean']:10.4f}"
            )


@app.command()
def sensitivity(config: ConfigOpt) -> None:
    """E11 — 개인 특이성 x 교정 길이 민감도 스윕을 실행합니다."""
    from audire.experiments.sensitivity import SensitivityConfig, run_sensitivity

    cfg = SensitivityConfig.load(config)
    result = run_sensitivity(cfg)
    summary = result["summary"]

    typer.echo(f"\nrun_id: {result['run_id']}   격자 칸 수: {summary['n_cells']}\n")
    typer.echo(
        f"{'집중도':>8s} {'교정길이':>9s} {'arm':26s} {'PR-AUC':>8s} "
        f"{'재현율':>8s} {'단어길이대비':>12s} {'이긴시드':>9s}"
    )
    typer.echo("-" * 90)
    for g in summary["grid"]:
        typer.echo(
            f"{g['dirichlet_concentration']:8.1f} {g['n_calibration_trials']:9d} "
            f"{g['arm']:26s} {g['pr_auc_mean']:8.4f} {g['misheard_recall_mean']:8.4f} "
            f"{g['recall_over_word_length_mean']:+12.4f} "
            f"{g['n_seeds_beating_word_length']:4d}/{g['n_seeds']:<4d}"
        )
    head = summary["headline"]
    typer.echo(
        f"\n개인화가 모든 시드에서 단어길이 휴리스틱을 이긴 칸: "
        f"{head['n_cells_where_personalization_always_beats_word_length']}"
        f"/{head['n_combined_cells']}"
    )
    for w in head["winning_cells"]:
        typer.echo(
            f"  집중도={w['dirichlet_concentration']} 교정길이={w['n_calibration_trials']} "
            f"이득={w['gain']:+.4f}"
        )
    typer.echo(f"\n단서: {head['caveat']}")


@app.command()
def runs() -> None:
    """List recorded experiment runs with their provenance."""
    from audire.experiments.registry import load_runs

    entries = load_runs()
    if not entries:
        typer.echo("no runs recorded yet; try `audire evaluate -c experiments/configs/smoke.yaml`")
        raise typer.Exit(0)
    typer.echo(f"{'run_id':44s} {'status':10s} {'git':10s} {'dirty':6s} seeds")
    typer.echo("-" * 92)
    for r in entries:
        typer.echo(
            f"{r['run_id']:44s} {r['status']:10s} {r['git_sha'][:8]:10s} "
            f"{r['git_dirty']!s:6s} {r['seeds']}"
        )


@app.command()
def figures(
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Defaults to the latest run")
    ] = None,
    all_runs: Annotated[
        bool, typer.Option("--all", help="Regenerate for every completed run")
    ] = False,
) -> None:
    """Regenerate figures and tables from recorded experiment artifacts."""
    from audire.experiments.figures import regenerate

    paths = regenerate(run_id=run_id, all_runs=all_runs)
    for p in paths:
        typer.echo(f"wrote {p}")
    if not paths:
        typer.echo("nothing to regenerate; run `audire evaluate` first")


@app.command()
def profile_summary(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Print a stored hearing profile's derived measures and what is missing."""
    from audire.profile.schema import HearingProfile

    p = HearingProfile.model_validate_json(path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(p.summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
