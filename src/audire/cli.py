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
from audire.config.paths import artifacts_dir, ensure_runtime_dirs, private_dir

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
    # The model column is essential: the same contrast is computed for every model
    # family, and without it two different rows look like a duplicated result.
    for c in sorted(summary["contrasts"], key=lambda c: (c["model"], c["metric"], c["arm"])):
        d = c["difference"]
        typer.echo(
            f"  {c['metric']:7s} {c['model']:20s} {c['arm']:24s} - {c['reference']:22s} = "
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
    # 예산과 모델을 반드시 함께 표시한다. 빼면 서로 다른 조건이 중복 행처럼 보인다.
    typer.echo(
        f"{'집중도':>7s} {'교정':>5s} {'예산':>5s} {'model':18s} {'arm':26s} "
        f"{'PR-AUC':>8s} {'재현율':>8s} {'최하위':>8s} {'길이대비':>10s} {'이긴시드':>9s}"
    )
    typer.echo("-" * 122)
    for g in summary["grid"]:
        typer.echo(
            f"{g['dirichlet_concentration']:7.1f} {g['n_calibration_trials']:5d} "
            f"{g['budget']:5.2f} {g['model']:18s} {g['arm']:26s} "
            f"{g['pr_auc_mean']:8.4f} {g['misheard_recall_mean']:8.4f} "
            f"{g['worst_listener_recall_mean']:8.4f} "
            f"{g['recall_over_word_length_mean']:+10.4f} "
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


@app.command("model-compare")
def model_compare(config: ConfigOpt) -> None:
    """E22 — 후보 모델 계열과 교정 방법의 청취자 수준 비교."""
    from audire.experiments.model_comparison import ModelComparisonConfig, run_model_comparison

    cfg = ModelComparisonConfig.load(config)
    result = run_model_comparison(cfg)
    summary = result["summary"]
    primary = f"{cfg.primary_budget:g}"

    typer.echo(f"\nrun_id: {result['run_id']}   시드: {summary['seeds']}\n")
    typer.echo(
        f"{'model':20s} {'교정':9s} {'PR-AUC':>15s} {'Brier':>8s} {'ECE':>8s} "
        f"{f'재현율@{primary}':>12s} {'중앙값':>8s} {'최하위':>8s} "
        f"{'길이대비':>10s} {'이긴시드':>9s} {'폴백':>5s}"
    )
    typer.echo("-" * 125)
    # 지는 후보도 그대로 인쇄합니다. 표에서 빼는 것은 선택 편향입니다.
    for row in sorted(summary["table"], key=lambda r: -r["gain_over_word_length"]["mean"]):
        pr = row["pr_auc"]
        mark = "" if row["output_is_probability"] else "*"
        typer.echo(
            f"{row['model']:20s} {row['calibration_requested']:9s} "
            f"{pr['mean']:6.4f}+-{pr['sd']:.4f} {row['brier']['mean']:8.4f}{mark:1s}"
            f"{row['ece']['mean']:8.4f}{mark:1s}"
            f"{row[f'recall_at_{primary}']['mean']:11.4f} "
            f"{row['recall_median_listener']['mean']:8.4f} "
            f"{row['recall_worst_listener']['mean']:8.4f} "
            f"{row['gain_over_word_length']['mean']:+10.4f} "
            f"{row['n_seeds_beating_word_length']:4d}/{row['n_seeds']:<4d} "
            f"{row['n_folds_fell_back']:5d}"
        )
    typer.echo("\n* Brier/ECE 는 랭킹 점수라 교정 전에는 의미가 없습니다 (순위 지표만 유효).")

    head = summary["headline"]
    typer.echo(
        f"\n모든 시드에서 단어길이를 이긴 후보: "
        f"{head['n_candidates_beating_word_length_on_every_seed']}/{head['n_candidates']}"
    )
    if head["best_candidate"]:
        b = head["best_candidate"]
        typer.echo(
            f"  최고: {b['model']} (교정={b['calibration_requested']}) "
            f"길이대비 {b['gain_over_word_length']:+.4f}"
        )
    else:
        # 음성 결과를 빈칸으로 남기지 않습니다. 이것도 결과입니다.
        typer.echo("  최고: 없음 — 모든 시드에서 단어길이를 이긴 후보가 하나도 없습니다.")
    if head["reference_logistic"]:
        r = head["reference_logistic"]
        typer.echo(
            f"  참조 로지스틱: 길이대비 {r['gain_over_word_length']:+.4f} "
            f"({r['n_seeds_beating_word_length']}개 시드에서 우세)"
        )
    paired = head["vs_reference_logistic"]
    if paired is None:
        typer.echo("  새 계열이 참조 기저선을 이겼는가: 판정 불가 (자격을 갖춘 후보 없음)")
    elif paired["best_is_the_reference"]:
        typer.echo("  최고 후보가 곧 참조 기저선입니다 — 새 계열이 이기지 못했습니다.")
    else:
        # 평균 차이가 아니라 시드별 짝지은 차이로 판정합니다.
        typer.echo(
            f"  참조 대비 짝지은 차이: {paired['paired_gain_mean']:+.5f} "
            f"+-{paired['paired_gain_sd']:.5f}  "
            f"({paired['n_seeds_best_beats_reference']}/{paired['n_seeds']} 시드에서 우세)"
        )
        typer.echo(f"  모든 시드에서 참조를 이겼는가: {paired['beats_on_every_seed']}")
    for row in summary["table"]:
        for reason in row["fallback_reasons"]:
            typer.echo(f"  교정 폴백 [{row['model']}/{row['calibration_requested']}]: {reason}")
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


@app.command("build-model")
def build_model(
    config: ConfigOpt,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Private joblib path for the fitted deployment scorer"),
    ] = None,
) -> None:
    """Fit the selected logistic family on all configured synthetic seed cohorts."""
    from audire.risk.artifact import fit_deployment_artifact

    target = out or (private_dir() / "models" / "audire-logistic.joblib")
    artifact = fit_deployment_artifact(config)
    model_path, sidecar = artifact.save(target)
    typer.echo(f"wrote model: {model_path}")
    typer.echo(f"wrote provenance: {sidecar}")
    typer.echo(artifact.metadata["caveat"])


@app.command("asr-eval")
def asr_eval(config: ConfigOpt) -> None:
    """Run the fixed public-corpus ASR WER/CER and timestamp regression."""
    from audire.asr.evaluation import ASREvalConfig, run_asr_evaluation

    cfg = ASREvalConfig.load(config)
    result = run_asr_evaluation(cfg)
    metrics = result["summary"]["metrics"]
    typer.echo(f"run_id: {result['run_id']}")
    typer.echo(
        f"WER={metrics['wer']['rate']:.4f}  "
        f"CER(no spaces)={metrics['cer_no_spaces']['rate']:.4f}  "
        f"RTF={metrics['real_time_factor']:.3f}"
    )
    typer.echo(
        f"timing problems: {metrics['n_timing_problems']} across "
        f"{metrics['n_utterances']} utterances"
    )
    typer.echo(result["summary"]["claim_scope"])


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
