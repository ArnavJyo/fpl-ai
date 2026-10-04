"""Render backtest artifacts after weights and metrics have been frozen.

Matplotlib creates standalone PNG charts. The HTML embeds them and uses local
table filtering/sorting, so the report opens offline without a server or CDN.
"""

from __future__ import annotations

import base64
import html
import os
import tempfile
from pathlib import Path

import pandas as pd



def write_charts(report: dict, predictions: pd.DataFrame, directory: Path) -> list[Path]:
    """Plot test predictions on shared scales and show epoch selection separately."""
    # Keep font-cache writes in the permitted temporary directory and use a
    # noninteractive backend. No browser or GUI is launched by this command.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fpl-ai-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = predictions.loc[predictions.split == "test"]
    target = report["target"]
    values = test[[target, "predicted_points", "baseline_points"]].to_numpy(dtype=float)
    low, high = float(values.min()) - 1, float(values.max()) + 1
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, score, title, color in zip(axes, ["predicted_points", "baseline_points"],
                                       ["Neural network", "Rolling-points baseline"], ["#c65b31", "#207a73"]):
        axis.scatter(test[target], test[score], s=24, alpha=0.4, color=color, edgecolors="none")
        axis.plot([low, high], [low, high], color="#6b7280", linestyle="--", linewidth=1, label="Perfect prediction")
        axis.set(xlim=(low, high), ylim=(low, high), xlabel="Actual points", ylabel="Predicted points", title=title)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
        axis.legend(loc="upper left", fontsize=8)
    figure.suptitle(f"Exploratory assessment · GW {', '.join(map(str, report['test_gameweeks']))} · {len(test)} player rows")
    scatter_path = directory / "predicted_vs_actual.png"
    figure.savefig(scatter_path, dpi=160)
    plt.close(figure)

    history = report["epoch_history"]
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    axes[0].plot(epochs, [record["training_loss"] for record in history], color="#207a73")
    axes[0].set(title=f"Training {report['options']['loss']} loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [record["validation_metrics"]["mae"] for record in history], color="#c65b31")
    axes[1].axvline(report["best_epoch"], color="#6b7280", linestyle="--", label=f"Selected epoch {report['best_epoch']}")
    axes[1].set(title="Validation MAE chooses the epoch", xlabel="Epoch", ylabel="MAE (points)")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.15)
    figure.suptitle("The assessment week is never used for early stopping")
    curve_path = directory / "learning_curve.png"
    figure.savefig(curve_path, dpi=160)
    plt.close(figure)
    return [scatter_path, curve_path]


def _metric_rows(report: dict) -> str:
    rows = []
    for name in ["validation", "test"]:
        scores = report[name]
        numbers = [scores[model][metric] for metric in ["mae", "rmse", "mean_error"] for model in ["nn", "baseline"]]
        rows.append(f"<tr><th>{name.title()}</th><td>{scores['rows']}</td>" +
                    "".join(f"<td>{value:.3f}</td>" for value in numbers) + "</tr>")
    return "".join(rows)


def _ranking_rows(report: dict) -> str:
    rows = []
    for week in report["test"]["per_gameweek"]:
        for model, label in [("nn", "NN"), ("baseline", "Baseline")]:
            ranking = week["ranking"][model]
            correlation = ranking["spearman_rank_correlation"]
            rendered_correlation = "Undefined (constant scores)" if correlation is None else f"{correlation:.3f}"
            rows.append(f"<tr><td>{week['gameweek']}</td><th>{label}</th><td>{ranking['top_k']}</td>"
                        f"<td>{ranking['actual_points_total']:.1f}</td><td>{ranking['optimal_actual_points_total']:.1f}</td>"
                        f"<td>{ranking['points_shortfall']:.1f}</td><td>{ranking['precision_at_k']:.0%}</td>"
                        f"<td>{rendered_correlation}</td></tr>")
    return "".join(rows)


def _prediction_rows(report: dict, predictions: pd.DataFrame) -> str:
    rows = []
    columns = ["split", "gameweek", "element_id", "player_name", report["target"],
               "predicted_points", "baseline_points", "nn_error", "baseline_error"]
    for values in predictions[columns].itertuples(index=False, name=None):
        cells = []
        for number, value in enumerate(values):
            if number >= 4:
                cells.append(f'<td data-value="{float(value)}">{float(value):.2f}</td>')
            elif number in [1, 2]:
                cells.append(f'<td data-value="{int(value)}">{int(value)}</td>')
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append(f'<tr data-split="{values[0]}">' + "".join(cells) + "</tr>")
    return "".join(rows)


def write_summary(report: dict, directory: Path) -> None:
    """Save a small, regenerated Markdown result record for documentation."""
    lines = ["# Exploratory backtest results", "", "Generated by `python -m src.models.backtest`.", "",
             f"- Training weeks: {report['train_gameweeks']} ({report['train_rows']} rows).",
             f"- Validation weeks: {report['validation_gameweeks']} ({report['validation']['rows']} rows).",
             f"- Assessment weeks: {report['test_gameweeks']} ({report['test']['rows']} rows).",
             f"- Selected epoch: {report['best_epoch']}; epochs run: {report['epochs_run']}.",
             f"- Input columns: {len(report['input_columns'])}; seed: {report['options']['seed']}.", "",
             "| Period | NN MAE | Baseline MAE | NN RMSE | Baseline RMSE |",
             "|---|---:|---:|---:|---:|"]
    for name in ["validation", "test"]:
        scores = report[name]
        lines.append(f"| {name.title()} | {scores['nn']['mae']:.3f} | {scores['baseline']['mae']:.3f} | "
                     f"{scores['nn']['rmse']:.3f} | {scores['baseline']['rmse']:.3f} |")
    lines.extend(["", "Lower MAE and RMSE are better. These are historical development results, not final future accuracy.", "",
                  "| Assessment GW | Ranking | Top-k | Actual points selected | Best actual top-k points | Overlap |",
                  "|---|---|---:|---:|---:|---:|"])
    for week in report["test"]["per_gameweek"]:
        for model, label in [("nn", "NN"), ("baseline", "Baseline")]:
            ranking = week["ranking"][model]
            lines.append(f"| {week['gameweek']} | {label} | {ranking['top_k']} | {ranking['actual_points_total']:.1f} | "
                         f"{ranking['optimal_actual_points_total']:.1f} | {ranking['precision_at_k']:.0%} |")
    lines.extend(["", "Ranking does not enforce squad constraints.", "", "Limitations:", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend(["", "Read [the interactive report](report.html), [metrics](metrics.json), or [player predictions](predictions.csv).", ""])
    (directory / "results.md").write_text("\n".join(lines))


def write_backtest_report(report: dict, predictions: pd.DataFrame, directory: Path) -> None:
    """Generate charts, offline HTML, and a readable result record."""
    charts = write_charts(report, predictions, directory)
    template = (Path(__file__).resolve().parent / "templates/backtest_report.html").read_text()
    replacements = {
        "TRAIN_WEEKS": ", ".join(map(str, report["train_gameweeks"])),
        "VALIDATION_WEEKS": ", ".join(map(str, report["validation_gameweeks"])),
        "TEST_WEEKS": ", ".join(map(str, report["test_gameweeks"])),
        "BEST_EPOCH": str(report["best_epoch"]), "EPOCHS_RUN": str(report["epochs_run"]),
        "SEED": str(report["options"]["seed"]), "INPUT_COUNT": str(len(report["input_columns"])),
        "ARCHITECTURE": html.escape(" → ".join(map(str, [len(report["input_columns"]), *report["options"]["hidden_layers"], 1]))),
        "METRIC_ROWS": _metric_rows(report), "RANKING_ROWS": _ranking_rows(report),
        "PREDICTION_ROWS": _prediction_rows(report, predictions),
        "LIMITATIONS": "".join(f"<li>{html.escape(value)}</li>" for value in report["limitations"]),
        "SCATTER_IMAGE": base64.b64encode(charts[0].read_bytes()).decode("ascii"),
        "CURVE_IMAGE": base64.b64encode(charts[1].read_bytes()).decode("ascii"),
        "TEST_MAE_WINNER": "NN" if report["test"]["nn"]["mae"] < report["test"]["baseline"]["mae"] else
                           "Tie" if report["test"]["nn"]["mae"] == report["test"]["baseline"]["mae"] else "Baseline",
        "TEST_RMSE_WINNER": "NN" if report["test"]["nn"]["rmse"] < report["test"]["baseline"]["rmse"] else
                            "Tie" if report["test"]["nn"]["rmse"] == report["test"]["baseline"]["rmse"] else "Baseline",
    }
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    (directory / "report.html").write_text(template)
    report["chart_files"] = [str(path) for path in charts]
    write_summary(report, directory)
