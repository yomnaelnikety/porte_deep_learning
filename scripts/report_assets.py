"""Export report assets from a completed validation run."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def latex_label(label):
    return label.replace("_", r"\_")


def write_table(path, rows, header, caption, layout):
    lines = [r"\begin{table}[htbp]", r"\centering\small", rf"\begin{{tabular}}{{{layout}}}",
             r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--report", type=Path, default=Path("report"))
    args = parser.parse_args()
    report, results = args.report, args.results
    tables, figures = report / "tables", report / "figures"
    for directory in tables, figures:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(results / "eda/eventflow_eda.pdf", figures / "eventflow_eda.pdf")
    eda = results / "eda"
    balance = pd.read_csv(eda / "event_balance.csv", index_col=0).iloc[:, 0]
    gaps = pd.read_csv(eda / "time_gaps.csv", index_col=0).iloc[:, 0]
    missing = pd.read_csv(eda / "missingness.csv", index_col=0).iloc[:, 0]
    most, least = latex_label(balance.idxmax()), latex_label(balance.idxmin())
    (tables / "data_summary.tex").write_text(
        f"In the fitting split, {most} is the most frequent event ({balance.max():.1%}) "
        f"and {least} is the least frequent ({balance.min():.1%}). "
        f"The median gap between consecutive events is {gaps.loc['50%']:.2f} timestamp units "
        f"and the 95th percentile is {gaps.loc['95%']:.2f}. "
        f"The highest missing fraction is {missing.max():.1%} "
        f"for {latex_label(missing.idxmax())}.\n".replace("%", r"\%"))
    log = pd.read_csv(results / "latest_experiments.csv")
    overview = log[log.run.isin(["uniform", "markov", "current_row_logreg",
                                 "full_raw", "selected_classical"])]
    write_table(tables / "classical_results.tex",
                [(latex_label(r.run), f"{r.validation_log_loss:.4f}") for r in overview.itertuples()],
                ["Run", "Validation log loss"], "Classical results on the shared v2 cutoffs.", "@{}lr@{}")
    ablations = pd.read_csv(results / "ablations.csv").set_index("run", drop=False)
    calibration_params = log[log.model == "LightGBM calibration"].set_index("run").hyperparameters
    alpha = json.loads(calibration_params.loc["prior_blend"])["alpha"]
    temperature = json.loads(calibration_params.loc["temperature"])["temperature"]
    chosen = json.loads(log.loc[log.run.eq("selected_classical"), "hyperparameters"].item())
    method_label = {"raw": "raw probabilities", "temperature": "temperature scaling",
                    "prior_blend": "prior blending"}[chosen["method"]]
    displayed = ablations.drop("prior_blend") if alpha == 0 else ablations
    rows = []
    for row in displayed.itertuples():
        precision = 5 if row.kind == "calibration" else 4
        rows.append((latex_label(row.run), f"{row.log_loss:.4f}",
                     f"{row.delta_vs_full:+.{precision}f}",
                     f"[{row.ci_95_low:+.{precision}f}, {row.ci_95_high:+.{precision}f}]"))
    write_table(tables / "classical_ablations.tex", rows,
                ["Run", "Log loss", r"$\Delta$ vs. raw", r"95\% paired CI"],
                "Ablations against the raw pair-free tree. Intervals resample machines.", "@{}lrrr@{}")
    scores = overview.set_index("run").validation_log_loss
    tuning = log[log.model == "LightGBM tuning"]
    winner = tuning.loc[tuning.inner_log_loss.idxmin()]
    tuning_details = [json.loads(params) for params in tuning.hyperparameters]
    chosen_tuning = json.loads(winner.hyperparameters)
    stopped = sum(details["patience_triggered"] for details in tuning_details)
    tuning_cap = chosen_tuning["n_estimators"]
    best_round = chosen_tuning["best_iteration"]
    evaluated_rounds = chosen_tuning["evaluated_rounds"]
    tails = pd.read_csv(results / "calibration_tails.csv").set_index("method")
    raw = tails.loc["raw"]
    temp = tails.loc["temperature"]
    pair_add, lag_drop, sensors, fewer, temp_ablation = (
        ablations.loc[name] for name in ["add_pair_back", "drop_lags", "drop_sensors",
                                        "four_fit_cutoffs", "temperature"])
    sensor_ci = ("with an interval spanning zero" if sensors.ci_95_low <= 0 <= sensors.ci_95_high
                 else "with an interval excluding zero")
    sensor_conclusion = (
        "The sensor history did not show a clear incremental effect. "
        if sensors.ci_95_low <= 0 <= sensors.ci_95_high
        else ("The sensor history helped at these settings. " if sensors.ci_95_low > 0
              else "The sensor history worsened loss at these settings. ")
    )
    calibration_ci = (
        "with a paired interval spanning zero" if temp_ablation.ci_95_low <= 0 <= temp_ablation.ci_95_high
        else "with a paired interval excluding zero")
    if temp_ablation.ci_95_high < 0:
        gain = -temp_ablation.delta_vs_full
        if gain < 0.001:
            temp_conclusion = (
                f"Temperature scaling gave a statistically detectable but negligible gain "
                f"(about {gain:.5f} log-loss points). "
            )
        else:
            temp_conclusion = "Temperature scaling lowered loss in this comparison. "
    elif temp_ablation.ci_95_low > 0:
        temp_conclusion = "Temperature scaling increased loss in this comparison. "
    else:
        temp_conclusion = "Temperature scaling showed no detectable log-loss gain in this comparison. "
    baseline_text = (
        f"The local transition table ({scores.loc['markov']:.3f}) and logistic regression "
        f"({scores.loc['current_row_logreg']:.3f}) are close to the brief's test references "
        r"of 1.81 and 1.75 \cite{brief}. This is a sanity check that the validation "
        "cutoffs capture a comparable level of difficulty."
    )
    feature_text = (
        f"With a cap of {tuning_cap} rounds, the patience rule triggered in {stopped} of "
        f"{len(tuning_details)} tuning trials. The inner holdout selected "
        f"{latex_label(winner.run)} at its best iteration of {best_round}, "
        f"after evaluating {evaluated_rounds} rounds. "
        f"Adding back the 196-category event pair changed validation loss by "
        f"{pair_add.delta_vs_full:+.4f}. With the pair excluded in both models, "
        f"dropping the six individual lags changed loss by {lag_drop.delta_vs_full:+.4f}. "
        f"Dropping sensor history changed loss by {sensors.delta_vs_full:+.4f}, "
        f"{sensor_ci}. The remaining features include current sensor readings. "
        + sensor_conclusion
        + f"Using four cutoffs per fitting machine changed loss by {fewer.delta_vs_full:+.4f} "
        "at the same tree settings."
    )
    calibration_text = (
        rf"Temperature scaling fitted \(T={temperature:.4f}\) and changed loss by "
        f"{temp_ablation.delta_vs_full:+.4f}, {calibration_ci}. "
        + temp_conclusion
        + f"The calibration split selected {method_label}. "
        f"For the three rare target classes, the share assigned under 0.5% probability "
        f"for the true event was {raw.rare_fraction_true_below_005:.1%} raw and "
        f"{temp.rare_fraction_true_below_005:.1%} after temperature scaling. "
    ).replace("%", r"\%")
    if alpha == 0:
        calibration_text += (
            r"Prior blending fitted \(\alpha=0\), so it reproduced the raw probabilities "
            "and is omitted from the table. "
        )
    else:
        calibration_text += rf"Prior blending fitted \(\alpha={alpha:.4f}\). "
    (tables / "classical_interpretation.tex").write_text(
        baseline_text + "\n\n" + feature_text + "\n\n" + calibration_text
        + "The paired intervals describe these fitted models and do not adjust for multiple "
        "comparisons or include variation across refits.\n")
    lengths = pd.read_csv(results / "errors_by_length.csv")
    by_class = pd.read_csv(results / "errors_by_class.csv")
    hardest = by_class.loc[by_class.log_loss.idxmax()]
    first, last = lengths.iloc[0], lengths.iloc[-1]
    confusion = pd.read_csv(results / "confusion_counts.csv", index_col=0)
    np.fill_diagonal(confusion.values, 0)
    pair = confusion.stack().nlargest(1)
    actual, predicted = map(latex_label, pair.index[0])
    worst = pd.read_csv(results / "highest_loss_examples.csv").iloc[0]
    (tables / "classical_errors.tex").write_text(
        f"For the selected classical model, prefixes of 90--129 events had validation "
        f"log loss {first.log_loss:.3f} (\\(n={first.n}\\)), versus {last.log_loss:.3f} "
        f"(\\(n={last.n}\\)) for 180--250 events. "
        f"{latex_label(hardest.target)} had the largest class-average loss "
        f"({hardest.log_loss:.3f} over {hardest.n} examples). "
        "The class mix can differ across length buckets. "
        f"The most common off-diagonal error was {actual} classified as {predicted} "
        f"({pair.item()} cases). The largest individual loss was {worst.loss:.2f}, "
        f"on a prefix of {worst.prefix_len} events where the model assigned "
        f"probability {worst.true_probability:.5f} to {latex_label(worst.target)}.\n")


if __name__ == "__main__":
    main()
