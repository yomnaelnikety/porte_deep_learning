"""Export the GRU tables and learning curve for the report.

Run after `train_deep tune`, `ablate` and (optionally) `compare`:
    python scripts/deep_report_assets.py
Writes report/tables/gru_*.tex and report/figures/gru_learning_curve.pdf.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_assets import latex_label, write_table

LABELS = {
    "no_time": "Remove time gaps", "no_sensors": "Remove sensors and missing flags",
    "no_context": "Remove categorical context", "context_16": "Last 16 events only",
    "context_64": "Last 64 events only", "add_relative_sensors": "Add machine-relative sensors",
    "other_seed": "Same model, different seed", "no_temperature": "No temperature scaling",
    "lightgbm_selected": "LightGBM", "average_gru_lightgbm": "Average of GRU and LightGBM",
}


def signed(value, digits=4):
    return f"{value:+.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", type=Path, default=Path("results/deep"))
    parser.add_argument("--report", type=Path, default=Path("report"))
    args = parser.parse_args()
    tables, figures = args.report / "tables", args.report / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    selected = json.loads((args.deep / "selected_gru.json").read_text())

    tuning = pd.read_csv(args.deep / "tuning_gru.csv").sort_values("early_stop_log_loss")
    rows = [(latex_label(r.run) + (" (selected)" if r.run == selected["run"] else ""),
             f"{r.n_parameters / 1000:.0f}k", str(r.best_epoch), f"{r.early_stop_log_loss:.4f}",
             f"{r.temperature:.3f}", f"{r.validation_log_loss_raw:.4f}", f"{r.validation_log_loss:.4f}")
            for r in tuning.itertuples()]
    write_table(tables / "gru_tuning.tex", rows,
                ["Run", "Params", "Epoch", "Early-stop", "$T$", "Val. raw", "Val. calibrated"],
                r"Recurrent model search. The selected run has the lowest early-stop loss; validation machines were not used for selection. "
                r"Validation uses the shared 10,400 cutoffs.", "@{}lrrrrrr@{}")

    ablations = pd.read_csv(args.deep / "ablations_gru.csv")
    rows = [(LABELS.get(r.run, latex_label(r.run)), f"{r.log_loss:.4f}", signed(r.delta_vs_full),
             f"[{signed(r.ci_95_low)}, {signed(r.ci_95_high)}]") for r in ablations.itertuples()]
    write_table(tables / "gru_ablations.tex", rows,
                ["Change from selected GRU", "Log loss", r"$\Delta$", r"95\% paired CI"],
                "One change at a time around the selected recurrent model. Positive "
                r"$\Delta$ means worse. Intervals resample validation machines.", "@{}lrrr@{}")

    compare = args.deep / "compare_gru_vs_classical.csv"
    if compare.exists():
        frame = pd.read_csv(compare)
        rows = [(LABELS.get(r.run, latex_label(r.run)), f"{r.log_loss:.4f}", signed(r.delta_vs_full),
                 f"[{signed(r.ci_95_low)}, {signed(r.ci_95_high)}]") for r in frame.itertuples()]
        write_table(tables / "gru_compare.tex", rows,
                    ["Model (vs. selected GRU)", "Log loss", r"$\Delta$", r"95\% paired CI"],
                    "Paired comparison on the shared validation cutoffs. The average is "
                    "exploratory: equal weights, nothing tuned on validation.", "@{}lrrr@{}")

    history = pd.read_csv(args.deep / "runs" / selected["run"] / "history.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(history.epoch, history.train_loss, marker="o", label="fit machines (all positions)")
    ax.plot(history.epoch, history.early_stop_loss, marker="o", label="early-stop machines (cutoffs)")
    ax.axvline(selected["metrics"]["best_epoch"], color="grey", ls="--", lw=1, label="kept epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("log loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "gru_learning_curve.pdf")

    m = selected["metrics"]
    reweight = args.deep / "selected_errors" / "length_reweighting.json"
    extra = ""
    if reweight.exists():
        r = json.loads(reweight.read_text())
        extra = (f" Reweighting validation cutoffs to the test length mix gives "
                 f"{r['reweighted_to_test_lengths']:.4f}.")
    (tables / "gru_summary.tex").write_text(
        f"The selected model, {latex_label(selected['run'])}, kept epoch {m['best_epoch']} of "
        f"{m['epochs_run']}. Temperature scaling fitted $T={m['temperature']:.3f}$ and moved "
        f"validation log loss from {m['validation_log_loss_raw']:.4f} to "
        f"{m['validation_log_loss']:.4f}; accuracy is {m['validation_accuracy']:.1%}.{extra}\n"
        .replace("%", r"\%"))
    print("Wrote GRU tables and figure to", args.report)


if __name__ == "__main__":
    main()
