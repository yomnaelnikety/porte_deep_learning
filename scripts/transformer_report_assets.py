"""Export compact CSV tables and a figure from completed Person C experiments.

Does not run training, read test labels, or invent results for unfinished runs.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/transformer"))
    p.add_argument("--generation", type=Path, default=Path("results/transformer/generation"))
    p.add_argument("--out", type=Path, default=Path("results/transformer/report_assets"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in sorted((args.results / "runs").glob("*/metrics.json")):
        row = json.loads(file.read_text())
        if row.get("smoke"):
            raise ValueError("Smoke predictive runs are not report evidence")
        cfg = json.loads(file.with_name("config.json").read_text())
        rows.append({**row, "event_only": cfg["event_only"], "hidden": cfg["hidden"],
                     "layers": cfg["layers"], "heads": cfg["heads"], "lr": cfg["lr"]})
    if rows:
        pd.DataFrame(rows).to_csv(args.out / "predictive_results.csv", index=False)
    selected_path = args.results / "selected_transformer.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())["run"]
        ablations_path = args.results / f"ablations_{selected}.csv"
        if ablations_path.exists():
            ablations = pd.read_csv(ablations_path).iloc[1:].copy()
            ablations["variant"] = [Path(p).parent.name.split("__", 1)[-1] for p in ablations.model]
            ablations.to_csv(args.out / "ablation_results.csv", index=False)
            labels = {"no_time": "Remove time gaps", "no_sensors": "Remove sensors",
                      "no_context": "Remove categorical context", "context16": "Short context: 16",
                      "context64": "Short context: 64", "other_seed": "Repeat with another seed"}
            fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
            delta = ablations.delta_vs_transformer.to_numpy()
            errors = np.vstack([delta-ablations.ci_95_low, ablations.ci_95_high-delta])
            ax.errorbar(delta, np.arange(len(ablations)), xerr=errors, fmt="o", capsize=4)
            ax.axvline(0, color="grey", linewidth=1)
            ax.set_yticks(np.arange(len(ablations)), [labels.get(n, n) for n in ablations.variant])
            ax.invert_yaxis()
            ax.set_xlabel("Change in validation log loss (positive = worse)")
            ax.set_title("Transformer ablations and seed sensitivity\n95% paired machine-bootstrap intervals")
            fig.savefig(args.out / "transformer_ablations.png", dpi=180)
            fig.savefig(args.out / "transformer_ablations.pdf")
            plt.close(fig)
    summary_path = args.generation / "summary.json"
    if not summary_path.exists():
        print("No generation summary found; predictive table only.")
        return
    summary = json.loads(summary_path.read_text())
    if summary["smoke"]:
        raise ValueError("Smoke generation is not report evidence; use a fully trained checkpoint")
    rows = []
    for metric, value in summary.items():
        if isinstance(value, dict) and "mean" in value:
            rows.append(dict(metric=metric, generated_mean=value["mean"],
                             sampling_std=value["sampling_std"],
                             real=summary["real_diversity"].get(metric, np.nan)))
    pd.DataFrame(rows).to_csv(args.out / "generation_diagnostics.csv", index=False)
    arrays = np.load(args.generation / "rollouts.npz")
    classes = arrays["classes"]
    real_counts = np.bincount(arrays["real"].ravel(), minlength=len(classes))
    gen_counts = np.bincount(arrays["generated"].ravel(), minlength=len(classes))
    real, gen = real_counts/real_counts.sum(), gen_counts/gen_counts.sum()
    pd.DataFrame(dict(event=classes, real_frequency=real, generated_frequency=gen)).to_csv(
        args.out / "event_frequencies.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    y = np.arange(len(classes))
    axes[0].barh(y-.18, real, height=.35, label="Real")
    axes[0].barh(y+.18, gen, height=.35, label="Generated")
    axes[0].set_yticks(y, classes, fontsize=8)
    axes[0].set_xlabel("Event frequency")
    axes[0].legend()
    axes[0].set_title("Marginal event frequencies")
    keys = [f"js_{n}gram" for n in (1, 2, 3)]
    axes[1].bar(["Unigrams", "Bigrams", "Trigrams"], [summary[k]["mean"] for k in keys],
                yerr=[summary[k]["sampling_std"] for k in keys], capsize=4)
    axes[1].set_ylabel("Jensen-Shannon divergence (nats)")
    axes[1].set_title("Marginal and ordering differences\nError bars: sampling SD, not confidence intervals")
    fig.savefig(args.out / "generation_diagnostics.png", dpi=180)
    fig.savefig(args.out / "generation_diagnostics.pdf")
    plt.close(fig)
    records = [json.loads(line) for line in (args.generation / "rollouts.jsonl").read_text().splitlines()]
    # Deterministic examples avoid choosing only the most flattering rollouts.
    examples = [{"sequence_id": r["sequence_id"], "prefix_len": r["prefix_len"],
                 "prefix_tail": r["prefix_tail"], "real": r["real"],
                 "sample_1": r["generated"][0], "sample_2": r["generated"][1]} for r in records[:3]]
    (args.out / "representative_examples.json").write_text(json.dumps(examples, indent=2))
    print(f"Wrote report assets to {args.out}")


if __name__ == "__main__":
    main()
