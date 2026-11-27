import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import MAX_PREFIX
from .generate import js_divergence


def lag_counts(sequences, lag):
    if lag < 1 or lag >= sequences.shape[1]:
        raise ValueError("Lag must be positive and shorter than the continuation")
    return Counter((int(a), int(b)) for row in sequences
                   for a, b in zip(row[:-lag], row[lag:]))


def repeated_run_lengths(sequences):
    lengths = []
    for row in sequences:
        longest = current = 1
        for a, b in zip(row[:-1], row[1:]):
            current = current + 1 if a == b else 1
            longest = max(longest, current)
        lengths.append(longest)
    return lengths


def training_windows(sequences, horizon):
    windows = set()
    count = 0
    for row in sequences:
        raw = np.asarray(row, dtype=np.uint8).tobytes()
        for i in range(len(raw)-horizon+1):
            windows.add(raw[i:i+horizon])
            count += 1
    return windows, count


def exact_match_rate(sequences, windows):
    return float(np.mean([np.asarray(row, dtype=np.uint8).tobytes() in windows for row in sequences]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/raw"))
    p.add_argument("--generation", type=Path, default=Path("results/transformer/generation"))
    args = p.parse_args()
    summary = json.loads((args.generation / "summary.json").read_text())
    arrays = np.load(args.generation / "rollouts.npz")
    real, generated, classes = arrays["real"], arrays["generated"], arrays["classes"].tolist()
    checkpoint = Path(summary["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != summary["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the one used to produce these rollouts")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fit_ids = set(saved["fit_ids"])
    evaluated = pd.read_csv(args.generation / "evaluated_manifest.csv")
    if fit_ids & set(evaluated.sequence_id) or classes != saved["encoder"]["classes"]:
        raise ValueError("Machine overlap or class-order mismatch")
    rows = []
    for j in range(generated.shape[1]):
        row = {"sample": j}
        for lag in (1, 5, 10):
            row[f"lag_{lag}_js"] = js_divergence(lag_counts(real, lag), lag_counts(generated[:, j], lag))
        row["longest_repeat_run_js"] = js_divergence(Counter(repeated_run_lengths(real)),
                                                      Counter(repeated_run_lengths(generated[:, j])))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.generation / "longer_structure_by_sample.csv", index=False)
    structure = {k: {"mean": float(frame[k].mean()), "sampling_std": float(frame[k].std(ddof=1))}
                 for k in frame if k != "sample"}
    print("Comparing continuations with the fitting machines' exposed event windows", flush=True)
    events = pd.read_csv(args.data / "eventflow_train.csv", usecols=["sequence_id", "event_index", "event_type"])
    events = events[events.sequence_id.isin(fit_ids)].sort_values(["sequence_id", "event_index"])
    mapping = {c: i for i, c in enumerate(classes)}
    # the training uses inputs through position 249 and targets through position 250
    encoded = (g.event_type.map(mapping).iloc[:MAX_PREFIX+1].to_numpy()
               for _, g in events.groupby("sequence_id", sort=False))
    windows, n_windows = training_windows(encoded, real.shape[1])
    memorization = {
        "n_fitting_machines": len(fit_ids), "horizon": real.shape[1],
        "training_exposure_limit_events": MAX_PREFIX+1,
        "n_training_windows": n_windows, "n_unique_training_windows": len(windows),
        "real_exact_training_window_match_rate": exact_match_rate(real, windows),
        "generated_exact_training_window_match_rate": exact_match_rate(generated.reshape(-1, real.shape[1]), windows),
        "note": "Exact matches can arise naturally in common event patterns; compare the real rate. "
                "No exact matches does not rule out partial or approximate memorization."
    }
    result = {"longer_structure": structure, "training_window_matches": memorization,
              "note": "Lag statistics test dependencies up to 10 events within the 20-event horizon. "
                      "They do not establish global sequence realism. SD measures sampling variation, not a confidence interval."}
    (args.generation / "supplemental_analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
