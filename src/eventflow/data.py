"""Shared data contract for classical and deep models."""

import numpy as np
import pandas as pd

SEED = 2210
PROTOCOL = "v2"
VALIDATION_CUTS = 8
MIN_PREFIX, MAX_PREFIX = 90, 250
CATEGORICAL = ["event_type", "subsystem", "shift_code", "operator_group",
               "device_family", "firmware_channel"]
PROB_PREFIX = "p__"


def load_events(path):
    header = pd.read_csv(path, nrows=0).columns
    dtype = {c: "float32" for c in header
             if c not in ["sequence_id", "event_index", "timestamp", *CATEGORICAL]}
    dtype.update({c: "category" for c in CATEGORICAL})
    df = pd.read_csv(path, dtype=dtype)
    df.sort_values(["sequence_id", "event_index"], inplace=True, ignore_index=True)
    if df[["sequence_id", "event_index", "timestamp", "event_type"]].isna().any().any():
        raise ValueError("Missing identifier, index, timestamp or event")
    if df.duplicated(["sequence_id", "event_index"]).any():
        raise ValueError("Duplicate event index within a sequence")
    if df.groupby("sequence_id", sort=False).timestamp.diff().lt(0).any():
        raise ValueError("Time is decreasing within a sequence")
    return df


def make_manifest(df, train_cuts=12, seed=SEED):
    """prefix_len L means use rows [:L], target row [L] (zero based)."""
    lengths = df.groupby("sequence_id").size()
    rng = np.random.default_rng(seed)
    ids = rng.permutation(lengths.index)
    val_ids = set(ids[:round(0.2 * len(ids))])
    extras_rng = np.random.default_rng(seed + 1)
    rows, rollouts = [], []
    for key, length in lengths.items():
        available = np.arange(MIN_PREFIX, min(MAX_PREFIX, length - 1) + 1)
        if not len(available):
            continue
        split = "validation" if key in val_ids else "train"
        count = 1 if key in val_ids else min(train_cuts, len(available))
        chosen = rng.choice(available, size=count, replace=False)
        if split == "validation":
            # Keep the original cutoff and the rollout RNG stream intact.
            remaining = np.setdiff1d(available, chosen)
            extra = extras_rng.choice(remaining, size=min(VALIDATION_CUTS-1, len(remaining)),
                                      replace=False)
            chosen = np.concatenate([chosen, extra])
        for cutoff in sorted(chosen):
            rows.append((key, cutoff, split))
        if key in val_ids and length >= MIN_PREFIX + 20:
            cutoff = rng.integers(MIN_PREFIX, min(MAX_PREFIX, length - 20) + 1)
            rollouts.append((key, cutoff, 20))
    return (pd.DataFrame(rows, columns=["sequence_id", "prefix_len", "split"]),
            pd.DataFrame(rollouts, columns=["sequence_id", "prefix_len", "horizon"]))


def multiclass_loss(y, prob, classes):
    """Probabilities must follow classes (the order in classes.json)."""
    prob = np.asarray(prob, dtype=float)
    if (prob.ndim != 2 or prob.shape != (len(y), len(classes))
            or not np.isfinite(prob).all() or (prob < 0).any()
            or not np.allclose(prob.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("Expected one normalized probability row per target")
    target = pd.Index(classes).get_indexer(y)
    if (target < 0).any():
        raise ValueError("Target missing from the class list")
    return float(-np.log(np.clip(prob[np.arange(len(y)), target], 1e-15, 1)).mean())


def inner_machine_split(manifest, seed=SEED):
    """Reserve disjoint machine sets for early stopping and calibration."""
    ids = sorted(manifest.loc[manifest.split == "train", "sequence_id"].unique())
    keys = np.random.default_rng(seed + 2).permutation(ids)
    n = round(0.1 * len(keys))
    rows = [(key, role) for role, subset in (
        ("early_stop", keys[:n]), ("calibration", keys[n:2*n]), ("fit", keys[2*n:]))
            for key in subset]
    return pd.DataFrame(rows, columns=["sequence_id", "role"]).sort_values("sequence_id")
