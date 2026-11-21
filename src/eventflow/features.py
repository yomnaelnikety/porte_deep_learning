"""Construct features from observed rows only, at requested prefix cutoffs."""

import warnings

import numpy as np
import pandas as pd

from .data import CATEGORICAL, MIN_PREFIX

SENSORS = ["load_reading", "temperature_reading", "temp_probe_b",
           "vibration_reading", "vibration_probe_b", "pressure_reading",
           "power_draw", "coolant_flow", "control_signal", "ambient_index",
           "efficiency_estimate", "acoustic_level"]
GROUPS = ["current", "lags", "counts", "time", "sensors"]


def build_features(events, manifest, classes, labeled=True):
    tables = []
    requested = {k: g.prefix_len.to_numpy() for k, g in manifest.groupby("sequence_id")}
    raw = [c for c in events.columns if c not in
           ["sequence_id", "event_index", "timestamp", *CATEGORICAL]]
    for key, seq in events.groupby("sequence_id", sort=False, observed=True):
        if key not in requested:
            continue
        end = requested[key] - 1
        if (end < MIN_PREFIX - 1).any() or (end + int(labeled) >= len(seq)).any():
            raise ValueError(f"Invalid prefix for {key}")
        event = seq.event_type.to_numpy()
        timestamp = seq.timestamp.to_numpy()
        features = {"sequence_id": key, "prefix_len": end + 1}
        if labeled:
            features["target"] = event[end + 1]
        for col in CATEGORICAL + raw:
            features[f"current__{col}"] = seq[col].to_numpy()[end]
        for lag in range(1, 7):
            features[f"lags__event_{lag}"] = event[end - lag]
        features["lags__pair"] = np.char.add(
            np.char.add(event[end-1].astype(str), "|"), event[end].astype(str))
        matches = event[:, None] == np.asarray(classes)[None, :]
        cumulative = np.vstack([np.zeros(len(classes), dtype=int), matches.cumsum(axis=0)])
        last = np.maximum.accumulate(np.where(matches, np.arange(len(seq))[:, None], -1), axis=0)
        for j, cls in enumerate(classes):
            for window in (8, 32, 90):
                features[f"counts__{window}_{cls}"] = (
                    cumulative[end + 1, j] - cumulative[end + 1 - window, j])
            seen = last[end, j]
            features[f"time__since_{cls}"] = np.where(
                seen >= 0, np.log1p(timestamp[end] - timestamp[np.maximum(0, seen)]), np.nan)
        gaps = np.diff(timestamp, prepend=timestamp[0])
        for lag in range(4):
            features[f"time__gap_{lag}"] = np.log1p(gaps[end - lag])
        sensors = seq[SENSORS].to_numpy()
        for window in (8, 32):
            positions = np.arange(window)
            values = sensors[end[:, None] - window + 1 + positions]
            valid = np.isfinite(values)
            y = np.where(valid, values, 0)
            x = positions[None, :, None]
            n = valid.sum(axis=1)
            sx, sy = (valid * x).sum(axis=1), y.sum(axis=1)
            denominator = n * (valid * x**2).sum(axis=1) - sx**2
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = np.where(n > 0, sy / n, np.nan)
                slope = np.where(denominator > 0, (n * (x*y).sum(axis=1) - sx*sy) /
                                 denominator, np.nan)
            for j, col in enumerate(SENSORS):
                features[f"sensors__mean_{window}_{col}"] = mean[:, j]
                features[f"sensors__slope_{window}_{col}"] = slope[:, j]
        tables.append(pd.DataFrame(features))
    if not tables:
        raise ValueError("Manifest does not match any sequence")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        features = pd.concat(tables, ignore_index=True)
    if len(features) != len(manifest):
        raise ValueError("Some manifest entries could not be resolved")
    return features.merge(manifest, on=["sequence_id", "prefix_len"], validate="one_to_one")
