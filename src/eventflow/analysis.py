"""Calibration and paired uncertainty for EventFlow."""

import csv
import os
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def example_loss(y, prob, classes):
    target = pd.Index(classes).get_indexer(y)
    if (target < 0).any():
        raise ValueError("Unknown target")
    return -np.log(np.clip(prob[np.arange(len(y)), target], 1e-15, 1))


def calibrate(prob, spec):
    if spec["method"] == "temperature":
        return softmax(np.log(np.clip(prob, 1e-15, 1)) / spec["temperature"], axis=1)
    if spec["method"] == "prior_blend":
        alpha = spec["alpha"]
        return (1 - alpha) * prob + alpha * np.asarray(spec["prior"])
    return prob


def fit_calibrators(y, prob, classes):
    counts = pd.Series(y).value_counts().reindex(classes, fill_value=0).to_numpy() + 1
    prior = (counts / counts.sum()).tolist()

    def loss(spec):
        return example_loss(y, calibrate(prob, spec), classes).mean()

    specs = [{"method": "raw"}]
    for method, key, bounds, extra in [
            ("temperature", "temperature", (.6, 3.0), {}),
            ("prior_blend", "alpha", (0.0, .30), {"prior": prior})]:
        def candidate(value):
            return {"method": method, key: float(value), **extra}

        opt = minimize_scalar(lambda value: loss(candidate(value)), bounds=bounds,
                              method="bounded")
        specs.append(min((candidate(v) for v in [*bounds, opt.x]), key=loss))
    return [(spec, float(loss(spec))) for spec in specs]


class MachineBootstrap:
    """Resample machines with replacement and keep their cutoffs together."""

    def __init__(self, ids, seed, draws=5000):
        self.codes, machines = pd.factorize(ids, sort=True)
        self.counts = np.bincount(self.codes)
        self.sample = np.random.default_rng(seed).integers(
            len(machines), size=(draws, len(machines)), dtype=np.int32)

    def interval(self, paired_difference):
        totals = np.bincount(self.codes, weights=paired_difference)
        draws = totals[self.sample].sum(axis=1) / self.counts[self.sample].sum(axis=1)
        return np.quantile(draws, [.025, .975])


LOG_FIELDS = ["run", "model", "features", "hyperparameters", "validation_log_loss",
              "seed", "n_train", "n_validation", "date_utc", "owner",
              "run_id", "timestamp_utc", "protocol", "inner_log_loss"]


def append_experiment(path, row):
    """Extend the original CSV schema once, then append each result as it finishes."""
    if path.exists():
        with path.open(newline="") as stream:
            previous = csv.DictReader(stream)
            if previous.fieldnames != LOG_FIELDS:
                if previous.fieldnames != LOG_FIELDS[:10]:
                    raise ValueError("Unrecognized experiment log columns")
                with NamedTemporaryFile(mode="w", newline="", dir=path.parent, delete=False) as tmp:
                    writer = csv.DictWriter(tmp, fieldnames=LOG_FIELDS)
                    writer.writeheader()
                    for index, old in enumerate(previous, 1):
                        writer.writerow({**old, "run_id": f"v1-{index:03}", "protocol": "v1"})
                os.replace(tmp.name, path)
    write_header = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
