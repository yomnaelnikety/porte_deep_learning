"""Summarize fitting systems for the report."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import SENSORS


def explore(events, out):
    out.mkdir(parents=True, exist_ok=True)
    groups = events.groupby("sequence_id", sort=False, observed=True)
    sizes = groups.size()
    gaps = groups.timestamp.diff().dropna()
    events.event_type.value_counts(normalize=True).rename("fraction").to_csv(
        out / "event_balance.csv")
    events.isna().mean().rename("missing_fraction").to_csv(out / "missingness.csv")
    sizes.describe().to_csv(out / "sequence_lengths.csv", header=["length"])
    gaps.describe(percentiles=[.05, .5, .95, .99]).to_csv(out / "time_gaps.csv", header=["gap"])
    sensors = events[SENSORS]
    sensors.corr().to_csv(out / "sensor_correlations.csv")
    pd.DataFrame({
        "median_consecutive_abs_change": groups[SENSORS].diff().abs().median(),
        "iqr": sensors.quantile(.75) - sensors.quantile(.25),
    }).to_csv(out / "sensor_variability.csv")
    nxt = groups.event_type.shift(-1)
    pd.crosstab(events.event_type, nxt, normalize="index").to_csv(out / "event_transitions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    events.event_type.value_counts(normalize=True).sort_values().plot.barh(
        ax=axes[0], color="#266c86")
    axes[0].set_xlabel("Fraction of training events")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="y", labelsize=7)
    axes[1].hist(np.log1p(gaps), bins=50, color="#266c86")
    axes[1].set(xlabel="log(1 + inter-event time)", ylabel="Count")
    fig.savefig(out / "eventflow_eda.pdf")
    plt.close(fig)
