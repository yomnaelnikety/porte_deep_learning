

import numpy as np
import pandas as pd
import pytest

from eventflow.seq_data import CONTEXT, IGNORE, Encoder

CLASSES = ["A", "B", "C"]


def make_events(lengths, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m, n in enumerate(lengths):
        t = np.cumsum(rng.uniform(0.1, 2.0, n))
        for i in range(n):
            rows.append(dict(
                sequence_id=f"M{m:02}", event_index=i, timestamp=t[i],
                event_type=CLASSES[(i + m) % 3], subsystem="S" + str(i % 2),
                severity=float(i % 4), shift_code=str(i % 3), operator_group="G1",
                device_family="F" + str(m % 2), firmware_channel="FW1",
                load_reading=np.nan if i % 17 == 0 else float(m + i / 10),
                temperature_reading=float(rng.normal())))
    df = pd.DataFrame(rows)
    for c in ["event_type", "subsystem", "shift_code", "operator_group",
              "device_family", "firmware_channel"]:
        df[c] = df[c].astype("category")
    return df


@pytest.fixture
def setup():
    events = make_events([130, 140, 160, 125])
    encoder = Encoder.fit(events, ["M00", "M01"], CLASSES)
    return events, encoder, encoder.transform(events)


def test_targets_are_next_event(setup):
    events, _, store = setup
    m = store.index["M02"]
    rows = slice(store.offsets[m], store.offsets[m] + store.lengths[m])
    classes = events.event_type.astype(str).to_numpy()[rows]
    expected = [CLASSES.index(c) for c in classes[1:]]
    assert store.target[rows][:-1].tolist() == expected
    assert store.target[rows][-1] == IGNORE  # last event has no next event


def test_prefix_window_never_sees_the_label(setup):
    events, _, store = setup
    cols = store.columns(["time", "sensor"])
    m, L = store.index["M02"], 100
    batch = store.prefix_batch([m], [L], cols)
    assert batch["last"][0] == L - 1
    seq = events[events.sequence_id == "M02"].event_type.astype(str).to_numpy()
    fed = batch["event"][0, : L]
    assert fed.tolist() == [CLASSES.index(c) + 1 for c in seq[:L]]
    assert batch["label"][0] == CLASSES.index(seq[L])


def test_short_context_window(setup):
    _, _, store = setup
    m = store.index["M01"]
    batch = store.prefix_batch([m, m], [100, 120], store.columns(["sensor"]), context=32)
    assert batch["event"].shape[1] == 32
    full = store.prefix_batch([m], [120], store.columns(["sensor"]))
    assert (batch["event"][1] == full["event"][0, 88:120]).all()


def test_normalization_uses_fitting_machines_only(setup):
    events, encoder, _ = setup
    fit = events[events.sequence_id.isin(["M00", "M01"])]
    assert np.isclose(encoder.mean[0], fit.load_reading.mean())
    changed = events.copy()
    changed.loc[changed.sequence_id == "M03", "load_reading"] += 1000
    assert Encoder.fit(changed, ["M00", "M01"], CLASSES).mean == encoder.mean


def test_missing_values_become_zero_with_flag(setup):
    events, _, store = setup
    row = int(np.flatnonzero(events.load_reading.isna())[0])
    names = store.numeric_names
    assert store.numeric[row, names.index("sensor__load_reading")] == 0
    assert store.numeric[row, names.index("missing__load_reading")] == 1


def test_relative_sensor_is_causal(setup):
    events, encoder, store = setup
    later = events.copy()
    m2 = later.sequence_id == "M02"
    later.loc[m2 & (later.event_index >= 80), "temperature_reading"] += 50
    changed = encoder.transform(later)
    col = store.numeric_names.index("relative__temperature_reading")
    start = store.offsets[store.index["M02"]]
    assert np.allclose(store.numeric[start:start + 80, col],
                       changed.numeric[start:start + 80, col])


def test_full_context_training_targets(setup):
    _, _, store = setup
    m = store.index["M00"]
    batch = store.train_batch([m], store.columns(["sensor"]))
    trained = np.flatnonzero(batch["target"][0] != IGNORE)
    assert trained[0] == 89  # step 89 predicts row 90 (prefix 90)
    assert trained[-1] == store.lengths[m] - 2


def test_short_context_training_targets(setup):
    _, _, store = setup
    rng = np.random.default_rng(1)
    batch = store.train_batch(np.arange(4), store.columns(["sensor"]), context=32, rng=rng, tail=16)
    for i in range(4):
        trained = np.flatnonzero(batch["target"][i] != IGNORE)
        assert len(trained) and trained[0] >= 31 and trained[-1] < 48


def test_unknown_category_maps_to_zero(setup):
    events, encoder, _ = setup
    other = events.copy()
    other["subsystem"] = other.subsystem.cat.add_categories(["NEW"])
    other.loc[other.index[:3], "subsystem"] = "NEW"
    store = encoder.transform(other)
    assert (store.context[:3, CONTEXT.index("subsystem")] == 0).all()
