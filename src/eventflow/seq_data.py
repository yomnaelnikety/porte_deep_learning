
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CATEGORICAL, MAX_PREFIX, MIN_PREFIX

# Categorical context embedded at every step. event_type is handled separately,
# and severity (0-3) is treated as a category rather than a number.
CONTEXT = ["subsystem", "severity", "shift_code", "operator_group",
           "device_family", "firmware_channel"]
NOT_SENSORS = {"sequence_id", "event_index", "timestamp", "severity", *CATEGORICAL}
IGNORE = -100  # target value that PyTorch's cross_entropy skips



def load_protocol(results):
    """Read the shared protocol files: class order, cutoffs and the inner machine split."""
    results = Path(results)
    classes = json.loads((results / "classes.json").read_text())
    manifest = pd.read_csv(results / "prefix_manifest.csv")
    inner = pd.read_csv(results / "classical_inner_split.csv")
    roles = dict(zip(inner.sequence_id, inner.role))
    manifest["role"] = [roles.get(k, "validation") if s == "train" else "validation"
                        for k, s in zip(manifest.sequence_id, manifest.split)]
    return classes, manifest




@dataclass
class Encoder:
    classes: list
    sensors: list
    vocabs: dict = field(default_factory=dict)
    mean: list = field(default_factory=list)
    std: list = field(default_factory=list)
    gap_mean: float = 0.0
    gap_std: float = 1.0

    @classmethod
    def fit(cls, events, fit_ids, classes):
        """Learn vocabularies and normalization from fitting machines only."""
        fit = events[events.sequence_id.isin(set(fit_ids))]
        if fit.empty:
            raise ValueError("No fitting machines found in the event table")
        sensors = [c for c in events.columns if c not in NOT_SENSORS]
        vocabs = {c: sorted(_as_text(fit[c]).unique().tolist()) for c in CONTEXT}
        values = fit[sensors].to_numpy(dtype=np.float64)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std[~(std > 1e-8)] = 1.0
        gaps = _log_gaps(fit)
        return cls(classes=list(classes), sensors=sensors, vocabs=vocabs,
                   mean=mean.tolist(), std=std.tolist(),
                   gap_mean=float(gaps.mean()), gap_std=float(gaps.std() or 1.0))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def numeric_names(self):
        return (["time__log_gap", "position__index"]
                + [f"sensor__{c}" for c in self.sensors]
                + [f"missing__{c}" for c in self.sensors]
                + [f"relative__{c}" for c in self.sensors])

    def transform(self, events, labeled=True):
        """Encode an event table (sorted by machine and index) into a SequenceStore."""
        if labeled and not set(events.event_type.astype(str)) <= set(self.classes):
            raise ValueError("Event table contains classes outside classes.json")
        codes, keys = pd.factorize(events.sequence_id, sort=False)
        if (np.diff(codes) < 0).any():
            raise ValueError("Rows must be grouped by sequence_id (use load_events)")
        lengths = np.bincount(codes)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])

        event_map = {c: i + 1 for i, c in enumerate(self.classes)}
        event = events.event_type.astype(str).map(event_map).fillna(0).to_numpy(np.int64)

        context = np.zeros((len(events), len(CONTEXT)), dtype=np.int64)
        for j, col in enumerate(CONTEXT):
            mapping = {v: i + 1 for i, v in enumerate(self.vocabs[col])}
            context[:, j] = _as_text(events[col]).map(mapping).fillna(0).to_numpy(np.int64)

        position = np.arange(len(events)) - np.repeat(offsets, lengths)
        raw = events[self.sensors].to_numpy(dtype=np.float64)
        missing = np.isnan(raw)
        scaled = np.where(missing, 0.0, (raw - np.asarray(self.mean)) / np.asarray(self.std))
        relative = _causal_relative(scaled, ~missing, codes)
        gaps = (_log_gaps(events) - self.gap_mean) / self.gap_std
        numeric = np.column_stack([gaps, position / MAX_PREFIX, scaled,
                                   missing.astype(np.float64), relative]).astype(np.float32)

        # Row r's label is the class of row r + 1 in the same machine (or IGNORE).
        target = np.full(len(events), IGNORE, dtype=np.int64)
        if labeled:
            same = np.zeros(len(events), dtype=bool)
            same[:-1] = codes[1:] == codes[:-1]
            target[same] = event[1:][same[:-1]] - 1
        return SequenceStore(keys=np.asarray(keys), offsets=offsets, lengths=lengths,
                             event=event, context=context, numeric=numeric, target=target,
                             numeric_names=self.numeric_names())


def _as_text(series):
    """Turn a column into strings, writing whole numbers without '.0' (severity 2.0 -> '2')."""
    values = series.astype(object)
    return values.map(lambda v: "<nan>" if pd.isna(v) else
                      str(int(v)) if isinstance(v, (float, np.floating)) and float(v).is_integer()
                      else str(v))


def _log_gaps(events):
    """log(1 + time since the previous event of the same machine); 0 for the first event."""
    gaps = events.groupby("sequence_id", sort=False, observed=True).timestamp.diff()
    return np.log1p(gaps.fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64))


def _causal_relative(scaled, valid, codes):
    """Each sensor minus the running mean of that machine's earlier and current values.

    Only rows up to and including the current one are used, so the feature never
    looks into the future. It removes each machine's own baseline level.
    """
    total = np.zeros_like(scaled)
    count = np.zeros_like(scaled)
    values = np.where(valid, scaled, 0.0)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    for s, e in zip(starts, ends):
        total[s:e] = np.cumsum(values[s:e], axis=0)
        count[s:e] = np.cumsum(valid[s:e], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        running = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return np.where(valid, scaled - running, 0.0)


# store

@dataclass
class SequenceStore:
    """All machines concatenated row-wise; machine i owns rows offsets[i]:offsets[i]+lengths[i]."""
    keys: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    event: np.ndarray
    context: np.ndarray
    numeric: np.ndarray
    target: np.ndarray
    numeric_names: list

    def __post_init__(self):
        # One all-zero padding row at the end lets index -1 mean "padding".
        self._event = np.r_[self.event, 0]
        self._context = np.vstack([self.context, np.zeros((1, self.context.shape[1]), np.int64)])
        self._numeric = np.vstack([self.numeric, np.zeros((1, self.numeric.shape[1]), np.float32)])
        self._target = np.r_[self.target, IGNORE]
        self.index = {k: i for i, k in enumerate(self.keys)}

    def machines(self, ids):
        return np.array([self.index[k] for k in ids], dtype=np.int64)

    def columns(self, groups):
        """Indices of numeric columns whose name starts with one of the groups."""
        return [i for i, n in enumerate(self.numeric_names) if n.split("__")[0] in groups]

    def _gather(self, rows, cols):
        pad = rows < 0
        rows = np.where(pad, len(self.event), rows)
        return dict(event=self._event[rows], context=self._context[rows],
                    numeric=self._numeric[rows][..., cols], target=self._target[rows])

    # evaluation windows
    def prefix_batch(self, machine, prefix_len, cols, context=0):
        """Windows ending at the last observed row of each prefix.

        A prefix of length L uses rows 0..L-1 and is labeled with row L. With
        context=K > 0 only the last K observed rows are fed to the model.
        Windows are right-padded; `last` is the position whose output is used.
        """
        machine = np.asarray(machine)
        prefix_len = np.asarray(prefix_len)
        if (prefix_len > self.lengths[machine]).any() or (prefix_len < 1).any():
            raise ValueError("Prefix longer than the observed sequence")
        start = prefix_len - context if context else np.zeros_like(prefix_len)
        start = np.maximum(start, 0)
        size = prefix_len - start
        width = int(size.max())
        steps = np.arange(width)[None, :]
        rows = self.offsets[machine][:, None] + start[:, None] + steps
        rows = np.where(steps < size[:, None], rows, -1)
        batch = self._gather(rows, cols)
        batch["last"] = size - 1
        # Label of the prefix = target stored on its last observed row.
        batch["label"] = self._target[self.offsets[machine] + prefix_len - 1]
        return batch

    #  training batches 
    def train_batch(self, machine, cols, context=0, rng=None, tail=32,
                    min_prefix=MIN_PREFIX, max_prefix=MAX_PREFIX):
        """Many-to-many training batch: a label at every valid step.

        Full context (context=0): feed rows 0..e-1 of each machine and train on
        every step whose prefix length r + 1 lies in [min_prefix, max_prefix].
        The model is causal, so the output at step r only depends on rows 0..r,
        which is exactly the prediction for a prefix of length r + 1.

        Short context (0 < context < min_prefix): feed a random chunk of
        context + tail rows and train only on steps that have seen at least
        `context` rows inside the chunk, so the model learns from roughly the
        same amount of history it will get at evaluation time.
        """
        machine = np.asarray(machine)
        n = self.lengths[machine]
        last_input = np.minimum(n - 1, max_prefix)   # rows < last_input have a label
        if context == 0:
            start = np.zeros_like(n)
            stop = last_input
            first_trained = np.zeros_like(n)
        else:
            if context >= min_prefix:
                raise ValueError("Use context=0 or a context shorter than min_prefix")
            rng = rng or np.random.default_rng()
            low = np.maximum(0, min_prefix - context)
            high = last_input - context              # inclusive upper bound for start
            start = low + (rng.random(len(n)) * (high - low + 1)).astype(np.int64)
            stop = np.minimum(start + context + tail, last_input)
            first_trained = np.full_like(n, context - 1)
        size = stop - start
        width = int(size.max())
        steps = np.arange(width)[None, :]
        rows = self.offsets[machine][:, None] + start[:, None] + steps
        valid = steps < size[:, None]
        batch = self._gather(np.where(valid, rows, -1), cols)
        prefix = start[:, None] + steps + 1
        keep = valid & (steps >= first_trained[:, None]) & (prefix >= min_prefix)
        batch["target"] = np.where(keep, batch["target"], IGNORE)
        return batch
