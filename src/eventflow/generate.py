from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_events
from .seq_data import Encoder, load_protocol
from .train_deep import Config, _build_model, _device


@torch.no_grad()
def sample_events(model, prefix, horizon, samples, temperature, seed, device):
    if horizon < 1 or samples < 1 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Positive horizon, samples and finite temperature required")
    prefix = np.asarray(prefix, dtype=np.int64)
    if prefix.ndim != 1 or not len(prefix) or (prefix < 1).any():
        raise ValueError("Expected a nonempty, unpadded event prefix")
    model.eval()
    events = torch.as_tensor(prefix, device=device).repeat(samples, 1)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    generated = []
    for _ in range(horizon):
        context = torch.empty((*events.shape, 0), dtype=torch.long, device=device)
        numeric = torch.empty((*events.shape, 0), dtype=torch.float32, device=device)
        logits = model(events, context, numeric)[:, -1]
        prob = torch.softmax(logits.float() / temperature, dim=-1).cpu()
        next_class = torch.multinomial(prob, 1, generator=rng)
        generated.append(next_class[:, 0].numpy())
        events = torch.cat([events, (next_class + 1).to(device)], dim=1)
    return np.stack(generated, axis=1)


def ngram_counts(sequences, n):
    return Counter(tuple(row[i:i+n]) for row in sequences for i in range(len(row)-n+1))


def js_divergence(a, b):
    keys = sorted(a.keys() | b.keys())
    if not keys or not sum(a.values()) or not sum(b.values()):
        raise ValueError("Cannot compare empty distributions")
    p = np.array([a[k] for k in keys], dtype=float)
    q = np.array([b[k] for k in keys], dtype=float)
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    def kl(x):
        positive = x > 0
        return np.sum(x[positive] * np.log(x[positive] / m[positive]))
    return float((kl(p) + kl(q)) / 2)


def diversity(sequences):
    result = {"duplicate_rate": 1 - len({tuple(s) for s in sequences}) / len(sequences)}
    for n in (2, 3):
        counts = ngram_counts(sequences, n)
        result[f"unique_{n}gram_ratio"] = len(counts) / sum(counts.values())
    return result


def diagnostics(real, generated):
    real, generated = np.asarray(real), np.asarray(generated)
    if (real.ndim != 2 or generated.ndim != 3 or
            generated.shape[0] != len(real) or generated.shape[2] != real.shape[1]
            or not len(real) or generated.shape[1] < 2 or real.shape[1] < 3):
        raise ValueError("Need matched prefixes, horizon >= 3, and >= 2 rollouts each")
    real_div = diversity(real)
    rows = []
    for j in range(generated.shape[1]):
        sample = generated[:, j, :]
        row = {f"js_{n}gram": js_divergence(ngram_counts(real, n), ngram_counts(sample, n))
               for n in (1, 2, 3)}
        row.update(diversity(sample))
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {key: {"mean": float(frame[key].mean()),
                     "sampling_std": float(frame[key].std(ddof=1))} for key in frame}
    summary["real_diversity"] = real_div
    summary["within_prefix_duplicate_rate"] = float(np.mean([
        1 - len({tuple(s) for s in group}) / len(group) for group in generated]))
    return summary, frame


def run_generation(args):
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.samples < 2 or args.temperature <= 0 or not np.isfinite(args.temperature):
        raise ValueError("Use at least two samples and positive finite temperature")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**saved["config"])
    if cfg.cell != "transformer" or not cfg.event_only or saved["numeric"] or cfg.context:
        raise ValueError("Generation requires a full-context, event-only Transformer checkpoint")
    encoder = Encoder.from_dict(saved["encoder"])
    classes, protocol = load_protocol(args.results)
    if classes != encoder.classes:
        raise ValueError("Checkpoint class order differs from the shared protocol")
    manifest = pd.read_csv(args.results / "rollout_manifest.csv")
    if args.limit:
        manifest = manifest.head(args.limit).copy()
    valid_ids = set(protocol.loc[protocol.role == "validation", "sequence_id"])
    if not set(manifest.sequence_id) <= valid_ids:
        raise ValueError("Rollouts must use held-out validation machines only")
    if "fit_ids" not in saved or set(saved["fit_ids"]) & set(manifest.sequence_id):
        raise ValueError("Missing fitting provenance or overlap with rollout machines")
    horizons = manifest.horizon.unique()
    if len(horizons) != 1 or horizons[0] < 3 or manifest.duplicated(["sequence_id", "prefix_len"]).any():
        raise ValueError("Expected unique prefixes with a common horizon >= 3")
    horizon = int(horizons[0])
    events = load_events(args.data / "eventflow_train.csv")
    events = events[events.sequence_id.isin(set(manifest.sequence_id))]
    sequences = {k: g.event_type.astype(str).map({c: i+1 for i, c in enumerate(classes)}).to_numpy()
                 for k, g in events.groupby("sequence_id", sort=False)}
    device = _device(args.device)
    model = _build_model(cfg, encoder, 0).to(device)
    model.load_state_dict(saved["state"])
    temperature = saved["temperature"] * args.temperature
    real, generated, records = [], [], []
    for i, row in enumerate(manifest.itertuples(index=False)):
        seq, length = sequences[row.sequence_id], int(row.prefix_len)
        if length < 1 or length + horizon > len(seq):
            raise ValueError("Real continuation extends beyond the observed training sequence")
        suffix = seq[length:length+horizon] - 1
        sampled = sample_events(model, seq[:length], horizon, args.samples,
                                temperature, args.seed+i, device)
        real.append(suffix)
        generated.append(sampled)
        records.append(dict(sequence_id=row.sequence_id, prefix_len=length,
                            prefix_tail=[classes[v-1] for v in seq[max(0,length-10):length]],
                            real=[classes[v] for v in suffix],
                            generated=[[classes[v] for v in s] for s in sampled]))
        if (i+1) % 25 == 0:
            print(f"Generated {i+1}/{len(manifest)} prefixes", flush=True)
    real, generated = np.asarray(real), np.asarray(generated)
    summary, replicate_metrics = diagnostics(real, generated)
    summary.update(n_prefixes=len(real), samples_per_prefix=args.samples,
                   n_generated=len(real)*args.samples, horizon=horizon,
                   calibration_temperature=saved["temperature"], sampling_multiplier=args.temperature,
                   effective_temperature=temperature, seed=args.seed, smoke=bool(saved.get("smoke")),
                   checkpoint=str(args.checkpoint),
                   checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   note="JS uses natural logs. Diversity uses matched corpus sizes. Sampling SD is not a CI. "
                        "Diversity alone does not establish absence of memorization; n-grams do not establish long-range realism.")
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "rollouts.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    replicate_metrics.to_csv(args.output / "diagnostics_by_sample.csv", index=False)
    manifest.to_csv(args.output / "evaluated_manifest.csv", index=False)
    np.savez_compressed(args.output / "rollouts.npz", real=real, generated=generated,
                        classes=np.asarray(classes))
    print(json.dumps(summary, indent=2), flush=True)
