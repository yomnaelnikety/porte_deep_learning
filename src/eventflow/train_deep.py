

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .analysis import LOG_FIELDS, MachineBootstrap, append_experiment, example_loss
from .data import PROB_PREFIX, PROTOCOL, SEED, load_events, multiclass_loss
from .deep_models import EventSequenceModel
from .seq_data import CONTEXT, IGNORE, Encoder, load_protocol

LENGTH_BINS = [89, 129, 179, 250]  # same buckets as the classical error analysis


# settings

@dataclass
class Config:
    cell: str = "gru"
    hidden: int = 128
    layers: int = 1
    dropout: float = 0.2
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 64  # machines per batch (each gives up to 161 targets)
    max_epochs: int = 40
    patience: int = 4  # epochs without early-stop improvement
    context: int = 0   # 0 = whole prefix (up to 250 events)
    tail: int = 32  # extra trained steps per chunk when context > 0
    use_time: bool = True
    use_sensors: bool = True
    use_context: bool = True
    relative: bool = False  # machine-relative sensors (off unless it helps)
    event_dim: int = 32
    context_dim: int = 8
    seed: int = SEED
    heads: int = 4  # Transformer only
    ff_multiplier: int = 4  # Transformer only
    event_only: bool = False  # generation: no unavailable future covariates
    eval_batch_size: int = 256

    def numeric_groups(self):
        if self.event_only:
            return []
        groups = ["position"]
        if self.use_time:
            groups.append("time")
        if self.use_sensors:
            groups += ["sensor", "missing"]
        if self.relative:
            groups.append("relative")
        return groups

    def feature_label(self):
        if self.event_only:
            return "events_only"
        parts = ["events"] + [name for name, on in [
            ("context", self.use_context), ("time", self.use_time),
            ("sensors", self.use_sensors), ("relative", self.relative)] if on]
        if self.context:
            parts.append(f"last{self.context}")
        return "+".join(parts)


# One change at a time around the default ("coordinate search").
TUNING = {
    "gru_h128": {},
    "gru_h64": {"hidden": 64},
    "gru_h256": {"hidden": 256},
    "gru_h128_2layers": {"layers": 2},
    "gru_h128_drop0.4": {"dropout": 0.4},
    "gru_h128_lr1e-3": {"lr": 1e-3},
    "lstm_h128": {"cell": "lstm"},
}
QUICK_TUNING = ["gru_h128", "gru_h64", "lstm_h128"]

ABLATIONS = {
    "no_time": {"use_time": False},
    "no_sensors": {"use_sensors": False},
    "no_context": {"use_context": False},
    "context_16": {"context": 16},
    "context_64": {"context": 64},
    "add_relative_sensors": {"relative": True},
    "other_seed": {"seed": SEED + 1},   # how much results move from training noise alone
}


# data

class Workspace:
    """Loads the data once so several models can be trained in one process."""

    def __init__(self, args):
        self.args = args
        self.out = args.out
        self.out.mkdir(parents=True, exist_ok=True)
        self.classes, manifest = load_protocol(args.results)
        events = load_events(args.data / "eventflow_train.csv")
        if args.smoke:
            manifest = _smoke_subset(manifest)
            events = events[events.sequence_id.isin(set(manifest.sequence_id))]
        fit_ids = manifest.loc[manifest.role == "fit", "sequence_id"].unique()
        print(f"Encoding {events.sequence_id.nunique()} machines", flush=True)
        self.encoder = Encoder.fit(events, fit_ids, self.classes)
        self.store = self.encoder.transform(events)
        self.fit = self.store.machines(fit_ids)
        self.sets = {role: manifest[manifest.role == role].reset_index(drop=True)
                     for role in ("early_stop", "calibration", "validation")}
        self.device = _device(args.device)
        print(f"Device: {self.device}", flush=True)


def _smoke_subset(manifest):
    rng = np.random.default_rng(0)
    keep = []
    for role, n in (("fit", 200), ("early_stop", 40), ("calibration", 40), ("validation", 60)):
        ids = manifest.loc[manifest.role == role, "sequence_id"].unique()
        keep += rng.choice(ids, size=min(n, len(ids)), replace=False).tolist()
    return manifest[manifest.sequence_id.isin(set(keep))]


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _tensors(batch, device):
    return (torch.from_numpy(batch["event"]).to(device),
            torch.from_numpy(batch["context"]).to(device),
            torch.from_numpy(batch["numeric"]).to(device))


def _build_model(cfg, encoder, n_numeric):
    if cfg.cell == "transformer":
        from .transformer import EventTransformer
        return EventTransformer(
            n_classes=len(encoder.classes),
            context_sizes=[len(encoder.vocabs[c]) for c in CONTEXT],
            n_numeric=n_numeric, hidden=cfg.hidden, layers=cfg.layers,
            dropout=cfg.dropout, event_dim=cfg.event_dim, context_dim=cfg.context_dim,
            use_context=cfg.use_context and not cfg.event_only,
            heads=cfg.heads, ff_multiplier=cfg.ff_multiplier)
    return EventSequenceModel(
        n_classes=len(encoder.classes),
        context_sizes=[len(encoder.vocabs[c]) for c in CONTEXT],
        n_numeric=n_numeric, cell=cfg.cell, hidden=cfg.hidden, layers=cfg.layers,
        dropout=cfg.dropout, event_dim=cfg.event_dim, context_dim=cfg.context_dim,
        use_context=cfg.use_context)


@torch.no_grad()
def predict_logits(model, store, frame, cols, cfg, device, batch_size=None):
    """Logits for each (sequence_id, prefix_len) row of `frame`, in frame order."""
    model.eval()
    batch_size = batch_size or cfg.eval_batch_size
    machine = store.machines(frame.sequence_id)
    length = frame.prefix_len.to_numpy()
    order = np.argsort(length, kind="stable")      # similar lengths -> less padding
    n_classes = model.head[-1].out_features
    logits = np.empty((len(frame), n_classes), dtype=np.float32)
    labels = np.empty(len(frame), dtype=np.int64)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = store.prefix_batch(machine[idx], length[idx], cols, cfg.context)
        out = model(*_tensors(batch, device))
        last = torch.from_numpy(batch["last"]).to(device)
        logits[idx] = out[torch.arange(len(idx), device=device), last].float().cpu().numpy()
        labels[idx] = batch["label"]
    return logits, labels


def softmax(logits, temperature=1.0):
    z = logits.astype(np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    """Find T > 0 minimizing log loss of softmax(logits / T) on held-out machines.

    T > 1 softens overconfident predictions, T < 1 sharpens underconfident ones.
    Dividing all logits by the same T keeps their order, so the top class and the accuracy are unchanged; only the confidence changes."""
    z = torch.from_numpy(logits.astype(np.float64))
    y = torch.from_numpy(labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


# training

def train_model(ws, cfg, run):
    """Train one configuration, calibrate it, evaluate it and log it."""
    folder = ws.out / "runs" / run
    if folder.exists() or (ws.args.models / f"{run}.pt").exists():
        raise FileExistsError(f"Run {run} already exists; choose a new run name")
    if cfg.event_only and cfg.cell != "transformer":
        raise ValueError("event_only is supported by the Transformer pipeline")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    cols = ws.store.columns(cfg.numeric_groups())
    model = _build_model(cfg, ws.encoder, len(cols)).to(ws.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {run}: {cfg.feature_label()} | {n_params:,} parameters", flush=True)

    history, best_loss, best_state, best_epoch, waited = [], np.inf, None, 0, 0
    started = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tick, total, count = time.time(), 0.0, 0
        order = rng.permutation(ws.fit)
        for start in range(0, len(order), cfg.batch_size):
            batch = ws.store.train_batch(order[start:start + cfg.batch_size], cols,
                                         cfg.context, rng, cfg.tail)
            target = torch.from_numpy(batch["target"]).to(ws.device)
            n = int((target != IGNORE).sum())
            if n == 0:
                continue
            logits = model(*_tensors(batch, ws.device))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                                   ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoids exploding gradients
            opt.step()
            total += loss.item() * n
            count += n
        stop_logits, stop_y = predict_logits(model, ws.store, ws.sets["early_stop"], cols, cfg, ws.device)
        stop_loss = float(F.cross_entropy(torch.from_numpy(stop_logits), torch.from_numpy(stop_y)))
        lr = opt.param_groups[0]["lr"]
        schedule.step(stop_loss)
        history.append(dict(epoch=epoch, train_loss=total / max(count, 1), early_stop_loss=stop_loss,
                            lr=lr, seconds=time.time() - tick))
        flag = ""
        if stop_loss < best_loss - 1e-4:
            best_loss, best_state, best_epoch, waited, flag = (
                stop_loss, copy.deepcopy(model.state_dict()), epoch, 0, "  *best")
        else:
            waited += 1
        print(f"{run} epoch {epoch:2d}  train {total / max(count, 1):.4f}  "
              f"early-stop {stop_loss:.4f}  lr {lr:.1e}  {time.time() - tick:4.0f}s{flag}", flush=True)
        if waited >= cfg.patience:
            break
    model.load_state_dict(best_state)

    # Temperature on the calibration machines, then score validation once.
    cal_logits, cal_y = predict_logits(model, ws.store, ws.sets["calibration"], cols, cfg, ws.device)
    temperature = fit_temperature(cal_logits, cal_y)
    val = ws.sets["validation"]
    val_logits, val_y = predict_logits(model, ws.store, val, cols, cfg, ws.device)
    targets = np.asarray(ws.classes)[val_y]
    raw = multiclass_loss(targets, softmax(val_logits), ws.classes)
    calibrated = multiclass_loss(targets, softmax(val_logits, temperature), ws.classes)
    metrics = dict(
        run=run, early_stop_log_loss=best_loss, best_epoch=best_epoch, epochs_run=len(history),
        temperature=temperature, validation_log_loss_raw=raw, validation_log_loss=calibrated,
        validation_accuracy=float((val_logits.argmax(1) == val_y).mean()),
        n_parameters=n_params, train_targets_per_epoch=count, minutes=(time.time() - started) / 60,
        smoke=bool(ws.args.smoke))
    print(f"{run}: early-stop {best_loss:.4f} | T={temperature:.3f} | "
          f"validation raw {raw:.4f} -> calibrated {calibrated:.4f}", flush=True)

    folder = ws.out / "runs" / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    np.savez_compressed(folder / "validation_logits.npz", logits=val_logits, labels=val_y,
                        sequence_id=val.sequence_id.to_numpy(), prefix_len=val.prefix_len.to_numpy())
    predictions = val[["sequence_id", "prefix_len"]].copy()
    predictions["target"] = targets
    predictions[[PROB_PREFIX + c for c in ws.classes]] = softmax(val_logits, temperature)
    predictions.to_csv(folder / "validation_predictions.csv", index=False)
    ws.args.models.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=asdict(cfg), state=model.state_dict(), encoder=ws.encoder.to_dict(),
                    temperature=temperature, numeric=[ws.store.numeric_names[i] for i in cols],
                    fit_ids=ws.store.keys[ws.fit].tolist(),
                    smoke=bool(ws.args.smoke)),
               ws.args.models / f"{run}.pt")
    _log(ws, cfg, run, metrics)
    return metrics


def _log(ws, cfg, run, metrics):
    stamp = datetime.now(timezone.utc).isoformat()
    details = {**asdict(cfg), **{k: metrics[k] for k in (
        "best_epoch", "epochs_run", "temperature", "validation_log_loss_raw", "n_parameters")}}
    row = dict(run=run, model=cfg.cell.upper(), features=cfg.feature_label(),
               hyperparameters=json.dumps(details), validation_log_loss=metrics["validation_log_loss"],
               seed=cfg.seed, n_train=metrics["train_targets_per_epoch"],
               n_validation=len(ws.sets["validation"]),
               date_utc=stamp[:10], owner=ws.args.owner,
               run_id=f"{ws.args.batch_id}-{run}", timestamp_utc=stamp, protocol=PROTOCOL,
               inner_log_loss=metrics["early_stop_log_loss"])
    assert set(row) == set(LOG_FIELDS)
    log_name = "experiments_transformer.csv" if cfg.cell == "transformer" else "experiments_gru.csv"
    append_experiment(ws.out / log_name, row)


# analyses

def _load_run(out, run):
    folder = out / "runs" / run
    data = np.load(folder / "validation_logits.npz", allow_pickle=True)
    metrics = json.loads((folder / "metrics.json").read_text())
    return data, metrics


def selected_analysis(ws, run):
    """Error analysis for the chosen GRU, reusing the classical pipeline's function and file names."""
    from .run import error_analysis  # imported here because run.py needs LightGBM
    data, metrics = _load_run(ws.out, run)
    prob = softmax(data["logits"], metrics["temperature"])
    frame = ws.sets["validation"][["sequence_id", "prefix_len"]].copy()
    frame["target"] = np.asarray(ws.classes)[data["labels"]]
    folder = ws.out / "selected_errors"
    folder.mkdir(exist_ok=True)
    error_analysis(frame, prob, ws.classes, folder)

    # Validation cutoffs are shorter on average than test prefixes. Reweight the
    # validation loss so each length bucket has its test share (inputs only, no labels).
    test = pd.read_csv(ws.args.data / "eventflow_test.csv", usecols=["sequence_id"])
    test_codes = pd.cut(test.groupby("sequence_id").size(), LENGTH_BINS, labels=False).to_numpy()
    val_codes = pd.cut(frame.prefix_len, LENGTH_BINS, labels=False).to_numpy()
    n_bins = len(LENGTH_BINS) - 1
    test_share = np.bincount(test_codes, minlength=n_bins) / len(test_codes)
    val_share = np.bincount(val_codes, minlength=n_bins) / len(val_codes)
    weight = (test_share / val_share)[val_codes]
    loss = example_loss(frame.target, prob, ws.classes)
    names = [f"{a + 1}-{b}" for a, b in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:])]
    summary = dict(run=run, validation_log_loss=float(loss.mean()),
                   reweighted_to_test_lengths=float(np.average(loss, weights=weight)),
                   validation_length_share=dict(zip(names, val_share.round(4).tolist())),
                   test_length_share=dict(zip(names, test_share.round(4).tolist())))
    (folder / "length_reweighting.json").write_text(json.dumps(summary, indent=2))
    print(f"Validation {summary['validation_log_loss']:.4f}; reweighted to test lengths "
          f"{summary['reweighted_to_test_lengths']:.4f}", flush=True)


def paired_rows(ws, reference_run, others):
    """Paired machine-bootstrap intervals, same method and seed as the classical pipeline."""
    ref, ref_metrics = _load_run(ws.out, reference_run)
    targets = np.asarray(ws.classes)[ref["labels"]]
    ref_loss = example_loss(targets, softmax(ref["logits"], ref_metrics["temperature"]), ws.classes)
    bootstrap = MachineBootstrap(ref["sequence_id"], SEED + 3)
    rows = []
    for name, kind, prob in others:
        change = example_loss(targets, prob, ws.classes) - ref_loss
        lo, hi = bootstrap.interval(change)
        rows.append(dict(run=name, kind=kind, log_loss=ref_loss.mean() + change.mean(),
                         delta_vs_full=change.mean(), ci_95_low=lo, ci_95_high=hi))
    return rows


def _selected(out):
    path = out / "selected_gru.json"
    if not path.exists():
        raise SystemExit("Run `python -m eventflow.train_deep tune` first")
    return json.loads(path.read_text())


#  commands

def cmd_train(ws, args):
    overrides = json.loads(args.config) if args.config else {}
    train_model(ws, replace(Config(), **overrides), args.run)


def cmd_tune(ws, args):
    names = QUICK_TUNING if args.quick else list(TUNING)
    results = []
    for name in names:
        folder = ws.out / "runs" / name
        if args.resume and (folder / "metrics.json").exists():
            print(f"Skipping {name} (already trained)")
            results.append(json.loads((folder / "metrics.json").read_text()))
            continue
        results.append(train_model(ws, replace(Config(), **TUNING[name]), name))
    # Choose on the early-stop machines, never on validation.
    best = min(results, key=lambda m: m["early_stop_log_loss"])
    config = json.loads((ws.out / "runs" / best["run"] / "config.json").read_text())
    (ws.out / "selected_gru.json").write_text(json.dumps(dict(run=best["run"], config=config,
                                                              metrics=best), indent=2))
    pd.DataFrame(results).to_csv(ws.out / "tuning_gru.csv", index=False)
    print(f"\nSelected {best['run']} (early-stop {best['early_stop_log_loss']:.4f}, "
          f"validation {best['validation_log_loss']:.4f})")
    selected_analysis(ws, best["run"])


def cmd_ablate(ws, args):
    chosen = _selected(ws.out)
    base = Config(**chosen["config"])
    others = []
    for name, change in ABLATIONS.items():
        if all(getattr(base, k) == v for k, v in change.items()):
            continue  # the selected model already has this setting
        run = f"{chosen['run']}__{name}"
        folder = ws.out / "runs" / run
        if not (args.resume and (folder / "metrics.json").exists()):
            train_model(ws, replace(base, **change), run)
        data, metrics = _load_run(ws.out, run)
        others.append((name, "architecture_or_input", softmax(data["logits"], metrics["temperature"])))
    data, _ = _load_run(ws.out, chosen["run"])
    others.append(("no_temperature", "calibration", softmax(data["logits"])))
    rows = paired_rows(ws, chosen["run"], others)
    pd.DataFrame(rows).to_csv(ws.out / "ablations_gru.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


def cmd_compare(ws, args):
    """GRU versus the selected LightGBM on the same 10,400 validation rows."""
    chosen = _selected(ws.out)
    data, metrics = _load_run(ws.out, chosen["run"])
    other = pd.read_csv(args.other)
    keys = pd.DataFrame(dict(sequence_id=data["sequence_id"], prefix_len=data["prefix_len"]))
    other = keys.merge(other, on=["sequence_id", "prefix_len"], how="left", validate="one_to_one")
    lgbm = other[[PROB_PREFIX + c for c in ws.classes]].to_numpy()
    if np.isnan(lgbm).any():
        raise ValueError("The other predictions do not cover all validation rows")
    gru = softmax(data["logits"], metrics["temperature"])
    rows = paired_rows(ws, chosen["run"], [
        ("lightgbm_selected", "model", lgbm),
        # Exploratory: equal weights, nothing tuned on validation.
        ("average_gru_lightgbm", "ensemble", (gru + lgbm) / 2),
    ])
    pd.DataFrame(rows).to_csv(ws.out / "compare_gru_vs_classical.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def cmd_predict(args):
    """Test probabilities from a saved checkpoint (trained on the fitting machines)."""
    run = args.run or _selected(args.out)["run"]
    saved = torch.load(args.models / f"{run}.pt", map_location="cpu", weights_only=False)
    cfg = Config(**saved["config"])
    encoder = Encoder.from_dict(saved["encoder"])
    test = load_events(args.data / "eventflow_test.csv")
    store = encoder.transform(test, labeled=False)
    cols = [store.numeric_names.index(n) for n in saved["numeric"]]
    model = _build_model(cfg, encoder, len(cols))
    model.load_state_dict(saved["state"])
    frame = pd.DataFrame(dict(sequence_id=store.keys, prefix_len=store.lengths))
    logits, _ = predict_logits(model, store, frame, cols, cfg, torch.device("cpu"))
    prob = softmax(logits, saved["temperature"])
    result = pd.DataFrame(prob, columns=[PROB_PREFIX + c for c in encoder.classes])
    result.insert(0, "sequence_id", frame.sequence_id)
    path = args.out / f"test_predictions_{run}.csv"
    result.to_csv(path, index=False)
    print(f"Wrote {path}. Score it once, at the end, with:\n"
          f"  python -m eventflow.run score --predictions {path}")


def cmd_smoke(args):
    """Small, fast run of every stage, written to models/smoke (ignored by git)."""
    args.out = args.models / "smoke"
    args.models = args.models / "smoke"
    args.smoke = True
    ws = Workspace(args)
    fast = dict(hidden=32, max_epochs=2, patience=1)
    for name, change in [("smoke_gru", {}), ("smoke_lstm_ctx16", {"cell": "lstm", "context": 16})]:
        train_model(ws, replace(Config(), **fast, **change), name)
    rows = paired_rows(ws, "smoke_gru", [("lstm_ctx16", "check", softmax(
        _load_run(ws.out, "smoke_lstm_ctx16")[0]["logits"]))])
    print(pd.DataFrame(rows).to_string(index=False))
    args.run = "smoke_gru"
    cmd_predict(args)
    print("\nSmoke test passed. The real commands write to results/deep and models/.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["smoke", "train", "tune", "ablate", "compare", "predict"])
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="shared protocol files")
    parser.add_argument("--out", type=Path, default=Path("results/deep"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--owner", default="Eldana Ashirova")
    parser.add_argument("--run", default=None, help="run name (train/predict)")
    parser.add_argument("--config", default=None,
                        help='JSON overrides for train, e.g. \'{"hidden": 64}\'')
    parser.add_argument("--quick", action="store_true", help="tune only 3 configurations")
    parser.add_argument("--resume", action="store_true", help="skip runs that already finished")
    parser.add_argument("--other", type=Path, default=Path("results/validation_predictions.csv"),
                        help="LightGBM validation predictions for compare")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    args = parser.parse_args()
    args.smoke = False
    args.batch_id = uuid4().hex[:8]
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.command == "smoke":
        return cmd_smoke(args)
    if args.command == "predict":
        return cmd_predict(args)
    if args.command == "train" and not args.run:
        args.run = "gru_default"
    ws = Workspace(args)
    {"train": cmd_train, "tune": cmd_tune, "ablate": cmd_ablate, "compare": cmd_compare}[args.command](ws, args)


if __name__ == "__main__":
    main()
