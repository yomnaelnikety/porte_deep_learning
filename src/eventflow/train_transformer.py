import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from .analysis import MachineBootstrap, example_loss
from .data import PROB_PREFIX, SEED, multiclass_loss
from .generate import run_generation
from .train_deep import Config, Workspace, train_model, cmd_predict


BASE = Config(cell="transformer", layers=2, lr=1e-3, batch_size=32, eval_batch_size=32)
TUNING = {"transformer_h128": {}, "transformer_h64": {"hidden": 64},
          "transformer_1layer": {"layers": 1}, "transformer_lr3e-4": {"lr": 3e-4}}
ABLATIONS = {"no_time": {"use_time": False}, "no_sensors": {"use_sensors": False},
             "no_context": {"use_context": False}, "context16": {"context": 16},
             "context64": {"context": 64}, "other_seed": {"seed": SEED+1}}


def checked_config(overrides, generator=False):
    cfg = replace(BASE, **overrides)
    if cfg.cell != "transformer" or cfg.hidden < 1 or cfg.heads < 1 or cfg.hidden % cfg.heads:
        raise ValueError("Expected Transformer with positive hidden divisible by heads")
    if min(cfg.layers, cfg.ff_multiplier, cfg.batch_size, cfg.eval_batch_size,
           cfg.max_epochs, cfg.patience) < 1 or cfg.lr <= 0 or not 0 <= cfg.dropout < 1:
        raise ValueError("Invalid training dimensions or optimization settings")
    if cfg.context < 0 or cfg.context >= 90:
        raise ValueError("Context must be 0 (full prefix) or 1..89")
    if generator:
        cfg = replace(cfg, event_only=True, use_time=False, use_sensors=False,
                      use_context=False, relative=False, context=0)
    return cfg


def fit_run(ws, cfg, name, resume=False):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Run must be a plain directory name")
    folder = ws.out / "runs" / name
    if resume and (folder / "metrics.json").exists():
        previous = json.loads((folder / "config.json").read_text())
        if asdict(Config(**previous)) != asdict(cfg) or not (ws.args.models / f"{name}.pt").exists():
            raise ValueError("Resume requires identical configuration and an existing checkpoint")
        return json.loads((folder / "metrics.json").read_text())
    return train_model(ws, cfg, name)


def comparison(args, run, filename=None):
    ref = pd.read_csv(args.out / "runs" / run / "validation_predictions.csv")
    classes = json.loads((args.results / "classes.json").read_text())
    cols = [PROB_PREFIX+c for c in classes]
    keys = ["sequence_id", "prefix_len"]
    if ref.duplicated(keys).any():
        raise ValueError("Duplicate reference keys")
    multiclass_loss(ref.target, ref[cols].to_numpy(), classes)
    loss = example_loss(ref.target, ref[cols].to_numpy(), classes)
    bootstrap = MachineBootstrap(ref.sequence_id.to_numpy(), SEED+3)
    rows = [dict(model=run, log_loss=float(loss.mean()), delta_vs_transformer=0.0,
                 ci_95_low=0.0, ci_95_high=0.0)]
    for path in args.other:
        other = pd.read_csv(path)
        aligned = ref[keys].merge(other, on=keys, how="left", validate="one_to_one")
        if aligned[cols].isna().any().any() or len(other) != len(ref):
            raise ValueError(f"Prediction keys do not match: {path}")
        if "target" in aligned and not np.array_equal(aligned.target, ref.target):
            raise ValueError("Mismatched target labels")
        multiclass_loss(ref.target, aligned[cols].to_numpy(), classes)
        delta = example_loss(ref.target, aligned[cols].to_numpy(), classes) - loss
        low, high = bootstrap.interval(delta)
        rows.append(dict(model=str(path), log_loss=float(loss.mean()+delta.mean()),
                         delta_vs_transformer=float(delta.mean()), ci_95_low=low, ci_95_high=high))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / (filename or f"comparison_{run}.csv"), index=False)
    print(frame.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["smoke", "train", "tune", "ablate", "train-generator",
                                       "generate", "compare", "predict"])
    p.add_argument("--data", type=Path, default=Path("data/raw"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("results/transformer"))
    p.add_argument("--models", type=Path, default=Path("models/transformer"))
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--owner", default="Yomna El Nikety")
    p.add_argument("--run")
    p.add_argument("--config", default="{}", help="JSON configuration overrides")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/transformer/generation"))
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0, help="Multiplier on calibrated temperature")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--limit", type=int, default=0, help="First N rollout prefixes (0 = all)")
    p.add_argument("--other", nargs="+", type=Path, default=[Path("results/validation_predictions.csv"),
                   Path("results/deep/selected_errors/validation_predictions.csv")])
    args = p.parse_args()
    if args.threads < 1 or args.limit < 0:
        p.error("threads must be positive and limit nonnegative")
    torch.set_num_threads(args.threads)
    args.smoke = args.command == "smoke"
    args.batch_id = uuid4().hex[:8]
    selected = args.out / "selected_transformer.json"
    if args.command == "generate":
        if not args.checkpoint:
            p.error("generate requires --checkpoint")
        return run_generation(args)
    if args.command in ("compare", "predict"):
        args.run = args.run or json.loads(selected.read_text())["run"]
        return comparison(args, args.run) if args.command == "compare" else cmd_predict(args)
    overrides = json.loads(args.config)
    if args.smoke:
        args.out = args.models / ("smoke_" + args.batch_id)
        args.models = args.out / "checkpoints"
    ws = Workspace(args)
    if args.smoke:
        fast = dict(hidden=32, layers=1, max_epochs=2, patience=2, batch_size=16, eval_batch_size=16)
        for name, generator in [("smoke_transformer", False), ("smoke_generator", True)]:
            fit_run(ws, checked_config(fast, generator), name)
        args.checkpoint = args.models / "smoke_generator.pt"
        args.output = args.out / "generation"
        args.limit, args.samples = 4, 3
        run_generation(args)
        print(f"Smoke checks complete: {args.out}. These are not reportable experiments.")
    elif args.command in ("train", "train-generator"):
        generator = args.command == "train-generator"
        fit_run(ws, checked_config(overrides, generator),
                args.run or ("generator_event_only" if generator else "transformer_default"), args.resume)
    elif args.command == "tune":
        metrics = []
        for name, delta in TUNING.items():
            metrics.append(fit_run(ws, checked_config({**overrides, **delta}), name, args.resume))
        best = min(metrics, key=lambda r: r["early_stop_log_loss"])
        cfg = json.loads((args.out / "runs" / best["run"] / "config.json").read_text())
        selected.write_text(json.dumps(dict(run=best["run"], config=cfg, metrics=best), indent=2))
        pd.DataFrame(metrics).to_csv(args.out / "tuning_transformer.csv", index=False)
        print(f"Selected {best['run']} by early-stop loss")
    elif args.command == "ablate":
        chosen = json.loads(selected.read_text())
        cfg = Config(**chosen["config"])
        paths = []
        for name, change in ABLATIONS.items():
            run = chosen["run"]+"__"+name
            fit_run(ws, replace(cfg, **change), run, args.resume)
            paths.append(args.out / "runs" / run / "validation_predictions.csv")
        args.other = paths
        comparison(args, chosen["run"], filename=f"ablations_{chosen['run']}.csv")


if __name__ == "__main__":
    main()
