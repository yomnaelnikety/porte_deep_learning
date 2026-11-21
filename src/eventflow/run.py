"""Run Person A experiments and export shared evaluation artifacts."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .analysis import (MachineBootstrap, append_experiment, calibrate, example_loss,
                       fit_calibrators)
from .data import (PROTOCOL, SEED, PROB_PREFIX, inner_machine_split, load_events,
                   make_manifest, multiclass_loss)
from .eda import explore
from .features import GROUPS, build_features

FINAL_EXCLUDE = ("lags__pair",)
GRID = [(7, 100, 2.0), (15, 100, 2.0), (7, 220, 10.0),
        (15, 220, 10.0), (31, 220, 10.0)]
EARLY_STOPPING_PATIENCE = 20
TREE_DEFAULTS = {"n_estimators": 600, "learning_rate": .04, "colsample_bytree": .85,
                 "max_bin": 63}


def columns(frame, groups):
    return [c for c in frame if c.split("__")[0] in groups]


def tree(frame, groups, params, jobs, stop=None, exclude=()):
    selected = [c for c in columns(frame, groups) if c not in exclude]
    categories = {c: sorted(frame[c].dropna().unique())
                  for c in selected if frame[c].dtype == object}
    model = LGBMClassifier(objective="multiclass", random_state=SEED, n_jobs=jobs,
                           verbosity=-1, deterministic=True, force_col_wise=True,
                           **{**TREE_DEFAULTS, **params})
    options = {}
    if stop is not None:
        options = dict(eval_set=[(tree_inputs(stop, selected, categories), stop.target)],
                       eval_metric="multi_logloss",
                       callbacks=[early_stopping(EARLY_STOPPING_PATIENCE, first_metric_only=True, verbose=False)])
    model.fit(tree_inputs(frame, selected, categories), frame.target, **options)
    return model, selected, categories


def tree_prob(bundle, frame, classes):
    model, cols, cats = bundle
    if list(model.classes_) != classes:
        raise ValueError("Tree class order does not match the shared class list")
    return model.predict_proba(tree_inputs(frame, cols, cats))


def tree_inputs(frame, selected, categories):
    x = frame[selected].copy()
    for c, values in categories.items():
        x[c] = pd.Categorical(x[c], categories=values)
    return x


def logistic(train, val):
    cols = columns(train, ["current"])
    cat = [c for c in cols if train[c].dtype == object]
    num = [c for c in cols if c not in cat]
    transform = ColumnTransformer([
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()), num),
    ])
    model = make_pipeline(transform, LogisticRegression(max_iter=400))
    model.fit(train[cols], train.target)
    return model.predict_proba(val[cols]), list(model.classes_)


def error_analysis(val, prob, classes, out):
    info = val[["sequence_id", "prefix_len", "target"]].reset_index(drop=True).copy()
    info["predicted"] = np.asarray(classes)[prob.argmax(axis=1)]
    info["confidence"] = prob.max(axis=1)
    target_idx = pd.Index(classes).get_indexer(info.target)
    info["true_probability"] = prob[np.arange(len(info)), target_idx]
    info["loss"] = -np.log(info.true_probability.clip(lower=1e-15))
    info["correct"] = info.predicted == info.target
    stats = {"n": ("loss", "size"), "log_loss": ("loss", "mean"),
             "accuracy": ("correct", "mean")}
    info.groupby("target").agg(**stats).to_csv(out / "errors_by_class.csv")
    buckets = pd.cut(info.prefix_len, bins=[89, 129, 179, 250])
    info.groupby(buckets, observed=True).agg(**stats).to_csv(out / "errors_by_length.csv")
    bands = pd.cut(info.confidence, bins=np.linspace(0, 1, 11), include_lowest=True)
    info.groupby(bands, observed=True).agg(
        **stats, mean_confidence=("confidence", "mean")).to_csv(out / "confidence_bins.csv")
    pd.crosstab(info.target, info.predicted).reindex(
        index=classes, columns=classes, fill_value=0).to_csv(out / "confusion_counts.csv")
    info.nlargest(25, "loss").to_csv(out / "highest_loss_examples.csv", index=False)
    info[["sequence_id", "prefix_len"]].join(pd.DataFrame(
        prob, columns=[PROB_PREFIX + c for c in classes])).to_csv(
        out / "validation_predictions.csv", index=False)


def develop(events, manifest, classes, args):
    out = args.out
    inner = inner_machine_split(manifest)
    explore(events[events.sequence_id.isin(inner.loc[inner.role == "fit", "sequence_id"])],
            out / "eda")
    print("Building prefix features", flush=True)
    frame = build_features(events, manifest, classes)
    local = frame.merge(inner, on="sequence_id", how="inner", validate="many_to_one")
    train, stop, cal = (local[local.role == role] for role in ["fit", "early_stop", "calibration"])
    val = frame[frame.split == "validation"]
    experiment = uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()
    records = []

    def record(name, model, groups, params, prob, n_train=len(train), inner_loss=None):
        loss = multiclass_loss(val.target, prob, classes)
        record_row = dict(run=name, model=model, features="+".join(groups),
                          hyperparameters=json.dumps({k: v for k, v in params.items() if k != "prior"}),
                          validation_log_loss=loss,
                          seed=SEED, n_train=n_train, n_validation=len(val),
                          date_utc=timestamp[:10], owner=args.owner,
                          run_id=f"{experiment}-{len(records)+1:02}", timestamp_utc=timestamp,
                          protocol=PROTOCOL, inner_log_loss=inner_loss)
        append_experiment(out / "experiments.csv", record_row)
        records.append(record_row)
        print(f"{name:<28} {loss:.5f}", flush=True)
        return loss

    record("uniform", "uniform", [], {}, np.full((len(val), len(classes)), 1/len(classes)), 0)
    event_col = "current__event_type"
    counts = pd.crosstab(train[event_col], train.target).reindex(
        index=classes, columns=classes, fill_value=0).astype(float) + 1
    transitions = counts.div(counts.sum(axis=1), axis=0)
    record("markov", "transition table", ["current event"], {"alpha": 1},
           transitions.loc[val[event_col]].to_numpy())
    prob, lr_classes = logistic(train, val)
    assert lr_classes == classes
    record("current_row_logreg", "logistic regression", ["current"], {"C": 1}, prob)

    trials = []
    for leaves, child, l2 in GRID[:1] if args.quick else GRID:
        params = dict(num_leaves=leaves, min_child_samples=child, reg_lambda=l2)
        bundle = tree(train, GROUPS, params, args.jobs, stop=stop, exclude=FINAL_EXCLUDE)
        model = bundle[0]
        rounds = model.best_iteration_
        inner_loss = model.best_score_["valid_0"]["multi_logloss"]
        prob = tree_prob(bundle, val, classes)
        name = f"leaves{leaves}_child{child}_l2_{l2:g}"
        evaluated_rounds = len(model.evals_result_["valid_0"]["multi_logloss"])
        tuning_details = {
            **TREE_DEFAULTS, **params, "best_iteration": rounds,
            "evaluated_rounds": evaluated_rounds, "patience": EARLY_STOPPING_PATIENCE,
            "patience_triggered": evaluated_rounds - rounds >= EARLY_STOPPING_PATIENCE,
            "exclude": FINAL_EXCLUDE,
        }
        record(name, "LightGBM tuning", GROUPS, tuning_details, prob, inner_loss=inner_loss)
        trials.append((inner_loss, rounds, params, bundle, prob))
    _, rounds, params, bundle, raw = min(trials, key=lambda trial: trial[0])
    fixed = {**TREE_DEFAULTS, **params, "n_estimators": rounds}
    feature_config = {**fixed, "exclude": FINAL_EXCLUDE}
    record("full_raw", "LightGBM", GROUPS, feature_config, raw)
    raw_losses = example_loss(val.target, raw, classes)
    bootstrap = MachineBootstrap(val.sequence_id, SEED+3)
    ablations = []

    def comparison(name, kind, prob):
        change = example_loss(val.target, prob, classes) - raw_losses
        lo, hi = bootstrap.interval(change)
        ablations.append((name, kind, raw_losses.mean()+change.mean(), change.mean(), lo, hi))

    rng = np.random.default_rng(SEED + 4)
    thin_idx = rng.permutation(train.index)
    thinner = train.loc[thin_idx].groupby("sequence_id", sort=False).head(4).sort_index()
    variants = [(f"drop_{g}", [x for x in GROUPS if x != g], train, FINAL_EXCLUDE)
                for g in GROUPS[1:]]
    variants += [("add_pair_back", GROUPS, train, ()),
                 ("four_fit_cutoffs", GROUPS, thinner, FINAL_EXCLUDE)]
    for name, groups, subset, excluded in variants:
        variant = tree(subset, groups, fixed, args.jobs, exclude=excluded)
        prob = tree_prob(variant, val, classes)
        record(name, "LightGBM ablation", groups, {**fixed, "exclude": excluded}, prob, len(subset))
        comparison(name, "feature_or_data", prob)

    cal_raw = tree_prob(bundle, cal, classes)
    choices = fit_calibrators(cal.target, cal_raw, classes)
    spec, _ = min(choices, key=lambda row: row[1])
    tail = []
    target_idx = pd.Index(classes).get_indexer(val.target)
    rare = val.target.isin(["CONTROL_ALERT", "SENSOR_RESET", "PRESSURE_WARNING"]).to_numpy()
    for choice, loss in choices:
        name = choice["method"]
        prob = calibrate(raw, choice)
        if name != "raw":
            record(name, "LightGBM calibration", GROUPS, {**feature_config, **choice}, prob, inner_loss=loss)
            comparison(name, "calibration", prob)
        true_prob = prob[np.arange(len(val)), target_idx]
        tail.append(dict(method=name, selected=name == spec["method"], calibration_log_loss=loss,
                         validation_log_loss=multiclass_loss(val.target, prob, classes),
                         fraction_all_below_005=float((prob < .005).mean()),
                         fraction_true_below_005=float((true_prob < .005).mean()),
                         rare_n=int(rare.sum()),
                         rare_fraction_true_below_005=float((true_prob[rare] < .005).mean()),
                         min_true_probability=float(true_prob.min())))
    selected = calibrate(raw, spec)
    loss = record("selected_classical", "LightGBM", GROUPS, {**feature_config, **spec}, selected)
    pd.DataFrame(records).to_csv(out / "latest_experiments.csv", index=False)
    pd.DataFrame(ablations, columns=["run", "kind", "log_loss", "delta_vs_full",
                                     "ci_95_low", "ci_95_high"]).to_csv(out / "ablations.csv", index=False)
    pd.DataFrame(tail).to_csv(out / "calibration_tails.csv", index=False)
    (out / "selected_classical.json").write_text(json.dumps(
        dict(groups=GROUPS, params=fixed, exclude=list(FINAL_EXCLUDE),
             calibration=spec, protocol=PROTOCOL,
             validation_log_loss=loss), indent=2))
    error_analysis(val, selected, classes, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "develop", "predict", "score"])
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--owner", default="Walae Laamoum")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="Use one tree tuning configuration")
    parser.add_argument("--predictions", type=Path, help="Optional external predictions for score")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage == "score":
        classes = json.loads((args.out / "classes.json").read_text())
        pred = pd.read_csv(args.predictions or args.out / "test_predictions.csv")
        labels = pd.read_csv(args.data / "eventflow_test_labels.csv")
        merged = labels.merge(pred, on="sequence_id", how="outer", validate="one_to_one",
                              indicator=True)
        if not merged["_merge"].eq("both").all():
            raise ValueError("Prediction IDs must match the label file exactly")
        loss = multiclass_loss(merged.next_event, merged[[PROB_PREFIX+c for c in classes]], classes)
        print(f"Final test log loss: {loss:.6f}")
        (args.out / "final_test_score.json").write_text(json.dumps(
            {"log_loss": loss, "n": len(merged)}, indent=2))
        return

    events = load_events(args.data / "eventflow_train.csv")
    manifest, rollouts = make_manifest(events)
    fitting_ids = manifest.loc[manifest.split == "train", "sequence_id"]
    classes = sorted(events.loc[events.sequence_id.isin(fitting_ids), "event_type"].unique())
    if len(classes) != 14:
        raise ValueError(f"Expected 14 classes, found {len(classes)}")
    manifest.to_csv(args.out / "prefix_manifest.csv", index=False)
    rollouts.to_csv(args.out / "rollout_manifest.csv", index=False)
    inner_machine_split(manifest).to_csv(args.out / "classical_inner_split.csv", index=False)
    (args.out / "classes.json").write_text(json.dumps(classes, indent=2))
    if args.stage == "develop":
        develop(events, manifest, classes, args)
    if args.stage == "predict":
        selected = json.loads((args.out / "selected_classical.json").read_text())
        if selected.get("protocol") != PROTOCOL or selected.get("exclude") != list(FINAL_EXCLUDE):
            raise ValueError("Run develop first")
        fit = build_features(events, manifest.drop(columns="split"), classes)
        model, cols, cats = tree(fit, selected["groups"], selected["params"], args.jobs,
                                exclude=selected["exclude"])
        assert list(model.classes_) == classes
        test = load_events(args.data / "eventflow_test.csv")
        cuts = test.groupby("sequence_id").size().rename("prefix_len").reset_index()
        test_features = build_features(test, cuts, classes, labeled=False)
        prob = calibrate(tree_prob((model, cols, cats), test_features, classes),
                         selected["calibration"])
        result = pd.DataFrame(prob, columns=[PROB_PREFIX + c for c in classes])
        result.insert(0, "sequence_id", test_features.sequence_id)
        result.to_csv(args.out / "test_predictions.csv", index=False)
        joblib.dump((model, cols, cats, classes, selected["calibration"]), args.out / "classical.joblib")


if __name__ == "__main__":
    main()
