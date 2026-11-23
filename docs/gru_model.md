# Sequence loader and recurrent model (GRU)

Builds on the shared data and evaluation protocol (`docs/shared_protocol.md`) without changing it.

## Files

| File | What it does |
| --- | --- |
| `src/eventflow/seq_data.py` | Turns event logs into per-step arrays. Fits vocabularies and normalization on fitting machines only. Builds training batches and evaluation windows. Only needs numpy and pandas. |
| `src/eventflow/deep_models.py` | `EventSequenceModel`: embeddings, then a linear layer with ReLU, then a GRU or LSTM, then 14 logits per step. |
| `src/eventflow/train_deep.py` | Commands: `smoke`, `train`, `tune`, `ablate`, `compare`, `predict`. Also handles temperature scaling and the experiment log. |
| `tests/test_seq_data.py` | Leakage and alignment checks on a synthetic log. |
| `scripts/deep_report_assets.py` | Builds the GRU tables and learning curve for the report. |
| `report/sections/02b_gru.tex`, `04b_gru_results.tex` | Report text for the model and its results. |
| `requirements-deep.txt` | PyTorch and pytest. |
| `models/` | Checkpoints. Not tracked by git (see `models/.gitignore`). |

Outputs go to `results/deep/`. That includes `experiments_gru.csv`, which uses the same columns as the classical experiment log, plus `tuning_gru.csv`, `ablations_gru.csv`, `selected_gru.json`, `selected_errors/` and one folder per run under `runs/`.

## How to run

```bash
source .venv/bin/activate
python -m pip install -r requirements-deep.txt
python -m pytest tests -q                      # 9 loader checks, a few seconds
python -m eventflow.train_deep smoke           # whole pipeline on 340 machines, ~1 min
python -m eventflow.train_deep tune            # 7 configurations (--quick runs 3)
python -m eventflow.train_deep ablate          # 7 retrained variants + calibration check
python -m eventflow.train_deep compare         # GRU vs LightGBM, paired CIs
python scripts/deep_report_assets.py           # report tables and figure
```

Or run all of the above in one go (logs go to `logs_gru/`):

```bash
bash run_gru.sh
```

Use `--resume` with `tune` or `ablate` to skip runs that already finished. Use `--device cuda` on a GPU, for example Colab. On a laptop CPU, expect very roughly 3–10 minutes per run.

Test set, only once, after the team has chosen its final model:

```bash
python -m eventflow.train_deep predict
python -m eventflow.run score --predictions results/deep/test_predictions_<run>.csv
```

## Design and why

**Same protocol as the classical model.** Classes come from `results/classes.json`, cutoffs from `results/prefix_manifest.csv`, and roles from `results/classical_inner_split.csv`. The 4,160 fitting machines are used for gradient updates. The 520 early-stop machines choose the epoch and the configuration. The 520 calibration machines fit the temperature. The 1,300 validation machines (10,400 cutoffs) are used only for reporting, so the GRU and LightGBM numbers are directly comparable.

**Inputs at each step**

- Event class: a learned 32-dimensional embedding. Embeddings let similar events end up with similar vectors.
- Six categorical fields: 8-dimensional embeddings each. Severity is treated as a category. Values not seen in fitting map to id 0.
- `log1p` of the time gap: gaps are skewed (median 0.96, 95th percentile 2.93), and the log compresses the long tail. It is standardized afterwards.
- Position in the log, divided by 250.
- All 16 measurements, including the 4 telemetry channels, standardized with fitting-machine statistics. Missing values are set to 0, with a separate 0/1 flag.
- Optional machine-relative sensors: each value minus the running mean of that machine's own earlier values. This is causal, so it never uses future values. It is tested as an addition in the ablations.

**Why many-to-many training.** A GRU is causal, so the output at step t is the prediction for prefix length t+1. One pass over a machine therefore supervises every prefix length from 90 to 250, about 95 targets per machine, instead of 12 sampled cutoffs. The classical ablation found that going from 12 to 4 cutoffs cost +0.035, so more training prefixes help.

**Why no packing.** With right padding and a causal model, padding comes *after* the real events and cannot change their outputs. We read the output at the last real step (`last` in `prefix_batch`).

**Why GRU vs LSTM.** Both use gates, which avoid the vanishing gradients of a plain RNN over 250 steps. A GRU has update and reset gates and a single state. An LSTM has input, forget and output gates plus a cell state. The GRU has about 25% fewer recurrent parameters. The search tries both.

**Overfitting controls:** dropout, weight decay, early stopping on separate machines, gradient clipping, and a learning rate that halves when the early-stop loss stalls.

**Temperature scaling.** We minimize log loss of softmax(z / T) over a single number T on the calibration machines. T > 1 means the network was overconfident. The ranking of the logits is unchanged, so accuracy is unchanged too. Log loss punishes confident mistakes hard, so calibration can matter. For LightGBM, T was 0.995, a negligible change.

**Ablations, one change at a time:** remove time gaps, remove sensors, remove categorical context, use only the last 16 or 64 events, add relative sensors, and remove temperature scaling. We also retrain the same model with a different seed. Bootstrap intervals only capture which validation machines were sampled, not training randomness. The seed row shows how much a retrained network moves on its own. An ablation effect smaller than that is not convincing.

**Paired bootstrap.** This is the shared `MachineBootstrap` with the same seed. It resamples whole validation machines 5,000 times, keeping each machine's 8 cutoffs together, and takes the 2.5th and 97.5th percentiles of the mean loss difference.

**Length reweighting.** Validation cutoffs are shorter than test prefixes: 51% of validation cutoffs fall in 90–129, against 32% of test prefixes. `selected_errors/length_reweighting.json` reweights validation loss to the test length mix. This uses test prefix lengths only, never labels.

## Results (validation, 10,400 shared cutoffs)

| Model | Log loss |
| --- | --- |
| Uniform / previous-event / logistic regression | 2.639 / 1.771 / 1.723 |
| LightGBM | 1.519 |
| LSTM, 128 units | 1.454 |
| GRU, 128 units, 1 layer | 1.451 |
| **GRU, 128 units, 2 layers (selected by early-stop loss)** | **1.452** |
| Average of GRU and LightGBM | 1.459 |

- The GRU beats LightGBM by 0.067, 95% CI [0.059, 0.076].
- Averaging the two does not help.
- Reweighted to the test length mix, the GRU's loss is 1.446.

Ablations (change from the selected GRU; a different seed alone moves it by −0.005):

| Change | Δ log loss |
| --- | --- |
| Last 16 events only | **+0.068** (history matters most) |
| Remove time gaps | +0.008 |
| Last 64 events only | +0.005 |
| Remove sensors | +0.004 (interval spans 0) |
| Remove categorical context | −0.005 |
| Add machine-relative sensors | +0.0004 |
| Temperature scaling | no effect, because T = 1.01 and the model was already calibrated |

Weak spots: the rare classes CONTROL_ALERT, PRESSURE_WARNING and ROUTINE_CHECK are never ranked first, and the largest losses come from confident COOLING_FAULT predictions when SENSOR_RESET happened.

## Reusing the loader for the second deep model

You can reuse the loader directly:

```python
from eventflow.seq_data import Encoder, load_protocol
from eventflow.data import load_events
classes, manifest = load_protocol("results")
events = load_events("data/raw/eventflow_train.csv")
encoder = Encoder.fit(events, manifest.query("role == 'fit'").sequence_id.unique(), classes)
store = encoder.transform(events)
batch = store.train_batch(machines, store.columns(["position", "time", "sensor", "missing"]))
# batch["event"] [B,T], batch["context"] [B,T,6], batch["numeric"] [B,T,F], batch["target"] [B,T]
```

A Transformer needs a causal attention mask so that step t cannot see later steps, and the padding should be masked too. With that in place, the same many-to-many training and `prefix_batch` evaluation work unchanged. For generation, the event-plus-time inputs are `store.columns(["position", "time"])`, because measurements are not available for generated future steps.
