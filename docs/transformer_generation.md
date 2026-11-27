# Transformer and event generation

This implementation starts from `origin/deep-model` and shares its encoder,
machine partitions, class order, training loop, calibration, and validation
cutoffs. No pretrained weights or external data are used. Raw data remain outside
version control. The commands below reproduce the training and analysis workflow.

## Setup and quick verification

From the repository root, using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-deep.txt
python -m pip install -e .
python -m pytest tests -q
python -m eventflow.train_transformer smoke --data ..
```

Here `--data ..` uses the course CSV files in the parent folder. Change it if your
data live elsewhere. `PYTHONPATH=src` also works instead of editable installation.
Smoke runs use 340 machines and two epochs per small model. Their checkpoints,
training outputs and sampled continuations go under ignored `models/transformer/`.
They verify execution, not competitive performance. The smoke command never reads
either test file.

## Reproducible experiment commands

```bash
# Four predictive configurations; selection uses early-stop loss only.
python -m eventflow.train_transformer tune --data .. --resume
# Retrain six controlled variants of the selected configuration.
python -m eventflow.train_transformer ablate --data .. --resume
# Shared validation comparison against recorded LightGBM and GRU probabilities.
python -m eventflow.train_transformer compare
# Separately train a full-context event-only generator.
python -m eventflow.train_transformer train-generator --data .. \
  --out results/transformer/generator_training
# Ten sampled 20-event continuations for every shared held-out rollout prefix.
python -m eventflow.train_transformer generate --data .. \
  --checkpoint models/transformer/generator_event_only.pt
# Longer-lag structure and exact matches to fitting-machine event windows.
python -m eventflow.generation_analysis --data ..
# Tables, diagnostic plot, and deterministic example selection from these outputs.
python scripts/transformer_report_assets.py
```

Use `--device cuda` for a CUDA GPU, `--device mps` for Apple GPU, or the default
CPU when CUDA is unavailable. Default CPU threads are four. Training and inference
batch sizes are 32 to limit attention memory. To run one configuration:

```bash
python -m eventflow.train_transformer train --data .. --run transformer_trial \
  --config '{"hidden":64,"layers":2,"max_epochs":20,"lr":0.001}'
```

Names cannot overwrite existing runs. Resume requires the same configuration
and a checkpoint; it skips completed runs, not interrupted epochs. Use a new name
for a new experiment. Keep data and protocol files unchanged when resuming.
`results/transformer/` contains run configurations, learning histories, logits,
validation probability CSVs, experiment records, selection and comparisons.
Checkpoint files under `models/transformer/` also contain fitted preprocessing,
temperature, fitting-machine IDs, and a smoke flag. Only load trusted checkpoints.

Generation exports `summary.json`, per-sample diagnostics, the evaluated manifest,
all real/generated event sequences in JSONL, and numeric arrays. The checkpoint's
SHA-256 identifies the exact model used. `--limit N` produces a clearly counted
subset for debugging; use all prefixes for the main analysis. Existing generation
directories are protected from overwriting; pass another `--output` for a new run.

After team model selection, `predict --run NAME --data ..` exports test probabilities
from a saved fitting-only checkpoint. It does not refit on all training machines
and does not read labels. Final test scoring remains a separate team decision.

## Model and evaluation methodology

### Architecture and prediction target

The Transformer combines event and categorical-context embeddings with
standardized numerical inputs. Sinusoidal positional encoding is followed by
pre-normalized self-attention and feed-forward blocks with residual connections.
Positional encoding represents event order, while time-gap inputs represent
elapsed physical time. Unlike the GRU's recurrent hidden state, self-attention
weights earlier event representations directly. Attention cost grows
quadratically with history length.

At position t, the model outputs 14 logits predicting event t+1. Event input IDs
are 1–14, with 0 reserved for padding; output labels are 0–13. The shared loader
trains on eligible prefix lengths 90–250. Cross-entropy minimizes the negative
log probability assigned to the true class. Evaluation uses the output at the
last real position of each prefix.

A causal mask prevents attention to future events during many-to-many training.
A padding mask excludes artificial zero tokens, and padded targets are ignored
by the loss. Tests verify that changing future events or covariates leaves
earlier logits unchanged.

### Data partitions and probability calibration

Model weights and preprocessing are fitted on fitting machines. Separate
early-stopping machines select the epoch and configuration; calibration machines
fit a single temperature. The 10,400 shared validation cutoffs are used for
reporting. Since cutoffs from the same machine are dependent, comparison
intervals resample whole machines. Positive `delta_vs_transformer` values indicate
that the compared model has higher log loss than the Transformer.

Calibrated probabilities are computed as `softmax(logits / T)`. A temperature
above 1 produces less concentrated probabilities without changing class rankings.
For generation, this fitted temperature is multiplied by a fixed user-specified
sampling multiplier, which defaults to 1. Sampling temperature was not tuned on
the reported held-out rollouts.

### Autoregressive event generation

A separate event-only Transformer generates multi-step continuations. Future
sensor measurements and categorical context are unavailable after a sampled
event, so this model excludes context embeddings and all numerical inputs,
including time gaps. It retains deterministic positional encoding. At each
step, an event is sampled from the predicted categorical distribution and
appended to the history used for the next prediction.

Generation diagnostics include:

- Unigram Jensen–Shannon divergence in natural-log units for event frequencies.
- Bigram and trigram divergence for local ordering, without crossing sequence
  boundaries.
- Unique n-gram ratios and duplicate rates for diversity. Each replicate uses
  one continuation per prefix to match the real corpus size. Within-prefix
  duplicates are measured separately.

Sampling standard deviations describe variation between replicates, not
confidence intervals over machines. Matching local n-grams does not establish
long-range realism, and diversity alone does not rule out memorization.
Longer-lag statistics and training-window matches provide additional diagnostics,
reported below. The generator does not predict timestamps, so timing diagnostics
are outside its scope.

### Limitations

Generated histories may extend beyond the 250 positions supervised during
training. Sinusoidal positions are defined at these lengths but do not guarantee
generalization. Training uses observed histories, whereas generation feeds back
sampled events, allowing errors to accumulate. Short-context training chunks
contain up to `context + tail` inputs, so later targets can see more than the
nominal context; these are approximate history ablations. Hardware and training
randomness may affect reproducibility.

## Local implementation verification

The implementation was checked with the pinned dependencies on Python 3.12 and
CPU. The tests cover the shared loader, causal masking, padding, finite gradients,
event-only inputs, seeded autoregressive sampling, order-sensitive diagnostics,
matched-size diversity, checkpoint loading, held-out provenance, and prediction
alignment. The real-data smoke run trained both small models for two epochs on
340 machines and generated 12 continuations (four prefixes, three samples each,
20 steps). Outputs are in `models/transformer/smoke_306daf94/` and are deliberately
excluded from version control. This verifies execution, not final predictive
performance. Full predictive tuning, baseline comparison, five ablations, a
second-seed repeat, and the full generation study are now complete. Saved prediction and rollout artifacts were also verified against their
manifests and source data. The current test suite can be run with the setup
command above.

## Full predictive results (24 November 2026)

Four configurations were trained locally on the 4,160 fitting machines. The 520
early-stop machines selected the epoch and configuration; 520 separate machines
fitted temperature scaling. Every model below is evaluated on the same 10,400
cutoffs from 1,300 validation machines. The GRU and LightGBM scores were recomputed
from their existing saved predictions; those models were not retrained here.

| Transformer configuration | Best early-stop loss | Calibrated validation loss | Best epoch / epochs run |
| --- | ---: | ---: | ---: |
| 128 units, 2 layers, learning rate 0.001 (selected) | 1.4942 | 1.4748 | 32 / 36 |
| 64 units, 2 layers | 1.5213 | 1.4998 | 39 / 40 |
| 128 units, 1 layer | 1.5262 | 1.5038 | 22 / 26 |
| 128 units, 2 layers, learning rate 0.0003 | 1.5050 | 1.4911 | 39 / 40 |

The selected Transformer has four attention heads, 414,062 parameters, temperature
1.055 and validation accuracy 52.67%. Its uncalibrated validation loss was 1.4760.
The full search took approximately 35.54 minutes of recorded CPU training and
evaluation time. Two configurations reached the 40-epoch cap; this is evidence
about the tested search, not a claim that all Transformer possibilities are exhausted.

| Model | Validation log loss | Difference from Transformer | Paired 95% interval |
| --- | ---: | ---: | --- |
| GRU, 128 units, 2 layers | 1.4522 | -0.0227 | [-0.0295, -0.0158] |
| Selected Transformer | 1.4748 | 0 | Reference |
| LightGBM | 1.5193 | +0.0445 | [+0.0362, +0.0527] |

The Transformer improves on LightGBM, but the GRU remains best among these models.
Intervals use 5,000 paired resamples of whole validation machines, preserving
their eight cutoffs together. They are conditional on these fitted models and do
not measure training-seed variability. Neither interval includes zero. Test
prefixes and labels were not used for these runs or this comparison.

Evidence is under `results/transformer/`: `tuning_transformer.csv`,
`selected_transformer.json`, `comparison_transformer_h128.csv`, run-specific
histories/predictions, `experiments_transformer.csv`, `execution_manifest.json`
(source/data hashes and environment), and `verification.json`. Checkpoints are
local under `models/transformer/`. To reproduce the comparison:

```bash
python -m eventflow.train_transformer compare
```

## Completed generation study

The event-only generator uses the default 128-unit, two-layer, four-head
Transformer configuration, trained on the same 4,160 fitting machines. Early
stopping selected epoch 30 (34 epochs run). The calibration temperature is 1.0731;
raw/calibrated validation log losses are 1.4979/1.4959. Its metrics and experiment
log are under `results/transformer/generator_training/`.

Using every prefix in the shared rollout manifest, the study produced 10 sampled
20-event continuations for each of 1,300 validation machines: 13,000 continuations
and 260,000 sampled events. The sampling multiplier was fixed at 1.0, giving
effective temperature 1.0731; seed 2210 was fixed before evaluation. No future
measurements, categorical context, or time gaps enter generation.

| Diagnostic | Generated versus real |
| --- | ---: |
| Event-frequency JS divergence (nats) | 0.000692 |
| Bigram JS divergence | 0.002938 |
| Trigram JS divergence | 0.029879 |
| Unique bigram ratio | 0.007931 generated / 0.007895 real |
| Unique trigram ratio | 0.077778 generated / 0.076410 real |
| Duplicate rate, matched-size corpora | 0.00769% generated / 0% real |
| Duplicate rate within the 10 samples of a prefix | 0% |
| Event-pair JS at lag 5 | 0.003891 |
| Event-pair JS at lag 10 | 0.005732 |
| Longest repeated-event run distribution JS | 0.004114 |
| Exact match to an exposed 20-event training window | 1.4308% generated / 1.3846% real |

For the main distribution/diversity metrics, each of ten replicates takes one
generated continuation per prefix, matching the 1,300 real continuations. Values
are replicate means; the JSON and CSV outputs also give sampling standard
deviations. These are not confidence intervals. The exact-match comparison uses
all generated continuations versus the 685,322 distinct 20-event windows among
the first 251 events of fitting machines (the generator's training exposure).

**Interpretation.** Overall event frequencies are close, but RECOVERY is more
frequent in generated sequences (32.29% versus 30.82%), while VIBRATION_WARNING
is less frequent (10.48% versus 11.37%). Trigram divergence is larger than bigram
divergence; this reflects both richer ordering statistics and finite-sample
sparsity, so it should not be interpreted as a directly comparable error rate.
Unique n-gram ratios are similar, and samples from the same prefix vary. The
small absolute bigram ratios do not by themselves imply low diversity: with 14
classes there are only 196 possible bigrams among 24,700 observed bigrams per
replicate. The similar exact training-match rates in generated and real data give no obvious
signal of excess exact copying, but do not rule out approximate memorization.
Lagged pairs and repeated runs test aspects of longer structure within 20 events,
not globally valid operational behavior or physical timing.

**Representative example.** For TR00004 at prefix length 110, the last observed
event is COOLING_FAULT. The real continuation begins COOLING_FAULT, MAINTENANCE_CHECK,
COOLING_FAULT, COMPONENT_REPLACED and enters a long RECOVERY run at step 9. The
first sampled continuation also starts with the same first three events but
enters RECOVERY at step 12. The second sample follows a different path, including
COOLING_ACTIVATED/TEMP_WARNING alternation and ending in COOLING_FAULT. This shows
persistent regimes and varied futures; a generated sequence is not required to
match the one realized future exactly. Full, deterministically selected examples
are in `report_assets/representative_examples.json`.

Source evidence: `results/transformer/generation/summary.json`,
`diagnostics_by_sample.csv`, `supplemental_analysis.json`, `rollouts.jsonl`,
`rollouts.npz`, and `evaluated_manifest.csv`. The report-assets script produces
the checked plot `results/transformer/report_assets/generation_diagnostics.png`
and its PDF version, plus tables and examples. The generator outputs preserve
the checkpoint hash. Generation settings were not tuned on these results.

## Completed ablations and seed check

All variants use the selected full model's settings, changing only the named
configuration field and retaining the same fitting, early-stop, calibration and
validation machine partitions. Early stopping and temperature scaling are rerun
for each variant. Positive differences mean worse validation log loss.

| Variant | Validation loss | Change from full model | Paired 95% interval |
| --- | ---: | ---: | --- |
| Full model, seed 2210 | 1.47483 | 0 | Reference |
| Remove time gaps | 1.48560 | +0.01076 | [+0.00602, +0.01532] |
| Remove sensors and missingness flags | 1.48817 | +0.01334 | [+0.00806, +0.01850] |
| Remove categorical context | 1.47629 | +0.00145 | [-0.00226, +0.00516] |
| Short context: 16 events | 1.52149 | +0.04666 | [+0.03964, +0.05399] |
| Short context: 64 events | 1.49129 | +0.01646 | [+0.01126, +0.02169] |
| Full model, seed 2211 | 1.48014 | +0.00531 | [+0.00119, +0.00953] |

**Interpretation.** Removing time gaps or sensors worsened these fitted models.
The categorical-context effect is small and its interval spans zero, giving no
clear evidence of a benefit conditional on the other inputs. The 16-event
pipeline deteriorates most, and 64 events performs better than 16. The full-model
repeat changes loss by 0.00531, larger than the observed categorical-context
effect. Two seeds are a useful sensitivity check, not a reliable estimate of the
entire distribution of training outcomes. The original selected model is retained;
these validation results were not used to choose a new configuration or seed.

**Context-study limitation.** The existing loader trains short-context variants
on sampled chunks, with up to 32 additional supervised positions, so later
training positions can see more than the nominal context. It also supplies fewer
targets per epoch: about 109,000, versus 395,461 for full-history training. Thus
their differences from the full model combine context and training-sampling
changes. The 16- and 64-event variants share this sampling scheme. The full-versus-short
comparisons are therefore approximate history ablations; history length alone
cannot explain the entire difference.

Intervals use 5,000 paired resamples of validation machines, keeping each
machine's cutoffs together. They are conditional on the fitted networks, do not
include training-seed uncertainty, and are not adjusted for multiple comparisons.
One trained instance per feature ablation cannot fully separate feature effects
from optimization randomness.

The separate table `results/transformer/ablations_transformer_h128.csv` preserves
the original GRU/LightGBM comparison. Per-run histories, configurations,
checkpoints and validation predictions are saved; `analysis_verification.json`
records the checks. Report-ready tables and the visually checked ablation plot
are under `results/transformer/report_assets/`. The fixed experiment and sampling
plan, commands and source hashes are in `analysis_execution_manifest.json`.
