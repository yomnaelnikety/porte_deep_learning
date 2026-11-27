# EventFlow · AI2210

Person A's classical pipeline plus the shared data and evaluation setup and the team report source. The team can add its GRU/LSTM and Transformer/TCN as separate modules using the same validation manifest.

## Run on your computer

Use Python 3.11 or 3.12. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use `.venv\Scripts\Activate.ps1` on Windows PowerShell. If importing LightGBM on macOS reports that `libomp.dylib` is missing and Homebrew is installed, run `brew install libomp`.

Extract the three original course CSV files directly into `data/raw/`, or pass `--data /path/to/extracted/AI2210_EventFlow_Dataset`. For example, with the original ZIP one folder above this repo:

```bash
unzip -j ../AI2210_EventFlow_Dataset.zip '*/eventflow_*.csv' -d data/raw
python -m eventflow.run develop --jobs 4
python scripts/report_assets.py
```

The development command takes several minutes on CPU. It reads only `eventflow_train.csv`. Add `--quick --out quick_results` for a smaller tuning search that leaves the full results intact. Use `python -m eventflow.run prepare` to generate only the shared manifests and class order.

## What runs

| Stage | Inputs and outputs |
| --- | --- |
| Shared split | Fixed seed 2210. Split 80/20 by machine. Sample up to 12 fitting cutoffs and eight validation cutoffs per machine. A history must have at least 90 observed events and a subsequent target. |
| EDA | `results/eda/` contains fitting-only class frequencies, length and gap summaries, missingness, transition probabilities, sensor correlations and consecutive changes. |
| Baselines | Uniform probabilities, a smoothed previous-event transition table, and a current-row logistic regression adapted from the course notebook. All use the shared cutoffs. |
| Feature experiments | Tune LightGBM with training-only early stopping. Compare individual feature drops, adding back the categorical event pair, four versus twelve fit cutoffs, and two calibration methods. Report paired intervals over machines. |
| Error analysis | Compare log loss and accuracy by class and prefix length. Inspect confusion counts, 25 largest per-example losses, and confidence versus accuracy by bin. |
| Report | `report/main.tex` is the main Overleaf file. The asset script regenerates the data paragraph, figure and classical tables from local results. |

The v2 manifest keeps the original machine assignments and cutoffs. It adds seven cutoffs per validation machine. The generation manifest is unchanged. The 5,200 training machines are partitioned for the classical model into 4,160 fitting, 520 early-stopping and 520 calibration machines. The tuning grid covers 7, 15 and 31 leaves, `min_child_samples` of 100 or 220 and L2 of 2 or 10 at a learning rate of 0.04. The inner stopping set chooses the configuration and its best iteration. Tuning evaluates up to 600 rounds with patience of 20 and logs the rounds evaluated and whether patience triggered. The separate calibration set selects raw probabilities, temperature scaling or a blend with its smoothed class prior. The shared validation set is used for scoring and comparisons in this pipeline.

Feature and cutoff-count ablations hold the selected tree settings and boosting rounds fixed. Removing historical sensor summaries leaves current sensor values in place. Tuning and final prediction exclude the event pair and keep the six individual lags. Only the pair add-back variant restores the pair. Paired 95% percentile intervals use 5,000 machine resamples of the per-cutoff log-loss difference against the same raw full model. Negative differences favor the variant. Each interval is conditional on the fitted models and does not correct for multiple comparisons.

## Team handoff

The Transformer and generation pipeline is documented in
[docs/transformer_generation.md](docs/transformer_generation.md), including commands,
leakage controls, evaluation methods, and recorded results. Start with
`python -m eventflow.train_transformer smoke --data ..` after installing both
requirements files and the package.

Use [docs/shared_protocol.md](docs/shared_protocol.md) to integrate the deep models. Record their experiments in the same schema as `results/experiments.csv`. The classical script appends one row per completed result with a timestamp and run ID. It preserves v1 rows and tags them with their original protocol. The v2 label identifies the shared validation manifest. The first v2 batch included the event pair in its main model, the second excluded it with a 260-round cap, and the third retained the exclusion with a 600-round cap. `results/latest_experiments.csv` is regenerated for the report. Give other models their own log files and combine the tables for the team report. View the generated CSVs in the notebook or a read-only viewer so the exports remain reproducible.

Once the team chooses its final model, use the test prefixes. If the classical model wins, refit it using the fixed cutoffs from both local splits and produce probabilities. The refit uses the chosen round count. It respects the saved feature exclusions and applies the method stored in the selected configuration. Any calibration parameters were fitted on the separate 520-machine calibration set using predictions from the original 4,160-machine tree. Their effect after transfer to the refit has not been measured:

```bash
python -m eventflow.run predict
python -m eventflow.run score
```

The `predict` stage reads training data and test prefixes. The separate `score` stage accesses the final labels. To score an exported deep model, run `python -m eventflow.run score --predictions path/to/final_predictions.csv`. The official approximate test references are 2.64 uniform, 1.81 previous-event, and 1.75 current-row logistic regression. Compare models directly on the shared validation split and report final test results separately.

## GitHub and Overleaf

Review the tracked files after staging. The raw data, test labels, model binaries and the local environment are ignored. Keep the source, manifests, report source and experiment results with the project. Enable teaching team access as specified in the brief.

```bash
git init
git add .
git status --short
git commit -m "Add EventFlow classical pipeline and report skeleton"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

For an existing team repository, copy the files into a feature branch and open a pull request. For Overleaf, upload the standalone report ZIP using **New Project → Upload Project**, or upload the contents of `report/` with `main.tex` at the project root.

The optional notebook uses a Jupyter kernel from the same environment. Install JupyterLab separately if needed.

Course baseline attribution: AI2210 teaching team's EventFlow baseline notebook. Follow the course policy on AI assistance and disclose assistance where required.
