# Scenario Playground Step 12 Walk-Forward Validation Report

## Status and scope

Step 12 is complete.

This step remained necessary because the Step 11 XGBoost result came from one
chronological validation period. A trustworthy methodology decision requires
repeated out-of-sample evidence, target-overlap-aware purging, fold-local
preprocessing, and a direct check that the model remains useful for an
eight-input Scenario Playground.

Only the S&P 500 21-trading-session return was modeled. Event-aware sampling,
the approved feature and target definitions, and the 584-row/58-feature raw
matrix remained frozen. No final-test target was parsed. No other target,
Step 13 work, production artifact, Supabase access, backend, frontend, or
server work was performed.

## Frozen data boundary

| Item | Result |
|---|---:|
| Approved matrix rows | 584 |
| Retained pre-final-test rows available to Step 12 | 464 |
| Rows never parsed because they belong to locked final test | 114 |
| Locked final-test target rows parsed | 0 |
| Features | 58 |
| Target trained | `sp500` only |
| Outer folds | 5 |
| Validation rows per fold | 60 |
| Out-of-sample predictions per strategy | 300 |

The bounded matrix reader inherited from Step 10 parses only the pre-final-test
portion needed for development and validation. The locked final-test target
values are not available to the Step 12 code.

## Walk-forward design

The design is an expanding-window walk-forward:

```text
Build the next chronological 60-row validation block
Take all earlier rows as candidate training history
Remove every candidate whose target_end_date reaches the validation boundary
Create a separate purged inner time split for model tuning
Fit preprocessing and the model using that fold's training rows only
Predict the outer validation block
Advance the cutoff and repeat
```

The purge rule is:

```text
training target_end_date < first validation target_baseline_date
```

This uses canonical market-session target dates. It does not remove an
arbitrary 21 event-aware rows.

| Fold | Purged train samples | Train rows | Outer purge rows | Validation samples | Validation rows | Last validation target end |
|---:|---|---:|---:|---|---:|---|
| 1 | 2013-08-05 to 2017-02-16 | 160 | 4 | 2017-03-24 to 2018-06-29 | 60 | 2018-07-27 |
| 2 | 2013-08-05 to 2018-06-04 | 222 | 2 | 2018-07-09 to 2019-10-07 | 60 | 2019-11-01 |
| 3 | 2013-08-05 to 2019-09-09 | 280 | 4 | 2019-10-11 to 2021-01-14 | 60 | 2021-02-11 |
| 4 | 2013-08-05 to 2020-12-18 | 342 | 2 | 2021-01-27 to 2022-06-13 | 60 | 2022-07-12 |
| 5 | 2013-08-05 to 2022-05-12 | 401 | 3 | 2022-06-15 to 2023-11-06 | 60 | 2023-12-04 |

All five outer folds have a nonempty purge. Every model/history fit also uses
a separately purged inner validation tail; inner purge counts range from one
to four rows.

## Compared methodologies

### Ridge

Each Ridge fit uses the Step 10 pipeline:

```text
median imputer
standard scaler
Ridge
```

All preprocessing is refitted inside the current fold. Alpha is selected on
the fold's purged inner chronological validation tail from:

```text
0.01, 0.1, 1, 10, 100, 1,000, 10,000, 100,000, 1,000,000
```

The selected full-history alphas by outer fold are:

```text
10,000; 1,000; 1,000; 1,000; 1,000,000
```

### XGBoost

Step 12 carries forward the conservative Step 11 stump architecture:

```text
maximum depth:        1
learning rate:        0.03
minimum child weight: 8
row sample:           0.85
column sample:        0.75
L1 alpha:             0.10
L2 lambda:            10
maximum rounds:       500
early stopping:       40 rounds
seed:                 20260725
threads:              1
```

The boosting-round count is selected separately inside every fold. For
full-history XGBoost it varies substantially:

```text
75; 23; 26; 196; 1 tree
```

The one-tree fifth fold is evidence that the learned tree structure is not
stable across historical periods.

### History strategies

Both model families were evaluated with:

- `full_history`: all purged expanding training rows;
- `recent_5y`: rows within five years of the fold cutoff, with full history
  used when fewer than 100 recent rows would remain;
- `recency_weighted`: all purged rows with a three-year exponential
  half-life.

In folds 1 and 2, the available history is shorter than five years, so the
full-history and recent-five-year datasets are identical. The recent window
becomes distinct in folds 3 through 5.

### Naive baselines

The same three baseline meanings used in Step 10 are evaluated:

- zero forward return;
- current fold training-target mean;
- trailing 21-observation S&P 500 return known at the target baseline date.

The trailing baseline uses the cached Yahoo `^GSPC` `Close` field and looks
back exactly 21 source observations. It never reads after the prediction
baseline.

## MAE by fold

All values are S&P 500 return percentage points.

| Fold | Zero | Train mean | Trailing 21 | Ridge full | Ridge 5y | Ridge weighted | XGB full | XGB 5y | XGB weighted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.2387 | 1.9537 | 2.8843 | **1.9336** | **1.9336** | 1.9349 | 1.9715 | 1.9715 | 1.9357 |
| 2 | 3.4591 | 3.3762 | 5.0827 | **3.2575** | **3.2575** | 3.2826 | 3.3652 | 3.3652 | 3.3874 |
| 3 | 5.9910 | 5.5365 | 6.8990 | **4.8931** | 5.0138 | 5.1058 | 5.4492 | 5.5932 | 5.5835 |
| 4 | 3.2789 | 3.2054 | 5.1371 | 3.6114 | 3.3426 | 3.3885 | **3.1898** | 3.2123 | 3.2687 |
| 5 | 4.3173 | 4.1811 | 6.0875 | 4.1800 | 4.1832 | 4.1819 | 4.1820 | **4.0440** | 4.1833 |

Full-history Ridge is a fold winner in folds 1, 2, and 3. XGBoost wins the
last two folds, but its fold 3 error is materially worse.

## Aggregate repeated out-of-sample performance

| Strategy | MAE | RMSE | OOS R-squared | Correlation | Directional | Fold MAE SD | Worst-fold MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge recent 5y | **3.5462** | **4.9360** | **0.0492** | **0.2373** | 65.33% | 1.1495 | 5.0138 |
| Ridge full history | 3.5751 | 5.0274 | 0.0136 | 0.1683 | **66.67%** | **1.1068** | **4.8931** |
| Ridge recency weighted | 3.5787 | 4.9406 | 0.0474 | 0.2388 | 65.67% | 1.1749 | 5.1058 |
| XGBoost full history | 3.6315 | 5.0233 | 0.0153 | 0.1258 | 66.33% | 1.2873 | 5.4492 |
| XGBoost recent 5y | 3.6372 | 5.0189 | 0.0170 | 0.1323 | 64.33% | 1.3248 | 5.5932 |
| Training mean | 3.6506 | 5.0658 | -0.0015 | -0.0383 | 65.67% | 1.3220 | 5.5365 |
| XGBoost recency weighted | 3.6717 | 5.0405 | 0.0085 | 0.1074 | 63.67% | 1.3392 | 5.5835 |
| Zero | 3.8570 | 5.1381 | -0.0303 | undefined | 0.00% | 1.4033 | 5.9910 |
| Trailing 21 observations | 5.2181 | 7.3477 | -1.1070 | -0.1044 | 54.33% | 1.5044 | 6.8990 |

The results are weak in absolute forecasting terms:

- all positive model OOS R-squared values are small;
- the best model MAE only modestly improves on the training-mean baseline;
- realized returns have much greater variation than model predictions;
- event-aware observations and their 21-session labels overlap, so the 300
  predictions are not 300 independent economic experiments.

The models nevertheless contain a small amount of repeated predictive signal,
and Ridge is the more stable family.

## Full-history versus recent-history decision

Recent-five-year Ridge has the lowest pooled MAE, but its advantage over
full-history Ridge is only approximately 0.82%.

Full-history Ridge has:

- a lower worst-fold MAE: 4.8931 versus 5.0138;
- lower fold MAE dispersion: 1.1068 versus 1.1495;
- higher directional accuracy: 66.67% versus 65.33%;
- three fold wins versus two for recent-five-year Ridge;
- all available historical observations rather than a shorter sample.

The approved tie rule is:

```text
Prefer full history when its MAE is within 1% of the within-family
minimum and both its worst-fold MAE and fold dispersion are lower.
```

Full-history Ridge passes this rule. Recency weighting does not provide a
consistent improvement. The approved history strategy is therefore:

```text
full_history
```

## VIX-regime performance

| Strategy | VIX regime | Rows | MAE | RMSE | OOS R-squared | Directional |
|---|---|---:|---:|---:|---:|---:|
| Training mean | Below 20 | 182 | 3.0648 | 4.6027 | -0.0214 | 64.29% |
| Ridge full | Below 20 | 182 | **3.0207** | **4.5184** | **0.0157** | 64.29% |
| XGBoost full | Below 20 | 182 | 3.0409 | 4.5545 | -0.0001 | **67.03%** |
| Training mean | 20 to below 30 | 91 | **4.1918** | 5.0334 | -0.0186 | 65.93% |
| Ridge full | 20 to below 30 | 91 | 4.2481 | 5.3056 | -0.1317 | **68.13%** |
| XGBoost full | 20 to below 30 | 91 | 4.2040 | **5.0258** | **-0.0155** | 62.64% |
| Training mean | 30 or above | 27 | 5.7753 | 7.5463 | -0.0679 | 74.07% |
| Ridge full | 30 or above | 27 | **5.0439** | **6.9523** | **0.0936** | **77.78%** |
| XGBoost full | 30 or above | 27 | 5.6837 | 7.4438 | -0.0391 | 74.07% |

Ridge's strongest relative result is in the high-VIX group. That group has
only 27 rows, so the finding is directionally useful but not conclusive.
Neither learned model improves every metric in every regime.

## Stability and overfitting

Mean diagnostics across the five folds are:

| Variant | Training MAE | Inner-validation MAE | Outer-validation MAE | Outer minus training |
|---|---:|---:|---:|---:|
| Ridge full | 2.4041 | 3.0921 | 3.5751 | 1.1710 |
| Ridge recent 5y | 2.6088 | 3.3470 | 3.5462 | **0.9374** |
| Ridge weighted | 2.4464 | 3.1337 | 3.5787 | 1.1324 |
| XGBoost full | 2.3047 | 3.1379 | 3.6315 | 1.3269 |
| XGBoost recent 5y | 2.4240 | 3.3889 | 3.6372 | 1.2133 |
| XGBoost weighted | 2.3476 | 3.1699 | 3.6717 | 1.3241 |

Every model performs worse on unseen outer folds than on its own training
rows. XGBoost has a larger average full-history generalization gap than Ridge,
and its selected tree count ranges from 1 to 196. These results do not support
promoting XGBoost based on the Step 11 validation-period MAE improvement.

## Scenario-shock sensitivity method

Every model/history variant is checked in all five folds.

For each fold:

1. Select real training rows nearest the 10th, 50th, and 90th percentiles of
   VIX.
2. Keep each row's non-scenario market state unchanged.
3. Set all eight scenario inputs to neutral zero.
4. Vary one input at a time across five points spanning that fold's observed
   training minimum and maximum, also including zero.
5. Record prediction span, end-to-end direction, discontinuities,
   monotonicity, and finite/bounded output checks.

A span of at least 0.05 S&P return percentage points is classified as a
meaningful response for this diagnostic. This threshold is an audit rule, not
a claim of economic significance.

## Scenario-shock sensitivity findings

### Approved full-history Ridge

| Input | Folds with meaningful response | Median prediction span | Maximum span | Meaningful direction |
|---|---:|---:|---:|---|
| Fed Funds Rate Change | 4 of 5 | 0.2517 | 0.3860 | mixed |
| CPI Surprise | 3 of 5 | 0.0863 | 0.3110 | positive |
| Oil Price Change | 4 of 5 | 0.6102 | 2.8333 | positive |
| GDP Growth Surprise | 3 of 5 | 0.0527 | 1.2345 | mixed |
| Unemployment Change | 2 of 5 | 0.0088 | 2.4866 | positive |
| PMI Change | 3 of 5 | 0.0684 | 2.0967 | negative |
| DXY Change | 3 of 5 | 0.1137 | 0.3441 | positive |
| VIX Change | 3 of 5 | 0.9385 | 2.5895 | negative |

The fold-level responsiveness result is:

| Fold | Ridge inputs with a meaningful response | XGBoost inputs with a meaningful response |
|---:|---|---|
| 1 | Fed Funds, Oil | none |
| 2 | Fed Funds, CPI, Oil, GDP, PMI, DXY, VIX | none |
| 3 | all eight inputs | VIX |
| 4 | all eight inputs | VIX |
| 5 | none | none |

Ridge has three properties that make it more suitable than the tested
XGBoost for the Scenario Playground:

- every one of the eight controls has a meaningful effect in at least one
  fold;
- each fold's Ridge response is smooth and monotonic because Ridge is linear;
- six of the eight inputs retain the same meaningful-response direction
  across folds.

It also has an important limitation:

```text
No input exceeds the meaningful-response threshold in all five folds.
Fed Funds and GDP response directions change across folds.
Neither model family crosses the threshold for any shock in the newest fold.
Several consistent signs should not be interpreted as causal effects.
```

### XGBoost

For full-history XGBoost:

- VIX Change is meaningful in only two of five folds;
- the other seven inputs are meaningful in zero folds;
- Oil Price Change reaches a maximum span of about 0.0451 percentage points,
  below the audit threshold;
- all other non-VIX inputs have a zero maximum span.

The recent-five-year XGBoost also responds meaningfully only to VIX in two
folds. The recency-weighted variant responds meaningfully only to VIX in one
fold.

The Step 11 limitation therefore remains and is stronger under repeated
evaluation:

```text
XGBoost continues to ignore most Scenario Playground inputs.
```

## Final methodology decision

The approved model-selection methodology for Step 13 is:

```text
Use purged expanding walk-forward evidence.
Fit every preprocessing step and model inside each fold.
Evaluate average error, worst-fold error, fold dispersion,
directional accuracy, prediction magnitude, VIX regimes,
overfitting, and eight-input scenario responsiveness.
Do not select a family or history window from one MAE result.
```

The approved S&P methodology is:

```text
model family:     Ridge
history strategy: full history
```

Only Ridge proceeds to Step 13. XGBoost is not promoted.

This approves a methodology to carry into Step 13; it does not approve a
finished model for production inference. Ridge's scenario effects may be
evaluated as learned historical associations, but the product must not
describe them as stable causal effects. The lack of meaningful latest-fold
response is a deployment-level blocker that Step 13 must continue to expose
for every target. If stable or causal shock interpretation is a product
requirement, the present data and models remain methodologically insufficient.

## Files created or modified

### Code, test, notebook, requirements, and documentation

```text
scenario_modeling/src/walk_forward_validation.py
scenario_modeling/tests/test_walk_forward_validation.py
scenario_modeling/notebooks/10_walk_forward_validation.ipynb
scenario_modeling/requirements.txt
markdown_files/scenario_modeling/12_WALK_FORWARD_VALIDATION_REPORT.md
markdown_files/scenario_modeling/README.md
```

No new dependency was installed. The requirements file's scope comment was
advanced through Step 12; all packages used by Step 12 were already pinned.

The notebook was executed top-to-bottom using the registered
`insightpulse-backend` kernel backed by `backend/.venv`. All seven code cells
completed and the notebook contains no error output.

### Generated local outputs

All generated outputs are under:

```text
data/scenario_playground/interim/models/walk_forward/
```

| File | Shape or role |
|---|---|
| `sp500_fold_manifest.csv` | 5 folds x 13 columns |
| `sp500_fit_audit.csv` | 30 fold/model/history fits x 16 columns |
| `sp500_fold_predictions.csv` | 2,700 prediction rows x 9 columns |
| `sp500_fold_metrics.csv` | 45 fold/strategy rows x 22 columns |
| `sp500_aggregate_metrics.csv` | 9 strategies x 26 columns |
| `sp500_regime_metrics.csv` | 27 strategy/regime rows x 22 columns |
| `sp500_overfit_diagnostics.csv` | 30 model fits x 20 columns |
| `sp500_sensitivity.csv` | 720 fold/state/input rows x 24 columns |
| `sp500_sensitivity_summary.csv` | 48 model/input rows x 16 columns |
| `sp500_methodology_comparison.csv` | 9 strategies x 30 columns |
| `sp500_walk_forward_summary.json` | Decision, provenance, hashes, guards, versions, and gate |

No Ridge pipeline, XGBoost Booster, or other production model artifact was
exported.

## Validation results

The Step 12 acceptance tests verify:

- five chronological expanding folds;
- exact outer purge counts and nonempty inner purges;
- all six model/history variants and all three naive baselines;
- 300 finite out-of-sample predictions per strategy;
- fold-local fit and preprocessing scope;
- full/recent/weighted history comparisons;
- deterministic repeated construction;
- all eight scenario inputs checked in all model variants;
- the repeated Ridge-versus-XGBoost decision;
- 584 matrix rows and 58 features preserved;
- zero locked final-test target rows parsed;
- the Step 12 gate.

Validation commands and results:

```text
backend\.venv\Scripts\python.exe -m compileall -q scenario_modeling
PASS

backend\.venv\Scripts\python.exe -m unittest \
  scenario_modeling.tests.test_walk_forward_validation -v
9 tests passed

backend\.venv\Scripts\python.exe -m unittest discover \
  -s scenario_modeling\tests -p "test_*.py" -v
86 tests passed in 136.600 seconds

backend\.venv\Scripts\python.exe -m pip check
No broken requirements found
```

Additional checks passed:

- all 36 pinned modeling requirements match installed versions;
- all 10 hashed CSV outputs match their recorded SHA-256 values;
- notebook JSON parses;
- all seven notebook code cells have consecutive execution counts;
- the notebook has zero error outputs;
- its recorded executable is `backend/.venv/Scripts/python.exe`;
- its kernelspec is `insightpulse-backend`;
- generated local outputs are ignored by Git.

## Remaining limitations

- Only S&P 500 has received walk-forward model selection.
- The final test remains locked and has not estimated final unbiased
  performance.
- Predictive gains over a training-mean baseline are modest.
- OOS R-squared is low.
- Event-aware rows and labels overlap within validation folds, although no
  training label crosses a validation boundary.
- The 27-row high-VIX subset is small.
- Ridge shock magnitude is not meaningful in every fold, and Fed Funds and
  GDP signs are unstable.
- Neither candidate has a meaningful scenario-shock response in the newest
  fold; production inference remains unapproved.
- The model estimates predictive associations, not intervention effects.
- The inherited PMI proxy and continuous `CL=F` limitations remain.

These limitations do not block Step 13 all-target training, provided the same
purged methodology and non-causal interpretation are preserved.

## Completion gate

```text
STEP_12_WALK_FORWARD_COMPLETE_READY_FOR_STEP_13_ALL_TARGETS
```
