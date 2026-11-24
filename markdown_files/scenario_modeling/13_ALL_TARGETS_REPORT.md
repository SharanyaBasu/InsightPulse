# Scenario Playground Step 13 All-Target Ridge Report

## Authoritative correction to the Step 14 serving decision

The Step 13 training, walk-forward metrics, naive comparisons, coefficients,
sensitivity diagnostics, and classifications in this report remain unchanged.
They must not be rewritten to make any model appear stronger.

The downstream interpretation of those classifications is corrected:

```text
evaluation classification != model availability
```

All 12 target-specific Ridge configurations remain available for callable
research inference. The previous `NAIVE_FALLBACK_REQUIRED` label remains an
honest statement that a target's Ridge model did not reliably outperform its
best naive benchmark. It now controls confidence metadata and warnings; it no
longer instructs Step 14 to replace that model's output with a naive value.

Accordingly:

- S&P 500, Energy, and Consumer Discretionary retain
  `APPROVED_WITH_LOW_CONFIDENCE`;
- the other nine targets retain their existing weak evaluation evidence and
  `NAIVE_FALLBACK_REQUIRED` assessment;
- all 12 independently fitted Ridge models must return numeric baseline,
  scenario, and incremental research predictions;
- zero, training-mean, and trailing strategies remain comparison benchmarks,
  not inference-output substitutes;
- no confidence category is promoted;
- all scenario responses remain noncausal, research-only associations.

The corrected all-target policy was subsequently implemented and validated in
Step 14. See `14_SCENARIO_INFERENCE_REPORT.md` for the 12-pipeline inference
results. This does not alter any Step 13 metric or classification.

## Status and scope

Step 13's all-target training and evaluation are complete. Its downstream
research-serving decision is amended by the authoritative correction above.

The step remained necessary because Step 12 selected a methodology using only
the S&P 500 target. Step 13 applies that frozen methodology to all 12 required
targets and decides, target by target, whether the resulting Ridge model is
credible enough for experimental scenario inference.

The implementation preserves:

- full-history Ridge only;
- the frozen 584-row, 58-feature event-aware matrix;
- the same five expanding walk-forward folds used in Step 12;
- target-end-date-aware outer and inner purging;
- fold-local median imputation, scaling, alpha selection, and fitting;
- all 12 frozen 21-trading-session targets;
- all eight frozen scenario-shock inputs;
- three target-specific naive comparators;
- noncausal interpretation of every scenario response.

Only the 464 retained pre-final-test rows were evaluated. The 114 locked
final-test target rows were not parsed or accessed. No model binary or
production artifact was exported.

## Important decisions

### Model family and history

Every target uses the Step 12-approved full-history Ridge pipeline:

```text
fold-local median imputer
fold-local standard scaler
Ridge
```

Ridge alpha is selected separately for every target and every outer fold by a
purged chronological inner split. The search grid remains:

```text
0.01, 0.1, 1, 10, 100, 1,000, 10,000, 100,000, 1,000,000
```

The final pretest alpha for each target is selected using the same inner
procedure on the 464-row pretest period and then refitted on those 464 rows.
These refits freeze research configurations for Step 14; they are not exported
as deployable models.

### Naive baselines

Every Ridge result is compared on the identical outer-fold rows with:

```text
naive_zero
naive_training_mean
naive_recent_trailing_21_observation
```

The trailing baseline is target-specific and uses only source observations
available by the row's target baseline date.

### Evaluation classifications

The classifications are deterministic measures of validation strength. Under
the corrected policy they do not remove a Ridge configuration from the
research-inference interface:

- `NOT_APPROVED`: predictions are pathological, sensitivity outputs are not
  finite/sensible, or fewer than four shocks respond meaningfully in any fold.
- `NAIVE_FALLBACK_REQUIRED`: Ridge is not within 2% of the best naive MAE and
  RMSE and does not beat the best naive MAE in at least two folds.
- `APPROVED_FOR_EXPERIMENTAL_INFERENCE`: Ridge passes the minimum gate, beats
  the best naive on both errors, has positive pooled out-of-sample R-squared,
  wins at least three folds, and has broad, stable scenario responsiveness.
- `APPROVED_WITH_LOW_CONFIDENCE`: Ridge passes the minimum predictive and
  scenario gates but not every strong experimental-approval gate.

A meaningful response is at least 0.05 percentage points for price-return
targets or 1 basis point for the 10-year-yield target. Sensitivity is tested at
low-, median-, and high-VIX representative states over each fold's observed
training range.

The sensible-range check compares a model's shock-response span with the
historical range of that target in the current training fold. This permits
large but historically supported crisis responses while still rejecting
nonfinite, nonmonotonic, or out-of-range behavior.

## Frozen walk-forward design

Each target has 300 out-of-sample Ridge predictions: five folds with 60
validation rows each.

| Fold | Training samples | Train rows | Outer purge rows | Validation samples | Validation rows |
|---:|---|---:|---:|---|---:|
| 1 | 2013-08-05 to 2017-02-16 | 160 | 4 | 2017-03-24 to 2018-06-29 | 60 |
| 2 | 2013-08-05 to 2018-06-04 | 222 | 2 | 2018-07-09 to 2019-10-07 | 60 |
| 3 | 2013-08-05 to 2019-09-09 | 280 | 4 | 2019-10-11 to 2021-01-14 | 60 |
| 4 | 2013-08-05 to 2020-12-18 | 342 | 2 | 2021-01-27 to 2022-06-13 | 60 |
| 5 | 2013-08-05 to 2022-05-12 | 401 | 3 | 2022-06-15 to 2023-11-06 | 60 |

The purge rule is:

```text
training target_end_date < first validation target_baseline_date
```

Every one of the 60 target/fold fits also has a nonempty inner purge.

## All-target walk-forward results

MAE and RMSE are percentage points for price-return targets and basis points
for `ten_year_yield_bps`. `MAE delta` is the percentage improvement over the
best naive MAE; a negative number means Ridge is worse. `Wins` is the number of
five folds in which Ridge MAE beats that target's best pooled-MAE naive
strategy.

| Target | Final alpha | MAE | RMSE | OOS R² | Direction | Best naive (MAE) | MAE delta | Wins | Classification |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| S&P 500 | 10,000 | 3.575 | 5.027 | 0.014 | 66.7% | train mean (3.651) | +2.07% | 4 | `APPROVED_WITH_LOW_CONFIDENCE` |
| Nasdaq | 10,000 | 5.299 | 7.096 | -0.383 | 66.7% | train mean (4.681) | -13.20% | 3 | `NAIVE_FALLBACK_REQUIRED` |
| 10-year yield | 1,000,000 | 19.765 | 26.599 | -0.084 | 44.7% | zero (18.943) | -4.34% | 1 | `NAIVE_FALLBACK_REQUIRED` |
| DXY | 100 | 1.446 | 1.911 | -0.042 | 54.3% | zero (1.430) | -1.10% | 0 | `NAIVE_FALLBACK_REQUIRED` |
| Gold | 10 | 2.989 | 3.775 | -0.029 | 54.3% | train mean (2.912) | -2.64% | 3 | `NAIVE_FALLBACK_REQUIRED` |
| Oil | 1,000,000 | 14.941 | 24.647 | -0.409 | 46.7% | zero (10.761) | -38.84% | 0 | `NAIVE_FALLBACK_REQUIRED` |
| Technology | 1,000,000 | 5.648 | 7.706 | -0.490 | 68.7% | train mean (4.924) | -14.70% | 4 | `NAIVE_FALLBACK_REQUIRED` |
| Energy | 10,000 | 7.211 | 10.590 | -0.028 | 60.7% | zero (7.279) | +0.93% | 3 | `APPROVED_WITH_LOW_CONFIDENCE` |
| Financials | 100,000 | 5.104 | 7.062 | -0.119 | 54.0% | train mean (4.659) | -9.53% | 2 | `NAIVE_FALLBACK_REQUIRED` |
| Utilities | 1,000 | 4.256 | 5.763 | -0.188 | 59.7% | train mean (3.896) | -9.23% | 2 | `NAIVE_FALLBACK_REQUIRED` |
| Healthcare | 100,000 | 3.328 | 4.361 | -0.008 | 58.7% | train mean (3.310) | -0.54% | 1 | `NAIVE_FALLBACK_REQUIRED` |
| Consumer discretionary | 10,000 | 4.800 | 6.561 | -0.018 | 61.7% | train mean (4.802) | +0.04% | 4 | `APPROVED_WITH_LOW_CONFIDENCE` |

No target meets the strongest experimental-inference standard. The three
low-confidence targets meet the minimum gate, but each has an important
weakness:

- S&P 500 has positive pooled OOS R² and beats both pooled naive errors, but
  none of its eight shocks crosses the meaningful-response threshold in the
  newest fold.
- Energy is responsive to all eight shocks in all folds, but pooled OOS R² is
  negative, RMSE is 1.25% worse than zero, and only three shock directions are
  stable.
- Consumer discretionary is approximately tied with the training-mean naive
  baseline, has negative pooled OOS R², and no shock is meaningful in the
  newest fold.

## Selected regularization and prediction ranges

The fold-alpha sequence is ordered from fold 1 through fold 5. Prediction
ranges are the 300 outer-fold Ridge predictions.

| Target | Fold-selected alphas | Final pretest alpha | Prediction range |
|---|---|---:|---:|
| S&P 500 | 10,000; 1,000; 1,000; 1,000; 1,000,000 | 10,000 | -1.139 to 7.166% |
| Nasdaq | 1,000,000; 100; 10,000; 100; 1,000,000 | 10,000 | -0.590 to 16.809% |
| 10-year yield | 10,000; 1,000; 1,000,000; 10,000; 1,000 | 1,000,000 | -15.120 to 12.692 bps |
| DXY | 1,000,000; 1,000,000; 1,000,000; 1,000,000; 10,000 | 100 | 0.078 to 0.533% |
| Gold | 1,000; 1,000,000; 10,000; 10,000; 1,000 | 10 | -0.483 to 4.249% |
| Oil | 1,000,000; 1,000; 1; 1,000; 100,000 | 1,000,000 | -74.443 to 107.812% |
| Technology | 1,000,000; 1,000; 1,000; 10; 10,000 | 1,000,000 | -1.736 to 24.864% |
| Energy | 1,000; 100; 100; 100; 100 | 10,000 | -12.060 to 26.749% |
| Financials | 10,000; 10; 1,000; 1,000; 1,000,000 | 100,000 | -9.581 to 14.686% |
| Utilities | 100; 100,000; 1,000,000; 1,000; 1,000 | 1,000 | -1.190 to 10.019% |
| Healthcare | 1,000,000; 100; 1,000,000; 10,000; 100,000 | 100,000 | -2.259 to 2.239% |
| Consumer discretionary | 1,000,000; 1,000; 1,000; 1,000; 1,000,000 | 10,000 | -0.872 to 6.949% |

The variation in selected alpha across folds is retained as evidence of model
instability. Very large final alphas for the yield, oil, and technology models
show that their pretest refits require heavy shrinkage.

## Feature importance

For Ridge, feature importance is the signed coefficient after fold-local
standardization. The five largest absolute standardized coefficients from
each final pretest research fit are:

| Target | Five largest standardized coefficients |
|---|---|
| S&P 500 | VIX change -0.0454; S&P z-score -0.0328; S&P drawdown -0.0315; unemployment rate +0.0310; VIX level +0.0297 |
| Nasdaq | VIX change -0.0445; unemployment rate +0.0419; S&P drawdown -0.0377; S&P volatility +0.0365; VIX level +0.0344 |
| 10-year yield | oil z-score +0.0023; DXY change +0.0022; DXY drawdown -0.0021; real-rate proxy -0.0020; CPI YoY +0.0018 |
| DXY | DXY change +0.4287; oil 200-day distance +0.3188; S&P drawdown +0.2382; unemployment rate -0.2293; CPI MoM -0.2163 |
| Gold | oil 200-day distance -2.4214; S&P z-score +2.0359; gold 200-day distance -1.9313; Nasdaq z-score -1.5723; S&P drawdown -1.4305 |
| Oil | oil shock +0.0049; GDP surprise -0.0025; S&P 21-day return +0.0022; oil drawdown -0.0020; oil 200-day distance -0.0019 |
| Technology | VIX change -0.0005; S&P drawdown -0.0005; S&P volatility +0.0004; S&P z-score -0.0004; VIX level +0.0004 |
| Energy | VIX level +0.0836; oil shock +0.0759; real GDP growth +0.0756; Nasdaq z-score -0.0703; S&P 50-day distance -0.0666 |
| Financials | VIX change -0.0059; unemployment change +0.0043; oil shock +0.0039; GDP surprise -0.0036; S&P z-score -0.0034 |
| Utilities | CPI MoM +0.2403; VIX change -0.2251; unemployment change +0.1773; 10y–2y slope +0.1419; oil 5-day return +0.1388 |
| Healthcare | S&P 50-day distance -0.0043; Nasdaq 50-day distance -0.0042; VIX level +0.0035; S&P volatility +0.0035; gold volatility +0.0034 |
| Consumer discretionary | VIX change -0.0518; unemployment rate +0.0396; S&P volatility +0.0347; oil shock +0.0345; S&P 200-day distance -0.0345 |

Coefficient magnitude does not establish economic causality. Correlated
features can also move importance between columns and folds.

## Error stability and regime results

`Fold MAE SD` and `worst fold` summarize temporal instability. Regime columns
are Ridge MAE on the same pooled walk-forward predictions grouped by the VIX
level known at each sample.

| Target | Fold MAE SD | Worst fold MAE | Low VIX MAE | Medium VIX MAE | High VIX MAE |
|---|---:|---:|---:|---:|---:|
| S&P 500 | 1.107 | 4.893 | 3.021 | 4.248 | 5.044 |
| Nasdaq | 2.092 | 7.868 | 4.027 | 6.703 | 9.134 |
| 10-year yield | 8.621 | 33.895 | 17.971 | 22.989 | 20.992 |
| DXY | 0.542 | 2.199 | 1.240 | 1.771 | 1.739 |
| Gold | 0.408 | 3.434 | 2.782 | 3.310 | 3.303 |
| Oil | 15.159 | 41.847 | 7.814 | 22.790 | 36.530 |
| Technology | 2.372 | 9.200 | 4.484 | 7.066 | 8.720 |
| Energy | 3.136 | 10.908 | 5.883 | 9.138 | 9.671 |
| Financials | 1.484 | 6.953 | 4.349 | 6.106 | 6.814 |
| Utilities | 1.349 | 5.509 | 3.783 | 4.667 | 6.054 |
| Healthcare | 0.705 | 4.415 | 3.183 | 3.065 | 5.191 |
| Consumer discretionary | 1.505 | 6.039 | 3.922 | 5.757 | 7.498 |

Most equity errors increase in higher-volatility regimes. Oil is the clearest
failure: MAE rises from 7.814 percentage points in low VIX to 36.530 in high
VIX. This, its continuous-futures roll exposure, and its severe pooled
underperformance support the zero fallback.

## Scenario responsiveness and sign stability

The notation below is:

```text
input = responsive folds out of 5, direction
L = meaningful in the latest fold
mixed = both positive and negative meaningful responses across folds
```

| Target | Eight-input response audit |
|---|---|
| S&P 500 | CPI 3/5 +; DXY 3/5 +; Fed 4/5 mixed; GDP 3/5 mixed; oil 4/5 +; PMI 3/5 -; unemployment 2/5 +; VIX 3/5 - |
| Nasdaq | CPI 3/5 mixed; DXY 2/5 +; Fed 3/5 -; GDP 2/5 mixed; oil 3/5 +; PMI 2/5 -; unemployment 2/5 mixed; VIX 3/5 - |
| 10-year yield | CPI 2/5 mixed L; DXY 2/5 + L; Fed 1/5 + L; GDP 2/5 mixed L; oil 3/5 + L; PMI 1/5 -; unemployment 3/5 + L; VIX 2/5 mixed L |
| DXY | CPI 1/5 + L; DXY 1/5 + L; Fed 1/5 - L; GDP 1/5 + L; oil 1/5 - L; PMI 0/5; unemployment 1/5 - L; VIX 0/5 |
| Gold | CPI 4/5 - L; DXY 4/5 - L; Fed 3/5 mixed L; GDP 3/5 mixed L; oil 3/5 - L; PMI 3/5 + L; unemployment 3/5 - L; VIX 4/5 + L |
| Oil | CPI 3/5 mixed; DXY 3/5 -; Fed 4/5 + L; GDP 3/5 - L; oil 4/5 + L; PMI 4/5 - L; unemployment 4/5 mixed L; VIX 4/5 mixed L |
| Technology | CPI 3/5 +; DXY 3/5 +; Fed 4/5 - L; GDP 4/5 mixed L; oil 4/5 + L; PMI 3/5 - L; unemployment 4/5 mixed L; VIX 4/5 - L |
| Energy | CPI 5/5 mixed L; DXY 5/5 - L; Fed 5/5 mixed L; GDP 5/5 mixed L; oil 5/5 + L; PMI 5/5 mixed L; unemployment 5/5 mixed L; VIX 5/5 - L |
| Financials | CPI 4/5 +; DXY 3/5 +; Fed 4/5 mixed; GDP 3/5 mixed; oil 4/5 +; PMI 3/5 -; unemployment 3/5 +; VIX 3/5 - |
| Utilities | CPI 3/5 mixed L; DXY 3/5 mixed L; Fed 3/5 mixed L; GDP 3/5 + L; oil 3/5 + L; PMI 3/5 mixed L; unemployment 3/5 mixed L; VIX 3/5 mixed L |
| Healthcare | CPI 1/5 +; DXY 1/5 +; Fed 2/5 -; GDP 2/5 -; oil 1/5 -; PMI 2/5 -; unemployment 2/5 +; VIX 2/5 - |
| Consumer discretionary | CPI 3/5 mixed; DXY 3/5 +; Fed 3/5 mixed; GDP 3/5 mixed; oil 3/5 +; PMI 2/5 -; unemployment 3/5 mixed; VIX 3/5 - |

All 1,440 individual sensitivity checks are finite, monotonic, and within the
target-specific sensible range. This only proves numerical behavior over the
observed training ranges. It does not prove that a response is stable,
economically correct, or causal.

| Target | Inputs responsive in any fold | Responsive in at least 3 folds | Responsive in latest fold | Stable response signs | Stable scenario-coefficient signs |
|---|---:|---:|---:|---:|---:|
| S&P 500 | 8 | 7 | 0 | 6 | 4 |
| Nasdaq | 8 | 4 | 0 | 5 | 5 |
| 10-year yield | 8 | 2 | 7 | 5 | 4 |
| DXY | 6 | 0 | 6 | 6 | 6 |
| Gold | 8 | 8 | 8 | 6 | 4 |
| Oil | 8 | 8 | 6 | 5 | 3 |
| Technology | 8 | 8 | 6 | 6 | 5 |
| Energy | 8 | 8 | 8 | 3 | 3 |
| Financials | 8 | 8 | 0 | 6 | 5 |
| Utilities | 8 | 8 | 8 | 2 | 2 |
| Healthcare | 8 | 0 | 0 | 8 | 4 |
| Consumer discretionary | 8 | 7 | 0 | 4 | 4 |

The distinction between response-sign stability and coefficient-sign
stability is intentional. The first applies the meaningful-response threshold;
the second examines the raw coefficient sign across every fold, including very
small coefficients.

## Residual findings

| Target | Worst sample | Actual | Prediction | Maximum absolute error |
|---|---|---:|---:|---:|
| S&P 500 | 2020-02-21 | -28.846% | 0.196% | 29.042 pp |
| Nasdaq | 2020-02-21 | -27.163% | 1.059% | 28.222 pp |
| 10-year yield | 2020-02-10 | -111.000 bps | -2.414 bps | 108.586 bps |
| DXY | 2022-11-07 | -6.765% | 0.223% | 6.988 pp |
| Gold | 2020-03-16 | 13.540% | 1.383% | 12.156 pp |
| Oil | 2020-04-29 | 173.177% | 61.247% | 111.930 pp |
| Technology | 2022-06-06 | -9.814% | 19.592% | 29.406 pp |
| Energy | 2020-02-21 | -53.309% | -3.209% | 50.100 pp |
| Financials | 2020-02-21 | -37.006% | 0.528% | 37.534 pp |
| Utilities | 2020-02-21 | -26.022% | 1.173% | 27.195 pp |
| Healthcare | 2020-03-27 | 23.146% | 0.980% | 22.166 pp |
| Consumer discretionary | 2020-02-21 | -31.579% | 0.897% | 32.476 pp |

The largest errors cluster around the 2020 crisis. Technology's worst miss is
different: one fold selected a low alpha and produced an overly large positive
prediction in June 2022. The later pretest refit selected alpha 1,000,000, but
the out-of-sample miss remains part of the assessment.

## Final evaluation classifications and benchmark assignments

### `APPROVED_FOR_EXPERIMENTAL_INFERENCE`

None.

### `APPROVED_WITH_LOW_CONFIDENCE`

- `sp500`
- `energy`
- `consumer_discretionary`

These models may proceed only as clearly labeled research outputs with their
limitations and uncertainty preserved.

### `NAIVE_FALLBACK_REQUIRED`

| Target | Best naive comparison benchmark |
|---|---|
| Nasdaq | fold-training mean |
| 10-year yield | zero 21-session yield change |
| DXY | zero 21-session return |
| Gold | fold-training mean |
| Oil | zero 21-session return |
| Technology | fold-training mean |
| Financials | fold-training mean |
| Utilities | fold-training mean |
| Healthcare | fold-training mean |

The nine Ridge fits remain weak-evidence research models. Their listed naive
strategies are the benchmarks that won the original comparison; they are not
the primary Step 14 output.

The corrected Step 14 workflow must fit and call these nine Ridge
configurations, return numeric Ridge baseline, scenario, and incremental
values, and attach explicit warnings that the models did not reliably beat
their best naive benchmarks. The benchmark predictions may remain available
in separately named comparison fields.

### `NOT_APPROVED`

None failed the minimum numerical/pathology gate. This does not override the
nine stricter naive-fallback decisions.

## Final-test decision

The active Step 13 instruction lists test metrics as a desired per-target
output, but it does not explicitly authorize the one-time opening of the
locked final test. The user required explicit authorization before that
access. Therefore:

```text
final-test rows declared: 114
final-test target rows parsed: 0
final-test access count: 0
test metrics: NOT_COMPUTED_LOCKED_FINAL_TEST_NOT_AUTHORIZED
```

No configuration, classification, or fallback used final-test evidence.

## Files and generated outputs

### Code and notebook

- `scenario_modeling/src/all_target_models.py`
  - builds the frozen folds;
  - performs per-target inner alpha selection and fold-local fitting;
  - produces Ridge and naive predictions;
  - computes pooled, fold, regime, residual, coefficient, and sensitivity
    diagnostics;
  - applies deterministic classifications;
  - persists research evidence without serializing models.
- `scenario_modeling/tests/test_all_target_models.py`
  - contains ten Step 13 acceptance tests.
- `scenario_modeling/notebooks/11_all_target_models.ipynb`
  - reruns and persists the all-target evaluation;
  - displays folds, metrics, regimes, sensitivities, configurations, residuals,
    classifications, and acceptance checks;
  - was executed top-to-bottom with the registered
    `InsightPulse backend .venv` kernel.

### Generated local evidence

All generated outputs are under:

```text
data/scenario_playground/interim/models/all_targets/
```

| Output | Rows | Purpose |
|---|---:|---|
| `all_targets_fold_manifest.csv` | 5 | Frozen folds and purge evidence |
| `all_targets_trailing_baseline_audit.csv` | 3,600 | Point-in-time trailing-naive provenance |
| `all_targets_alpha_search.csv` | 648 | Inner alpha candidates for fold and pretest fits |
| `all_targets_fit_audit.csv` | 60 | Fold-local fit and preprocessing scope |
| `all_targets_fold_predictions.csv` | 14,400 | 12 targets × 4 strategies × 300 OOS rows |
| `all_targets_fold_metrics.csv` | 240 | Fold metrics by target and strategy |
| `all_targets_aggregate_metrics.csv` | 48 | Pooled metrics by target and strategy |
| `all_targets_naive_comparison.csv` | 12 | Best-naive and fold-win comparisons |
| `all_targets_regime_metrics.csv` | 144 | Low-, medium-, and high-VIX metrics |
| `all_targets_residual_summary.csv` | 12 | Residual and worst-error summaries |
| `all_targets_fold_coefficients.csv` | 3,480 | 58 coefficients × 12 targets × 5 folds |
| `all_targets_coefficient_stability.csv` | 696 | Cross-fold coefficient stability |
| `all_targets_sensitivity.csv` | 1,440 | 12 × 5 × 3 states × 8 shocks |
| `all_targets_sensitivity_summary.csv` | 96 | Per-target/per-shock response stability |
| `all_targets_final_configurations.csv` | 12 | Frozen pretest research configurations |
| `all_targets_final_coefficients.csv` | 696 | Final pretest research-fit coefficients |
| `all_targets_target_assessment.csv` | 12 | Complete target decisions and limitations |
| `all_targets_summary.json` | 1 | Machine-readable gate and lock evidence |

These outputs are local, reproducible research data and remain ignored by Git.

## Validation

The following checks pass:

- all 12 targets are present and ordered according to the frozen contract;
- only full-history Ridge is used;
- the five exact Step 12 folds are reused;
- all outer and inner purges are nonempty;
- all 60 target/fold fits use fold-local preprocessing;
- all four strategies have 300 finite OOS predictions per target;
- all 3,600 trailing baselines pass their point-in-time checks;
- all 3,480 fold coefficients and 696 stability records are present;
- all eight shocks are audited for every target;
- all 1,440 sensitivity outputs are finite and sensible;
- all 12 targets receive an allowed classification;
- no Ridge prediction set is pathological;
- all 12 final pretest configurations are frozen without model binaries;
- the event-aware design and 58-feature matrix are preserved;
- locked final-test target rows parsed and accessed both equal zero;
- the executed notebook has seven successful code cells and no error output;
- the notebook uses the `insightpulse-backend` kernelspec;
- Python modules compile;
- the focused Step 13 suite passes 10 of 10 tests;
- the complete Scenario Modeling suite passes;
- no new third-party dependency was needed.

Primary commands:

```powershell
backend\.venv\Scripts\python.exe -m scenario_modeling.src.all_target_models
backend\.venv\Scripts\python.exe -m unittest scenario_modeling.tests.test_all_target_models -v
backend\.venv\Scripts\python.exe -m unittest discover -s scenario_modeling/tests -p "test_*.py"
```

## Remaining limitations

- No target earned `APPROVED_FOR_EXPERIMENTAL_INFERENCE`.
- Nine targets did not reliably beat their best naive benchmark. Their Ridge
  outputs are callable only as weak-evidence experimental research outputs,
  not as validated production forecasts.
- The three previously promoted Ridge models remain low-confidence research
  models.
- Scenario response signs are unstable for several targets and often disappear
  in the latest fold.
- Most pooled OOS R-squared values are negative.
- Crisis-period errors are large.
- Oil uses a continuous `CL=F` futures series and retains roll effects.
- PMI remains the disclosed Kansas City Fed manufacturing proxy.
- Economic relationships are predictive associations, not causal effects.
- The final test remains locked, so there is no untouched final-test estimate.
- No production model binaries, live-state builder, inference API, uncertainty
  interval, or application integration exists yet.

## Completion and corrected Step 14 handoff

Step 13's all-target training and evaluation work is complete:

```text
STEP_13_ALL_TARGET_RIDGE_EVALUATION_COMPLETE_READY_FOR_CORRECTED_ALL_TARGET_STEP_14
```

The corrected Step 14 research interface must:

- build 12 independent target-specific Ridge pipelines;
- return finite numeric Ridge baseline, scenario, and incremental values for
  every target;
- preserve each target's selected alpha, unit, preprocessing, feature order,
  metrics, classification, and warnings;
- retain naive strategies as separately identified comparison benchmarks only;
- state prominently that nine models failed the prior promotion gate and that
  all 12 outputs are research-only;
- do not claim causal or production-ready effects;
- keep the final test locked unless a later instruction explicitly authorizes
  its one-time use.

The corrected Step 14 was subsequently completed without altering this
historical evidence, exporting production artifacts, accessing Supabase,
modifying backend or frontend code, or starting a server.
