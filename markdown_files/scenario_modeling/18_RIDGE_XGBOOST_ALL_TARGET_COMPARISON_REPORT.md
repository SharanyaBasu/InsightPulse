# Scenario Playground All-Target Ridge versus XGBoost Research Report

## A. Executive summary

Task 6 is complete. This report verifies saved Task 5 results and independently applies the six frozen promotion gates to all 12 targets. No model is retrained, refitted, exported, or deployed.

Research selection result: SELECT_XGBOOST for Oil; RETAIN_RIDGE for the other 11 targets. Oil alone passes all six gates. This result is preserved as the research outcome; the subsequent release decision in Section H defers a mixed-model bundle. The released scenario-playground-ridge-v1.0.0 bundle remains all-Ridge and unchanged.

Oil's pooled MAE improves 27.986190% over Ridge with 4/5 fold wins. Its advantage over zero is only 0.013311893%. Responses are concentrated in Oil, DXY and VIX; all five individual macro-shock sweeps are flat. Selection does not imply high confidence or comprehensive macroeconomic scenario support.

Evidence convention: numeric tables/counts are verified from saved CSV/JSON results; derived values are identified. Gate calculations use full saved scalar precision. Display rounding does not relax a threshold. Gate 6 suitability and release recommendations are research judgments. Return errors/spans use percentage points of return; ten_year_yield_bps uses basis points. No raw-error average is calculated across targets.

## B. Experiment specification

| Item | Frozen specification |
| --- | --- |
| Dataset | 584 event-aware rows; 58 features; 12 targets |
| Authorized pretest | 464 retained rows; bounded loader parses 470 prefix rows and excludes six boundary-purge rows |
| Locked final test | 114 rows; zero target values accessed |
| Features | 45 market/state + eight scenario shocks + five release flags |
| Horizon | Exactly 21 canonical trading sessions |
| History | Full eligible expanding history |
| Outer validation | Five original folds; 60 rows each; 300 unique sample dates per strategy |
| Ridge | Authoritative Step 13 predictions; fold-specific selected alpha reconstructed only for verified diagnostics |
| Benchmarks | naive_zero; naive_training_mean; naive_recent_trailing_21_observation |
| Inner selection | Purged chronological inner split; MAE; simplest architecture within 1% of minimum |
| Early stopping/refit | Approved patience; best_iteration + 1 rounds; refit complete eligible outer training history |
| Preprocessing | Fold-local median imputation/scaling for Ridge; native XGBoost missing handling |

Both inner and outer purges require training target_end_date < first validation target_baseline_date. Canonical target dates are used, not removal of 21 event-aware rows. Outer-validation labels are never used in fitting, early stopping, or candidate selection.

| Fold | Train dates | Train rows | Outer purge | Validation dates | n | Last target end |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2013-08-05 to 2017-02-16 | 160 | 4 | 2017-03-24 to 2018-06-29 | 60 | 2018-07-27 |
| 2 | 2013-08-05 to 2018-06-04 | 222 | 2 | 2018-07-09 to 2019-10-07 | 60 | 2019-11-01 |
| 3 | 2013-08-05 to 2019-09-09 | 280 | 4 | 2019-10-11 to 2021-01-14 | 60 | 2021-02-11 |
| 4 | 2013-08-05 to 2020-12-18 | 342 | 2 | 2021-01-27 to 2022-06-13 | 60 | 2022-07-12 |
| 5 | 2013-08-05 to 2022-05-12 | 401 | 3 | 2022-06-15 to 2023-11-06 | 60 | 2023-12-04 |

| Architecture/rank | Depth | eta | Child weight | Row sample | Column sample | L1 | L2 | gamma | Max rounds | Patience |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regularized_stumps/1 | 1 | 0.03 | 8 | 0.85 | 0.75 | 0.1 | 10 | 0 | 500 | 40 |
| regularized_depth2/2 | 2 | 0.025 | 8 | 0.8 | 0.7 | 0.25 | 15 | 0.05 | 600 | 40 |
| strongly_regularized_depth2/3 | 2 | 0.02 | 15 | 0.7 | 0.65 | 0.75 | 30 | 0.1 | 700 | 50 |

CPU hist; nthread=1; max_bin=128; seed=20260725; objective reg:squarederror; eval_metric=mae. The all-target helper explicitly uses max_bin=128; the original historical Step 12 S&P-only path retains its original default. Those original paths were preserved, not retrospectively equated.

| Run | Inner candidate fits | Outer refits | Ridge reconstructions | Matched pairs | Seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 180 | 60 | 60 | 60 | 221.982 |
| 2 | 180 | 60 | 60 | 60 | 217.77 |

Each run has 240 XGBoost training calls: 180 candidate evaluations and 60 selected refits. Selected architecture counts: regularized_stumps=37, regularized_depth2=13, strongly_regularized_depth2=10. The identical second run was verification, not another methodology search; it was not persisted.

## C. Reproducibility and provenance

Branch: `khalil/scenario_playground_model_2`. HEAD: `ee9b6c7db2f5795608378a956644a7fc5673cbf9`. The experiment ran before the research modules were placed under version control. Source hashes below identify the actual experiment code independently of the later preservation commit.

| Source/configuration path | SHA-256 |
| --- | --- |
| `scenario_modeling/requirements.txt` | `bc3424bb1b1f650b941284ca4c59529c180ae9f7e915e7cb93e46c9997e38050` |
| `scenario_modeling/src/all_target_models.py` | `ee6527a4e184cb821c719077156bb314bd7a95ae83f22ecfaad59b5f50392121` |
| `scenario_modeling/src/all_target_xgboost_comparison.py` | `120708e81ac9abb7479fd457865463c201f03f3066097c517bcb73927467d2a1` |
| `scenario_modeling/src/evaluation.py` | `c2694af0cf61c8e0b54442cffff194fb1baaf1c47f618831bc58c2a996dec3ea` |
| `scenario_modeling/src/matched_scenario_sensitivity.py` | `a5f46d0e0babd201d61ff0cbfaf477e92690de0a50f9ed541d52f614ae2e8260` |
| `scenario_modeling/src/model_matrix.py` | `c10bbae99fb304abd646be120b60f0356771c6a93bf8ad03747f20ae8b9d46c9` |
| `scenario_modeling/src/ridge_baselines.py` | `c0797ab306e356cd081efab6e5836400f4e95bcafef9a89092b9adde8020340f` |
| `scenario_modeling/src/walk_forward_validation.py` | `5bb2eec4e26207aa20814053f791cd82bbec3cabd2bc568f25e6e2e8faea5d2e` |
| `scenario_modeling/src/xgboost_prototype.py` | `a1773cf06873b9bfcb321c15a95f5d8f1a6bb08b03c2ff8fc302508ef5796e8f` |

Authorized-pretest fingerprint: `7f14589043fb7ae6d93a172595c477593e6ef5e8bf0f8f87cdde43c94db2e008`. It matches approved Task 1 identity. Serialization is `insightpulse-retained-pretest-floathex-v1`: chronological retained rows, sample_date then ordered features and targets, Python float.hex binary64 strings, compact UTF-8 JSON without BOM/trailing newline. This establishes current input identity, not historical full-input byte identity. Task 6 verifies recorded identity without reading/hashing the canonical matrix.

| Evidence/identity | Result or SHA-256 |
| --- | --- |
| Frozen split manifest | `b7805d38f675ab2f18f69323f0790d5428f1d9fc70f2b71c37773a9cbc6aeaab` |
| Serialized outer fold manifest | `b357f963e5c75dc250be1e3b08f9a202d5c4051ba4fa1aef06322e7d2eded472` |
| Saved Step 13 summary | `78176bc9630539935bf1c889c4c510abaf92ee9b4d68e7d908b4d81743ea5001` |
| Task 5 checksum index | `e89320d0599bc533e612fa0ac450347987cf0a5626f6c958d6c323d86afed076` |
| Output verification | 18/18 indexed files; exactly 19 files including index |
| Benchmark identity | 14,400 original records and their combined-table copies remain exact |
| Focused tests recorded in Task 5 | 94 passed; zero failures; synthetic-only guard |
| Two-run agreement | 17/17 tables bit-exact; numeric difference 0; selections/round counts identical |
| Ridge reconstruction | 60/60 passed; 3,600 saved validation predictions checked |
| Final-test target access | 0 in Task 5; 0 in Task 6 |

Original Step 13 CSV evidence checksums:

| Original output key | SHA-256 |
| --- | --- |
| aggregate_metrics | `44b5367bc572847bdb60796549e690a083bb77ce1bc33c215ccbab2fabe7ac47` |
| alpha_search | `38cd2a8851e1ab357bfc0c7b527e379436c29245eac4291ac477231d015ab092` |
| coefficient_stability | `286bfe3bd0fcd3678cac1e9ec6ec7388c957518c6aa75e4384d278a78026b1ef` |
| final_coefficients | `cd8875804692d14b453cd8ed057d28569a6d3db09cdd0c4eeab9c96f58f0d61a` |
| final_configurations | `4f8a045a7ead9b0c65f3ce7320b47bd81dce1fa9e0acbd9b2c299421210c1fa1` |
| fit_audit | `4d5c415f59e2a775330056c0830ba2a72b45a92d3db64044b567c97100903710` |
| fold_coefficients | `a0c5ad7dc557b16dc232cd3f13cfed597214c47214870408245db59b62e6ed84` |
| fold_manifest | `b357f963e5c75dc250be1e3b08f9a202d5c4051ba4fa1aef06322e7d2eded472` |
| fold_metrics | `a25fc6b33f0c9805e7065402b6b9a0569724c13ad188e6a9389c9552856d93e6` |
| fold_predictions | `69dbbcad8b8ced0f06e72f540f4d30b7cfc1adc99efd9a28e02261f7366e99ad` |
| naive_comparison | `bcaffa1a22ec7d202e656902715c1a067b099071749b3ea3f654f133e799845b` |
| regime_metrics | `ddd25898d22197749a64d66e80d6aee9e32a93de98d7a87b0b4558d5ff20ef7f` |
| residual_summary | `240ea738f72ea227fcae1a0c3c6b91a0a34d86aa949763033743808906ff225c` |
| sensitivity | `a91c5c3b177d490fb2bfda8a271c34e9777a4841c5b73d6bf9a29db334add5d7` |
| sensitivity_summary | `cffe2bc8f70fafb0d71f1770123f785ab7e40fe2ca73bee2eee965944303b69e` |
| target_assessment | `8f564b05ad7361bc949da6cf275215a32742177191e88fc63a7e76142c2313fd` |
| trailing_baseline_audit | `3303079c091a10fcacf8e9bdbcc7a3e34387ab32f871fad5148d21e0bc3adf68` |

| Environment | Recorded version |
| --- | --- |
| joblib | 1.5.3 |
| numpy | 2.4.3 |
| pandas | 3.0.1 |
| python | 3.14.3 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.0 |
| xgboost | 3.3.0 |

Original target/fold Ridge alphas, median imputation, scaling, feature order, and training boundaries were reused without alpha search. Maximum absolute reconstruction difference is 3.85938392355e-10 (Oil, fold 3); all pass abs(diff) <= 1e-10 + 1e-11*abs(saved_prediction). The relative component permits this largest absolute difference. Reconstructed predictions never replace authoritative accuracy evidence.

CSV serialization verification: scalar columns use 12 significant digits; JSON-valued audit fields can retain fuller precision. The in-memory sensitivity validator requires exact arithmetic and cannot be applied unchanged after CSV reload. Saved increment/grid/range arithmetic was independently checked using the sum of 12-significant-digit rounding bounds and binary64 allowance. Maximum scenario-minus-baseline-minus-increment residual is 7.54994677887e-11; all residuals are within those bounds. Neutral increments remain exactly zero. This is serialization accounting, not a changed promotion threshold or a change to evidence files.

The Task 6 analysis guard rejects canonical-matrix reads, model fits, dataset builders, network calls and filesystem writes. The sole documentation file is created separately using an explicit patch. No experiment is rerun. All original source/evidence/artifact hashes recorded by Task 5 remain unchanged.

## D. Complete predictive comparisons

### Pooled performance and fold stability

Each model/target uses the same 300 aligned observations. R=Ridge full history; X=XGBoost full history. Direction accuracy is exact sign agreement, not investment performance. Correlation is undefined for constant predictors; saved reasons are explicit.

| Target | Model | MAE | RMSE | OOS R² | Direction % | Correlation | Fold MAE SD | Worst MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sp500 | R | 3.57513 | 5.02737 | 0.0136463 | 66.6667 | 0.168336 | 1.10678 | 4.89314 |
| sp500 | X | 3.70407 | 5.02445 | 0.0147936 | 58.3333 | 0.15817 | 1.24204 | 5.2507 |
| nasdaq | R | 5.29865 | 7.09598 | -0.382691 | 66.6667 | -0.103179 | 2.09166 | 7.86802 |
| nasdaq | X | 4.7136 | 6.09869 | -0.021347 | 63.6667 | 0.0295442 | 1.63216 | 6.70756 |
| ten_year_yield_bps | R | 19.7651 | 26.5991 | -0.0842112 | 44.6667 | -0.120226 | 8.62052 | 33.8947 |
| ten_year_yield_bps | X | 20.1668 | 27.2508 | -0.137987 | 41 | -0.248523 | 8.84573 | 34.815 |
| dxy | R | 1.44616 | 1.91149 | -0.0419577 | 54.3333 | -0.18448 | 0.542256 | 2.19917 |
| dxy | X | 1.46806 | 1.9226 | -0.0541035 | 49.3333 | -0.0771799 | 0.532819 | 2.23561 |
| gold | R | 2.98913 | 3.77543 | -0.0290137 | 54.3333 | 0.0783975 | 0.408378 | 3.43397 |
| gold | X | 2.95323 | 3.84807 | -0.0689912 | 54.6667 | 0.058193 | 0.505928 | 3.41696 |
| oil | R | 14.9413 | 24.647 | -0.409007 | 46.6667 | 0.353266 | 15.1588 | 41.8467 |
| oil | X | 10.7598 | 20.548 | 0.0206794 | 48.6667 | 0.160349 | 6.10641 | 21.2064 |
| technology | R | 5.64831 | 7.70619 | -0.489791 | 68.6667 | -0.0307065 | 2.37174 | 9.20031 |
| technology | X | 4.95998 | 6.41464 | -0.0322615 | 66.3333 | 0.0261823 | 1.58776 | 6.47195 |
| energy | R | 7.2111 | 10.5901 | -0.027781 | 60.6667 | 0.327866 | 3.13635 | 10.9083 |
| energy | X | 7.25988 | 10.3897 | 0.0107603 | 56 | 0.161221 | 3.30968 | 12.3713 |
| financials | R | 5.10356 | 7.06193 | -0.118751 | 54 | 0.108104 | 1.48397 | 6.95296 |
| financials | X | 4.86313 | 6.92748 | -0.0765564 | 55.3333 | 0.0215743 | 1.51292 | 7.33012 |
| utilities | R | 4.25588 | 5.76265 | -0.188346 | 59.6667 | 0.054982 | 1.34891 | 5.50919 |
| utilities | X | 3.91531 | 5.34035 | -0.0205563 | 56.3333 | 0.0565384 | 1.35463 | 5.51766 |
| healthcare | R | 3.3282 | 4.3613 | -0.00811914 | 58.6667 | 0.0584162 | 0.705304 | 4.41473 |
| healthcare | X | 3.29746 | 4.29083 | 0.0241984 | 57 | 0.169592 | 0.67493 | 4.31221 |
| consumer_discretionary | R | 4.80036 | 6.56133 | -0.0182702 | 61.6667 | 0.0775601 | 1.50474 | 6.03921 |
| consumer_discretionary | X | 4.74976 | 6.46813 | 0.0104516 | 62.6667 | 0.103377 | 1.50937 | 6.1193 |

R² is negative for 11/12 Ridge targets and 7/12 XGBoost targets. Positive values are small. Lower error than another weak model does not establish reliable forecasting skill.

### Individual fold MAE

Five-value sequences are F1 to F5. Wins use strict MAE comparison with no practical-tie margin.

| Target | Ridge F1–F5 | XGB F1–F5 | MAE improvement % | X wins | R wins |
| --- | --- | --- | --- | --- | --- |
| sp500 | 1.93361, 3.2575, 4.89314, 3.61137, 4.18001 | 1.96471, 3.39835, 5.2507, 3.43585, 4.47075 | -3.60673 | 1 | 4 |
| nasdaq | 2.44263, 4.23466, 6.56449, 7.86802, 5.38347 | 2.37849, 4.38798, 6.70756, 4.37656, 5.71738 | 11.0416 | 2 | 3 |
| ten_year_yield_bps | 11.0902, 16.2324, 16.8009, 20.8072, 33.8947 | 11.2497, 17.4504, 16.791, 20.528, 34.815 | -2.03259 | 2 | 3 |
| dxy | 1.58989, 0.68957, 1.32026, 1.4319, 2.19917 | 1.59422, 0.754686, 1.32608, 1.42972, 2.23561 | -1.51472 | 1 | 4 |
| gold | 2.85197, 2.38129, 3.43397, 3.00529, 3.27316 | 2.27182, 2.60152, 3.40921, 3.06664, 3.41696 | 1.20108 | 2 | 3 |
| oil | 5.9022, 8.15716, 41.8467, 11.1497, 7.6505 | 5.90191, 8.29461, 21.2064, 10.8512, 7.54468 | 27.9862 | 4 | 1 |
| technology | 2.66214, 4.71124, 5.73455, 9.20031, 5.93332 | 2.66953, 4.73915, 6.47195, 4.45872, 6.46054 | 12.1866 | 1 | 4 |
| energy | 4.0229, 4.3645, 10.9083, 9.9092, 6.85056 | 3.98194, 4.91703, 12.3713, 8.32891, 6.70027 | -0.676505 | 3 | 2 |
| financials | 3.05118, 6.04525, 6.95296, 4.82492, 4.64349 | 3.54767, 4.10732, 7.33012, 4.07805, 5.2525 | 4.71103 | 2 | 3 |
| utilities | 5.03071, 2.15189, 5.50919, 3.71181, 4.87583 | 2.58828, 2.62695, 5.51766, 3.77431, 5.06935 | 8.00244 | 1 | 4 |
| healthcare | 2.48737, 3.03327, 4.41473, 3.26889, 3.43671 | 2.48476, 2.98834, 4.31221, 3.22722, 3.47477 | 0.923496 | 4 | 1 |
| consumer_discretionary | 2.57133, 3.91202, 5.73752, 5.74173, 6.03921 | 2.55316, 3.96161, 6.1193, 5.06421, 6.05053 | 1.0541 | 2 | 3 |

### Naive benchmarks

All five strategies align on target/fold/sample date and actual target values. Trailing is the original recent-trailing-21-observation benchmark. Naive models remain evaluation benchmarks, not substitutes for callable Ridge research outputs.

| Target | R MAE | X MAE | Zero MAE | Training mean MAE | Trailing MAE | X beats all |
| --- | --- | --- | --- | --- | --- | --- |
| sp500 | 3.57513 | 3.70407 | 3.85698 | 3.6506 | 5.21811 | FAIL |
| nasdaq | 5.29865 | 4.7136 | 4.96364 | 4.68099 | 6.44054 | FAIL |
| ten_year_yield_bps | 19.7651 | 20.1668 | 18.9433 | 19.1454 | 25.81 | FAIL |
| dxy | 1.44616 | 1.46806 | 1.43048 | 1.4465 | 2.00525 | FAIL |
| gold | 2.98913 | 2.95323 | 2.92579 | 2.91239 | 4.39206 | FAIL |
| oil | 14.9413 | 10.7598 | 10.7612 | 10.8596 | 14.3115 | PASS |
| technology | 5.64831 | 4.95998 | 5.32803 | 4.92423 | 6.8403 | FAIL |
| energy | 7.2111 | 7.25988 | 7.27865 | 7.29397 | 10.3897 | PASS |
| financials | 5.10356 | 4.86313 | 4.74376 | 4.6594 | 7.22792 | FAIL |
| utilities | 4.25588 | 3.91531 | 3.98534 | 3.89628 | 6.05422 | FAIL |
| healthcare | 3.3282 | 3.29746 | 3.41064 | 3.31022 | 4.87778 | PASS |
| consumer_discretionary | 4.80036 | 4.74976 | 4.97754 | 4.80206 | 6.93139 | PASS |

### VIX regimes

The frozen support minimum is 20 observations. All 180 saved regime/strategy rows meet support; unsupported estimates are not substituted. Rows below retain target-specific units.

| Target | VIX regime | n | R MAE | X MAE | R RMSE | X RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| sp500 | low_vix_lt_20 | 182 | 3.02072 | 3.16842 | 4.51844 | 4.51688 |
| sp500 | medium_vix_20_to_30 | 91 | 4.24814 | 4.29941 | 5.30555 | 5.18354 |
| sp500 | high_vix_ge_30 | 27 | 5.04389 | 5.30822 | 6.95228 | 7.23991 |
| nasdaq | low_vix_lt_20 | 182 | 4.02742 | 3.74926 | 5.63704 | 5.20794 |
| nasdaq | medium_vix_20_to_30 | 91 | 6.70323 | 6.00836 | 8.45479 | 6.95833 |
| nasdaq | high_vix_ge_30 | 27 | 9.13377 | 6.85011 | 10.2155 | 8.20074 |
| ten_year_yield_bps | low_vix_lt_20 | 182 | 17.9711 | 18.1227 | 24.6513 | 24.6647 |
| ten_year_yield_bps | medium_vix_20_to_30 | 91 | 22.9889 | 24.1316 | 29.4649 | 31.4302 |
| ten_year_yield_bps | high_vix_ge_30 | 27 | 20.9924 | 20.5831 | 28.9643 | 28.6534 |
| dxy | low_vix_lt_20 | 182 | 1.2405 | 1.27009 | 1.60751 | 1.62727 |
| dxy | medium_vix_20_to_30 | 91 | 1.77056 | 1.78139 | 2.35264 | 2.35659 |
| dxy | high_vix_ge_30 | 27 | 1.7391 | 1.74653 | 2.12709 | 2.12227 |
| gold | low_vix_lt_20 | 182 | 2.78219 | 2.72594 | 3.51214 | 3.57366 |
| gold | medium_vix_20_to_30 | 91 | 3.30999 | 3.29213 | 4.05749 | 4.18177 |
| gold | high_vix_ge_30 | 27 | 3.30267 | 3.34318 | 4.44317 | 4.41642 |
| oil | low_vix_lt_20 | 182 | 7.81426 | 7.58335 | 10.4993 | 10.2374 |
| oil | medium_vix_20_to_30 | 91 | 22.7899 | 9.75941 | 33.727 | 12.9577 |
| oil | high_vix_ge_30 | 27 | 36.5295 | 35.5427 | 46.6135 | 58.4722 |
| technology | low_vix_lt_20 | 182 | 4.48364 | 4.14464 | 6.08879 | 5.62438 |
| technology | medium_vix_20_to_30 | 91 | 7.06633 | 6.23811 | 9.42139 | 7.44998 |
| technology | high_vix_ge_30 | 27 | 8.7198 | 6.14818 | 10.5249 | 7.54308 |
| energy | low_vix_lt_20 | 182 | 5.88289 | 5.82814 | 8.64148 | 8.75982 |
| energy | medium_vix_20_to_30 | 91 | 9.1375 | 8.68178 | 12.2898 | 11.292 |
| energy | high_vix_ge_30 | 27 | 9.67149 | 12.1186 | 15.2872 | 15.8869 |
| financials | low_vix_lt_20 | 182 | 4.34851 | 4.03989 | 6.23252 | 6.17256 |
| financials | medium_vix_20_to_30 | 91 | 6.10604 | 5.89885 | 7.63483 | 7.09811 |
| financials | high_vix_ge_30 | 27 | 6.81445 | 6.92164 | 9.78881 | 10.3241 |
| utilities | low_vix_lt_20 | 182 | 3.78338 | 3.20784 | 5.13066 | 4.40557 |
| utilities | medium_vix_20_to_30 | 91 | 4.66742 | 4.68448 | 6.10671 | 5.93054 |
| utilities | high_vix_ge_30 | 27 | 6.05385 | 6.09179 | 8.11485 | 8.2164 |
| healthcare | low_vix_lt_20 | 182 | 3.18338 | 3.15767 | 4.14911 | 4.06999 |
| healthcare | medium_vix_20_to_30 | 91 | 3.06499 | 3.09135 | 3.73189 | 3.7933 |
| healthcare | high_vix_ge_30 | 27 | 5.19145 | 4.93442 | 6.95429 | 6.66432 |
| consumer_discretionary | low_vix_lt_20 | 182 | 3.92203 | 3.83747 | 5.65094 | 5.56644 |
| consumer_discretionary | medium_vix_20_to_30 | 91 | 5.75676 | 5.57559 | 7.24856 | 6.94664 |
| consumer_discretionary | high_vix_ge_30 | 27 | 7.4975 | 8.11594 | 9.27403 | 9.66173 |

### Training-to-validation generalization

Saved five-fold ranges are reported without refitting. Positive validation-minus-training gaps indicate larger validation error. These diagnostics introduce no additional promotion gate.

| Target | Model | Training MAE range | Validation MAE range | Validation minus training MAE |
| --- | --- | --- | --- | --- |
| sp500 | R | 1.93823 to 2.97613 | 1.93361 to 4.89314 | -0.270455 to 2.66868 |
| sp500 | X | 1.63197 to 2.21391 | 1.96471 to 5.2507 | 0.126912 to 3.61873 |
| nasdaq | R | 2.12698 to 3.76354 | 2.44263 to 7.86802 | -0.334889 to 4.94398 |
| nasdaq | X | 1.0273 to 3.07113 | 2.37849 to 6.70756 | -0.191643 to 3.69449 |
| ten_year_yield_bps | R | 12.8627 to 14.8031 | 11.0902 to 33.8947 | -3.71291 to 19.5557 |
| ten_year_yield_bps | X | 11.3749 to 14.3355 | 11.2497 to 34.815 | -1.31891 to 23.4401 |
| dxy | R | 1.30089 to 1.49216 | 0.68957 to 2.19917 | -0.802591 to 0.898277 |
| dxy | X | 0.887172 to 1.45502 | 0.754686 to 2.23561 | -0.132486 to 1.009 |
| gold | R | 2.98136 to 3.34984 | 2.38129 to 3.43397 | -0.958802 to 0.300226 |
| gold | X | 1.56059 to 3.14873 | 2.27182 to 3.41696 | -0.0820831 to 0.711228 |
| oil | R | 4.85506 to 10.1861 | 5.9022 to 41.8467 | -2.53561 to 36.9916 |
| oil | X | 5.92895 to 10.1139 | 5.90191 to 21.2064 | -2.5692 to 14.4515 |
| technology | R | 2.35427 to 3.68054 | 2.66214 to 9.20031 | 0.0558753 to 6.4745 |
| technology | X | 1.21474 to 2.91059 | 2.66953 to 6.47195 | 0.0724823 to 3.59792 |
| energy | R | 3.11503 to 4.80896 | 4.0229 to 10.9083 | 0.379401 to 7.64187 |
| energy | X | 2.44627 to 4.19543 | 3.98194 to 12.3713 | 0.622858 to 8.53184 |
| financials | R | 2.15333 to 4.02928 | 3.05118 to 6.95296 | 0.0240472 to 3.89192 |
| financials | X | 2.54447 to 3.35507 | 3.54767 to 7.33012 | 1.0032 to 4.18712 |
| utilities | R | 2.43639 to 3.1216 | 2.15189 to 5.50919 | -0.813099 to 2.69669 |
| utilities | X | 2.04876 to 3.06415 | 2.58828 to 5.51766 | 0.383348 to 2.70843 |
| healthcare | R | 2.25709 to 3.07161 | 2.48737 to 4.41473 | -0.274634 to 1.64005 |
| healthcare | X | 2.09332 to 2.79068 | 2.48476 to 4.31221 | -0.265965 to 2.21889 |
| consumer_discretionary | R | 2.36093 to 3.68717 | 2.57133 to 6.03921 | -0.0214381 to 3.04662 |
| consumer_discretionary | X | 2.37675 to 3.5062 | 2.55316 to 6.1193 | 0.0218848 to 3.27387 |

## E. Scenario responsiveness

### Matched design

Each target/model has 120 checks: five folds × three real training states × eight single-shock sweeps. States are eligible-training rows nearest the 10th/50th/90th VIX percentiles, with first-occurrence ties. Identical ordered 58-feature batches are used across model families. The 45 state features and five release flags stay fixed; baseline uses eight zero shocks; scenario changes one shock; incremental=response minus baseline.

Grids contain up to five distinct observed training values, observed endpoints and observed zero, using deterministic support ranks. Unobserved zero is only the neutral baseline, not inserted into the supported grid. Fed-rate grids are sparse in folds 1–2 (0,25 bps) and fold 3 (-25,0,25 bps), explaining 13,824 detailed points rather than five points for every check.

Near-flat means response span <= 0.02 × training-target IQR. This is a preliminary numerical diagnostic, not statistical significance or economic materiality. Normalized spans are dimensionless. All saved IQR statuses are AVAILABLE; unavailable/zero/nonfinite handling remains explicit in source.

### Breadth, magnitude and stability

| Target | Model | Active shocks /8 | Non-flat /120 | Raw span range | Span/IQR range | Sign-reversal groups | Max turns | Max adjacent jump |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sp500 | R | 8 | 66 | 0.000219658 to 2.83335 | 5.15211e-05 to 0.676378 | 6 | 0 | 2.04896 |
| sp500 | X | 2 | 13 | 0 to 1.42239 | 0 to 0.336328 | 0 | 0 | 0.683014 |
| nasdaq | R | 8 | 54 | 6.52347e-05 to 7.54209 | 1.4971e-05 to 1.40501 | 9 | 0 | 5.40815 |
| nasdaq | X | 3 | 15 | 0 to 0.934126 | 0 to 0.218729 | 0 | 2 | 0.864425 |
| ten_year_yield_bps | R | 8 | 63 | 0.000814585 to 13.6253 | 3.54167e-05 to 0.592404 | 12 | 0 | 8.2856 |
| ten_year_yield_bps | X | 2 | 8 | 0 to 3.6476 | 0 to 0.158591 | 0 | 0 | 3.05839 |
| dxy | R | 7 | 21 | 1.3259e-05 to 0.153703 | 5.15504e-06 to 0.0715913 | 0 | 0 | 0.0946231 |
| dxy | X | 2 | 14 | 0 to 0.739694 | 0 to 0.28199 | 0 | 1 | 0.430003 |
| gold | R | 8 | 66 | 4.79333e-05 to 1.25288 | 9.57532e-06 to 0.251548 | 6 | 0 | 0.770314 |
| gold | X | 3 | 8 | 0 to 0.295135 | 0 to 0.0514217 | 0 | 1 | 0.198704 |
| oil | R | 8 | 72 | 3.16518e-05 to 44.9164 | 2.6837e-06 to 3.5966 | 9 | 0 | 27.3918 |
| oil | X | 3 | 16 | 0 to 16.2409 | 0 to 1.30046 | 0 | 0 | 15.1718 |
| technology | R | 8 | 78 | 9.53532e-05 to 9.91932 | 2.31658e-05 to 1.8226 | 3 | 0 | 7.43915 |
| technology | X | 3 | 20 | 0 to 1.57745 | 0 to 0.282375 | 0 | 2 | 1.06744 |
| energy | R | 8 | 114 | 0.085556 to 22.0098 | 0.0111862 to 2.70855 | 12 | 0 | 18.1361 |
| energy | X | 3 | 26 | 0 to 3.74208 | 0 to 0.460504 | 0 | 2 | 2.33727 |
| financials | R | 8 | 72 | 0.000589492 to 5.5432 | 0.000111861 to 1.02347 | 6 | 0 | 4.56759 |
| financials | X | 1 | 6 | 0 to 2.87736 | 0 to 0.531263 | 0 | 1 | 1.55759 |
| utilities | R | 8 | 72 | 7.27429e-05 to 3.77499 | 1.49837e-05 to 0.687194 | 18 | 0 | 3.11059 |
| utilities | X | 2 | 12 | 0 to 0.589295 | 0 to 0.111033 | 1 | 0 | 0.589295 |
| healthcare | R | 8 | 30 | 2.1381e-05 to 4.37078 | 4.46214e-06 to 0.968149 | 0 | 0 | 3.32271 |
| healthcare | X | 2 | 11 | 0 to 0.639617 | 0 to 0.133486 | 0 | 0 | 0.45059 |
| consumer_discretionary | R | 8 | 60 | 3.91383e-05 to 3.3283 | 1.07425e-05 to 0.705599 | 6 | 0 | 2.57509 |
| consumer_discretionary | X | 1 | 3 | 0 to 0.525797 | 0 to 0.111469 | 0 | 0 | 0.340702 |

An active shock exceeds the near-flat diagnostic in at least one state/fold. A sign-reversal group has opposite non-flat endpoint signs across states or folds; groups overlap and are not independent failures. Ridge has such flags for 10 targets; XGBoost has one for Utilities. Changing sign or being piecewise-constant/nonmonotonic is permitted and is not by itself pathological or a causal-sign violation.

XGBoost non-flat responses occur only for DXY, Oil and VIX somewhere in the experiment. No five-macro-shock check exceeds the near-flat threshold. This materially limits macroeconomic what-if usefulness even when Gate 6's minimum passes. Flat predictions do not imply no economic effect.

### VIX-state differences

| Target | Model | State | Non-flat /40 | Max raw span | Max span/IQR |
| --- | --- | --- | --- | --- | --- |
| sp500 | R | low_vix | 22 | 2.83335 | 0.676378 |
| sp500 | R | median_vix | 22 | 2.83335 | 0.676378 |
| sp500 | R | high_vix | 22 | 2.83335 | 0.676378 |
| sp500 | X | low_vix | 4 | 1.40888 | 0.336328 |
| sp500 | X | median_vix | 5 | 1.42239 | 0.336328 |
| sp500 | X | high_vix | 4 | 1.40888 | 0.336328 |
| nasdaq | R | low_vix | 18 | 7.54209 | 1.40501 |
| nasdaq | R | median_vix | 18 | 7.54209 | 1.40501 |
| nasdaq | R | high_vix | 18 | 7.54209 | 1.40501 |
| nasdaq | X | low_vix | 5 | 0.781146 | 0.155834 |
| nasdaq | X | median_vix | 5 | 0.781053 | 0.151329 |
| nasdaq | X | high_vix | 5 | 0.934126 | 0.218729 |
| ten_year_yield_bps | R | low_vix | 21 | 13.6253 | 0.592404 |
| ten_year_yield_bps | R | median_vix | 21 | 13.6253 | 0.592404 |
| ten_year_yield_bps | R | high_vix | 21 | 13.6253 | 0.592404 |
| ten_year_yield_bps | X | low_vix | 3 | 2.15461 | 0.0936787 |
| ten_year_yield_bps | X | median_vix | 2 | 3.6476 | 0.158591 |
| ten_year_yield_bps | X | high_vix | 3 | 3.22607 | 0.140264 |
| dxy | R | low_vix | 7 | 0.153703 | 0.0715913 |
| dxy | R | median_vix | 7 | 0.153703 | 0.0715913 |
| dxy | R | high_vix | 7 | 0.153703 | 0.0715913 |
| dxy | X | low_vix | 5 | 0.739694 | 0.28199 |
| dxy | X | median_vix | 5 | 0.62423 | 0.27159 |
| dxy | X | high_vix | 4 | 0.67673 | 0.27159 |
| gold | R | low_vix | 22 | 1.25288 | 0.251548 |
| gold | R | median_vix | 22 | 1.25288 | 0.251548 |
| gold | R | high_vix | 22 | 1.25288 | 0.251548 |
| gold | X | low_vix | 2 | 0.263343 | 0.0458826 |
| gold | X | median_vix | 3 | 0.28479 | 0.0496194 |
| gold | X | high_vix | 3 | 0.295135 | 0.0514217 |
| oil | R | low_vix | 24 | 44.9164 | 3.5966 |
| oil | R | median_vix | 24 | 44.9164 | 3.5966 |
| oil | R | high_vix | 24 | 44.9164 | 3.5966 |
| oil | X | low_vix | 5 | 16.2409 | 1.30046 |
| oil | X | median_vix | 6 | 16.2409 | 1.30046 |
| oil | X | high_vix | 5 | 16.2409 | 1.30046 |
| technology | R | low_vix | 26 | 9.91932 | 1.8226 |
| technology | R | median_vix | 26 | 9.91932 | 1.8226 |
| technology | R | high_vix | 26 | 9.91932 | 1.8226 |
| technology | X | low_vix | 6 | 1.57745 | 0.282375 |
| technology | X | median_vix | 7 | 1.53245 | 0.27432 |
| technology | X | high_vix | 7 | 1.33539 | 0.239044 |
| energy | R | low_vix | 38 | 22.0098 | 2.70855 |
| energy | R | median_vix | 38 | 22.0098 | 2.70855 |
| energy | R | high_vix | 38 | 22.0098 | 2.70855 |
| energy | X | low_vix | 9 | 1.21892 | 0.181665 |
| energy | X | median_vix | 9 | 2.13809 | 0.254315 |
| energy | X | high_vix | 8 | 3.74208 | 0.460504 |
| financials | R | low_vix | 24 | 5.5432 | 1.02347 |
| financials | R | median_vix | 24 | 5.5432 | 1.02347 |
| financials | R | high_vix | 24 | 5.5432 | 1.02347 |
| financials | X | low_vix | 2 | 2.87736 | 0.531263 |
| financials | X | median_vix | 2 | 2.87736 | 0.531263 |
| financials | X | high_vix | 2 | 2.87736 | 0.531263 |
| utilities | R | low_vix | 24 | 3.77499 | 0.687194 |
| utilities | R | median_vix | 24 | 3.77499 | 0.687194 |
| utilities | R | high_vix | 24 | 3.77499 | 0.687194 |
| utilities | X | low_vix | 4 | 0.589294 | 0.111033 |
| utilities | X | median_vix | 3 | 0.589294 | 0.111033 |
| utilities | X | high_vix | 5 | 0.589295 | 0.111033 |
| healthcare | R | low_vix | 10 | 4.37078 | 0.968149 |
| healthcare | R | median_vix | 10 | 4.37078 | 0.968149 |
| healthcare | R | high_vix | 10 | 4.37078 | 0.968149 |
| healthcare | X | low_vix | 3 | 0.366293 | 0.076444 |
| healthcare | X | median_vix | 4 | 0.639617 | 0.133486 |
| healthcare | X | high_vix | 4 | 0.316783 | 0.0656078 |
| consumer_discretionary | R | low_vix | 20 | 3.3283 | 0.705599 |
| consumer_discretionary | R | median_vix | 20 | 3.3283 | 0.705599 |
| consumer_discretionary | R | high_vix | 20 | 3.3283 | 0.705599 |
| consumer_discretionary | X | low_vix | 1 | 0.525797 | 0.111469 |
| consumer_discretionary | X | median_vix | 1 | 0.525797 | 0.111469 |
| consumer_discretionary | X | high_vix | 1 | 0.525797 | 0.111469 |

All 2,880 checks are finite, repeatable and exactly neutral-consistent. Suspicious-adjacent-jump and excessive-span flags are zero under the existing Step 13 envelope max(training-target range, 10 return pp or 50 yield bps), with adjacent jumps checked against half that envelope. This does not assert smoothness or certify economic validity.

### Oil detailed promotion assessment

The following scalars preserve saved precision, rather than the rounded Task 5 chat summary. Comparisons use the same 300 observations.

| Calculation | Saved or derived value | Verdict |
| --- | --- | --- |
| Ridge pooled RMSE | 24.6470228118 | reference |
| XGB pooled RMSE | 20.5480375073 | reference |
| XGB/Ridge RMSE (derived) | 0.83369247735115613100566278756123 | PASS |
| Permitted XGB RMSE = 1.02 × Ridge (derived) | 25.139963268036 | PASS |
| Worst Ridge MAE | 41.8466597308 | reference |
| Worst XGB MAE | 21.2063865239 | reference |
| XGB/Ridge worst MAE (derived) | 0.50676413984583014382742791702716 | PASS |
| Permitted XGB worst MAE = 1.10 × Ridge (derived) | 46.031325703880 | PASS |

| Fold | Ridge MAE | XGB MAE | XGB strictly lower |
| --- | --- | --- | --- |
| 1 | 5.90219526335 | 5.90191071144 | yes |
| 2 | 8.15715720459 | 8.29460536775 | no |
| 3 | 41.8466597308 | 21.2063865239 | yes |
| 4 | 11.1497389916 | 10.8512311857 | yes |
| 5 | 7.65049983711 | 7.54468351336 | yes |

| Oil strategy | Pooled MAE | XGB beats benchmark |
| --- | --- | --- |
| Ridge | 14.9412502055 | reference |
| XGBoost | 10.7597634604 | reference |
| naive_zero | 10.7611959793 | yes |
| naive_training_mean | 10.8595572106 | yes |
| naive_recent_trailing_21_observation | 14.311518638 | yes |

Derived zero-benchmark advantage is only 0.0014325189 return pp (0.013311893053110099119036494806345% relative). Fold 1 is won by only 0.00028455191 return pp. The largest improvement is in fold 3. The strict inequality passes; no new minimum margin or statistical-significance gate is invented. Narrow margins and regime concentration limit confidence.

| Oil XGB shock | Non-flat /15 | Raw span range (return pp) | Span/IQR range | Max adjacent jump | Max turns |
| --- | --- | --- | --- | --- | --- |
| cpi_surprise_pct | 0 | 0 to 0 | 0 to 0 | 0 | 0 |
| dxy_change_pct | 1 | 0 to 0.346312522888 | 0 to 0.0311017316814 | 0.346312522888 | 0 |
| fed_funds_rate_change_bps | 0 | 0 to 0 | 0 to 0 | 0 | 0 |
| gdp_growth_surprise_pct | 0 | 0 to 0 | 0 to 0 | 0 | 0 |
| oil_price_change_pct | 12 | 0 to 16.2408661842 | 0 to 1.30046011159 | 15.1718034148 | 0 |
| pmi_change_points | 0 | 0 to 0 | 0 to 0 | 0 | 0 |
| unemployment_change_pct | 0 | 0 to 0 | 0 to 0 | 0 | 0 |
| vix_change_points | 3 | 0 to 0.265954971313 | 0 to 0.0212958981219 | 0.265954971313 | 0 |

Oil has 16 non-flat and 104 near-flat checks. Own-oil responses are non-flat in 12/15 cases, DXY in 1/15 and VIX in 3/15. Five macro-shock responses are exactly flat in these tested grids. There are no non-flat sign-reversal groups, within-grid direction changes, suspicious jumps or invalid/nonrepeatable outputs. The maximum own-oil adjacent jump is sizable but within the frozen warning envelope; it is disclosed without inventing a new failure threshold.

Oil Gate 6 is PASS as a research judgment: finite/deterministic outputs, exact neutral isolation, non-negligible response to some supported shocks, and no unresolved pathological sensitivity under the approved diagnostics. This is narrow scenario suitability, not evidence that all eight inputs are learned well. Weak macro coverage remains a prominent future-bundle warning. Oil recommendation: SELECT_XGBOOST with weak-evidence experimental confidence, not production or causal approval.

## F. Promotion-gate matrix

G1: >=5% pooled MAE improvement. G2: >=3/5 strict MAE fold wins. G3: XGB RMSE <=1.02×Ridge. G4: XGB worst MAE <=1.10×Ridge. G5: strictly lower MAE than each aligned naive benchmark. G6: finite/reproducible/neutral-consistent outputs, some supported non-negligible response, and no unresolved pathological sensitivity. No new numeric promotion threshold or causal monotonicity rule is introduced.

| Target | Unit | R MAE | X MAE | MAE gain % | X wins | G1 | G2 | G3 | G4 | G5 | G6 | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sp500 | percent | 3.57513 | 3.70407 | -3.60673 | 1 | FAIL | FAIL | PASS | PASS | FAIL | PASS | RETAIN_RIDGE |
| nasdaq | percent | 5.29865 | 4.7136 | 11.0416 | 2 | PASS | FAIL | PASS | PASS | FAIL | PASS | RETAIN_RIDGE |
| ten_year_yield_bps | basis_points | 19.7651 | 20.1668 | -2.03259 | 2 | FAIL | FAIL | FAIL | PASS | FAIL | PASS | RETAIN_RIDGE |
| dxy | percent | 1.44616 | 1.46806 | -1.51472 | 1 | FAIL | FAIL | PASS | PASS | FAIL | PASS | RETAIN_RIDGE |
| gold | percent | 2.98913 | 2.95323 | 1.20108 | 2 | FAIL | FAIL | PASS | PASS | FAIL | PASS | RETAIN_RIDGE |
| oil | percent | 14.9413 | 10.7598 | 27.9862 | 4 | PASS | PASS | PASS | PASS | PASS | PASS | SELECT_XGBOOST |
| technology | percent | 5.64831 | 4.95998 | 12.1866 | 1 | PASS | FAIL | PASS | PASS | FAIL | PASS | RETAIN_RIDGE |
| energy | percent | 7.2111 | 7.25988 | -0.676505 | 3 | FAIL | PASS | PASS | FAIL | PASS | PASS | RETAIN_RIDGE |
| financials | percent | 5.10356 | 4.86313 | 4.71103 | 2 | FAIL | FAIL | PASS | PASS | FAIL | PASS | RETAIN_RIDGE |
| utilities | percent | 4.25588 | 3.91531 | 8.00244 | 1 | PASS | FAIL | PASS | PASS | FAIL | PASS | RETAIN_RIDGE |
| healthcare | percent | 3.3282 | 3.29746 | 0.923496 | 4 | FAIL | PASS | PASS | PASS | PASS | PASS | RETAIN_RIDGE |
| consumer_discretionary | percent | 4.80036 | 4.74976 | 1.0541 | 2 | FAIL | FAIL | PASS | PASS | PASS | PASS | RETAIN_RIDGE |

G6 PASS is the frozen minimum, not broad macro coverage or high confidence. Utilities' bounded state/fold sign change is disclosed but not automatically pathological. No missing evidence requires a NEEDS_REVIEW status. Eleven decisions follow clear failed mandatory gates rather than ambiguously missing results, so RETAIN_RIDGE is used instead of INCONCLUSIVE_RETAIN_RIDGE.

- `sp500` — RETAIN_RIDGE; failed gates 1, 2, 5. Worse pooled MAE, one win and a training-mean loss outweigh the marginal RMSE improvement.

- `nasdaq` — RETAIN_RIDGE; failed gates 2, 5. Meaningful pooled improvement does not overcome two fold wins and a training-mean loss.

- `ten_year_yield_bps` — RETAIN_RIDGE; failed gates 1, 2, 3, 5. Worse MAE and a RMSE ratio above 102%, plus failed naive and fold-win gates.

- `dxy` — RETAIN_RIDGE; failed gates 1, 2, 5. Worse MAE/RMSE, one win, losses to zero and training mean.

- `gold` — RETAIN_RIDGE; failed gates 1, 2, 5. MAE gain below 5%, two wins, naive losses; RMSE worsens despite passing its 102% allowance.

- `technology` — RETAIN_RIDGE; failed gates 2, 5. Pooled improvement does not overcome one fold win and a training-mean loss.

- `energy` — RETAIN_RIDGE; failed gates 1, 4. Three wins coexist with worse pooled MAE and a worst-fold ratio above 110%.

- `financials` — RETAIN_RIDGE; failed gates 1, 2, 5. The 4.711% gain misses 5%; two wins and naive losses also fail.

- `utilities` — RETAIN_RIDGE; failed gates 2, 5. Pooled improvement coexists with one win, a training-mean loss and a disclosed response-sign warning.

- `healthcare` — RETAIN_RIDGE; failed gates 1. Four fold wins and naive wins do not overcome improvement below 5%.

- `consumer_discretionary` — RETAIN_RIDGE; failed gates 1, 2. Small gain and naive wins do not meet the 5% or three-fold requirements.

Retaining Ridge preserves the current research family, not a strong-forecasting endorsement. Original weak-evidence/confidence warnings remain. No global family is selected; each target is assessed independently.

## G. Research limitations

- Retrospective purged walk-forward evidence with prior researcher exposure to some validation periods is not a completely untouched independent test.
- Unique validation dates do not imply independent observations: 21-session target windows overlap within and across folds. No significance claim follows from pooled errors or fold counts.
- Weak absolute performance, negative/small R², near-constant tree predictions, benchmark losses and disagreement among MAE/RMSE/direction metrics remain visible.
- Oil's advantage over zero and its first-fold win are tiny. Much of the gain over Ridge is concentrated in the high-error third fold.
- No locked final-test targets are evaluated; no final-test confirmation exists for these recommendations.
- PMI Change is the Kansas City Fed manufacturing proxy, not national ISM PMI. Earlier public-source, proxy and point-in-time decisions are preserved rather than reopened.
- The authorized current-input fingerprint is verified, but historical complete-input byte identity is incomplete. Untracked source hashes must accompany the recorded commit.
- Three representative training states and single-shock supported grids do not certify joint-shock interactions, extrapolation or live-state readiness.
- Flat macro-shock outputs materially limit XGBoost's macroeconomic what-if utility and must not be interpreted as evidence of no real economic effect.
- Responses are predictive associations, not identified causal effects; no policy-impact, investment-performance or production-readiness claim is warranted.
- Gate 6 and SELECT_XGBOOST do not upgrade forecasting confidence. RETAIN_RIDGE also does not upgrade original Ridge evidence.

## H. Research recommendation and final release decision

### Preserved research recommendation

Under the frozen research selection criteria, XGBoost was selected for Oil and Ridge was retained for the other 11 targets. A separately versioned mixed experimental bundle was therefore a technically valid research recommendation. Any such implementation would need to preserve all eight shocks, 12 targets, the 21-session horizon, baseline/state/release-flag isolation, units, confidence warnings and PMI proxy disclosure. Oil warnings would need to retain its tiny zero-baseline margin, low R², regime concentration and flat macro responses.

### Final release decision

The mixed Ridge/XGBoost release is deferred. The decision reflects the limited absolute predictive improvement over simple benchmarks, XGBoost's limited scenario responsiveness outside a small subset of inputs, and project timeline constraints. Ridge v1.0.0 remains the active released model for all 12 targets.

This final release decision does not alter the research gate results or claim that Ridge outperformed XGBoost for Oil under those gates. It only determines which verified bundle remains active. No new version, refit hyperparameter, training/export task, loader or API change is authorized here. Historical fold models remain research evidence rather than an exported application artifact.

## I. Evidence references and preservation

Actual saved Task 5 files and checksums:

| Existing path | SHA-256 |
| --- | --- |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_authoritative_predictions.csv` | `69dbbcad8b8ced0f06e72f540f4d30b7cfc1adc99efd9a28e02261f7366e99ad` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_candidate_audit.csv` | `ab06b6d1cc5c79532550ad6e490fea55b67584dab6b9fe1fc4a1b5a9fa0673c2` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_family_comparison.csv` | `5c39ef157a3b95b7a4cb2669e16c7451682f53151ddad0bf304fe4bdd8d53f91` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_fold_manifest.csv` | `b357f963e5c75dc250be1e3b08f9a202d5c4051ba4fa1aef06322e7d2eded472` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_fold_metrics.csv` | `5b88a8f28b426e34a3911a4de31b673d0e56d789b6d7cc3bd3d7ae3e548f1a72` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_generalization.csv` | `1b75540a5a8a50969e0e74e5a6443acfb070b61c315029db28323b73a4290ff7` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_naive_comparison.csv` | `fe0edfd3ed02be18cf6be5756f0ad3ae5c97a8e083511e4b131f0e13c9816897` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_pooled_metrics.csv` | `7c0c2cb490ebbff528a8daeaea430455c8733d685945f24db933689b1d7afd53` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_predictions.csv` | `31a60a552da1f0dbe26711ffb19f87818b3e5fff4565bd1d32362e6a4591db30` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_regime_metrics.csv` | `8fd08d1b045091f953daef56ffa67c515269417b0aab715f769d512d89ea2190` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_ridge_fit_audit.csv` | `4d5c415f59e2a775330056c0830ba2a72b45a92d3db64044b567c97100903710` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_ridge_reconstruction_audit.csv` | `6299db2282be8e0e055622d1a7f3d08415d19556a936d9c90de2551cd164020f` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_sensitivity_detail.csv` | `4c25497ab5b726303633c0b7450b09f3aaee0854cb17c47f78d46ac91246fd0b` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_sensitivity_flags.csv` | `3dd068783433942441865ced40bf18b0f69fefd428b991f8f12ba4c5444a035d` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_sensitivity_summary.csv` | `ab5886bec6ab8221dcb9e6d3ab73aade76a500e380d14acc446be559ad036617` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_summary.json` | `ddafb3020a3ccbba2a15d0c5040acac3f4385ba4dbbc1d24532b94355a08de24` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_xgboost_fit_audit.csv` | `cba14063f3c5104ea2ebd48d5e664d610a4fd6f04bdcc11bcb09e81d00487afb` |
| `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_xgboost_predictions.csv` | `970133690035fadaadbf0ec563ec60c2dd7b1f99b66a65864de6a06a875d2eee` |

The checksum index is `data/scenario_playground/interim/models/all_target_xgboost_comparison/comparison_checksums.json`. Its digest appears in C. Saved table counts:

| Table | Rows |
| --- | --- |
| authoritative_predictions | 14400 |
| candidate_audit | 180 |
| family_comparison | 12 |
| fold_manifest | 5 |
| fold_metrics | 300 |
| generalization | 120 |
| naive_comparison | 72 |
| pooled_metrics | 60 |
| predictions | 18000 |
| regime_metrics | 180 |
| ridge_fit_audit | 60 |
| ridge_reconstruction_audit | 60 |
| sensitivity_detail | 13824 |
| sensitivity_flags | 24 |
| sensitivity_summary | 2880 |
| xgboost_fit_audit | 60 |
| xgboost_predictions | 3600 |

Relevant finalized documentation:

- `markdown_files/scenario_modeling/01_MODEL_CONTRACT.md`
- `markdown_files/scenario_modeling/02_DATA_SOURCE_INVENTORY.md`
- `markdown_files/scenario_modeling/09_LEAKAGE_AUDIT_REPORT.md`
- `markdown_files/scenario_modeling/11_XGBOOST_PROTOTYPE_REPORT.md`
- `markdown_files/scenario_modeling/12_WALK_FORWARD_VALIDATION_REPORT.md`
- `markdown_files/scenario_modeling/13_ALL_TARGETS_REPORT.md`
- `markdown_files/scenario_modeling/14_SCENARIO_INFERENCE_REPORT.md`
- `markdown_files/scenario_modeling/15_EXPLANATIONS_REPORT.md`
- `markdown_files/scenario_modeling/16_CONFIDENCE_DIAGNOSTICS_REPORT.md`
- `markdown_files/scenario_modeling/17_ARTIFACT_EXPORT_REPORT.md`

Relevant source modules:

- `scenario_modeling/src/all_target_xgboost_comparison.py`
- `scenario_modeling/src/matched_scenario_sensitivity.py`
- `scenario_modeling/src/walk_forward_validation.py`
- `scenario_modeling/src/all_target_models.py`
- `scenario_modeling/src/ridge_baselines.py`
- `scenario_modeling/src/evaluation.py`
- `scenario_modeling/src/xgboost_prototype.py`
- `scenario_modeling/src/scenario_runtime.py`

Original evidence directories: `data/scenario_playground/interim/models/xgboost_prototype/`, `data/scenario_playground/interim/models/walk_forward/`, `data/scenario_playground/interim/models/all_targets/`. Existing bundle: `scenario_modeling/artifacts/scenario_playground_ridge/v1.0.0/`.

Task 6 creates only this Markdown report. Task 5 results, original sources/evidence, dataset definitions, folds, Ridge runtime/bundle/release assets, backend/frontend and Git staging/branch state remain unchanged. No training, final-test-target read, new artifact export, deployment, dependency installation or network call is performed.

Final status: `TASK_6_COMPLETE`.
