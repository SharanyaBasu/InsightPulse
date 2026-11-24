# Scenario Modeling Documentation

This folder contains the decisions, reports, and implementation context produced while following the corresponding instruction files in `model_implementation/`. The numbered reports are chronological records: later reports and the final release decision below supersede future-work statements in earlier reports without rewriting the historical evidence.

## Current release decision

The finalized Ridge-versus-XGBoost research selected XGBoost for Oil under the frozen research criteria and retained Ridge for the other 11 targets. The mixed-model release was subsequently deferred because its absolute predictive improvement was limited, its scenario responsiveness was narrow, and the project timeline favored preserving the verified integration path. Ridge v1.0.0 remains the active released model for all 12 targets.

- `SCENARIO_PLAYGROUND_DEMO_GUIDE.md` - concise, beginner-friendly explanation of the model goal, inputs, features, data, training, outputs, interpretation, limitations, and common demo questions.
- `01_MODEL_CONTRACT.md` - Step 1 model contract.
- `02_DATA_SOURCE_INVENTORY.md` - Step 2 source inventory and source decisions.
- `03_RAW_DATA_DOWNLOAD_REPORT.md` - generated Step 3 dataset, coverage, metadata, and failure report.
- `03_STEP_3_PROBLEMS_AND_RESOLUTIONS.md` - high-level explanation of the problems encountered during Step 3 and how they were resolved.
- `04_SAMPLING_DESIGN_REPORT.md` - Step 4 event-aware sampling decision and timing rules.
- `05_MARKET_TARGETS_REPORT.md` - Step 5 construction and validation of all twelve 21-session targets.
- `06_SCENARIO_SHOCKS_REPORT.md` - Step 6 historical scenario-shock definitions, outputs, and point-in-time checks.
- `07_STATE_FEATURES_REPORT.md` - Step 7 macro, market, risk, and trend state-feature construction.
- `08_POINT_IN_TIME_MATRIX_REPORT.md` - Step 8 raw model-matrix assembly, role separation, missing-row policy, and chronological split manifest.
- `09_LEAKAGE_AUDIT_REPORT.md` - Step 9 exploratory review, leakage checklist, oil-volatility correction, and Ridge training gate.
- `10_RIDGE_BASELINES_REPORT.md` - Step 10 train-only naive and Ridge baselines, validation predictions, metrics, and final-test lock evidence.
- `11_XGBOOST_PROTOTYPE_REPORT.md` - Step 11 conservative S&P 500 XGBoost candidate comparison, diagnostics, sensitivity findings, and walk-forward gate.
- `12_WALK_FORWARD_VALIDATION_REPORT.md` - Step 12 purged expanding-window comparison, history-window decision, scenario-sensitivity audit, and approved Ridge methodology.
- `13_ALL_TARGETS_REPORT.md` - Step 13 all-target Ridge evaluation, unchanged validation evidence, and the corrected decision that confidence classifications do not suppress any of the 12 Ridge models from research inference.
- `14_SCENARIO_INFERENCE_REPORT.md` - completed corrected Step 14 callable research inference with 12 independent Ridge pipelines, numeric all-target baseline/scenario/incremental outputs, unchanged confidence evidence, and benchmark-only naive comparisons.
- `15_EXPLANATIONS_REPORT.md` - Step 15 exact standardized Ridge contribution decomposition, deterministic drivers, all-target shock sensitivity sweeps, reconciliation evidence, and frozen explanation schema.
- `16_CONFIDENCE_DIAGNOSTICS_REPORT.md` - Step 16 non-compensatory input-range, state-distance, walk-forward, regime, residual, freshness, confidence, warning, and postponed-XGBoost diagnostics with the frozen Step 17 metadata schema.
- `17_ARTIFACT_EXPORT_REPORT.md` - Step 17 versioned joblib bundle, lean offline runtime, SHA-256 manifest, fresh-process no-retraining verification, notebook execution, and team handoff evidence.
- `18_RIDGE_XGBOOST_ALL_TARGET_COMPARISON_REPORT.md` - finalized all-target Ridge-versus-XGBoost research comparison, frozen promotion-gate results, matched scenario-responsiveness evidence, and the subsequent decision to defer a mixed-model release.
