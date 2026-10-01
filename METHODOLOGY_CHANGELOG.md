# Methodology Changelog

## 2026-09-25 to 2026-09-26: Original Audit and Reproduction

**Methodology version: ORIGINAL_UNCHANGED. No statistical methodology change made.**

### Preserved

- Original source ZIP, research files and saved outputs, checked with SHA-256.
- Original positive-fee and league eligibility filters, highest-fee/maximum-playing-time deduplication, name matching threshold and two exceptions.
- Original prior/same-season mapping, age, contract, league, COVID, per-90 and target formulas.
- Original row-random train/test split, seed42, five-fold CV, missing-data treatment and encoders.
- Original candidate-only VIF loop, backward AIC/BIC selection, OLS/LinearRegression, RF and XGBoost parameters.
- Original residual definitions, pooled summaries, conditional age grid and season-within-random-test calculations.

### Execution Adaptations Only

- Redirected source reads and generated writes to preserved snapshot and isolated reproduction directories.
- Extracted Python chunks mechanically from original R Markdown in their original order.
- Used the preserved combined FBref table as one restart point; separately replayed original live cache requests and merge.
- Exported original in-memory match tables, stage counts, source provenance, design matrices and original report computations for audit evidence.
- Resolved OpenMP loading using an existing local library; configured local thread limits and temporary/cache directories.
- Fixed knitr runner working-directory handling without altering analytical expressions.
- Captured plots/tables/Markdown rather than recompiling a full report PDF; preserved all original PDFs.
- Compared saved and regenerated data/results; computed requested full-design rank/VIF diagnostics without using them to alter any fitted model.

### Findings Not Yet Implemented as Corrections

- Ast_per90 and Succ_Take_per90 were removed, not npxG.
- Negative 2018-19 actual-minus-predicted residual means overprediction.
- Full design has44 columns including intercept and rank41; candidate VIFs do not assess full-design dependencies.
- Model4 preprocessing/selection precede CV; validation is non-nested.
- Same-season proxies, contract timing, snapshot league labels, repeated names and unequal Model1/model2-4 samples limit interpretation.

### Scope Stop

No corrected dataset, revised model, new position analysis, new residual-premium test, or Sloan writing has been started. Any subsequent methodological change requires an explicit new version with its rationale, inputs, validation plan and results, rather than replacing this original baseline.

`RESULTS_MANIFEST.csv` records artifact provenance, file hashes and verification status. It is an artifact manifest, not evidence that every report claim is correct. Its own recursive hash and volatile temporary/OS files are deliberately excluded.

## Phase 2: DATA_AND_DESIGN_ONLY


This section supersedes the Phase 1 scope-stop statement for data/design work only. All original sources, reproduced models and Phase 1 evidence remain unchanged.

- Completed row-level timing, snapshot-contract, all-212 fuzzy identity and historical club-season evidence audits.
- Preserved 1,407-row original-comparable values; created 1,195-row strict-timing, no-contract, 1,194-row identity-screened, and 946-row league-corroborated variants.
- Recommended the last variant as a conservative restricted-sample primary, Variant 4 as broader-coverage sensitivity. League evidence restriction is not proof of misclassification or a complete historical reconstruction.
- Removed contract columns from primary for unverified historical timing, not fit improvement. No rows lost solely to column removal.
- Removed three deterministic predictors from future structural designs; verified full rank and reported full-design VIF, including high polynomial-age VIF.
- Preserved every exclusion in a row ledger. No original performance values, fees, identity values or league labels silently repaired.
- Future fold-specific preprocessing, nested selection and temporal/player-ID validation remain unperformed. No final models, residual tests, position models or abstract. Original performance metrics cannot be attributed to corrected variants.

See 02_data/final/PRIMARY_DATASET_RECOMMENDATION.md and 01_audit/*_AUDIT.md Phase 2 reports. RESULTS_MANIFEST.csv labels new artifacts separately.

## Phase 3: CORRECTED_MODELING_VALIDATION


- Replaced the two-sided-eligibility primary recommendation with destination-confirmed n=1,115; V4 n=1,194 and V5 n=946 remain sensitivities. Original source values were preserved.
- Primary STRUCTURAL_A excludes uncertain origin predictors and all contracts; STRUCTURAL_B adds corroborated origin fixed effects only in matched V5 contrasts. Fixed hand-ranked league_level_diff is excluded.
- Fixed age centering at 25, empirically verified equivalent fitted values. Inner validation chose quadratic-only AGE_SPEC_1; no holdout outcomes selected age.
- Preserved original traditional/advanced candidate definitions; all four models share 892/223 holdout IDs. Fold-local imputation, encoding, scaling, rank handling, VIF/AIC/BIC selection replace globally preprocessed/selected CV.
- Completed nested development-only CV, all-player GroupKFold, actual-date-checked expanding-window forecasts, dataset and historically unverified contract sensitivities. Unseen future season uses training-weighted average effect, explicitly documented.
- Sensitivity age is fixed quadratic-only, not chosen from a dataset's evaluation outcomes. Contract comparison holds both IDs and training-selected performance variables constant.
- Model 4 corrected holdout R2=0.210239; nested-CV mean=0.402793; player-grouped mean=0.313596; pooled temporal=0.165907. No switch to a more favorable sample or specification after seeing results.
- Generated model artifacts, observation-level predictions, split/selection logs, reports and QA. No position-specific importance, market-premium interpretation or Sloan abstract. STOP.

## Phase 4: PER90_EXPOSURE_CORRECTION


- Independently verified the Neco Williams/Pablo Sarabia saved rates and predictions. They account for 51.056697% of 2022-23 Model4 squared error. Exact cached minutes are 8 and 22, distinct from rounded 90s times 90.
- User-prespecified primary cutoff recorded 90s >=3 before revised fits: 1,059 rows; sensitivities >=5:1,029 and >=10:908. All exclusions use exposure only; no rate edits or named-player exceptions.
- Imported the unchanged Phase3 estimator code. Fixed STRUCTURAL_A and AGE_SPEC_1; no contracts/origin predictors. Repeated common holdout, development-only fold-local selection CV, player-grouped and chronological validation under each threshold.
- MIN270 Model4 holdout R2=0.413790; nested mean=0.433762; grouped mean=0.428356; pooled temporal=0.434510. 2022-23 changes from -0.315284 (241 rows) to 0.413950 (228 rows). Old predictions already score 0.424164 on the retained 228, separating evaluation composition from refitting.
- Threshold fixed before revised results but motivated by prior OOS inspection: diagnostic correction, not a fresh confirmatory test. No threshold shopping; 450/900 remain sensitivities regardless of scores.
- No winsorization, additional error-based deletions, empirical Bayes or new model family. Prior Phase1-3 files preserved; Phase3 performance is historical and superseded for the new eligible performance-analysis population, not overwritten.
- Reports, error concentration, feature/exposure distributions, ledgers, fitted artifacts, OOF predictions and QA saved with new PER90/MIN names. No position analysis, market-premium interpretation or Sloan abstract. STOP.


## Position/Residual Exploratory Phase (After Exposure Correction)

MIN270 remains primary. Original six roles; grouped and temporal held-out permutation importance; identical-other-predictor xG ablation; small fixed role blocks; six separate FDR-adjusted interactions; valid GK block; targeted MIN450/MIN900 sensitivity; OOS-only residual tests and cross-source robustness. Conditional player-bootstrap intervals do not capture model-refitting uncertainty. No robust residual market-premium evidence. Recommend exposure-aware validation as a qualified primary story, not population-invariant improvement. Reports and heatmap are exploratory; no result freeze or abstract. Earlier artifacts preserved by SHA-256 verification.


## Final Analytical Freeze (2026-09-30T08:20:34.506622+00:00)

No new models or findings. MIN270, STRUCTURAL_A and AGE_SPEC_1 unchanged. Canonical byte-identical snapshots, source reconciliation, figure verification and SHA256 manifest completed. Fold means, pooled metrics and 847-row selection-CV scope explicitly separated. Exposure story retains population-restriction and retrospective-rule limitations. Two abstract visuals recommended: exposure stability and model validation. Null xG, interaction, residual and separate-model findings retained. No abstract or release. Reopen this changelog before any analytical change.
