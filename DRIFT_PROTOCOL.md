# DRIFT_PROTOCOL.md
# GreenMLOps - Drift Detection Protocol
# Locked: February 2026 

---

## Purpose

This document defines the complete drift detection specification for all four GreenMLOps
datasets. Every element here must appear verbatim in the paper's Methodology section.
This protocol is frozen before any experiment runs. Changing thresholds or windows after
seeing results invalidates the experimental integrity.

---

## Simulation Parameters

| Parameter            | Value                        |
|----------------------|------------------------------|
| Simulation duration  | 60 days                      |
| Drift check cadence  | Daily (every 24 simulated hours) |
| Reference window     | Clean baseline cached from training set before simulation begins (days 0-6 of training data, never from simulation stream)  |
| Rolling window       | Most recent 3 days           |
| Cooldown             | 3 days minimum between retraining events (MEDIUM and LOW urgency only - does NOT apply to Fraud/CRITICAL) |
| Drift injections     | 20 pre-defined timestamps per dataset |
| Seeds                | 3 (model initialization only, injection schedule is fixed) |

---

## Per-Dataset Specification

### 1. CIFAR-100 (ResNet-18)

| Property             | Value                                          |
|----------------------|------------------------------------------------|
| Task                 | Computer vision classification (100 classes)  |
| Model                | ResNet-18 fine-tuned from ImageNet weights     |
| Urgency class        | MEDIUM                                         |
| D_max                | 12 hours                                       |
| Max accuracy drop    | 2%                                             |
| Drift signal         | Embedding drift                                |
| Features monitored   | ResNet-18 penultimate layer (512-dim -> PCA-50)|
| Drift method         | MMD (primary) + KS test (secondary)            |
| MMD threshold        | MMD > mean + 2sigma of null distribution       |
| KS threshold         | p < 0.05                                       |
| Null distribution    | Computed from reference window subsamples (days 0-6 of training data, pre-simulation) |
| MMD kernel           | RBF, bandwidth = median heuristic on reference window |
| PCA                  | Fit once on reference window, never refit      |
| Embedding source     | Fixed baseline model checkpoint only           |

**Drift injection method:** Add Gaussian noise to input images. Noise sigma is fixed at
0.5 * per-channel std of the reference window. This value is computed once from the
reference window before simulation begins and held constant across all 20 injections
and all 3 seeds. No per-run calibration.

---

### 2. AG News (DistilBERT)

| Property             | Value                                              |
|----------------------|----------------------------------------------------|
| Task                 | NLP text classification (4 classes)               |
| Model                | DistilBERT fine-tuned from distilbert-base-uncased |
| Urgency class        | MEDIUM                                             |
| D_max                | 12 hours                                           |
| Max accuracy drop    | 2%                                                 |
| Drift signal         | Embedding drift                                    |
| Features monitored   | DistilBERT [CLS] token embeddings (768-dim -> PCA-50) |
| Drift method         | MMD (primary) + KS test (secondary)                |
| MMD threshold        | MMD > mean + 2sigma of null distribution           |
| KS threshold         | p < 0.05                                           |
| Null distribution    | Computed from reference window subsamples (days 0-6 of training data, pre-simulation)  |
| MMD kernel           | RBF, bandwidth = median heuristic on reference window |
| PCA                  | Fit once on reference window, never refit          |
| Embedding source     | Fixed baseline model checkpoint only               |

**Drift injection method:** Fixed 80% topic swap. At each injection timestamp, 80% of
samples in the current window are replaced with samples from a different class (World
window receives 80% Sports samples, Sports window receives 80% Business samples, cycling
in order). The 80% ratio and the class rotation order are fixed and do not vary across
injections or seeds.

---

### 3. Credit Card Fraud (XGBoost)

| Property             | Value                                      |
|----------------------|--------------------------------------------|
| Task                 | Binary classification (imbalanced)         |
| Model                | XGBoost with scale_pos_weight              |
| Urgency class        | CRITICAL                                   |
| D_max                | 0 hours (always retrain immediately)       |
| Max accuracy drop    | 0% (no delay permitted)                    |
| Drift signal         | Feature drift                              |
| Features monitored   | All 29 input features (V1-V28 + Amount)    |
| Drift method         | PSI (Population Stability Index)           |
| PSI threshold        | PSI > 0.2 (standard industry threshold)    |
| Tool                 | Evidently AI v0.4.16                       |

**Drift injection method:** Shift ALL 29 input features (V1-V28 + Amount) by exactly
1.5 * per-feature std of the reference window. Shift direction is positive for all
features. The multiplier (1.5) and feature set (all 29) are fixed and do not vary
across injections or seeds. Std values are computed once from the reference window
before simulation begins.

**Note:** Fraud is the experimental control. The scheduler always bypasses carbon
optimization for CRITICAL urgency. Expected result: zero carbon savings, zero accuracy
degradation. This validates that the system correctly handles security-sensitive workloads.

---

### 4. ETT / LSTM (Electricity Transformer Temperature)

| Property             | Value                                              |
|----------------------|----------------------------------------------------|
| Task                 | Time-series forecasting (OT prediction)           |
| Model                | LSTM                                               |
| Urgency class        | LOW                                                |
| D_max                | 24 hours                                           |
| Max accuracy drop    | 3%                                                 |
| Drift signal         | Feature drift                                      |
| Features monitored   | All 7 input features (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) |
| Drift method         | PSI (Population Stability Index)                   |
| PSI threshold        | PSI > 0.2 (standard industry threshold)            |
| Tool                 | Evidently AI v0.4.16                               |

**Drift injection method:** Shift ALL 7 input features (HUFL, HULL, MUFL, MULL, LUFL,
LULL, OT) by exactly 1.5 * per-feature std of the reference window. Shift direction is
positive for all features. The multiplier (1.5) and feature set (all 7) are fixed and
do not vary across injections or seeds. Std values are computed once from the reference
window before simulation begins. ETT has natural seasonal drift built in (monthly OT
range ~5 to ~27), so injections augment existing signal rather than creating artificial
patterns.

---

## Drift Detection Logic (All Datasets)

```
every 24 simulated hours:
    current_window = most recent 3 days of incoming data
    reference_window = clean baseline cached from training data before simulation starts (never updated)

    if dataset in [CIFAR-100, AG News]:
        embeddings_current = extract_embeddings(current_window, baseline_model)
        embeddings_ref     = extract_embeddings(reference_window, baseline_model)
        pca_current        = fitted_pca.transform(embeddings_current)
        pca_ref            = fitted_pca.transform(embeddings_ref)
        drift_score        = compute_mmd(pca_ref, pca_current, kernel="rbf")
        drift_detected     = drift_score > mmd_threshold

    if dataset in [Fraud, ETT]:
        drift_report   = evidently_psi(reference_window, current_window)
        drift_score    = max PSI across all monitored features
        drift_detected = drift_score > 0.2

    if dataset == Fraud:
        # CRITICAL urgency: D_max = 0, cooldown does not apply
        # every drift detection triggers immediate retraining with no delay and no cooldown check
        if drift_detected:
            issue_retraining_request(timestamp=now, urgency=CRITICAL)
    else:
        if drift_detected AND days_since_last_retrain >= 3:
            issue_retraining_request(timestamp=now, urgency=dataset.urgency)

    log_to_mlflow(
        drift_score=drift_score,
        drift_detected=drift_detected,
        days_since_last_retrain=days_since_last_retrain,
        accuracy_on_new_distribution=evaluate_on_post_drift_test_set(),
        carbon_intensity_at_check=caiso_carbon(now)
    )
```

---

## Accuracy During Wait - Definition

During the delay window [t0, t*], model accuracy is evaluated on a held-out test set
drawn from the NEW (post-drift) distribution. This is the performance degradation metric
reported in the paper.

It is NOT evaluated on the original distribution. Evaluating on the original distribution
would mask the actual cost of waiting and understate performance degradation.

Recording rule:
- Always record accuracy at t0 (drift detection day) and at t* (retraining day)
- If wait >= 24 hours: also record accuracy at every 24-hour interval between t0 and t*
- If wait < 24 hours (MEDIUM urgency, D_max=12h): record only t0 and t* - no intermediate
  readings possible within the same simulated day

This ensures at least 2 data points per retraining event regardless of delay length.

Post-drift test set construction: the held-out test set used for accuracy measurement
at each timestamp is constructed using the same injection transform or mixture rule
that was applied to the stream at that timestamp. Specifically:
- CIFAR-100: test images have the same Gaussian noise (sigma = 0.5 * reference std)
  applied as the injected stream at that day
- AG News: test set uses the same 80% topic swap ratio and class rotation as the
  injected stream at that day
- Fraud/ETT: test set features are shifted by the same 1.5 * reference std as the
  injected stream at that day

This ensures accuracy measurement reflects model performance on the actual distribution
the model is seeing in production, not a clean holdout that would understate degradation.

---

## Embedding Drift - PCA Implementation

Steps executed in this exact order, once per dataset before simulation begins:

1. Load fixed baseline model checkpoint (weights frozen after initial training)
2. Extract embeddings from clean reference baseline (cached from training set, 
    days 0-6, before any simulation stream begins)
3. Fit PCA to 50 dimensions on reference window embeddings
4. Freeze PCA - never refit during simulation
5. Compute null MMD distribution: MMD between 500 random subsample pairs drawn from
   the clean reference baseline (with replacement). Each pair uses N=256 samples
   (fixed constant; derived from samples_per_day=85 x 3 days rolling window, rounded
   to 256 for computational stability). Threshold = mean + 2sigma of these 500 MMD
   values. All computed before simulation begins - never recomputed mid-simulation.
6. At each daily check: transform current window with frozen PCA, compute MMD against
   frozen reference PCA embeddings, compare against threshold

Rationale for PCA-50: High-dimensional MMD (512-dim or 768-dim) has poor statistical
power and high variance. PCA-50 is fast, stable, and standard practice.

Fallback: If MMD behaves erratically after PCA-50, use KS test only on PCA-50 components.
KS test is run independently on each of the 50 PCA components. Drift is flagged if more
than 10% of components (i.e., >= 6 out of 50) show p < 0.05. This fraction-based rule
avoids false positives from isolated noisy components while remaining sensitive to
systematic distribution shift.

---

## Drift Injection Schedule

20 drift injection timestamps per dataset, fixed across all 3 seeds.
Only model initialization varies by seed - injection schedule does not.

Injection timestamps are evenly distributed across the 60-day simulation with
slight jitter to avoid artificial periodicity. Exact timestamps below:

### CIFAR-100 injection days (day index within 60-day simulation)
3, 6, 9, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 51, 54, 57, 59

### AG News injection days
3, 6, 9, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 51, 54, 57, 59

### Credit Card Fraud injection days
4, 7, 10, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 52, 55, 57, 59

### ETT injection days
4, 7, 10, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 52, 55, 57, 59

The 3 approaches are:
- Periodic: fixed weekly retraining, no drift detection, no carbon scheduling
- Drift-immediate: retrain at t0 (drift detection time), no carbon scheduling
- Carbon-aware: retrain at t* (optimal carbon window), urgency-based scheduling (ours)

Total retraining decisions: 20 events x 4 datasets x 3 approaches x 3 seeds = 720

---

## MLflow Logging Schema

Every daily drift check logs one row per dataset:

| Field                        | Type    | Description                                      |
|------------------------------|---------|--------------------------------------------------|
| timestamp                    | int     | Simulated day index (0-59)                       |
| dataset                      | str     | cifar100 / ag_news / fraud / ett                 |
| drift_score                  | float   | PSI or MMD value at this check                   |
| drift_detected               | bool    | Whether threshold was exceeded                   |
| days_since_last_retrain      | int     | Cooldown tracker                                 |
| retraining_triggered         | bool    | drift_detected AND cooldown satisfied (MEDIUM/LOW only) |
| accuracy_on_new_distribution | float   | Model accuracy on post-drift test set            |
| carbon_intensity_at_check    | float   | gCO2/kWh from CAISO at this timestamp            |

Note: For Fraud (CRITICAL), retraining_triggered = drift_detected. Cooldown does not apply.


Every retraining event logs one additional row:

| Field                | Type  | Description                                        |
|----------------------|-------|----------------------------------------------------|
| t0                   | int   | Day drift was detected                             |
| t_star               | int   | Day training was actually executed                 |
| urgency_class        | str   | CRITICAL / MEDIUM / LOW                            |
| carbon_immediate     | float | gCO2 if trained at t0                              |
| carbon_scheduled     | float | gCO2 actually emitted at t_star                    |
| carbon_saved_pct     | float | (carbon_immediate - carbon_scheduled) / carbon_immediate |
| wait_duration_hours  | float | (t_star - t0) * 24                                 |
| accuracy_during_wait | list  | Daily accuracy readings from t0 to t_star          |
| accuracy_post_retrain| float | Accuracy after retraining completes                |

---

## Version History

| Date          | Change                        | Author |
|---------------|-------------------------------|--------|
| Feb 2026      | Initial version created       | Arun   |
| Feb 2026      | Fix day indexing (0-59), pin injection strength, clarify accuracy-during-wait rule, exempt Fraud from cooldown | Arun |
| Feb 2026      | Add null MMD sample size (500 pairs), KS fallback fraction rule (>=6/50 components), post-drift test set construction per dataset | Arun |
| Feb 2026      | Fix reference window contamination (now pre-simulation cached baseline), fix day notation to 0-6 consistently | Arun |
| Feb 2026      | Fix PCA step numbering (7->6), pin rolling window N=256 samples, fix MLflow table formatting | Arun |

---