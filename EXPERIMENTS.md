# F1 Race Position Prediction — Experiment Report

**Training data:** 2020–2024 seasons (107 events, ~20K samples)
**Split:** 70/15/15 chronological by event (75 train, 16 val, 16 test)
**Target:** Race finishing position (normalized pos/20, MSE loss)
**Baseline:** Constant prediction MAE = 5.5 positions

---

## 1. Feature Tiers

Features are organized into three tiers, each adding signal on top of the previous:

| Tier | Dims | Description | Features |
|------|------|-------------|----------|
| **Core** | 46 | Standard weekend context | era (5), session_flags (7), target_type (3), grid_position (1), qualifying_times (3), practice_pace (6), practice_best_lap (3), practice_speed (4), compound_usage (3), tyre_stint_count (1), weather_race (6), sprint_result (1), sprint_grid (1), sprint_pace (2) |
| **Form** | 5 | Historical rolling stats | rolling_avg_finish (1), grid_finish_delta (1), championship_points (1), dnf_rate (1), history_available (1) |
| **Derived** | 8 | Advanced computed features | qualifying_gap_to_pole (1), qualifying_gap_to_ahead (1), practice_long_run_pace (2), practice_relative_pace (3), tyre_degradation (1) |
| **Total** | **59** | | |

---

## 2. Feature Tier Ablation

All runs use lookback=5, 100 epochs, patience=15.

### 2a. Core Only (46 dims)

| Model | Params | Test MAE | Test RMSE | Top-3 Acc | Val MAE |
|-------|--------|----------|-----------|-----------|---------|
| Linear | 671 | 3.236 | 4.197 | 0.269 | 3.326 |
| SingleLayerMLP | 671 | 3.219 | 4.152 | 0.264 | 3.289 |
| MLP (128x64) | 16,869 | 3.142 | 4.070 | 0.188 | 3.222 |
| Deep MLP (256x128x64) | — | *running* | — | — | — |

### 2b. Core + Form (51 dims)

Results from lookback=3 (consistent feature set across all 4 models):

| Model | Params | Test MAE | Test RMSE | Top-3 Acc | Val MAE |
|-------|--------|----------|-----------|-----------|---------|
| Linear | 676 | 3.142 | 4.087 | 0.262 | 3.187 |
| SingleLayerMLP | 676 | **3.102** | 4.092 | **0.299** | 3.134 |
| MLP (128x64) | 17,509 | 3.133 | 4.074 | 0.287 | 3.193 |
| Deep MLP (256x128x64) | 59,109 | 3.131 | **4.069** | 0.201 | 3.216 |

### 2c. All Tiers (59 dims)

| Model | Params | Test MAE | Test RMSE | Top-3 Acc | Val MAE |
|-------|--------|----------|-----------|-----------|---------|
| SingleLayerMLP | 684 | 3.138 | 4.083 | 0.269 | 3.111 |
| MLP (128x64) | 18,533 | 3.147 | **4.042** | 0.227 | 3.119 |
| Linear | — | *running* | — | — | — |
| Deep MLP | — | *running* | — | — | — |

### 2d. Feature Tier Impact (MLP 128x64)

| Tiers | Dims | Test MAE | Test RMSE | Delta MAE |
|-------|------|----------|-----------|-----------|
| Core only | 46 | 3.142 | 4.070 | baseline |
| Core + Form | 51 | 3.133 | 4.074 | -0.009 |
| All | 59 | 3.147 | 4.042 | +0.005 |

**Finding:** Adding form features (rolling avg finish, championship points, DNF rate) provides a small MAE improvement (~0.01 positions). The derived features (qualifying gaps, long-run pace, tyre degradation) improve RMSE but slightly worsen MAE, suggesting they help with tail predictions but add noise for typical cases. The overall feature tier effect is **modest** — most predictive signal comes from the core weekend features, especially grid position and qualifying times.

---

## 3. Model Architecture Comparison

Using core+form features (51 dims, lookback=3) for consistent comparison:

| Model | Architecture | Params | Test MAE | Test RMSE | Top-3 Acc |
|-------|-------------|--------|----------|-----------|-----------|
| Linear | embed → linear | 676 | 3.142 | 4.087 | 0.262 |
| SingleLayerMLP | embed → 128 → 1 | 676 | **3.102** | 4.092 | **0.299** |
| MLP | embed → 128 → 64 → 1 | 17,509 | 3.133 | 4.074 | 0.287 |
| Deep MLP | embed → 256 → 128 → 64 → 1 | 59,109 | 3.131 | **4.069** | 0.201 |

**Finding:** The **SingleLayerMLP with just 676 parameters achieves the best MAE (3.10)** and top-3 accuracy (30%). Larger models (MLP, Deep MLP) have marginally better RMSE but worse overall accuracy despite having 25-87x more parameters. This strongly suggests the problem is **data-limited, not model-limited** — additional model capacity does not help and may hurt through overfitting.

---

## 4. Lookback Window

Comparing lookback=3 vs lookback=5 where consistent feature sets are available:

### Linear (51 dims, core+form)

| Lookback | Test MAE | Test RMSE | Top-3 Acc |
|----------|----------|-----------|-----------|
| 3 | 3.142 | 4.087 | 0.262 |
| 5 | 3.147 | 4.082 | 0.252 |

**Finding:** Increasing lookback from 3 to 5 races provides **no meaningful improvement**. The difference is within noise range (~0.005 positions). This makes sense — F1 form is volatile and looking back 3 races captures the relevant momentum window. Longer lookbacks may dilute signal with stale data.

Additional lookback experiments (lb=10) are still running.

---

## 5. Key Takeaways

1. **Best model:** SingleLayerMLP (676 params) with core+form features, MAE = 3.10 positions. Simple is better.

2. **Feature importance hierarchy:**
   - **Grid position & qualifying** are the dominant predictors (core tier provides most signal)
   - **Form features** add marginal value (~0.01 MAE improvement)
   - **Derived features** are a wash — help RMSE, hurt MAE

3. **Model complexity doesn't help:** 25-87x more parameters gives no MAE improvement. The linear model (676 params) achieves MAE=3.14, only 0.04 worse than the best. This is a **feature/data problem, not a modeling problem**.

4. **Lookback window is not critical:** 3 races of history is sufficient; longer windows don't improve predictions.

5. **Performance context:** MAE of ~3.1 positions means the model predicts within about 3 grid positions on average. For a 20-driver field, this is solid but far from race-winning accuracy. The main bottleneck is **unpredictable race-day events** (strategy, incidents, weather changes, reliability) that are not captured in pre-race features.

---

## 6. Remaining Experiments

The following experiments are still running and will be added when complete:

- Core-only Deep MLP (46 dims, lb5)
- Core+Form ablation suite (51 dims, lb5) — 4 models
- All-features ablation suite (59 dims, lb5) — 4 models
- Lookback sweep: lb=3, lb=5, lb=10 with all features + MLP

---

## 7. Possible Next Steps

- **Race-day features:** In-race telemetry, pit stop timing, safety car periods (requires live/post-race data rather than pre-race prediction)
- **Track-specific features:** Circuit characteristics (street vs permanent, high/low downforce)
- **Lap-time-relative normalization:** Normalize qualifying gaps and tyre degradation by lap time to make cross-track comparisons meaningful
- **Split-aware history:** Prevent any data leakage by only computing historical features from training events for train samples
- **Ensemble methods:** Combine predictions from multiple model types
- **Alternative targets:** Predict finishing order (ranking loss) instead of absolute position
