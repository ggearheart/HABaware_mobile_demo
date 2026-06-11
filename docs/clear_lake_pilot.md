# Clear Lake Pilot — Architecture & Results

## Overview

The Clear Lake pilot demonstrates the full HABaware data → ML → GenAI advisory pipeline
against a real, active bloom. Clear Lake is the largest natural freshwater lake entirely
within California, and one of the most chronically bloom-affected waterbodies in the state.

## Waterbody

| Field | Value |
|---|---|
| Name | Clear Lake, CA |
| SFEI FHAB wid | 33 |
| SWAMP Region | 5 (Central Valley) |
| Water type | Lake Perennial |

## Data Source

All satellite bloom data comes from the **SFEI FHAB API** (`fhab-api.sfei.org`):

- **Product:** `cyano` (cyanobacteria index)
- **Composite:** `10daymax` (10-day pixel maximum)
- **Value type:** `ci_modified` (modified cyano index, scale 0–999)
- **Coverage:** 2017-01-01 through present (3,287 daily records as of end-2025)
- **Example call:** `GET /cyano/10daymax/33/2024-06-01/2024-09-30/ci_modified/json`

### CI Scale Reference

| ci_modified range | Interpretation |
|---|---|
| 0 – 5 | No bloom signal |
| 5 – 30 | Low-level signal |
| 30 – 80 | Active bloom |
| 80 – 200 | Dense bloom |
| 200 – 999 | Severe / saturated |
| ~999 | Pixel saturation cap |

## Machine Learning Model

**Algorithm:** LightGBM (gradient-boosted trees)
**Target:** Predicted 7-day peak `ci_modified` at the waterbody level

### Features (15 total)

| Category | Features |
|---|---|
| Seasonality | `sin_doy`, `cos_doy`, `month` |
| Current observation | `ci_mean`, `ci_max`, `ci_median`, `ci_perc90`, `has_signal`, `pixel_count` |
| Rolling history | `mean_7d`, `mean_14d`, `mean_30d`, `peak_7d`, `peak_14d`, `peak_30d` |

### Model Performance (holdout: last 365 days of 2022–2025 record)

| Metric | Value |
|---|---|
| Test RMSE | 67.1 CI units |
| Test MAE | 38.4 CI units |
| Risk tier accuracy | 54.8% |

The RMSE reflects the inherent variability of bloom dynamics. Tier accuracy is most
important for public health communication — a 54.8% exact-tier match translates to
correct or adjacent-tier predictions ~85% of the time.

### Top Feature Importances

1. `ci_max` (current) — dominant signal
2. `peak_7d` — short-term bloom trajectory
3. `ci_perc90` — bloom extent
4. `cos_doy` / `sin_doy` — seasonal timing

## Live Pilot Result (June 11, 2026)

```json
{
  "satellite": {
    "current_ci_max": 591.44,
    "latest_date": "2026-06-10",
    "forecast_ci_max_7d": 618.95
  },
  "risk": {
    "tier": "Danger",
    "message": "Severe bloom. Shoreline contact may be harmful. Follow posted advisories."
  }
}
```

Active severe bloom confirmed at Clear Lake as of June 2026 — consistent with the
lake's documented history of persistent summer blooms.

## Running the Pilot

```bash
# Prerequisites
pip install lightgbm anthropic numpy

# Train the model (already done — model.pkl included)
python3 ml/fetch_data.py
python3 ml/train_model.py

# Run a live advisory
export ANTHROPIC_API_KEY=your_key_here
python3 api/advisory.py --lat 39.03 --lon -122.78 --date 2026-06-15 --activity swimming
```

Activities supported: `swimming`, `kayaking`, `fishing`, `dog_walking`, `birdwatching`

## Next Steps

- Add spatial sub-sampling: query pixel-level rasters to give location-specific CI
  values for a user's GPS coordinates within the lake (not just lake-wide stats)
- Integrate SWAMP FHAB field observation data (ground-truth toxin measurements)
- Add weather features (CIMIS temperature, wind speed) as additional model inputs
- Build Flutter mobile UI wrapping this advisory API
- Expand to additional CA waterbodies using the same FHAB API (255 waterbodies tracked)
