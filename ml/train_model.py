"""
Train a LightGBM model to forecast the 7-day peak cyanoindex at Clear Lake.
Inputs:  data/processed/clear_lake_features.csv  (cyanoindex + CIMIS weather if available)
Outputs: ml/model.pkl        (LightGBM booster + metadata)
         ml/model_report.json (evaluation metrics + feature importances)

The model automatically detects whether CIMIS weather features are present in the
feature CSV and includes them. Re-run ml/fetch_data.py after ml/cimis_data.py to
regenerate features with weather, then re-run this script to retrain.
"""

import csv
import json
import math
import pickle
import statistics
from pathlib import Path

FEATURES_FILE = Path(__file__).parent.parent / "data/processed/clear_lake_features.csv"
MODEL_OUT     = Path(__file__).parent / "model.pkl"
REPORT_OUT    = Path(__file__).parent / "model_report.json"

# Base satellite + seasonal features (always present)
BASE_FEATURE_COLS = [
    "sin_doy", "cos_doy", "month",
    "ci_mean", "ci_max", "ci_median", "ci_perc90",
    "has_signal", "pixel_count",
    "mean_7d", "mean_14d", "mean_30d",
    "peak_7d", "peak_14d", "peak_30d",
]

# CIMIS weather features (included when cimis_data.py has been run)
WEATHER_FEATURE_COLS = [
    "tmp_avg_c",       "tmp_max_c",       "tmp_min_c",
    "tmp_avg_7d",      "tmp_avg_14d",     "tmp_max_7d",
    "wind_spd_avg_ms", "wind_spd_7d",     "wind_spd_14d",
    "wind_run_7d",
    "sol_rad_avg_wm2", "sol_rad_7d",      "sol_rad_14d",
    "precip_mm",       "precip_7d",       "precip_14d",
    "eto_mm",          "eto_7d",
    "rh_avg_pct",      "rh_avg_7d",
    "calm_days_7d",
]
TARGET_COL = "target_max_7d"

# Risk tier thresholds (ci_modified units)
RISK_TIERS = [
    (0,    5,   "Low",       "No bloom signal detected. Standard precautions apply."),
    (5,    30,  "Moderate",  "Low-level bloom signal present. Sensitive individuals should exercise caution."),
    (30,   80,  "High",      "Active bloom signal. Avoid water contact. Keep pets away."),
    (80,   200, "Very High", "Dense bloom present. Do not enter the water."),
    (200,  999, "Danger",    "Severe bloom. Shoreline contact may be harmful. Follow posted advisories."),
]

def risk_tier(ci_value):
    for lo, hi, label, message in RISK_TIERS:
        if lo <= ci_value < hi:
            return label, message
    return "Danger", RISK_TIERS[-1][3]

def detect_feature_cols(sample_row):
    """Return the feature column list, adding weather cols if they exist in the CSV."""
    available = set(sample_row.keys())
    weather_present = [c for c in WEATHER_FEATURE_COLS if c in available and sample_row.get(c) not in (None, "", "None")]
    if weather_present:
        print(f"  CIMIS weather features detected: {len(weather_present)} columns will be included")
        return BASE_FEATURE_COLS + WEATHER_FEATURE_COLS
    else:
        print("  No CIMIS weather features found — using satellite + seasonal features only")
        print("  (Run ml/cimis_data.py with CIMIS_APP_KEY to add weather features)")
        return BASE_FEATURE_COLS


def load_data():
    with open(FEATURES_FILE) as f:
        rows = list(csv.DictReader(f))

    feature_cols = detect_feature_cols(rows[10] if len(rows) > 10 else rows[0])

    X, y, dates = [], [], []
    for r in rows:
        target = r.get(TARGET_COL)
        if target is None or target == "":
            continue
        row_x = []
        for c in feature_cols:
            val = r.get(c)
            try:
                row_x.append(float(val) if val not in (None, "", "None") else float("nan"))
            except (ValueError, TypeError):
                row_x.append(float("nan"))
        X.append(row_x)
        y.append(float(target))
        dates.append(r["date"])
    return X, y, dates, feature_cols

def rmse(actual, predicted):
    n = len(actual)
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n)

def mae(actual, predicted):
    n = len(actual)
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / n

def train():
    try:
        import lightgbm as lgb
    except ImportError:
        print("lightgbm not installed — run: pip install lightgbm")
        raise

    X, y, dates, feature_cols = load_data()
    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")

    # Chronological split — last 365 days as test set
    split = len(X) - 365
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    dates_test = dates[split:]

    import numpy as np
    X_train = np.array(X_train, dtype=np.float32)
    X_test  = np.array(X_test,  dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test  = np.array(y_test,  dtype=np.float32)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)

    params = {
        "objective":       "regression",
        "metric":          "rmse",
        "num_leaves":      63,
        "learning_rate":   0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":    5,
        "verbose":         -1,
        "n_jobs":          -1,
    }

    print("Training LightGBM...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_test, label=y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    preds = model.predict(X_test)
    test_rmse = rmse(y_test, preds)
    test_mae  = mae(y_test, preds)
    print(f"Test RMSE: {test_rmse:.3f}  MAE: {test_mae:.3f}")

    # Feature importances
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain").tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Tier accuracy on test set
    correct_tier = sum(
        1 for a, p in zip(y_test, preds)
        if risk_tier(a)[0] == risk_tier(p)[0]
    )
    tier_acc = correct_tier / len(y_test)
    print(f"Risk tier accuracy: {tier_acc:.1%}")

    # Save model
    model_bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "has_weather_features": any(c in feature_cols for c in WEATHER_FEATURE_COLS),
        "risk_tiers": RISK_TIERS,
        "baseline_ci": 0.9972436372799999,
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Model saved to {MODEL_OUT}")

    # Save report
    report = {
        "feature_set":   "satellite+weather" if any(c in feature_cols for c in WEATHER_FEATURE_COLS) else "satellite_only",
        "n_features":    len(feature_cols),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "test_rmse":     round(test_rmse, 4),
        "test_mae":      round(test_mae, 4),
        "tier_accuracy": round(tier_acc, 4),
        "feature_importance": {k: round(v, 2) for k, v in importance_sorted.items()},
        "risk_tiers": [
            {"range": f"{lo}-{hi}", "label": label, "message": msg}
            for lo, hi, label, msg in RISK_TIERS
        ],
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {REPORT_OUT}")

    return model_bundle

if __name__ == "__main__":
    train()
