"""
Fetch and preprocess Clear Lake cyanoindex data from the SFEI FHAB API,
joined with CIMIS weather features from Station #106 (Sanel Valley, 27km away).
Produces a cleaned CSV ready for feature engineering and model training.
"""

import json
import csv
import math
from pathlib import Path
from datetime import date, timedelta

RAW_FILE    = Path(__file__).parent.parent / "data/raw/clear_lake_cyanoindex_2017_2025.json"
WEATHER_CSV = Path(__file__).parent.parent / "data/processed/cimis_clear_lake_weather.csv"
OUT_FILE    = Path(__file__).parent.parent / "data/processed/clear_lake_features.csv"

BASELINE_CI = 0.9972436372799999  # value returned when no bloom signal present

def load_raw():
    with open(RAW_FILE) as f:
        return json.load(f)

def is_valid(record):
    """A record has real signal if max > baseline."""
    return record["pixel_count"] > 0 and record["max"] > BASELINE_CI

def doy(date_str):
    d = date.fromisoformat(date_str)
    return d.timetuple().tm_yday

def week_of_year(date_str):
    d = date.fromisoformat(date_str)
    return int(d.strftime("%V"))

def build_features(records):
    rows = []
    for i, r in enumerate(records):
        # Rolling stats over prior 7, 14, 30 days
        window_7  = [x["mean"] for x in records[max(0,i-7):i]  if is_valid(x)]
        window_14 = [x["mean"] for x in records[max(0,i-14):i] if is_valid(x)]
        window_30 = [x["mean"] for x in records[max(0,i-30):i] if is_valid(x)]

        peak_7  = max((x["max"]  for x in records[max(0,i-7):i]  if is_valid(x)), default=0)
        peak_14 = max((x["max"]  for x in records[max(0,i-14):i] if is_valid(x)), default=0)
        peak_30 = max((x["max"]  for x in records[max(0,i-30):i] if is_valid(x)), default=0)

        mean_7  = sum(window_7)  / len(window_7)  if window_7  else 0
        mean_14 = sum(window_14) / len(window_14) if window_14 else 0
        mean_30 = sum(window_30) / len(window_30) if window_30 else 0

        d = date.fromisoformat(r["date"])

        rows.append({
            "date":         r["date"],
            "year":         d.year,
            "doy":          doy(r["date"]),
            "week":         week_of_year(r["date"]),
            "month":        d.month,
            # Seasonal signal encoded as sine/cosine to avoid discontinuity at year wrap
            "sin_doy":      round(math.sin(2 * math.pi * doy(r["date"]) / 365), 6),
            "cos_doy":      round(math.cos(2 * math.pi * doy(r["date"]) / 365), 6),
            # Raw observations
            "ci_mean":      round(r["mean"],   6) if is_valid(r) else 0,
            "ci_max":       round(r["max"],    6) if is_valid(r) else 0,
            "ci_median":    round(r["median"], 6) if is_valid(r) else 0,
            "ci_perc90":    round(r["perc90"], 6) if is_valid(r) else 0,
            "pixel_count":  r["pixel_count"],
            "has_signal":   int(is_valid(r)),
            # Rolling features
            "mean_7d":      round(mean_7,  6),
            "mean_14d":     round(mean_14, 6),
            "mean_30d":     round(mean_30, 6),
            "peak_7d":      round(peak_7,  6),
            "peak_14d":     round(peak_14, 6),
            "peak_30d":     round(peak_30, 6),
            # Target: max ci_modified value in the following 7 days (forecast target)
            "target_max_7d": None,  # filled in second pass
        })

    # Fill forward-looking target
    for i, row in enumerate(rows):
        future = [r for r in records[i+1:i+8] if is_valid(r)]
        row["target_max_7d"] = round(max((r["max"] for r in future), default=0), 6)

    return rows

def load_weather():
    """Load CIMIS weather CSV into a dict keyed by date string. Returns {} if file missing."""
    if not WEATHER_CSV.exists():
        print(f"  Warning: CIMIS weather file not found ({WEATHER_CSV.name}). "
              "Run ml/cimis_data.py with CIMIS_APP_KEY set to add weather features.")
        return {}
    weather = {}
    with open(WEATHER_CSV) as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")
            if d:
                weather[d] = row
    print(f"  Loaded {len(weather)} CIMIS weather records from {WEATHER_CSV.name}")
    return weather


# Weather feature columns to join — only the pre-computed rolling versions
# (raw daily values are already incorporated into 7d/14d rolling features)
WEATHER_FEATURES = [
    "tmp_avg_c",        "tmp_max_c",        "tmp_min_c",
    "tmp_avg_7d",       "tmp_avg_14d",      "tmp_max_7d",
    "wind_spd_avg_ms",  "wind_spd_7d",      "wind_spd_14d",
    "wind_run_7d",
    "sol_rad_avg_wm2",  "sol_rad_7d",       "sol_rad_14d",
    "precip_mm",        "precip_7d",        "precip_14d",
    "eto_mm",           "eto_7d",
    "rh_avg_pct",       "rh_avg_7d",
    "calm_days_7d",
]


def join_weather(rows, weather):
    """Attach CIMIS features to each cyanoindex row, filling None if date missing."""
    n_matched = 0
    for row in rows:
        w = weather.get(row["date"], {})
        if w:
            n_matched += 1
        for col in WEATHER_FEATURES:
            val = w.get(col)
            try:
                row[col] = float(val) if val not in (None, "", "None") else None
            except (ValueError, TypeError):
                row[col] = None
    print(f"  Joined weather: {n_matched}/{len(rows)} dates matched")
    return rows


def main():
    records = load_raw()
    print(f"Loaded {len(records)} raw cyanoindex records")
    rows = build_features(records)
    valid = sum(1 for r in rows if r["has_signal"])
    print(f"Records with bloom signal: {valid} / {len(rows)}")

    weather = load_weather()
    rows = join_weather(rows, weather)
    weather_coverage = sum(1 for r in rows if r.get("tmp_avg_c") is not None)
    print(f"  Weather coverage: {weather_coverage}/{len(rows)} rows have temperature data")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written to {OUT_FILE}")

if __name__ == "__main__":
    main()
