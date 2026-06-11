"""
Fetch and preprocess Clear Lake cyanoindex data from the SFEI FHAB API.
Produces a cleaned CSV ready for feature engineering and model training.
"""

import json
import csv
import math
from pathlib import Path
from datetime import date, timedelta

RAW_FILE = Path(__file__).parent.parent / "data/raw/clear_lake_cyanoindex_2017_2025.json"
OUT_FILE = Path(__file__).parent.parent / "data/processed/clear_lake_features.csv"

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

def main():
    records = load_raw()
    print(f"Loaded {len(records)} raw records")
    rows = build_features(records)
    valid = sum(1 for r in rows if r["has_signal"])
    print(f"Records with bloom signal: {valid} / {len(rows)}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written to {OUT_FILE}")

if __name__ == "__main__":
    main()
