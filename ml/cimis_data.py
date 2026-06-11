"""
CIMIS weather data fetcher for the HABaware Clear Lake pilot.

Fetches daily weather from the California Irrigation Management Information System (CIMIS).
Primary station: #106 Sanel Valley, Mendocino County (27 km from Clear Lake center) — closest active station.
Fallback:       Spatial CIMIS at Clear Lake coordinates for ETo + solar radiation.

Produces:
  data/raw/cimis/cimis_station106_YYYY.json        — raw daily JSON per year
  data/processed/cimis_clear_lake_weather.csv      — cleaned daily weather features (2017–present)

Usage:
    export CIMIS_APP_KEY=your_key_here   # register free at https://www.cimis.water.ca.gov/
    python3 ml/cimis_data.py

The CIMIS Web API requires a free AppKey:
    https://www.cimis.water.ca.gov/Auth/Register.aspx
"""

import csv
import json
import os
import time
import urllib.request
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

CIMIS_BASE    = "https://et.water.ca.gov/api/data"
STATION_ID    = 106      # Sanel Valley, Mendocino — closest active station to Clear Lake
CL_LAT        = 39.0300  # Clear Lake center (for Spatial CIMIS fallback)
CL_LON        = -122.780

RAW_DIR = Path(__file__).parent.parent / "data/raw/cimis"
OUT_DIR = Path(__file__).parent.parent / "data/processed"
OUT_CSV = OUT_DIR / "cimis_clear_lake_weather.csv"

# Daily data items to fetch — chosen for bloom relevance
# Temperature: drives stratification + metabolic rates of cyanobacteria
# Wind: mixing suppression = bloom surface accumulation
# Solar radiation: photosynthesis driver for cyanobacteria
# Precipitation: nutrient flushing and dilution
# ETo: integrates temp + humidity + wind + radiation into one index
# Humidity: affects thermal stratification
DATA_ITEMS = [
    "day-air-tmp-avg",
    "day-air-tmp-max",
    "day-air-tmp-min",
    "day-rel-hum-avg",
    "day-sol-rad-avg",
    "day-wind-spd-avg",
    "day-wind-run",
    "day-precip",
    "day-eto",
]

# Lookup maps for parsing CIMIS JSON response
ITEM_KEY_MAP = {
    "day-air-tmp-avg":  "tmp_avg_c",
    "day-air-tmp-max":  "tmp_max_c",
    "day-air-tmp-min":  "tmp_min_c",
    "day-rel-hum-avg":  "rh_avg_pct",
    "day-sol-rad-avg":  "sol_rad_avg_wm2",
    "day-wind-spd-avg": "wind_spd_avg_ms",
    "day-wind-run":     "wind_run_km",
    "day-precip":       "precip_mm",
    "day-eto":          "eto_mm",
}


def get_app_key():
    key = os.environ.get("CIMIS_APP_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "CIMIS_APP_KEY not set.\n"
            "Register free at https://www.cimis.water.ca.gov/Auth/Register.aspx\n"
            "then: export CIMIS_APP_KEY=your_key_here"
        )
    return key


def fetch_year(year, app_key):
    """Fetch one calendar year of daily data from CIMIS station 106."""
    start = f"{year}-01-01"
    end   = f"{year}-12-31" if year < date.today().year else date.today().isoformat()

    params = urllib.parse.urlencode({
        "appKey":        app_key,
        "targets":       str(STATION_ID),
        "startDate":     start,
        "endDate":       end,
        "dataItems":     ",".join(DATA_ITEMS),
        "unitOfMeasure": "M",   # metric
    })
    url = f"{CIMIS_BASE}?{params}"

    req = urllib.request.Request(url, headers={
        "Accept":     "application/json",
        "User-Agent": "HABaware/1.0",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def parse_records(raw):
    """Convert raw CIMIS JSON into a list of flat dicts keyed by date."""
    records = []
    for provider in raw.get("Data", {}).get("Providers", []):
        for record in provider.get("Records", []):
            date_str = record.get("Date", "")[:10]
            row = {"date": date_str, "station_id": STATION_ID}
            for item in record.get("DayData", []):
                code  = item.get("Item", "")
                value = item.get("Value")
                qc    = item.get("Qc", "")
                key   = ITEM_KEY_MAP.get(code)
                if key:
                    try:
                        row[key] = float(value) if value not in (None, "", " ") else None
                    except (ValueError, TypeError):
                        row[key] = None
                    row[f"{key}_qc"] = qc
            if date_str:
                records.append(row)
    return records


def fetch_all_years(start_year=2017):
    """Fetch data for every year from start_year through today and cache as JSON."""
    app_key = get_app_key()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []

    for year in range(start_year, date.today().year + 1):
        cache_path = RAW_DIR / f"cimis_station{STATION_ID}_{year}.json"
        # Use cached file unless it's the current year (may be incomplete)
        if cache_path.exists() and year < date.today().year:
            with open(cache_path) as f:
                raw = json.load(f)
            print(f"  {year}: loaded from cache ({cache_path.name})")
        else:
            print(f"  {year}: fetching from CIMIS API...", end=" ", flush=True)
            raw = fetch_year(year, app_key)
            with open(cache_path, "w") as f:
                json.dump(raw, f)
            print("done")
            time.sleep(0.5)   # be polite to the API

        records = parse_records(raw)
        all_records.extend(records)
        print(f"         {len(records)} daily records")

    return all_records


def add_rolling_features(records):
    """Add 7- and 14-day rolling weather features relevant to bloom prediction."""
    from collections import deque

    def rolling_mean(window, key):
        vals = [r.get(key) for r in window if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def rolling_sum(window, key):
        vals = [r.get(key) for r in window if r.get(key) is not None]
        return round(sum(vals), 4) if vals else None

    enhanced = []
    buf = deque(maxlen=14)
    for r in records:
        buf.append(r)
        w7  = list(buf)[-7:]
        w14 = list(buf)

        r_out = dict(r)
        r_out["tmp_avg_7d"]    = rolling_mean(w7,  "tmp_avg_c")
        r_out["tmp_avg_14d"]   = rolling_mean(w14, "tmp_avg_c")
        r_out["tmp_max_7d"]    = rolling_mean(w7,  "tmp_max_c")
        r_out["wind_spd_7d"]   = rolling_mean(w7,  "wind_spd_avg_ms")
        r_out["wind_spd_14d"]  = rolling_mean(w14, "wind_spd_avg_ms")
        r_out["wind_run_7d"]   = rolling_sum(w7,   "wind_run_km")
        r_out["sol_rad_7d"]    = rolling_mean(w7,  "sol_rad_avg_wm2")
        r_out["sol_rad_14d"]   = rolling_mean(w14, "sol_rad_avg_wm2")
        r_out["precip_7d"]     = rolling_sum(w7,   "precip_mm")
        r_out["precip_14d"]    = rolling_sum(w14,  "precip_mm")
        r_out["eto_7d"]        = rolling_sum(w7,   "eto_mm")
        r_out["rh_avg_7d"]     = rolling_mean(w7,  "rh_avg_pct")
        # Calm-day count: days with wind < 2 m/s in last 7 days (calm = bloom accumulation risk)
        r_out["calm_days_7d"]  = sum(
            1 for x in w7
            if x.get("wind_spd_avg_ms") is not None and x["wind_spd_avg_ms"] < 2.0
        )
        enhanced.append(r_out)
    return enhanced


def write_csv(records):
    if not records:
        print("No records to write.")
        return
    # Collect all keys from first record as column order
    base_keys = list(records[0].keys())
    # Remove QC columns from base but keep them at end
    qc_keys  = [k for k in base_keys if k.endswith("_qc")]
    data_keys = [k for k in base_keys if not k.endswith("_qc")]
    fieldnames = data_keys + qc_keys

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"\nWritten {len(records)} rows to {OUT_CSV}")


def main():
    print(f"Fetching CIMIS daily weather — Station #{STATION_ID} (Sanel Valley, 27km from Clear Lake)")
    print(f"Data items: {', '.join(DATA_ITEMS)}\n")

    records = fetch_all_years(start_year=2017)
    print(f"\nTotal raw records: {len(records)}")

    records_with_rolling = add_rolling_features(records)
    write_csv(records_with_rolling)

    # Quick summary
    valid_tmp = [r["tmp_avg_c"] for r in records if r.get("tmp_avg_c") is not None]
    valid_wind = [r["wind_spd_avg_ms"] for r in records if r.get("wind_spd_avg_ms") is not None]
    if valid_tmp:
        print(f"Temp range: {min(valid_tmp):.1f}–{max(valid_tmp):.1f} °C  (mean {sum(valid_tmp)/len(valid_tmp):.1f})")
    if valid_wind:
        print(f"Wind range: {min(valid_wind):.2f}–{max(valid_wind):.2f} m/s (mean {sum(valid_wind)/len(valid_wind):.2f})")


if __name__ == "__main__":
    main()
