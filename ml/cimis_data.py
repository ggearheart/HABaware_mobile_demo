"""
Weather data fetcher for the HABaware Clear Lake pilot.

Primary source:  Open-Meteo Archive API (https://open-meteo.com)
                 Queried at Clear Lake coordinates. Free, no key, 1940–present.
                 Variables match CIMIS daily output closely.
CIMIS fallback:  et.water.ca.gov Station #106 Sanel Valley (27km away).
                 Set CIMIS_APP_KEY env var if the Open-Meteo source is unavailable.

Produces:
  data/raw/weather/openmeteo_clear_lake_YYYY.json      — raw daily JSON per year
  data/processed/cimis_clear_lake_weather.csv          — cleaned daily weather features

The output CSV is named cimis_clear_lake_weather.csv for compatibility with
fetch_data.py and train_model.py regardless of which source was used.

Usage:
    python3 ml/cimis_data.py
"""

import csv
import json
import math
import os
import time
import urllib.request
import urllib.parse
from collections import deque
from datetime import date, timedelta
from pathlib import Path

# ── Clear Lake coordinates ────────────────────────────────────────────────────
CL_LAT, CL_LON = 39.0300, -122.780
TIMEZONE        = "America/Los_Angeles"

# ── Open-Meteo Archive API ────────────────────────────────────────────────────
OM_BASE    = "https://archive-api.open-meteo.com/v1/archive"
OM_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
OM_VARIABLES = ",".join([
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "shortwave_radiation_sum",    # MJ/m² — convert to W/m² daily mean
    "wind_speed_10m_max",         # km/h — convert to m/s
    "wind_speed_10m_mean",        # km/h
    "precipitation_sum",          # mm
    "et0_fao_evapotranspiration", # mm — equivalent to CIMIS ETo
])

# ── CIMIS fallback ────────────────────────────────────────────────────────────
CIMIS_BASE       = "https://et.water.ca.gov/api/data"
CIMIS_STATION_ID = 106
CIMIS_ITEMS      = "day-air-tmp-avg,day-air-tmp-max,day-air-tmp-min,day-rel-hum-avg,day-sol-rad-avg,day-wind-spd-avg,day-wind-run,day-precip,day-eto"

RAW_DIR = Path(__file__).parent.parent / "data/raw/weather"
OUT_DIR = Path(__file__).parent.parent / "data/processed"
OUT_CSV = OUT_DIR / "cimis_clear_lake_weather.csv"


# ── Open-Meteo fetch ──────────────────────────────────────────────────────────

def fetch_openmeteo_year(year):
    start = f"{year}-01-01"
    end   = f"{year}-12-31" if year < date.today().year else (date.today() - timedelta(days=1)).isoformat()
    params = urllib.parse.urlencode({
        "latitude":  CL_LAT,
        "longitude": CL_LON,
        "start_date": start,
        "end_date":   end,
        "daily":      OM_VARIABLES,
        "timezone":   TIMEZONE,
    })
    url = f"{OM_BASE}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "HABaware/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def fetch_openmeteo_recent(days=16):
    """Fetch recent + forecast days from Open-Meteo forecast API."""
    params = urllib.parse.urlencode({
        "latitude":  CL_LAT,
        "longitude": CL_LON,
        "past_days":  days,
        "forecast_days": 7,
        "daily":      OM_VARIABLES,
        "timezone":   TIMEZONE,
    })
    url = f"{OM_FORECAST_BASE}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "HABaware/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def parse_openmeteo(raw):
    daily = raw.get("daily", {})
    dates = daily.get("time", [])
    records = []
    for i, date_str in enumerate(dates):
        def val(key):
            v = daily.get(key, [None] * (i+1))[i]
            return float(v) if v is not None else None

        tmp_avg    = val("temperature_2m_mean")
        tmp_max    = val("temperature_2m_max")
        tmp_min    = val("temperature_2m_min")
        rh_avg     = val("relative_humidity_2m_mean")
        rad_mj     = val("shortwave_radiation_sum")     # MJ/m²/day
        wind_max   = val("wind_speed_10m_max")          # km/h
        wind_mean  = val("wind_speed_10m_mean")         # km/h
        precip     = val("precipitation_sum")            # mm
        eto        = val("et0_fao_evapotranspiration")   # mm

        # Convert units to match CIMIS conventions
        sol_rad_wm2   = round(rad_mj * 1e6 / 86400, 2) if rad_mj is not None else None  # W/m²
        wind_spd_ms   = round(wind_mean / 3.6, 3)      if wind_mean is not None else None  # m/s
        wind_max_ms   = round(wind_max  / 3.6, 3)      if wind_max  is not None else None  # m/s
        # Wind run (km/day) approximated from mean speed
        wind_run_km   = round(wind_spd_ms * 86.4, 2)   if wind_spd_ms is not None else None

        records.append({
            "date":            date_str,
            "source":          "Open-Meteo (Clear Lake coords)",
            "station_id":      "OM-CL",
            "tmp_avg_c":       tmp_avg,
            "tmp_max_c":       tmp_max,
            "tmp_min_c":       tmp_min,
            "rh_avg_pct":      rh_avg,
            "sol_rad_avg_wm2": sol_rad_wm2,
            "wind_spd_avg_ms": wind_spd_ms,
            "wind_max_ms":     wind_max_ms,
            "wind_run_km":     wind_run_km,
            "precip_mm":       precip,
            "eto_mm":          eto,
        })
    return records


# ── CIMIS fetch (fallback) ────────────────────────────────────────────────────

def fetch_cimis_year(year, app_key):
    start = f"{year}-01-01"
    end   = f"{year}-12-31" if year < date.today().year else date.today().isoformat()
    params = urllib.parse.urlencode({
        "appKey":        app_key,
        "targets":       str(CIMIS_STATION_ID),
        "startDate":     start,
        "endDate":       end,
        "dataItems":     CIMIS_ITEMS,
        "unitOfMeasure": "M",
    })
    req = urllib.request.Request(
        f"{CIMIS_BASE}?{params}",
        headers={"Accept": "application/json", "User-Agent": "HABaware/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = json.loads(r.read())
    if "html" in str(raw)[:50].lower():
        raise RuntimeError("CIMIS API returned HTML (WAF block)")
    return raw

def parse_cimis(raw):
    item_map = {
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
    records = []
    for provider in raw.get("Data", {}).get("Providers", []):
        for record in provider.get("Records", []):
            row = {"date": record["Date"][:10], "source": f"CIMIS Station {CIMIS_STATION_ID}", "station_id": str(CIMIS_STATION_ID)}
            for item in record.get("DayData", []):
                key = item_map.get(item.get("Item", ""))
                if key:
                    v = item.get("Value")
                    try:
                        row[key] = float(v) if v not in (None, "", " ") else None
                    except (ValueError, TypeError):
                        row[key] = None
            # Ensure all expected keys present
            for k in item_map.values():
                row.setdefault(k, None)
            row.setdefault("tmp_min_c", None)
            row.setdefault("wind_max_ms", None)
            records.append(row)
    return records


# ── Rolling features ──────────────────────────────────────────────────────────

def add_rolling_features(records):
    buf = deque(maxlen=14)
    enhanced = []
    for r in records:
        buf.append(r)
        w7  = list(buf)[-7:]
        w14 = list(buf)

        def mean(window, key):
            vals = [x[key] for x in window if x.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def total(window, key):
            vals = [x[key] for x in window if x.get(key) is not None]
            return round(sum(vals), 4) if vals else None

        r_out = dict(r)
        r_out["tmp_avg_7d"]    = mean(w7,  "tmp_avg_c")
        r_out["tmp_avg_14d"]   = mean(w14, "tmp_avg_c")
        r_out["tmp_max_7d"]    = mean(w7,  "tmp_max_c")
        r_out["wind_spd_7d"]   = mean(w7,  "wind_spd_avg_ms")
        r_out["wind_spd_14d"]  = mean(w14, "wind_spd_avg_ms")
        r_out["wind_run_7d"]   = total(w7, "wind_run_km")
        r_out["sol_rad_7d"]    = mean(w7,  "sol_rad_avg_wm2")
        r_out["sol_rad_14d"]   = mean(w14, "sol_rad_avg_wm2")
        r_out["precip_7d"]     = total(w7,  "precip_mm")
        r_out["precip_14d"]    = total(w14, "precip_mm")
        r_out["eto_7d"]        = total(w7,  "eto_mm")
        r_out["rh_avg_7d"]     = mean(w7,  "rh_avg_pct")
        r_out["calm_days_7d"]  = sum(
            1 for x in w7
            if x.get("wind_spd_avg_ms") is not None and x["wind_spd_avg_ms"] < 2.0
        )
        enhanced.append(r_out)
    return enhanced


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_all_years(start_year=2017):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    cimis_key = os.environ.get("CIMIS_APP_KEY", "").strip()

    for year in range(start_year, date.today().year + 1):
        cache_path = RAW_DIR / f"openmeteo_clear_lake_{year}.json"

        if cache_path.exists() and year < date.today().year:
            with open(cache_path) as f:
                raw = json.load(f)
            records = parse_openmeteo(raw)
            print(f"  {year}: {len(records)} records (cached)")
        else:
            # Try Open-Meteo first
            try:
                print(f"  {year}: fetching Open-Meteo...", end=" ", flush=True)
                raw = fetch_openmeteo_year(year)
                with open(cache_path, "w") as f:
                    json.dump(raw, f)
                records = parse_openmeteo(raw)
                print(f"{len(records)} records ✓")
                time.sleep(0.3)
            except Exception as e:
                print(f"Open-Meteo failed ({e})")
                # Fallback to CIMIS if key available
                if cimis_key:
                    try:
                        print(f"  {year}: falling back to CIMIS...", end=" ", flush=True)
                        raw_c = fetch_cimis_year(year, cimis_key)
                        records = parse_cimis(raw_c)
                        print(f"{len(records)} records ✓")
                    except Exception as e2:
                        print(f"CIMIS also failed ({e2}) — skipping {year}")
                        records = []
                else:
                    records = []
        all_records.extend(records)

    return all_records


def main():
    print(f"Fetching weather at Clear Lake ({CL_LAT}°N, {abs(CL_LON)}°W)")
    print(f"Source: Open-Meteo archive API (CIMIS-equivalent variables)\n")

    records = fetch_all_years(start_year=2017)
    print(f"\nTotal raw records: {len(records)}")

    records = add_rolling_features(records)

    valid_tmp = [r["tmp_avg_c"] for r in records if r.get("tmp_avg_c") is not None]
    valid_wind = [r["wind_spd_avg_ms"] for r in records if r.get("wind_spd_avg_ms") is not None]
    calm = [r["calm_days_7d"] for r in records if r.get("calm_days_7d") is not None]
    print(f"Temp:  {min(valid_tmp):.1f}–{max(valid_tmp):.1f} °C  (mean {sum(valid_tmp)/len(valid_tmp):.1f})")
    print(f"Wind:  {min(valid_wind):.2f}–{max(valid_wind):.2f} m/s (mean {sum(valid_wind)/len(valid_wind):.2f})")
    print(f"Calm days (7d window): mean {sum(calm)/len(calm):.1f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"\nWritten {len(records)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
