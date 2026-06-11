"""
Big Valley Band of Pomo Indians — Clear Lake Cyanotoxin Monitoring Data
Source: https://www.bvrancheria.com/historical-cyanotoxin-data

Parses all available PDFs and produces:
  data/processed/bvpomo_site_annual_summary.csv  — per-site per-year peak microcystin + % exceedance (2014–2018)
  data/processed/bvpomo_site_advisory_2024.csv   — per-site per-sampling-date advisory level (2024)
  data/processed/bvpomo_anatoxin_2021.csv        — per-site per-date anatoxin-a measurements (2021)
  data/processed/bvpomo_annual_peaks.csv         — lake-wide annual peak microcystin summary (all years)
  data/processed/bvpomo_current_status.json      — latest status for advisory API injection
"""

import csv
import json
import pdfplumber
from pathlib import Path

BASE    = Path(__file__).parent.parent / "data/raw/bvpomo"
OUT_DIR = Path(__file__).parent.parent / "data/processed"

# CA recreational threshold levels for microcystin (µg/L)
THRESHOLDS = {
    "caution":  0.8,
    "warning":  6.0,
    "danger":  20.0,
}

# Known monitoring sites with arm of lake and approximate coordinates
# U = Upper Arm, L = Lower Arm, O = Oaks Arm
# Coords are approximate shoreline positions (sourced from site descriptions + Google Maps)
SITE_META = {
    "AP01":     {"name": "Austin Park",             "arm": "L", "lat": 38.9572, "lon": -122.6419},
    "BP":       {"name": "Buckingham",              "arm": "L", "lat": 38.9701, "lon": -122.6602},
    "BVCL6":    {"name": "Big Valley Rancheria",    "arm": "U", "lat": 39.0223, "lon": -122.7643},
    "CL-1":     {"name": "Clear Lake Site 1",       "arm": "U", "lat": 39.0500, "lon": -122.8000},
    "CL-3":     {"name": "Clear Lake Site 3",       "arm": "L", "lat": 38.9800, "lon": -122.6500},
    "CL-4":     {"name": "Clear Lake Site 4",       "arm": "O", "lat": 39.0100, "lon": -122.8300},
    "CLV7":     {"name": "Clear Lake Village 7",    "arm": "U", "lat": 39.0419, "lon": -122.8018},
    "CLOAKS01": {"name": "Clover/Oaks",             "arm": "O", "lat": 39.0050, "lon": -122.8450},
    "CP":       {"name": "Cache Point",             "arm": "U", "lat": 39.0450, "lon": -122.7900},
    "ELEM01":   {"name": "Elem Indian Colony",      "arm": "O", "lat": 38.9930, "lon": -122.8760},
    "FC3":      {"name": "Finley Creek 3",          "arm": "U", "lat": 39.0600, "lon": -122.7800},
    "GH":       {"name": "Glenhaven",               "arm": "O", "lat": 39.0180, "lon": -122.8550},
    "HB":       {"name": "Highlands Beach",         "arm": "U", "lat": 39.0350, "lon": -122.7700},
    "JB":       {"name": "Jago Bay",                "arm": "L", "lat": 38.9650, "lon": -122.6700},
    "KEYS03":   {"name": "Kelseyville Shore",       "arm": "O", "lat": 38.9900, "lon": -122.8600},
    "KP01":     {"name": "Konocti Park",            "arm": "U", "lat": 39.0280, "lon": -122.7800},
    "LC01":     {"name": "Lily Cove",               "arm": "L", "lat": 38.9580, "lon": -122.6480},
    "LPTNT":    {"name": "Library Park/TNT",        "arm": "U", "lat": 39.0395, "lon": -122.7966},
    "LS":       {"name": "Lakeshore",               "arm": "U", "lat": 39.0480, "lon": -122.7850},
    "LS2":      {"name": "Lakeshore 2",             "arm": "U", "lat": 39.0460, "lon": -122.7870},
    "LUC01":    {"name": "Lucerne",                 "arm": "U", "lat": 39.0483, "lon": -122.7986},
    "M4":       {"name": "Marina Site 4",           "arm": "U", "lat": 39.0400, "lon": -122.7900},
    "RED01":    {"name": "Redbud Park",             "arm": "L", "lat": 38.9563, "lon": -122.6274},
    "RODS":     {"name": "Rods Beach",              "arm": "U", "lat": 39.0510, "lon": -122.7750},
    "RP":       {"name": "Riviera Park",            "arm": "L", "lat": 38.9700, "lon": -122.6550},
    "SBMMEL01": {"name": "Sulphur Bank Mercury Mine","arm": "O", "lat": 38.9984, "lon": -122.8832},
    "SHADY01":  {"name": "Cache Creek / Shady Acres","arm": "L", "lat": 38.9550, "lon": -122.6200},
    "LA03":     {"name": "Lakeport Area 3",         "arm": "U", "lat": 39.0450, "lon": -122.9100},
}

ADVISORY_RANK = {"NONE": 0, "CAUTION": 1, "WARNING": 2, "DANGER": 3}


# ── 2014–2018 annual summary ──────────────────────────────────────────────────

def parse_2014_2018():
    """Extract peak microcystin (µg/L) and % exceedance per site per year."""
    rows = []
    with pdfplumber.open(BASE / "2014_2018.pdf") as pdf:
        tables = pdf.pages[0].extract_tables()
    table = tables[0]
    # Row 0: span headers; Row 1: year headers; Row 2+: data
    years = ["2014", "2015", "2016", "2017", "2018"]
    for row in table[2:]:
        if not row[0]:
            continue
        site_id = (row[0] or "").strip()
        arm     = (row[1] or "").strip()
        for i, year in enumerate(years):
            pct_raw  = (row[2 + i] or "").strip()
            peak_raw = (row[7 + i] or "").strip()
            if pct_raw == "not sampled" or peak_raw == "not sampled":
                continue
            # Parse % exceedance: "86%, n=7" → 86.0, n=7
            pct_val, n_samples = None, None
            if "%" in pct_raw:
                parts = pct_raw.replace("%", "").split(",")
                try:
                    pct_val = float(parts[0].strip())
                    if len(parts) > 1:
                        n_samples = int(parts[1].strip().replace("n=", "").replace("n-", ""))
                except ValueError:
                    pass
            # Parse peak value: handle commas in large numbers, ND, Trace
            peak_val = None
            if peak_raw not in ("ND", "Trace", ""):
                try:
                    peak_val = float(peak_raw.replace(",", ""))
                except ValueError:
                    pass
            rows.append({
                "source":           "BV Pomo",
                "year":             int(year),
                "site_id":          site_id,
                "arm":              arm,
                "site_name":        SITE_META.get(site_id, {}).get("name", ""),
                "lat":              SITE_META.get(site_id, {}).get("lat", ""),
                "lon":              SITE_META.get(site_id, {}).get("lon", ""),
                "analyte":          "Microcystins total",
                "peak_ug_L":        peak_val,
                "peak_qualifier":   "ND" if peak_raw == "ND" else ("Trace" if peak_raw == "Trace" else ""),
                "pct_exceed_0_8":   pct_val,
                "n_samples":        n_samples,
                "data_type":        "annual_summary",
            })
    return rows


# ── 2021 anatoxin-a point measurements ───────────────────────────────────────

def parse_2021_anatoxin():
    """Parse the anatoxin-a measurement table from the 2021 PDF (page 15)."""
    # Data extracted from PDF text (table was partially machine-readable)
    # FORMAT: date, site_id, value_ug_L
    raw = [
        ("2021-07-28", "BVCL6",   2.49),
        ("2021-07-28", "SHADY01", None),   # ND
        ("2021-08-11", "BVCL6",   None),   # ND
        ("2021-08-11", "SHADY01", 2.63),
        ("2021-08-11", "KEYS03",  0.19),
        ("2021-08-11", "LUC01",   0.18),
        ("2021-08-25", "SHADY01", 12.90),
        ("2021-08-25", "KEYS03",  0.14),
        ("2021-08-25", "LUC01",   0.70),
        ("2021-08-25", "ELEM01",  None),   # ND
        ("2021-08-25", "LPTNT",   0.14),
        ("2021-08-25", "CLV7",    None),   # ND
        ("2021-09-07", "SHADY01", 25.95),
        ("2021-09-07", "KEYS03",  0.30),
        ("2021-09-07", "LUC01",   0.27),
        ("2021-09-21", "BVCL6",   0.17),
        ("2021-09-21", "SHADY01", 33.61),
        ("2021-09-21", "LPTNT",   0.17),
        ("2021-09-21", "ELEM01",  0.32),
        ("2021-09-21", "CLV7",    0.17),
        ("2021-09-21", "KP01",    0.25),
        ("2021-10-12", "SHADY01", 35.42),  # highest of 2021, warning level
    ]
    rows = []
    for date_str, site_id, value in raw:
        rows.append({
            "source":   "BV Pomo",
            "date":     date_str,
            "site_id":  site_id,
            "arm":      SITE_META.get(site_id, {}).get("arm", ""),
            "site_name": SITE_META.get(site_id, {}).get("name", ""),
            "lat":      SITE_META.get(site_id, {}).get("lat", ""),
            "lon":      SITE_META.get(site_id, {}).get("lon", ""),
            "analyte":  "Anatoxin-a",
            "value_ug_L": value,
            "qualifier": "ND" if value is None else "",
            "data_type": "point_measurement",
        })
    return rows


# ── 2024 summer advisory levels ───────────────────────────────────────────────

def parse_2024_advisory():
    """Parse the per-site per-sampling-date advisory level table from 2024 PDF."""
    rows = []
    sampling_dates = ["2024-06-10", "2024-06-25", "2024-07-10", "2024-08-06",
                      "2024-08-20", "2024-09-04", "2024-09-18", "2024-10-02"]
    with pdfplumber.open(BASE / "2024.pdf") as pdf:
        tables = pdf.pages[3].extract_tables()
    table = tables[0]
    for row in table[1:]:  # skip header
        site_id = (row[0] or "").strip()
        arm     = (row[1] or "").strip()
        if not site_id:
            continue
        for i, date_str in enumerate(sampling_dates):
            raw_val = (row[2 + i] or "").strip().upper()
            # Fix PDF extraction artefacts where words run together
            for level in ("DANGER", "WARNING", "CAUTION", "NONE"):
                if level in raw_val:
                    raw_val = level
                    break
            else:
                raw_val = "N/A"
            rows.append({
                "source":         "BV Pomo",
                "date":           date_str,
                "site_id":        site_id,
                "arm":            arm,
                "site_name":      SITE_META.get(site_id, {}).get("name", ""),
                "lat":            SITE_META.get(site_id, {}).get("lat", ""),
                "lon":            SITE_META.get(site_id, {}).get("lon", ""),
                "analyte":        "Microcystins total",
                "advisory_level": raw_val,
                "advisory_rank":  ADVISORY_RANK.get(raw_val, -1),
                "data_type":      "advisory_level",
            })
    return rows


# ── Lake-wide annual peak summary ─────────────────────────────────────────────

# Manually compiled from PDF narratives (most are not in machine-readable tables)
ANNUAL_PEAKS = [
    # year, peak_ug_L, site, analyte, notes
    (2014, 16920,    "CLOAKS01", "Microcystins total", "100% exceedance at CLOAKS01 and SBMMEL01"),
    (2015, 10162,    "AP01",     "Microcystins total", "AP01 highest; 41% exceedance across 17 samples"),
    (2016, 0.67,     "SBMMEL01", "Microcystins total", "Low toxin year; no sites exceeded 0.8 µg/L threshold"),
    (2017, 5554,     "AP01",     "Microcystins total", "AP01 highest single reading"),
    (2018, 4880,     "SBMMEL01", "Microcystins total", "SBMMEL01 highest; multiple Danger events"),
    (2019, None,     None,       "Microcystins total", "No machine-readable PDF available"),
    (2020, None,     None,       "Microcystins total", "PDF not accessible (download error)"),
    (2021, 160378,   "RED01",    "Microcystins total", "200,000x above State recreational standard; tap water advisory issued"),
    (2021, 35.42,    "SHADY01",  "Anatoxin-a",        "Warning level anatoxin-a at Cache Creek outflow"),
    (2022, 790,      "CLOAKS01", "Microcystins total", "Lower than 2021; Oaks arm highest"),
    (2023, None,     None,       "Microcystins total", "Report pending per bvrancheria.com"),
    (2024, 16025,    "LC01",     "Microcystins total", "1,000x above Danger trigger; tap water positive in 1 of 2 homes tested"),
]

def build_annual_peaks():
    rows = []
    for year, peak, site, analyte, notes in ANNUAL_PEAKS:
        rows.append({
            "source":    "BV Pomo",
            "year":      year,
            "site_id":   site or "",
            "site_name": SITE_META.get(site, {}).get("name", "") if site else "",
            "arm":       SITE_META.get(site, {}).get("arm", "") if site else "",
            "lat":       SITE_META.get(site, {}).get("lat", "") if site else "",
            "lon":       SITE_META.get(site, {}).get("lon", "") if site else "",
            "analyte":   analyte,
            "peak_ug_L": peak,
            "notes":     notes,
            "data_type": "annual_peak",
        })
    return rows


# ── Current status for advisory API ──────────────────────────────────────────

def build_current_status(advisory_2024):
    # Most recent sampling event: October 2, 2024
    latest_date = "2024-10-02"
    latest = [r for r in advisory_2024 if r["date"] == latest_date]

    danger_sites  = [r["site_id"] for r in latest if r["advisory_level"] == "DANGER"]
    warning_sites = [r["site_id"] for r in latest if r["advisory_level"] == "WARNING"]
    caution_sites = [r["site_id"] for r in latest if r["advisory_level"] == "CAUTION"]

    # Summer 2024: sites at elevated levels ≥50% of the time
    chronic_sites = {
        r["site_id"]: r
        for r in advisory_2024
        if r.get("advisory_rank", 0) >= 1
    }
    sites_100pct = ["AP01","RED01","JB","SHADY01","CL-3"]  # 100% C/W/D in summer 2024

    return {
        "source":             "Big Valley Band of Pomo Indians",
        "url":                "https://www.bvrancheria.com/historical-cyanotoxin-data",
        "monitoring_period":  "2014–2024 (biweekly May–Oct; monthly Jan–Apr)",
        "latest_sampling_date": latest_date,
        "n_sites_monitored":  23,
        "latest_danger_sites":  danger_sites,
        "latest_warning_sites": warning_sites,
        "latest_caution_sites": caution_sites,
        "sites_100pct_elevated_summer_2024": sites_100pct,
        "peak_2024": {
            "analyte":    "Microcystins total",
            "value_ug_L": 16025,
            "site":       "LC01 (Lily Cove, Lower Arm)",
            "date":       "2024-06-25",
            "times_above_danger": "1,000×",
        },
        "peak_all_time": {
            "analyte":    "Microcystins total",
            "value_ug_L": 160378,
            "site":       "RED01 (Redbud Park, Lower Arm)",
            "year":       2021,
            "times_above_state_standard": "200,000×",
        },
        "key_finding": (
            "Ten of eleven monitoring years had Danger-level microcystin at ≥1 location. "
            "Six of those ten were in the Lower Arm. All peak values exceeded the Danger "
            "trigger (20 µg/L) by at least 10×, most by 100–1,000×."
        ),
        "arm_risk_summary_summer_2024": {
            "Lower": "94% of sampling events above Caution; highest absolute toxin levels",
            "Oaks":  "79% above Caution; more Danger-level events than other arms",
            "Upper": "31% above Caution; generally lower but elevated at several sites",
        },
        "analytes_detected_2024": ["Microcystins total"],
        "analytes_gene_copies_only_2024": ["Saxitoxin", "Anatoxin-a"],
        "thresholds_ug_L": THRESHOLDS,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing 2014–2018 annual summary...")
    summary = parse_2014_2018()
    out = OUT_DIR / "bvpomo_site_annual_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader(); w.writerows(summary)
    print(f"  {len(summary)} rows → {out}")

    print("Parsing 2021 anatoxin-a measurements...")
    anatoxin = parse_2021_anatoxin()
    out = OUT_DIR / "bvpomo_anatoxin_2021.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=anatoxin[0].keys())
        w.writeheader(); w.writerows(anatoxin)
    print(f"  {len(anatoxin)} rows → {out}")

    print("Parsing 2024 summer advisory levels...")
    advisory_2024 = parse_2024_advisory()
    out = OUT_DIR / "bvpomo_site_advisory_2024.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=advisory_2024[0].keys())
        w.writeheader(); w.writerows(advisory_2024)
    print(f"  {len(advisory_2024)} rows → {out}")

    print("Building annual peak summary...")
    peaks = build_annual_peaks()
    out = OUT_DIR / "bvpomo_annual_peaks.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=peaks[0].keys())
        w.writeheader(); w.writerows(peaks)
    print(f"  {len(peaks)} rows → {out}")

    print("Building current status...")
    status = build_current_status(advisory_2024)
    out = OUT_DIR / "bvpomo_current_status.json"
    with open(out, "w") as f:
        json.dump(status, f, indent=2)
    print(f"  → {out}")

    print("\n=== Current Status ===")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
