"""
SWAMP FHAB observation data loader for the HABaware Clear Lake pilot.

Reads the four CA Open Data CSVs and produces:
  1. clear_lake_observations.csv  — bloom reports with advisory levels, indexed by date
  2. clear_lake_toxins.csv        — lab results (microcystins, cylindrospermopsin, saxitoxin)
  3. current_status.json          — latest advisory status for use in the advisory API

Source: https://data.ca.gov/dataset/surface-water-freshwater-harmful-algal-blooms
Downloaded: 2026-06-02 (data freeze notice: no updates through 2026-06-25)
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

RAW_DIR  = Path(__file__).parent.parent / "data/raw/swamp_fhab"
OUT_DIR  = Path(__file__).parent.parent / "data/processed"

ADVISORY_RANK = {
    "Danger":           4,
    "Warning":          3,
    "Caution":          2,
    "General awareness": 1,
    "None":             0,
    "":                 0,
}


def load_csv(filename):
    with open(RAW_DIR / filename, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def is_clear_lake(row):
    text = " ".join([
        row.get("Water_Body_Name", ""),
        row.get("Case_Water_Body_Name", ""),
        row.get("Official_Water_Body_Name", ""),
    ]).lower()
    return "clear lake" in text and "canada" not in text

def parse_date(s):
    if not s or not s.strip():
        return None
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None

def advisory_rank(advisory_str):
    for label, rank in ADVISORY_RANK.items():
        if label.lower() in advisory_str.lower():
            return rank, label
    return 0, "None"


def build_observations():
    """
    Combine bloom_reports + hab_responses to produce a daily observation table
    with advisory level, bloom size, texture, and linked toxin flag.
    """
    reports    = [r for r in load_csv("bloom_reports.csv")  if is_clear_lake(r)]
    responses  = [r for r in load_csv("hab_responses.csv")  if is_clear_lake(r)]

    # Index responses by Bloom_Report_ID for advisory lookup
    resp_by_report = defaultdict(list)
    for r in responses:
        if r.get("Bloom_Report_ID"):
            resp_by_report[r["Bloom_Report_ID"]].append(r)

    rows = []
    for r in reports:
        obs_date = parse_date(r.get("Observation_Date", ""))
        if not obs_date:
            obs_date = parse_date(r.get("Bloom_Date_Created", ""))
        if not obs_date:
            continue

        # Advisory level: prefer linked response advisory, fall back to reported
        adv_text = r.get("Reported_Advisory_Types", "") or ""
        for resp in resp_by_report.get(r["Bloom_Report_ID"], []):
            if resp.get("Advisory_Detail"):
                adv_text = resp["Advisory_Detail"]
                break

        adv_rank, adv_label = advisory_rank(adv_text)

        # Active advisory: no end date = still open
        adv_active = False
        for resp in resp_by_report.get(r["Bloom_Report_ID"], []):
            adv_start = parse_date(resp.get("Advisory_Start_Date", ""))
            adv_end   = parse_date(resp.get("Advisory_End_Date", ""))
            if adv_start and not adv_end:
                adv_active = True
                break

        rows.append({
            "date":              obs_date.strftime("%Y-%m-%d"),
            "bloom_report_id":   r["Bloom_Report_ID"],
            "case_id":           r.get("Case_ID", ""),
            "advisory_level":    adv_label,
            "advisory_rank":     adv_rank,
            "advisory_active":   int(adv_active),
            "advisory_detail":   r.get("AdvisoryDetail", "") or r.get("Advisory_Detail_Description", ""),
            "bloom_size":        r.get("Bloom_Size", ""),
            "bloom_texture":     r.get("Bloom_Texture", ""),
            "bloom_location":    r.get("Bloom_Location", ""),
            "lat":               r.get("Bloom_Latitude", ""),
            "lon":               r.get("Bloom Longitude") or r.get("Bloom_Longitude", ""),
            "lab_data_linked":   r.get("Lab_Data_Linked_to_Bloom", "0"),
            "has_pictures":      r.get("Has_Pictures", ""),
            "county":            r.get("County", ""),
        })

    rows.sort(key=lambda x: x["date"])
    return rows


def build_toxins():
    results = [r for r in load_csv("hab_results.csv") if is_clear_lake(r)]
    target_analytes = {"Microcystins total", "Cylindrospermopsin", "Saxitoxin"}

    rows = []
    for r in results:
        if r.get("Analyte") not in target_analytes:
            continue
        sample_date = parse_date(r.get("Sample_Date", ""))
        if not sample_date:
            continue

        val = r.get("Measurement_Value", "").strip()
        try:
            val_float = float(val) if val else None
        except ValueError:
            val_float = None

        rows.append({
            "date":          sample_date.strftime("%Y-%m-%d"),
            "sample_id":     r.get("Sample_ID", ""),
            "bloom_report_id": r.get("Bloom_Report_ID", ""),
            "analyte":       r["Analyte"],
            "value":         val_float,
            "unit":          r.get("Measurement_Unit", ""),
            "sample_location": r.get("Sample_Location", ""),
            "lat":           r.get("Latitude", ""),
            "lon":           r.get("Longitude", ""),
            "method":        r.get("Method", ""),
            "data_type":     r.get("Data_Type", ""),
        })

    rows.sort(key=lambda x: x["date"])
    return rows


def build_current_status(obs_rows, toxin_rows):
    """
    Produce a concise current-status object for injection into the advisory prompt.
    """
    # Most recent bloom report
    latest_obs = obs_rows[-1] if obs_rows else {}

    # Any open (no-end-date) advisories
    responses = [r for r in load_csv("hab_responses.csv") if is_clear_lake(r)]
    open_advisories = []
    for r in responses:
        if r.get("Advisory_Start_Date") and not r.get("Advisory_End_Date"):
            open_advisories.append({
                "start":   r["Advisory_Start_Date"],
                "detail":  r.get("Advisory_Detail", ""),
                "type":    r.get("Response_Type", ""),
                "case_id": r.get("Case_ID", ""),
            })
    # Deduplicate by detail
    seen = set()
    unique_open = []
    for a in open_advisories:
        key = (a["detail"], a["case_id"])
        if key not in seen:
            seen.add(key)
            unique_open.append(a)

    # Open case status
    cases = [r for r in load_csv("hab_cases.csv") if is_clear_lake(r)]
    open_case = next((c for c in cases if c.get("Case_Status") == "Ongoing"), None)

    # Latest toxin values
    latest_toxins = {}
    for row in reversed(toxin_rows):
        if row["analyte"] not in latest_toxins and row["value"] is not None:
            latest_toxins[row["analyte"]] = {
                "date":  row["date"],
                "value": row["value"],
                "unit":  row["unit"],
            }

    # Advisory level trend: last 30 obs
    recent_30 = obs_rows[-30:] if len(obs_rows) >= 30 else obs_rows
    max_recent_rank = max((r["advisory_rank"] for r in recent_30), default=0)
    rank_to_label = {v: k for k, v in ADVISORY_RANK.items() if k}
    max_recent_label = rank_to_label.get(max_recent_rank, "None")

    status = {
        "waterbody":          "Clear Lake",
        "wid":                33,
        "data_as_of":         "2026-06-02",
        "open_case": {
            "case_id":   open_case["Case_ID"] if open_case else None,
            "year":      open_case["Case_Year"] if open_case else None,
            "status":    open_case["Case_Status"] if open_case else "No open case",
        },
        "open_advisories":    unique_open[:5],
        "latest_bloom_report": {
            "date":           latest_obs.get("date"),
            "advisory_level": latest_obs.get("advisory_level"),
            "bloom_size":     latest_obs.get("bloom_size"),
            "bloom_texture":  latest_obs.get("bloom_texture"),
            "advisory_detail": latest_obs.get("advisory_detail"),
        },
        "latest_toxin_results": latest_toxins,
        "peak_advisory_last_30_reports": max_recent_label,
        "total_clear_lake_bloom_reports": len(obs_rows),
    }
    return status


def main():
    print("Building Clear Lake SWAMP observation dataset...")
    obs   = build_observations()
    toxins = build_toxins()
    status = build_current_status(obs, toxins)

    # Write observations CSV
    if obs:
        obs_out = OUT_DIR / "clear_lake_observations.csv"
        with open(obs_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=obs[0].keys())
            w.writeheader()
            w.writerows(obs)
        print(f"  Observations: {len(obs)} rows → {obs_out}")

    # Write toxins CSV
    if toxins:
        tox_out = OUT_DIR / "clear_lake_toxins.csv"
        with open(tox_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=toxins[0].keys())
            w.writeheader()
            w.writerows(toxins)
        print(f"  Toxins: {len(toxins)} rows → {tox_out}")

    # Write current status JSON
    status_out = OUT_DIR / "clear_lake_current_status.json"
    with open(status_out, "w") as f:
        json.dump(status, f, indent=2)
    print(f"  Current status → {status_out}")

    print("\n=== Current Status ===")
    print(json.dumps(status, indent=2))

    return obs, toxins, status


if __name__ == "__main__":
    main()
