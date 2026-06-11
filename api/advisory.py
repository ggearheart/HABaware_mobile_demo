"""
HABaware Advisory API — Clear Lake pilot
Fetches live cyanoindex from SFEI FHAB, runs the LightGBM forecast,
then calls the Claude API to generate a plain-language risk advisory.

Usage:
    python advisory.py --lat 39.03 --lon -122.78 --date 2025-09-15 --activity swimming
"""

import argparse
import json
import math
import os
import pickle
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

SWAMP_STATUS_PATH  = Path(__file__).parent.parent / "data/processed/clear_lake_current_status.json"
SWAMP_OBS_PATH     = Path(__file__).parent.parent / "data/processed/clear_lake_observations.csv"
BVPOMO_STATUS_PATH = Path(__file__).parent.parent / "data/processed/bvpomo_current_status.json"
BVPOMO_PEAKS_PATH  = Path(__file__).parent.parent / "data/processed/bvpomo_annual_peaks.csv"
CIMIS_WEATHER_CSV  = Path(__file__).parent.parent / "data/processed/cimis_clear_lake_weather.csv"

CIMIS_STATION_ID   = 106   # kept for legacy reference; live data now comes from Open-Meteo

# Open-Meteo forecast API — no key required, queried at Clear Lake coordinates
OM_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
OM_VARIABLES     = ("temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
                    "relative_humidity_2m_mean,shortwave_radiation_sum,"
                    "wind_speed_10m_mean,wind_speed_10m_max,"
                    "precipitation_sum,et0_fao_evapotranspiration")
CL_LAT, CL_LON   = 39.0300, -122.780

FHAB_BASE   = "https://fhab-api.sfei.org"
CLEAR_LAKE_WID = 33
MODEL_PATH  = Path(__file__).parent.parent / "ml/model.pkl"

ACTIVITY_GUIDANCE = {
    "swimming":   "direct full-body water contact",
    "kayaking":   "paddling with splash exposure",
    "fishing":    "shoreline and hand contact with water",
    "dog_walking": "pet contact with water and shoreline",
    "birdwatching": "no water contact expected",
}


# ── FHAB API helpers ──────────────────────────────────────────────────────────

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "HABaware/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def get_recent_cyanoindex(wid, days=30):
    """Return the last `days` daily records of 10-day-max cyanoindex."""
    end   = date.today()
    start = end - timedelta(days=days)
    url = (f"{FHAB_BASE}/cyano/10daymax/{wid}"
           f"/{start.isoformat()}/{end.isoformat()}/ci_modified/json")
    return fetch_json(url)

def latest_valid(records, baseline=0.9972436372799999):
    """Return the most recent record that has real bloom signal."""
    for r in reversed(records):
        if r["max"] > baseline:
            return r
    return records[-1] if records else None


# ── Feature engineering (mirrors ml/fetch_data.py) ───────────────────────────

def build_features_from_records(records):
    """Build the 15-column feature vector expected by the model."""
    baseline = 0.9972436372799999

    def has_signal(r):
        return r["pixel_count"] > 0 and r["max"] > baseline

    def ci(r, field):
        return r[field] if has_signal(r) else 0.0

    valid = [r for r in records if has_signal(r)]
    recent = records[-1]
    d = date.fromisoformat(recent["date"])
    doy = d.timetuple().tm_yday

    window = lambda n: [ci(r, "mean") for r in records[-n:] if has_signal(r)]
    peaks  = lambda n: [ci(r, "max")  for r in records[-n:] if has_signal(r)]

    w7,  w14, w30 = window(7), window(14), window(30)
    p7,  p14, p30 = peaks(7),  peaks(14),  peaks(30)

    return [
        math.sin(2 * math.pi * doy / 365),
        math.cos(2 * math.pi * doy / 365),
        d.month,
        ci(recent, "mean"),
        ci(recent, "max"),
        ci(recent, "median"),
        ci(recent, "perc90"),
        int(has_signal(recent)),
        recent["pixel_count"],
        sum(w7)  / len(w7)  if w7  else 0,
        sum(w14) / len(w14) if w14 else 0,
        sum(w30) / len(w30) if w30 else 0,
        max(p7)  if p7  else 0,
        max(p14) if p14 else 0,
        max(p30) if p30 else 0,
    ]


# ── Risk tier lookup ──────────────────────────────────────────────────────────

def risk_tier(ci_value):
    tiers = [
        (0,   5,   "Low",       "No bloom signal detected. Standard precautions apply."),
        (5,   30,  "Moderate",  "Low-level bloom signal present. Sensitive individuals should exercise caution."),
        (30,  80,  "High",      "Active bloom signal. Avoid water contact. Keep pets away."),
        (80,  200, "Very High", "Dense bloom present. Do not enter the water."),
        (200, 999, "Danger",    "Severe bloom. Shoreline contact may be harmful. Follow posted advisories."),
    ]
    for lo, hi, label, msg in tiers:
        if lo <= ci_value < hi:
            return label, msg
    return "Danger", tiers[-1][3]


# ── SWAMP observation data ────────────────────────────────────────────────────

def load_swamp_status():
    """Load the pre-processed SWAMP current status, or return empty dict if missing."""
    if SWAMP_STATUS_PATH.exists():
        with open(SWAMP_STATUS_PATH) as f:
            return json.load(f)
    return {}

def load_recent_swamp_obs(n=10):
    """Return the n most recent Clear Lake bloom report rows as dicts."""
    import csv
    if not SWAMP_OBS_PATH.exists():
        return []
    with open(SWAMP_OBS_PATH) as f:
        rows = list(csv.DictReader(f))
    return rows[-n:] if len(rows) >= n else rows

def fetch_recent_weather(past_days=14):
    """Fetch recent + short-range forecast weather via Open-Meteo at Clear Lake coordinates."""
    params = urllib.parse.urlencode({
        "latitude":       CL_LAT,
        "longitude":      CL_LON,
        "past_days":      past_days,
        "forecast_days":  3,
        "daily":          OM_VARIABLES,
        "timezone":       "America/Los_Angeles",
    })
    try:
        req = urllib.request.Request(
            f"{OM_FORECAST_BASE}?{params}",
            headers={"User-Agent": "HABaware/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = json.loads(r.read())
    except Exception as e:
        print(f"  Open-Meteo fetch failed: {e}")
        return []

    daily = raw.get("daily", {})
    dates = daily.get("time", [])
    days_out = []
    for i, date_str in enumerate(dates):
        def val(key):
            v = daily.get(key, [None] * (i+1))[i]
            return float(v) if v is not None else None

        rad_mj   = val("shortwave_radiation_sum")
        wind_ms  = val("wind_speed_10m_mean")
        days_out.append({
            "date":            date_str,
            "tmp_avg_c":       val("temperature_2m_mean"),
            "tmp_max_c":       val("temperature_2m_max"),
            "tmp_min_c":       val("temperature_2m_min"),
            "rh_avg_pct":      val("relative_humidity_2m_mean"),
            "sol_rad_avg_wm2": round(rad_mj * 1e6 / 86400, 2) if rad_mj is not None else None,
            "wind_spd_avg_ms": round(wind_ms / 3.6, 3) if wind_ms is not None else None,
            "precip_mm":       val("precipitation_sum"),
            "eto_mm":          val("et0_fao_evapotranspiration"),
        })
    return days_out


def load_bvpomo_status():
    """Load BV Pomo current status and recent annual peaks."""
    import csv
    status = {}
    if BVPOMO_STATUS_PATH.exists():
        with open(BVPOMO_STATUS_PATH) as f:
            status = json.load(f)
    peaks = []
    if BVPOMO_PEAKS_PATH.exists():
        with open(BVPOMO_PEAKS_PATH) as f:
            peaks = list(csv.DictReader(f))
    return status, peaks


# ── GenAI advisory via Claude ─────────────────────────────────────────────────

def generate_advisory(lat, lon, visit_date, activity, ci_current, ci_forecast,
                       tier_label, tier_msg, recent_records, swamp_status, swamp_obs,
                       bvpomo_status, bvpomo_peaks, cimis_weather):
    """Call Claude to generate a plain-language risk advisory."""
    try:
        import anthropic
    except ImportError:
        return "[anthropic SDK not installed — run: pip install anthropic]"

    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ANTHROPIC_API_KEY not set]"

    client = anthropic.Anthropic(api_key=api_key)

    # Summarise recent trend for the prompt
    valid_recent = [r for r in recent_records if r["max"] > 0.9972]
    trend_lines = "\n".join(
        f"  {r['date']}: ci_max={r['max']:.1f}, ci_mean={r['mean']:.1f}"
        for r in valid_recent[-7:]
    )

    # Summarise SWAMP ground-truth data for the prompt
    open_case    = swamp_status.get("open_case", {})
    latest_obs   = swamp_status.get("latest_bloom_report", {})
    open_advs    = swamp_status.get("open_advisories", [])
    peak_adv     = swamp_status.get("peak_advisory_last_30_reports", "Unknown")
    toxins       = swamp_status.get("latest_toxin_results", {})

    swamp_block = f"""SWAMP FHAB ground-truth observation data (CA Water Boards, as of {swamp_status.get('data_as_of','unknown')}):
- Open case: Case {open_case.get('case_id','?')} (since {open_case.get('year','?')}, status: {open_case.get('status','?')})
- Latest field report: {latest_obs.get('date','?')} — "{latest_obs.get('advisory_detail','no detail')}" (size: {latest_obs.get('bloom_size','unknown') or 'not recorded'}, texture: {latest_obs.get('bloom_texture','unknown') or 'not recorded'})
- Peak advisory level (last 30 reports): {peak_adv}
- Open advisories: {len(open_advs)} on record with no end date"""

    if toxins:
        for analyte, data in toxins.items():
            swamp_block += f"\n- Latest {analyte}: {data['value']} {data['unit']} (sampled {data['date']})"
    else:
        swamp_block += "\n- Toxin lab results: not available in current dataset"

    # Recent field reports
    obs_lines = "\n".join(
        f"  {r['date']}: advisory={r['advisory_level'] or 'unspecified'}, "
        f"size={r['bloom_size'] or 'not recorded'}, detail={r['advisory_detail'] or 'none'}"
        for r in swamp_obs[-5:]
    )

    # BV Pomo block
    bvp = bvpomo_status
    bvpomo_block = ""
    if bvp:
        caution_sites = ", ".join(bvp.get("latest_caution_sites", [])[:5]) or "none"
        danger_sites  = ", ".join(bvp.get("latest_danger_sites",  [])[:5]) or "none"
        arm_summary   = bvp.get("arm_risk_summary_summer_2024", {})
        peak_2024     = bvp.get("peak_2024", {})
        peak_alltime  = bvp.get("peak_all_time", {})
        recent_peaks  = [p for p in bvpomo_peaks[-4:] if p.get("peak_ug_L")]

        bvpomo_block = f"""
Tribal monitoring data — Big Valley Band of Pomo Indians (source: bvrancheria.com, 2014–2024):
- Latest sampling: {bvp.get('latest_sampling_date','?')} across {bvp.get('n_sites_monitored','?')} shoreline sites
- Latest Danger sites: {danger_sites}
- Latest Caution sites: {caution_sites}
- Sites at elevated toxin levels 100% of summer 2024: {', '.join(bvp.get('sites_100pct_elevated_summer_2024',[]))}
- Arm risk (summer 2024): Lower={arm_summary.get('Lower','?')} | Oaks={arm_summary.get('Oaks','?')} | Upper={arm_summary.get('Upper','?')}
- Peak 2024: {peak_2024.get('value_ug_L','?')} µg/L microcystin at {peak_2024.get('site','?')} on {peak_2024.get('date','?')} ({peak_2024.get('times_above_danger','?')} above Danger trigger)
- All-time peak: {peak_alltime.get('value_ug_L','?')} µg/L microcystin at {peak_alltime.get('site','?')} in {peak_alltime.get('year','?')} ({peak_alltime.get('times_above_state_standard','?')} above state standard)
- Key finding: {bvp.get('key_finding','')}"""

        if recent_peaks:
            bvpomo_block += "\n- Recent annual lake-wide peaks (µg/L microcystin): " + \
                ", ".join(f"{p['year']}:{p['peak_ug_L']}" for p in recent_peaks)

    # CIMIS weather block
    cimis_block = ""
    if cimis_weather:
        recent_w = [w for w in cimis_weather if w.get("tmp_avg_c") is not None][-7:]
        if recent_w:
            tmp_vals  = [w["tmp_avg_c"]       for w in recent_w if w.get("tmp_avg_c") is not None]
            wind_vals = [w["wind_spd_avg_ms"]  for w in recent_w if w.get("wind_spd_avg_ms") is not None]
            rad_vals  = [w["sol_rad_avg_wm2"]  for w in recent_w if w.get("sol_rad_avg_wm2") is not None]
            prcp_vals = [w["precip_mm"]        for w in recent_w if w.get("precip_mm") is not None]
            calm_days = sum(1 for w in recent_w if w.get("wind_spd_avg_ms") is not None and w["wind_spd_avg_ms"] < 2.0)

            avg = lambda v: round(sum(v)/len(v), 1) if v else None
            cimis_block = f"""
Weather at Clear Lake (Open-Meteo), last 7 days:
- Avg air temp: {avg(tmp_vals)} °C  |  Avg wind speed: {avg(wind_vals)} m/s  |  Calm days (<2 m/s): {calm_days}/7
- Avg solar radiation: {avg(rad_vals)} W/m²  |  Total precip: {round(sum(prcp_vals),1) if prcp_vals else 'N/A'} mm
- Calm, warm, sunny conditions favor surface bloom accumulation."""

    system = """You are HABaware, a public health advisory assistant specializing in
harmful algal bloom (HAB) risk at California waterbodies. You communicate risk clearly,
accurately, and without either alarming or dismissing. You always ground your advice in
the provided data. You never invent toxin measurements or advisory levels not given to you.
Keep your advisory under 150 words."""

    user = f"""Generate a risk advisory for the following visit:

Location: Clear Lake, CA (lat={lat}, lon={lon})
Planned visit: {visit_date}
Activity: {activity} ({ACTIVITY_GUIDANCE.get(activity, activity)})

Satellite data (SFEI FHAB cyanoindex, ci_modified scale 0–999):
- Current ci_max: {ci_current:.1f}
- 7-day forecast ci_max: {ci_forecast:.1f}
- Risk tier: {tier_label}
- Tier message: {tier_msg}

Recent 7-day satellite trend:
{trend_lines}

{swamp_block}
{bvpomo_block}
{cimis_block}

Recent SWAMP field reports (most recent first):
{obs_lines}

Write a plain-language advisory addressed to the visitor. Include:
1. Overall risk level (cite both satellite and field data)
2. What this means practically for their specific activity
3. One specific, actionable recommendation
4. Remind them to check posted signs at the lake and the CA Water Boards advisory page."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


# ── Main ──────────────────────────────────────────────────────────────────────

def run_advisory(lat, lon, visit_date, activity="swimming"):
    print(f"\nFetching cyanoindex for Clear Lake (wid={CLEAR_LAKE_WID})...")
    records = get_recent_cyanoindex(CLEAR_LAKE_WID, days=45)
    print(f"  Retrieved {len(records)} records through {records[-1]['date']}")

    latest = latest_valid(records)
    ci_current = latest["max"] if latest else 0.0
    print(f"  Latest signal: {latest['date']} ci_max={ci_current:.1f}")

    print("Loading model and generating 7-day forecast...")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    features = build_features_from_records(records)
    import numpy as np
    ci_forecast = float(model.predict(np.array([features], dtype=np.float32))[0])
    ci_forecast = max(0.0, ci_forecast)
    print(f"  Forecast 7-day peak ci_max: {ci_forecast:.1f}")

    tier_label, tier_msg = risk_tier(ci_forecast)
    print(f"  Risk tier: {tier_label}")

    print("Loading SWAMP ground-truth data...")
    swamp_status = load_swamp_status()
    swamp_obs    = load_recent_swamp_obs(10)

    print("Fetching weather data (Open-Meteo, last 14 days + 3-day forecast)...")
    cimis_weather = fetch_recent_weather(past_days=14)
    if cimis_weather:
        valid_w = [w for w in cimis_weather if w.get("tmp_avg_c") is not None]
        print(f"  Retrieved {len(valid_w)} days with temperature data")

    print("Loading BV Pomo tribal monitoring data...")
    bvpomo_status, bvpomo_peaks = load_bvpomo_status()
    if bvpomo_status:
        print(f"  Latest sampling: {bvpomo_status.get('latest_sampling_date','?')}, "
              f"{len(bvpomo_status.get('latest_caution_sites',[]))} caution / "
              f"{len(bvpomo_status.get('latest_danger_sites',[]))} danger sites")
    if swamp_status:
        print(f"  Open case: {swamp_status.get('open_case',{}).get('case_id','?')} "
              f"({swamp_status.get('open_case',{}).get('status','?')})")
        print(f"  Peak advisory (last 30 reports): {swamp_status.get('peak_advisory_last_30_reports','?')}")
        print(f"  Latest field report: {swamp_status.get('latest_bloom_report',{}).get('date','?')}")

    print("Generating AI advisory...")
    advisory_text = generate_advisory(
        lat, lon, visit_date, activity,
        ci_current, ci_forecast, tier_label, tier_msg, records,
        swamp_status, swamp_obs, bvpomo_status, bvpomo_peaks, cimis_weather
    )

    result = {
        "location":     {"lat": lat, "lon": lon, "waterbody": "Clear Lake", "wid": CLEAR_LAKE_WID},
        "visit_date":   visit_date,
        "activity":     activity,
        "satellite": {
            "current_ci_max":    round(ci_current, 2),
            "latest_date":       latest["date"] if latest else None,
            "forecast_ci_max_7d": round(ci_forecast, 2),
        },
        "risk": {
            "tier":    tier_label,
            "message": tier_msg,
        },
        "weather": {
            "source":         "Open-Meteo (archive-api + forecast API, Clear Lake coords)",
            "days_retrieved": len([w for w in cimis_weather if w.get("tmp_avg_c") is not None]),
            "latest_date":    max((w["date"] for w in cimis_weather if w.get("date")), default=None),
            "latest_tmp_avg_c": next((w["tmp_avg_c"] for w in reversed(cimis_weather) if w.get("tmp_avg_c") is not None), None),
            "latest_wind_spd_ms": next((w["wind_spd_avg_ms"] for w in reversed(cimis_weather) if w.get("wind_spd_avg_ms") is not None), None),
            "note": "Includes 3-day forecast; historical training data via CIMIS-equivalent variables",
        },
        "bvpomo_tribal_monitoring": {
            "source":                    "Big Valley Band of Pomo Indians",
            "latest_sampling_date":      bvpomo_status.get("latest_sampling_date"),
            "n_sites_monitored":         bvpomo_status.get("n_sites_monitored"),
            "latest_danger_sites":       bvpomo_status.get("latest_danger_sites"),
            "latest_caution_sites":      bvpomo_status.get("latest_caution_sites"),
            "peak_2024_ug_L":            bvpomo_status.get("peak_2024", {}).get("value_ug_L"),
            "peak_alltime_ug_L":         bvpomo_status.get("peak_all_time", {}).get("value_ug_L"),
            "key_finding":               bvpomo_status.get("key_finding"),
        } if bvpomo_status else {},
        "swamp_ground_truth": {
            "data_as_of":              swamp_status.get("data_as_of"),
            "open_case_id":            swamp_status.get("open_case", {}).get("case_id"),
            "open_case_status":        swamp_status.get("open_case", {}).get("status"),
            "peak_advisory_30_reports": swamp_status.get("peak_advisory_last_30_reports"),
            "latest_field_report_date": swamp_status.get("latest_bloom_report", {}).get("date"),
            "latest_field_detail":     swamp_status.get("latest_bloom_report", {}).get("advisory_detail"),
            "total_bloom_reports":     swamp_status.get("total_clear_lake_bloom_reports"),
            "latest_toxins":           swamp_status.get("latest_toxin_results", {}),
        },
        "advisory": advisory_text,
    }

    print("\n" + "="*60)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HABaware Clear Lake advisory")
    parser.add_argument("--lat",      type=float, default=39.03)
    parser.add_argument("--lon",      type=float, default=-122.78)
    parser.add_argument("--date",     default=str(date.today()))
    parser.add_argument("--activity", default="swimming",
                        choices=list(ACTIVITY_GUIDANCE.keys()))
    args = parser.parse_args()
    run_advisory(args.lat, args.lon, args.date, args.activity)
