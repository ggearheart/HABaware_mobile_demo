"""
HABaware web preview server.
Serves the map interface and advisory API endpoints.

Usage:
    python3 app/server.py
"""

import csv
import json
import math
import os
import pickle
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── In-memory cache: avoids blank page when FHAB API is slow ─────────────────
_cache = {}   # key → {"data": ..., "ts": datetime}
CACHE_TTL_SECONDS = 300   # 5 min; FHAB updates daily so this is fine

def _cached(key, fn):
    """Return cached value if fresh, otherwise call fn() and cache result."""
    import time
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        return entry["data"], True
    result = fn()
    _cache[key] = {"data": result, "ts": time.time()}
    return result, False

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH         = ROOT / "ml/model.pkl"
SWAMP_STATUS_PATH  = ROOT / "data/processed/clear_lake_current_status.json"
SWAMP_OBS_PATH     = ROOT / "data/processed/clear_lake_observations.csv"
BVPOMO_STATUS_PATH = ROOT / "data/processed/bvpomo_current_status.json"
BVPOMO_PEAKS_PATH  = ROOT / "data/processed/bvpomo_annual_peaks.csv"
CI_RAW_PATH        = ROOT / "data/raw/clear_lake_cyanoindex_2017_2025.json"
WEATHER_CSV_PATH   = ROOT / "data/processed/cimis_clear_lake_weather.csv"

FHAB_BASE      = "https://fhab-api.sfei.org"
CLEAR_LAKE_WID = 33
OM_FORECAST    = "https://api.open-meteo.com/v1/forecast"
CL_LAT, CL_LON = 39.0300, -122.780

BASELINE_CI = 0.9972436372799999

# BV Pomo monitoring sites (28 sites with arm and approx coordinates)
BVPOMO_SITES = {
    "AP01":     {"name": "Austin Park",              "arm": "L", "lat": 38.9572, "lon": -122.6419},
    "BP":       {"name": "Buckingham",               "arm": "L", "lat": 38.9701, "lon": -122.6602},
    "BVCL6":    {"name": "Big Valley Rancheria",     "arm": "U", "lat": 39.0223, "lon": -122.7643},
    "CL-1":     {"name": "Clear Lake Site 1",        "arm": "U", "lat": 39.0500, "lon": -122.8000},
    "CL-3":     {"name": "Clear Lake Site 3",        "arm": "L", "lat": 38.9800, "lon": -122.6500},
    "CL-4":     {"name": "Clear Lake Site 4",        "arm": "O", "lat": 39.0100, "lon": -122.8300},
    "CLV7":     {"name": "Clear Lake Village 7",     "arm": "U", "lat": 39.0419, "lon": -122.8018},
    "CLOAKS01": {"name": "Clover/Oaks",              "arm": "O", "lat": 39.0050, "lon": -122.8450},
    "CP":       {"name": "Cache Point",              "arm": "U", "lat": 39.0450, "lon": -122.7900},
    "ELEM01":   {"name": "Elem Indian Colony",       "arm": "O", "lat": 38.9930, "lon": -122.8760},
    "FC3":      {"name": "Finley Creek 3",           "arm": "U", "lat": 39.0600, "lon": -122.7800},
    "GH":       {"name": "Glenhaven",                "arm": "O", "lat": 39.0180, "lon": -122.8550},
    "HB":       {"name": "Highlands Beach",          "arm": "U", "lat": 39.0350, "lon": -122.7700},
    "JB":       {"name": "Jago Bay",                 "arm": "L", "lat": 38.9650, "lon": -122.6700},
    "KEYS03":   {"name": "Kelseyville Shore",        "arm": "O", "lat": 38.9900, "lon": -122.8600},
    "KP01":     {"name": "Konocti Park",             "arm": "U", "lat": 39.0280, "lon": -122.7800},
    "LC01":     {"name": "Lily Cove",                "arm": "L", "lat": 38.9580, "lon": -122.6480},
    "LPTNT":    {"name": "Library Park/TNT",         "arm": "U", "lat": 39.0395, "lon": -122.7966},
    "LS":       {"name": "Lakeshore",                "arm": "U", "lat": 39.0480, "lon": -122.7850},
    "LS2":      {"name": "Lakeshore 2",              "arm": "U", "lat": 39.0460, "lon": -122.7870},
    "LUC01":    {"name": "Lucerne",                  "arm": "U", "lat": 39.0483, "lon": -122.7986},
    "M4":       {"name": "Marina Site 4",            "arm": "U", "lat": 39.0400, "lon": -122.7900},
    "RED01":    {"name": "Redbud Park",              "arm": "L", "lat": 38.9563, "lon": -122.6274},
    "RODS":     {"name": "Rods Beach",               "arm": "U", "lat": 39.0510, "lon": -122.7750},
    "RP":       {"name": "Riviera Park",             "arm": "L", "lat": 38.9700, "lon": -122.6550},
    "SBMMEL01": {"name": "Sulphur Bank Mercury Mine","arm": "O", "lat": 38.9984, "lon": -122.8832},
    "SHADY01":  {"name": "Cache Creek/Shady Acres",  "arm": "L", "lat": 38.9550, "lon": -122.6200},
    "LA03":     {"name": "Lakeport Area 3",          "arm": "U", "lat": 39.0450, "lon": -122.9100},
}

ARM_NAMES = {"U": "Upper Arm", "L": "Lower Arm", "O": "Oaks Arm"}

RISK_TIERS = [
    (0,   5,   "Low",       "#2e7d32"),
    (5,   30,  "Moderate",  "#558b2f"),
    (30,  80,  "High",      "#f57f17"),
    (80,  200, "Very High", "#e64a19"),
    (200, 999, "Danger",    "#b71c1c"),
]

def risk_tier(ci):
    for lo, hi, label, color in RISK_TIERS:
        if lo <= ci < hi:
            return label, color
    return "Danger", "#b71c1c"


# ── FHAB helpers ──────────────────────────────────────────────────────────────

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "HABaware/1.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        body = r.read()
    if body[:9].lower().startswith(b"<!doctype") or body[:5] == b"<html":
        raise ValueError("SFEI API returned HTML (Cloudflare WAF block)")
    return json.loads(body)


def load_historical_cyanoindex():
    """Load records from local historical JSON when SFEI API is blocked."""
    if CI_RAW_PATH.exists():
        with open(CI_RAW_PATH) as f:
            return json.load(f)
    return []

def get_recent_cyanoindex(days=45):
    end   = date.today()
    start = end - timedelta(days=days)
    url = (f"{FHAB_BASE}/cyano/10daymax/{CLEAR_LAKE_WID}"
           f"/{start.isoformat()}/{end.isoformat()}/ci_modified/json")
    return fetch_json(url)

def fetch_recent_weather(past_days=14):
    params = urllib.parse.urlencode({
        "latitude": CL_LAT, "longitude": CL_LON,
        "past_days": past_days, "forecast_days": 3,
        "daily": ("temperature_2m_mean,temperature_2m_max,relative_humidity_2m_mean,"
                  "shortwave_radiation_sum,wind_speed_10m_mean,precipitation_sum,"
                  "et0_fao_evapotranspiration"),
        "timezone": "America/Los_Angeles",
    })
    req = urllib.request.Request(f"{OM_FORECAST}?{params}", headers={"User-Agent": "HABaware/1.0"})
    with urllib.request.urlopen(req, timeout=12) as r:
        raw = json.loads(r.read())
    daily = raw.get("daily", {})
    dates = daily.get("time", [])
    result = []
    for i, d in enumerate(dates):
        def v(k): return daily.get(k, [None]*(i+1))[i]
        rad = v("shortwave_radiation_sum")
        w   = v("wind_speed_10m_mean")
        result.append({
            "date": d,
            "tmp_avg_c":       v("temperature_2m_mean"),
            "tmp_max_c":       v("temperature_2m_max"),
            "rh_avg_pct":      v("relative_humidity_2m_mean"),
            "sol_rad_avg_wm2": round(rad * 1e6 / 86400, 1) if rad else None,
            "wind_spd_avg_ms": round(w / 3.6, 2) if w else None,
            "precip_mm":       v("precipitation_sum"),
            "eto_mm":          v("et0_fao_evapotranspiration"),
        })
    return result


# ── Data quality scoring ───────────────────────────────────────────────────────

def score_satellite(records):
    """0–1 quality score based on recency and pixel density of latest CI data."""
    valid = [r for r in records if r.get("max", 0) > BASELINE_CI and r.get("pixel_count", 0) > 0]
    if not valid:
        return 0.1, "No valid satellite retrievals", None, None

    latest = valid[-1]
    days_old = (date.today() - date.fromisoformat(latest["date"])).days
    pixel_count = latest.get("pixel_count", 0)

    # Freshness score: full credit ≤3 days, degrades to 0 at 30 days (cloud cover)
    freshness = max(0.0, 1.0 - days_old / 30.0)
    # Coverage score: pixel_count > 500 = excellent, <50 = poor
    coverage = min(1.0, pixel_count / 500.0)
    score = 0.7 * freshness + 0.3 * coverage

    if days_old == 0:
        age_str = "Today"
    elif days_old == 1:
        age_str = "1 day ago"
    else:
        age_str = f"{days_old} days ago"

    detail = f"{age_str} · {pixel_count} px · ci_max={latest['max']:.0f}"
    return round(score, 3), detail, latest["date"], latest["max"]

def score_swamp():
    """0–1 quality score based on recency of latest SWAMP field observation."""
    if not SWAMP_STATUS_PATH.exists():
        return 0.0, "SWAMP data file not found", None
    with open(SWAMP_STATUS_PATH) as f:
        status = json.load(f)
    latest_date = status.get("latest_bloom_report", {}).get("date")
    if not latest_date:
        return 0.1, "No field reports available", None
    days_old = (date.today() - date.fromisoformat(latest_date)).days
    n_reports = status.get("total_clear_lake_bloom_reports", 0)
    # Score: fresh < 30d → 1.0, degrades to 0.3 at 1 year, 0.1 at 2+ years
    if days_old <= 30:
        score = 1.0
    elif days_old <= 90:
        score = 0.85
    elif days_old <= 180:
        score = 0.70
    elif days_old <= 365:
        score = 0.50
    else:
        score = max(0.1, 0.5 - (days_old - 365) / 365 * 0.4)

    if days_old < 30:
        age_str = f"{days_old}d ago"
    elif days_old < 365:
        age_str = f"{days_old // 30}mo ago"
    else:
        age_str = f"{days_old // 365}yr {(days_old % 365) // 30}mo ago"
    detail = f"{age_str} · {n_reports} total reports · Case {status.get('open_case',{}).get('case_id','?')}"
    return round(score, 3), detail, latest_date

def score_bvpomo():
    """0–1 quality score based on recency of tribal monitoring data."""
    if not BVPOMO_STATUS_PATH.exists():
        return 0.0, "BV Pomo data file not found", None
    with open(BVPOMO_STATUS_PATH) as f:
        status = json.load(f)
    latest_date = status.get("latest_sampling_date")
    n_sites = status.get("n_sites_monitored", 0)
    if not latest_date:
        return 0.1, "No sampling dates available", None
    days_old = (date.today() - date.fromisoformat(latest_date)).days
    # Same decay as SWAMP, but tribal data is seasonal (Oct is end of season)
    if days_old <= 30:
        score = 1.0
    elif days_old <= 90:
        score = 0.90
    elif days_old <= 180:
        score = 0.75
    elif days_old <= 270:
        score = 0.60
    elif days_old <= 365:
        score = 0.45
    else:
        score = max(0.1, 0.45 - (days_old - 365) / 365 * 0.35)

    if days_old < 30:
        age_str = f"{days_old}d ago"
    elif days_old < 365:
        age_str = f"{days_old // 30}mo ago"
    else:
        age_str = f"{days_old // 365}yr {(days_old % 365) // 30}mo ago"
    detail = f"{age_str} · {n_sites} sites · Lower/Oaks arms"
    return round(score, 3), detail, latest_date

def score_weather(weather_records):
    """Weather from Open-Meteo is always near-real-time."""
    valid = [w for w in weather_records if w.get("tmp_avg_c") is not None]
    if not valid:
        return 0.1, "Weather data unavailable"
    latest = valid[-1]
    days_old = (date.today() - date.fromisoformat(latest["date"])).days
    # Open-Meteo includes forecast days so "latest" may be future
    # Score slightly less than perfect to reflect it's not in-situ
    score = 0.95 if days_old <= 1 else max(0.5, 0.95 - days_old * 0.05)
    tmp = latest.get("tmp_avg_c")
    wind = latest.get("wind_spd_avg_ms")
    detail = f"Real-time · {tmp:.1f}°C · {wind:.1f} m/s wind" if tmp and wind else "Real-time"
    return round(score, 3), detail

def compute_overall_confidence(sat, obs, tribal, weather):
    """Weighted confidence: satellite dominates, field obs validate."""
    w = {"sat": 0.50, "obs": 0.20, "tribal": 0.20, "weather": 0.10}
    raw = w["sat"] * sat + w["obs"] * obs + w["tribal"] * tribal + w["weather"] * weather
    return round(raw * 100, 1)

def ci_confidence_interval(ci_forecast, rmse=67.1):
    """Return 68% and 95% confidence interval bounds around the CI forecast."""
    return {
        "ci_68_low":  round(max(0, ci_forecast - rmse), 1),
        "ci_68_high": round(ci_forecast + rmse, 1),
        "ci_95_low":  round(max(0, ci_forecast - 2 * rmse), 1),
        "ci_95_high": round(ci_forecast + 2 * rmse, 1),
        "rmse":       rmse,
    }

def tier_from_interval(ci_low, ci_high):
    """Return the set of risk tiers spanned by a CI band."""
    tiers_spanned = set()
    for lo, hi, label, _ in RISK_TIERS:
        if ci_low < hi and ci_high >= lo:
            tiers_spanned.add(label)
    return sorted(tiers_spanned, key=lambda t: next(i for i,(lo,hi,l,_) in enumerate(RISK_TIERS) if l==t))


# ── ML inference ──────────────────────────────────────────────────────────────

def run_forecast(records):
    if not MODEL_PATH.exists():
        return None, None
    import numpy as np
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    def has_sig(r): return r["pixel_count"] > 0 and r["max"] > BASELINE_CI
    def ci(r, field): return r[field] if has_sig(r) else 0.0

    latest = records[-1]
    d = date.fromisoformat(latest["date"])
    doy = d.timetuple().tm_yday

    w  = lambda n, f: [ci(r, f) for r in records[-n:] if has_sig(r)]
    w7m,  w14m, w30m = w(7,"mean"), w(14,"mean"), w(30,"mean")
    p7,   p14,  p30  = w(7,"max"),  w(14,"max"),  w(30,"max")

    feats = [
        math.sin(2*math.pi*doy/365), math.cos(2*math.pi*doy/365), d.month,
        ci(latest,"mean"), ci(latest,"max"), ci(latest,"median"), ci(latest,"perc90"),
        int(has_sig(latest)), latest["pixel_count"],
        sum(w7m)/len(w7m) if w7m else 0,
        sum(w14m)/len(w14m) if w14m else 0,
        sum(w30m)/len(w30m) if w30m else 0,
        max(p7) if p7 else 0, max(p14) if p14 else 0, max(p30) if p30 else 0,
    ]
    ci_fc = float(max(0, model.predict(np.array([feats], dtype=np.float32))[0]))
    return ci_fc, feats


# ── Location helpers ──────────────────────────────────────────────────────────

def infer_arm(lat, lon):
    """Rough arm assignment based on coordinates."""
    if lon < -122.73:
        return "O"  # Oaks arm is westernmost
    if lat < 38.99:
        return "L"  # Lower arm is southernmost
    return "U"

def nearest_bvpomo_sites(lat, lon, n=5):
    """Return n closest BV Pomo monitoring sites with distance."""
    R = 6371000  # Earth radius metres
    def dist(site):
        dlat = math.radians(site["lat"] - lat)
        dlon = math.radians(site["lon"] - lon)
        a = (math.sin(dlat/2)**2 +
             math.cos(math.radians(lat)) * math.cos(math.radians(site["lat"])) *
             math.sin(dlon/2)**2)
        return R * 2 * math.asin(math.sqrt(a))

    scored = []
    for sid, s in BVPOMO_SITES.items():
        d_m = dist(s)
        scored.append({"site_id": sid, "name": s["name"], "arm": s["arm"],
                       "lat": s["lat"], "lon": s["lon"], "dist_m": round(d_m)})
    return sorted(scored, key=lambda x: x["dist_m"])[:n]

def get_bvpomo_site_status():
    """Return dict of site_id → last advisory level from current status."""
    if not BVPOMO_STATUS_PATH.exists():
        return {}
    with open(BVPOMO_STATUS_PATH) as f:
        status = json.load(f)
    site_status = {}
    for sid in status.get("latest_danger_sites", []):
        site_status[sid] = "DANGER"
    for sid in status.get("latest_warning_sites", []):
        site_status[sid] = "WARNING"
    for sid in status.get("latest_caution_sites", []):
        site_status[sid] = "CAUTION"
    return site_status

def get_recent_ci_trend(records, n=30):
    """Return last n days as sparkline data: [{date, ci_max}]."""
    result = []
    for r in records[-n:]:
        ci_val = r["max"] if r.get("max", 0) > BASELINE_CI else 0
        result.append({"date": r["date"], "ci_max": round(ci_val, 1)})
    return result

# ── Location-based risk adjustment ────────────────────────────────────────────
# Derived from BV Pomo tribal monitoring 2014–2024 (% sampling events above Caution):
#   Lower Arm: 94%  Oaks Arm: 79%  Upper Arm: 31%
# Normalised to lake-wide mean (68%) → multipliers applied to satellite CI.
ARM_RISK_MULTIPLIER = {"U": 0.68, "L": 1.28, "O": 1.12}
ARM_RISK_BASIS = {
    "U": "31% of BV Pomo sampling events above Caution (2014–2024)",
    "L": "94% of BV Pomo sampling events above Caution; highest absolute toxin levels",
    "O": "79% of BV Pomo sampling events above Caution; elevated Danger-level events",
}

def location_adjusted_ci(ci_lakewide, arm):
    """Scale lake-wide CI by arm-specific risk multiplier from tribal monitoring history."""
    mult = ARM_RISK_MULTIPLIER.get(arm, 1.0)
    return round(ci_lakewide * mult, 1)


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sites")
def api_sites():
    """Return all BV Pomo monitoring sites with last known advisory status."""
    site_status = get_bvpomo_site_status()
    features = []
    for sid, s in BVPOMO_SITES.items():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
            "properties": {
                "site_id": sid,
                "name": s["name"],
                "arm": s["arm"],
                "arm_name": ARM_NAMES.get(s["arm"], s["arm"]),
                "advisory": site_status.get(sid, "NONE"),
            }
        })
    return jsonify({"type": "FeatureCollection", "features": features})

@app.route("/api/status")
def api_status():
    """Return current data quality, CI values, and forecast — no Claude call."""
    lat = float(request.args.get("lat", CL_LAT))
    lon = float(request.args.get("lon", CL_LON))

    # Satellite data — cached so a slow/unavailable FHAB API never blanks the page
    try:
        records, from_cache = _cached("cyanoindex", lambda: get_recent_cyanoindex(days=45))
        sat_score, sat_detail, sat_date, sat_ci_max = score_satellite(records)
        ci_forecast, _ = run_forecast(records)
        ci_trend = get_recent_ci_trend(records, n=30)
    except Exception as e:
        # Fall back to cached value, then local historical file
        entry = _cache.get("cyanoindex")
        if entry:
            records = entry["data"]
            sat_detail = "(cached) " + score_satellite(records)[1]
        else:
            records = load_historical_cyanoindex()
            if not records:
                return jsonify({"error": f"FHAB API unavailable and no local data: {e}"}), 503
            sat_detail = "(historical) SFEI API blocked — using 2017-2025 archive"
            _cache["cyanoindex"] = {"data": records, "ts": time.time() - CACHE_TTL_SECONDS + 60}
        sat_score, _, sat_date, sat_ci_max = score_satellite(records)
        ci_forecast, _ = run_forecast(records)
        ci_trend = get_recent_ci_trend(records, n=30)

    # SWAMP
    obs_score, obs_detail, obs_date = score_swamp()

    # BV Pomo
    tribal_score, tribal_detail, tribal_date = score_bvpomo()

    # Weather
    try:
        weather, _ = _cached("weather", lambda: fetch_recent_weather(past_days=14))
        wx_score, wx_detail = score_weather(weather)
    except Exception:
        entry = _cache.get("weather")
        weather = entry["data"] if entry else []
        wx_score, wx_detail = (score_weather(weather) if weather else (0.5, "Weather temporarily unavailable"))

    # Confidence
    overall_conf = compute_overall_confidence(sat_score, obs_score, tribal_score, wx_score)
    ci_fc = ci_forecast if ci_forecast is not None else (sat_ci_max or 0)
    # Location context
    arm = infer_arm(lat, lon)
    nearest = nearest_bvpomo_sites(lat, lon)

    # Apply arm-specific risk multiplier derived from BV Pomo historical data
    ci_current_adj  = location_adjusted_ci(sat_ci_max or 0, arm)
    ci_fc_adj       = location_adjusted_ci(ci_fc, arm)
    tier_label, tier_color = risk_tier(ci_fc_adj)
    ci_band = ci_confidence_interval(ci_fc_adj)
    tiers_spanned = tier_from_interval(ci_band["ci_68_low"], ci_band["ci_68_high"])

    # Recent weather summary (last 7 valid days)
    wx_recent = [w for w in weather if w.get("tmp_avg_c") is not None][-7:]
    if wx_recent:
        tmp_7d = round(sum(w["tmp_avg_c"] for w in wx_recent) / len(wx_recent), 1)
        wind_7d = round(sum(w["wind_spd_avg_ms"] for w in wx_recent if w.get("wind_spd_avg_ms")) /
                        max(1, sum(1 for w in wx_recent if w.get("wind_spd_avg_ms"))), 2)
        calm_7d = sum(1 for w in wx_recent if w.get("wind_spd_avg_ms", 99) < 2.0)
        precip_7d = round(sum(w.get("precip_mm") or 0 for w in wx_recent), 1)
        wx_summary = {"tmp_avg_7d": tmp_7d, "wind_avg_7d": wind_7d,
                      "calm_days_7d": calm_7d, "precip_7d_mm": precip_7d,
                      "latest": wx_recent[-1]}
    else:
        wx_summary = {}

    return jsonify({
        "as_of": date.today().isoformat(),
        "location": {
            "lat": lat, "lon": lon,
            "arm": arm, "arm_name": ARM_NAMES.get(arm, arm),
            "nearest_sites": nearest,
        },
        "forecast": {
            "ci_max_current":        round(sat_ci_max, 1) if sat_ci_max else 0,
            "ci_max_7d":             round(ci_fc, 1),
            "ci_max_current_adj":    round(ci_current_adj, 1),
            "ci_max_7d_adj":         round(ci_fc_adj, 1),
            "arm_multiplier":        ARM_RISK_MULTIPLIER.get(arm, 1.0),
            "arm_risk_basis":        ARM_RISK_BASIS.get(arm, ""),
            "tier":                  tier_label,
            "tier_color":            tier_color,
            "confidence_band":       ci_band,
            "tiers_68pct_band":      tiers_spanned,
        },
        "overall_confidence": overall_conf,
        "ci_trend": ci_trend,
        "data_quality": {
            "satellite": {
                "score":       sat_score,
                "pct":         round(sat_score * 100),
                "detail":      sat_detail,
                "latest_date": sat_date,
                "source":      "SFEI FHAB 10-day max cyanoindex",
                "weight_pct":  50,
            },
            "field_obs": {
                "score":       obs_score,
                "pct":         round(obs_score * 100),
                "detail":      obs_detail,
                "latest_date": obs_date,
                "source":      "CA SWAMP FHAB Program (229 reports)",
                "weight_pct":  20,
            },
            "tribal": {
                "score":       tribal_score,
                "pct":         round(tribal_score * 100),
                "detail":      tribal_detail,
                "latest_date": tribal_date,
                "source":      "BV Pomo tribal monitoring (2014–2024)",
                "weight_pct":  20,
            },
            "weather": {
                "score":       wx_score,
                "pct":         round(wx_score * 100),
                "detail":      wx_detail,
                "source":      "Open-Meteo (real-time, Clear Lake coords)",
                "weight_pct":  10,
                "summary":     wx_summary,
            },
        },
    })

@app.route("/api/advisory", methods=["POST"])
def api_advisory():
    """Full Claude advisory generation — may take ~5s."""
    body = request.json or {}
    lat      = float(body.get("lat", CL_LAT))
    lon      = float(body.get("lon", CL_LON))
    visit_dt = body.get("date", str(date.today()))
    activity = body.get("activity", "swimming")

    try:
        from api.advisory import run_advisory
        result = run_advisory(lat, lon, visit_dt, activity)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"HABaware preview server → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
