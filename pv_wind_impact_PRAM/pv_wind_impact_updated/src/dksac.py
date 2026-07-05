"""
src/dksac.py
============
Validation of the project's PV physics against **real measured generation** from
the **Desert Knowledge Australia Solar Centre (DKASC)**, Alice Springs — a real
solar facility whose actual metered output is published openly
(https://dkasolarcentre.com.au/download?location=alice-springs).

This module does NOT ship any measured values. It loads the *real* DKASC CSV that
the user downloads, maps its columns, applies the project's own PV physics to the
DKASC site's measured weather, and compares the model against the measured power.

Why a channel selector (the fix for "0 hourly records")
-------------------------------------------------------
A DKASC export is wide: ~50 per-array `*_Active_Power` columns (each a separate
PV system, ~5-17 kW) PLUS whole-site master meters (~200 kW), PLUS many blank /
archived columns.  The previous loader grabbed the *first* column whose name
contained "active" and "power" — which in this file is an **archived QCells
column that is entirely blank** — then dropped every row with a NaN there,
yielding 0 records and nan metrics.

This version instead enumerates every power column that actually carries data,
labels it (array vs whole-site master meter), and lets the caller pick which
measured PV signal to validate against.  The rated power is auto-estimated from
the *selected* channel so the model scale is self-consistent with the truth.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

import config

# Real DKA Solar Centre sites where measurements physically exist.
SITES = {
    "alice": {"name": "DKASC, Alice Springs", "lat": -23.7621, "lon": 133.8748,
              "elev": 558.0,
              "patterns": ["*dkasc*.csv", "*DKASC*.csv", "*alice*spring*.csv",
                           "*Alice*Spring*.csv",
                           "*DKA*.csv", "*dka*.csv", "*MasterMeter*.csv",
                           "*master*meter*.csv"]},
    "yulara": {"name": "Yulara Solar System", "lat": -25.2406, "lon": 130.9889,
               "elev": 492.0,
               "patterns": ["*yulara*.csv", "*Yulara*.csv", "*YULARA*.csv"]},
}
# Backward-compatible default (Alice Springs)
DKASC_SITE = SITES["alice"]
AUSTRALIA_LAT = (-44.0, -10.0)
AUSTRALIA_LON = (112.0, 154.0)

_SEARCH_DIRS = ["data", ".", "/mnt/user-data/uploads", os.path.expanduser("~")]

# Keyword sets used to locate the on-site weather channels.
WEATHER_KEYS = {
    "ghi":  ("global", "horizontal"),
    "dhi":  ("diffuse", "horizontal"),
    "poa":  ("global", "tilted"),
    "temp": ("temperature",),
    "rh":   ("humidity",),
    "wind": ("wind", "speed"),
    "rain": ("rainfall",),
}


def find_site_csv(site_key: str) -> str | None:
    """Locate a downloaded CSV for a given site (alice / yulara)."""
    site = SITES.get(site_key, SITES["alice"])
    for d in _SEARCH_DIRS:
        for p in site["patterns"]:
            hits = glob.glob(os.path.join(d, p))
            if hits:
                return sorted(hits)[0]
    return None


def find_dksac_csv() -> str | None:
    """Backward-compatible: locate the Alice Springs DKASC CSV."""
    return find_site_csv("alice")


def _find_col(cols, *keywords):
    low = {c: str(c).lower() for c in cols}
    for c in cols:
        if all(k in low[c] for k in keywords):
            return c
    return None


def _channel_label(col: str) -> str:
    """Human-readable label for a DKASC power column."""
    low = col.lower()
    if "mastermeter" in low:
        n = re.search(r"mastermeter(\d+)", low)
        return f"Whole site — Master Meter {n.group(1) if n else ''}".strip()
    m = re.search(r"_M(\d+)_(.+?)_(?:phase|phases)", col, re.IGNORECASE)
    if m:
        phases = m.group(2).replace("_", "")
        suffix = " (II)" if re.search(r"phase_ii", low) else ""
        if phases.isdigit():
            return f"Array M{m.group(1)} ({phases}-phase){suffix}"
        return f"Array M{m.group(1)}-{phases}{suffix}"
    if "qcells" in low:
        return "Q CELLS array"
    return col


def _is_power_col(name: str) -> bool:
    cl = str(name).lower()
    # Exclude non-real-power channels first (these can also contain the word
    # "power", e.g. "power_factor" / "apparent_power" / "reactive_power").
    for bad in ("factor", "reactive", "apparent", "energy",
                "var", "voltage", "current", "freq"):
        if bad in cl:
            return False
    # Accept any real/active power channel, however the array labels it:
    #   "*_Active_Power", "*_AC_Power", "*_Power_kW", "*Power(kW)", etc.
    if "active_power" in cl:
        return True
    if "active" in cl and "power" in cl:
        return True
    if "power" in cl and ("kw" in cl or "(kw" in cl):
        return True
    return False


def load_dksac_frame(path: str):
    """Read a real DKASC CSV and resample to hourly, keeping ALL data-bearing
    PV power columns plus the on-site weather.

    Returns
    -------
    hourly : DataFrame
        Hourly-mean frame: weather keys (ghi, dhi, poa, temp, rh, wind, rain)
        plus every measured `*_Active_Power` column that actually has data.
    weather_mapping : dict
        Which raw column each weather key resolved to (or None).
    power_cols : list[str]
        Data-bearing power columns present in `hourly`.
    """
    df = pd.read_csv(path)
    cols = list(df.columns)

    dtcol = _find_col(cols, "timestamp") or _find_col(cols, "time") or cols[0]
    dt = pd.to_datetime(df[dtcol], errors="coerce", utc=False)
    df = df.assign(_dt=dt).dropna(subset=["_dt"]).set_index("_dt")

    weather_mapping = {k: _find_col(cols, *kw) for k, kw in WEATHER_KEYS.items()}

    # Keep only power columns that carry real (non-blank, non-zero) generation.
    power_cols = []
    for c in cols:
        if not _is_power_col(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0 and float(np.nanmax(s.values)) > 0.05:
            power_cols.append(c)

    data = {}
    for k, col in weather_mapping.items():
        data[k] = pd.to_numeric(df[col], errors="coerce") if col is not None else np.nan
    for c in power_cols:
        data[c] = pd.to_numeric(df[c], errors="coerce")

    frame = pd.DataFrame(data, index=df.index)
    hourly = frame.resample("1h").mean(numeric_only=True)
    return hourly, weather_mapping, power_cols


def power_channels(hourly: pd.DataFrame, power_cols: list[str]):
    """Rank selectable measured-PV channels.

    Returns a list of (column, label, peak_kw). Single arrays come first
    (sorted by descending peak — best signal/noise for a single-orientation
    physics test), whole-site master meters last (a looser whole-facility check).
    """
    out = []
    for c in power_cols:
        if c not in hourly:
            continue
        s = hourly[c]
        if s.notna().sum() == 0:
            continue
        peak = float(np.nanpercentile(s.values, 99.5))
        if peak <= 0.05:
            continue
        out.append((c, _channel_label(c), peak))

    arrays = [t for t in out if not t[1].lower().startswith("whole site")]
    meters = [t for t in out if t[1].lower().startswith("whole site")]
    arrays.sort(key=lambda t: -t[2])
    meters.sort(key=lambda t: t[1])
    # Whole-site master meter(s) first => unbiased default. Individual arrays
    # follow; some DKASC arrays are tracking / off-axis and will NOT match a
    # single fixed-tilt irradiance sensor, so the aggregate is the safer default.
    return meters + arrays


def load_dksac_csv(path: str):
    """Backward-compatible loader. Returns (df, mapping) where df has measured PV
    power (`P_meas`, kW) from the best available channel plus on-site weather.

    Unlike the old version, the channel is chosen from columns that ACTUALLY
    contain data, so this never silently yields 0 rows because of a blank
    archived column.
    """
    hourly, wmap, power_cols = load_dksac_frame(path)
    chans = power_channels(hourly, power_cols)
    pcol = chans[0][0] if chans else None

    out = pd.DataFrame(index=hourly.index)
    out["P_meas"] = hourly[pcol] if pcol is not None else np.nan
    for k in ("ghi", "dhi", "poa", "temp", "rh", "wind", "rain"):
        out[k] = hourly[k] if k in hourly else np.nan
    out = out.dropna(subset=["P_meas"])

    mapping = dict(wmap)
    mapping["P_meas"] = pcol
    return out, mapping


def estimate_rated_kw(p_meas: pd.Series) -> float:
    """Robust estimate of the array's rated power from the measured peak."""
    vals = pd.to_numeric(p_meas, errors="coerce").values
    if not np.isfinite(vals).any():
        return 0.1
    val = float(np.nanpercentile(vals, 99.5))
    return max(val, 0.1)


def model_pv_from_dksac(d: pd.DataFrame, rated_kw: float) -> pd.Series:
    """Apply the project's core PV physics to DKASC measured weather → predicted kW.

    Uses the same equations documented for the project:
      cell temp (NOCT) -> thermal derate -> normalised output (POA/1000)*f_temp.
    POA uses the plant's measured tilted irradiance when available (most accurate),
    else GHI. Soiling is set to 1.0 because DKASC has no aerosol channel (stated
    openly as a limitation of this validation).
    """
    poa = d["poa"].where(d["poa"].notna(), d["ghi"])
    ghi = d["ghi"].where(d["ghi"].notna(), poa).fillna(0.0)
    temp = d["temp"].fillna(25.0)

    t_cell = temp + (config.PV_NOCT_C - 20.0) / 800.0 * ghi
    f_temp = (1.0 - config.PV_TEMP_COEFF * (t_cell - 25.0)).clip(0.70, 1.05)
    f_soil = 1.0
    norm = (poa / 1000.0) * f_temp * f_soil
    return (rated_kw * norm).clip(lower=0.0, upper=rated_kw)


def validation_metrics(p_meas: pd.Series, p_model: pd.Series, rated_kw: float) -> dict:
    """Compare model vs measured over operating (daylight) hours."""
    m = pd.DataFrame({"meas": p_meas, "model": p_model}).dropna()
    m = m[m["meas"] > 0.02 * rated_kw]            # operating hours only
    err = m["model"] - m["meas"]
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mbe = float(err.mean())
    mean_meas = float(m["meas"].mean()) or 1.0
    nrmse = rmse / mean_meas * 100.0
    ss_res = float((err ** 2).sum())
    ss_tot = float(((m["meas"] - m["meas"].mean()) ** 2).sum()) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    corr = float(m["meas"].corr(m["model"])) if len(m) > 2 else float("nan")
    return {"n": len(m), "mae": mae, "rmse": rmse, "mbe": mbe,
            "nrmse_pct": nrmse, "r2": r2, "corr": corr,
            "mean_meas": mean_meas, "rated_kw": rated_kw, "frame": m}


def verdict(nrmse_pct: float):
    """Map nRMSE to a (css_class, label) verdict."""
    if nrmse_pct <= 10:
        return "good", "✅ Strong match — the model is validated against real data"
    if nrmse_pct <= 20:
        return "warn", "🟡 Reasonable match — acceptable, with some explained error"
    return "bad", "🟠 Notable difference — must be justified"
