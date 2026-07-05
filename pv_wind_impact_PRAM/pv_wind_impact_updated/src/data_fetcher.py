"""
src/data_fetcher.py
===================
Two ways to obtain a *time-synchronised hourly* table of the 57 raw atmospheric
variables for one geographic location:

1. fetch_realtime_data(lat, lon, elevation, past_days, forecast_days)
       -> pulls REAL data from the free Open-Meteo API (no API key required).
          The series ends at the current hour, so it is genuinely "real-time
          synchronised" for the requested coordinates.

2. generate_synthetic_raw(lat, lon, elevation, n_days)
       -> a physically-plausible offline fallback with the EXACT same columns,
          used automatically when there is no internet connection (and for the
          test-suite / CI).

Both return a tidy pandas DataFrame whose columns are
``["Datetime"] + config.RAW_HOURLY_VARS`` -- i.e. the input to
``src/feature_engineering.build_feature_table``.

Only this module talks to the network.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

import config

# Open-Meteo endpoints (free, CC-BY 4.0, no authentication for non-commercial).
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# NASA POWER (NASA / U.S. government; the deck's recommended source for an
# India M.Tech dissertation).  Free, global, no API key.
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Which raw vars come from the Open-Meteo *air-quality* API rather than weather.
AIR_QUALITY_VARS = [
    "aerosol_optical_depth", "dust", "pm2_5", "pm10", "uv_index",
    "ozone", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
]
# Raw vars that are computed locally (not fetched from any API).
LOCALLY_FILLED_VARS = ["surface_albedo"]

_TIMEOUT = 60


# ===========================================================================
# Small HTTP helper
# ===========================================================================
def _http_get_json(url: str, params: dict) -> dict:
    """GET a URL with query params and parse the JSON response."""
    query = urllib.parse.urlencode(params, doseq=True)
    full = f"{url}?{query}"
    req = urllib.request.Request(full, headers={"User-Agent": "pv-wind-impact/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        payload = resp.read().decode("utf-8")
    data = json.loads(payload)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Open-Meteo error: {data.get('reason', 'unknown')}")
    return data


# ===========================================================================
# Optional helpers: geocoding + elevation lookup
# ===========================================================================
def geocode_place(name: str) -> dict | None:
    """Resolve a place name to {name, latitude, longitude, elevation, country}."""
    try:
        data = _http_get_json(GEOCODE_URL, {"name": name, "count": 1, "format": "json"})
    except Exception:
        return None
    results = data.get("results") or []
    if not results:
        return None
    r = results[0]
    return {
        "name": r.get("name"),
        "country": r.get("country"),
        "latitude": r.get("latitude"),
        "longitude": r.get("longitude"),
        "elevation": r.get("elevation"),
    }


def lookup_elevation(lat: float, lon: float) -> float | None:
    """Terrain elevation (m) for a coordinate via Open-Meteo's 90 m DEM."""
    try:
        data = _http_get_json(ELEVATION_URL, {"latitude": lat, "longitude": lon})
        elev = data.get("elevation")
        if isinstance(elev, list) and elev:
            return float(elev[0])
    except Exception:
        return None
    return None


# ===========================================================================
# 1. REAL-TIME data from Open-Meteo
# ===========================================================================
def fetch_realtime_data(
    latitude: float,
    longitude: float,
    elevation: float | None = None,
    past_days: int = config.DEFAULT_PAST_DAYS,
    forecast_days: int = config.DEFAULT_FORECAST_DAYS,
    use_archive: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Return an hourly DataFrame of the 57 raw variables for the given location.

    By default the *forecast* endpoint is used with ``past_days`` of recent
    history plus ``forecast_days`` of near-term forecast, so the series is
    synchronised to "now".  Set ``use_archive=True`` (with start/end dates) to
    pull a longer ERA5 reanalysis history instead.
    """
    weather_vars = [v for v in config.RAW_HOURLY_VARS
                    if v not in AIR_QUALITY_VARS and v not in LOCALLY_FILLED_VARS]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(weather_vars),
        "timezone": "auto",
        "wind_speed_unit": "ms",          # m/s (not km/h)
        "tilt": config.PV_TILT_DEG,        # for global_tilted_irradiance
        "azimuth": 0,                      # 0 = south-facing (REQUIRED by Open-Meteo
                                           # whenever global_tilted_irradiance is
                                           # requested; omitting it errors the call)
    }
    if elevation is not None:
        params["elevation"] = elevation

    if use_archive:
        if not (start_date and end_date):
            raise ValueError("Archive mode needs start_date and end_date (YYYY-MM-DD).")
        params["start_date"] = start_date
        params["end_date"] = end_date
        url = ARCHIVE_URL
    else:
        params["past_days"] = int(np.clip(past_days, 1, 92))
        params["forecast_days"] = int(np.clip(forecast_days, 0, 16))
        url = FORECAST_URL

    data = _http_get_json(url, params)
    hourly = data.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("Open-Meteo returned no hourly data for this location.")

    df = pd.DataFrame({"Datetime": pd.to_datetime(hourly["time"])})
    for var in weather_vars:
        series = hourly.get(var)
        df[var] = pd.to_numeric(pd.Series(series), errors="coerce") if series is not None else np.nan

    # --- Air-quality / aerosols from the Open-Meteo Air-Quality API ------
    aq = _fetch_air_quality(latitude, longitude, past_days, forecast_days,
                            use_archive, start_date, end_date)
    if aq is not None and len(aq):
        df = df.merge(aq, on="Datetime", how="left")

    # --- Surface albedo: bare ground ~0.2; fresh snow raises it ----------
    snow = df["snowfall"] if "snowfall" in df else pd.Series(0.0, index=df.index)
    df["surface_albedo"] = np.where(pd.to_numeric(snow, errors="coerce").fillna(0) > 0.2,
                                    0.55, 0.20)

    df.attrs["elevation"] = data.get("elevation", elevation)
    df.attrs["latitude"] = data.get("latitude", latitude)
    df.attrs["longitude"] = data.get("longitude", longitude)
    df.attrs["timezone"] = data.get("timezone", "auto")
    df.attrs["source"] = ("Open-Meteo archive (ERA5 reanalysis) + air quality"
                          if use_archive else
                          "Open-Meteo forecast (live, ECMWF/GFS) + air quality")

    return _clean_raw(df)


def _fetch_air_quality(latitude, longitude, past_days, forecast_days,
                       use_archive, start_date, end_date) -> pd.DataFrame | None:
    """Aerosols / pollutants from the Open-Meteo Air-Quality API (best effort)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(AIR_QUALITY_VARS),
        "timezone": "auto",
    }
    if use_archive and start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        params["past_days"] = int(np.clip(past_days, 1, 92))
        params["forecast_days"] = int(np.clip(forecast_days, 0, 7))
    try:
        data = _http_get_json(AIR_QUALITY_URL, params)
        hourly = data.get("hourly")
        if not hourly or "time" not in hourly:
            return None
        out = pd.DataFrame({"Datetime": pd.to_datetime(hourly["time"])})
        for var in AIR_QUALITY_VARS:
            series = hourly.get(var)
            out[var] = pd.to_numeric(pd.Series(series), errors="coerce") if series is not None else np.nan
        return out
    except Exception:
        return None                         # aerosols are optional; clean fills gaps


# ===========================================================================
# 1b. REAL data from NASA POWER  (NASA / U.S. government; deck-recommended)
# ===========================================================================
# Hourly POWER parameters we request (max 15 per call -> two batches).
_NASA_BATCH_1 = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI",
                 "ALLSKY_SFC_SW_DIFF", "TOA_SW_DWN", "ALLSKY_SFC_PAR_TOT",
                 "T2M", "T2MDEW", "TS", "RH2M", "QV2M", "PS",
                 "WS10M", "WD10M", "WS50M"]
_NASA_BATCH_2 = ["WD50M", "PRECTOTCORR"]
# Surface albedo is requested separately: it is not always offered at hourly
# resolution, and POWER rejects a whole batch if any one parameter is invalid —
# isolating it keeps precipitation / WD50M safe.
_NASA_BATCH_3 = ["ALLSKY_SRF_ALB"]


def _nasa_request(lat, lon, elevation, start, end, parameters) -> pd.DataFrame:
    """One NASA POWER hourly call -> DataFrame indexed by Datetime."""
    params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "LST",
    }
    if elevation is not None:
        params["site-elevation"] = elevation
    data = _http_get_json(NASA_POWER_URL, params)
    block = (data.get("properties") or {}).get("parameter") or {}
    if not block:
        raise RuntimeError("NASA POWER returned no parameter block.")
    frame = pd.DataFrame(block)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H")
    frame = frame.replace(-999.0, np.nan)
    return frame


def fetch_nasa_power(latitude: float, longitude: float,
                     elevation: float | None = None,
                     past_days: int = config.DEFAULT_PAST_DAYS,
                     forecast_days: int = 0) -> pd.DataFrame:
    """
    Build the canonical 57-variable hourly table from NASA POWER (hourly, RE
    community).  POWER supplies the core solar + meteorological variables; the
    rest of the canonical schema (multi-height temps/winds, aerosols, soil,
    derived state) is reconstructed from them with standard atmospheric
    relations.  POWER is near-real-time (a few days' latency), so the series
    ends a couple of days before "now".
    """
    end_dt = pd.Timestamp(datetime.utcnow()).normalize() - pd.Timedelta(days=2)
    start_dt = end_dt - pd.Timedelta(days=int(np.clip(past_days, 1, 365)))
    start, end = start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")

    f1 = _nasa_request(latitude, longitude, elevation, start, end, _NASA_BATCH_1)
    try:
        f2 = _nasa_request(latitude, longitude, elevation, start, end, _NASA_BATCH_2)
        nasa = f1.join(f2, how="left")
    except Exception:
        nasa = f1
    try:                                    # albedo is optional (may be daily-only)
        f3 = _nasa_request(latitude, longitude, elevation, start, end, _NASA_BATCH_3)
        nasa = nasa.join(f3, how="left")
    except Exception:
        pass
    nasa = nasa.sort_index()
    nasa = nasa.dropna(subset=["T2M"])                  # trim NRT gap at the tail
    if nasa.empty:
        raise RuntimeError("NASA POWER returned only fill values for this period.")

    n = len(nasa)
    g = lambda k, d=np.nan: (nasa[k].to_numpy() if k in nasa
                             else np.full(n, d))         # safe getter
    ghi = np.clip(g("ALLSKY_SFC_SW_DWN", 0.0), 0, None)
    dni = np.clip(g("ALLSKY_SFC_SW_DNI", 0.0), 0, None)
    dhi = np.clip(g("ALLSKY_SFC_SW_DIFF", 0.0), 0, None)
    toa = np.clip(g("TOA_SW_DWN", 0.0), 0, None)
    t2 = g("T2M", 25.0)
    rh = np.clip(g("RH2M", 50.0), 0, 100)
    dew = g("T2MDEW", t2 - 5.0)
    ts = g("TS", t2)
    ps_hpa = g("PS", 101.3) * 10.0                       # kPa -> hPa
    w10 = np.clip(g("WS10M", 3.0), 0, None)
    w50 = np.clip(g("WS50M", w10 * 1.2), 0, None)
    wd10 = g("WD10M", 0.0) % 360
    wd50 = g("WD50M", wd10) % 360
    precip = np.clip(g("PRECTOTCORR", 0.0), 0, None)
    alb_raw = g("ALLSKY_SRF_ALB", np.nan)

    # wind shear exponent from the two measured heights
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.log(np.clip(w50, 0.1, None) / np.clip(w10, 0.1, None)) / np.log(50.0 / 10.0)
    alpha = np.clip(np.nan_to_num(alpha, nan=0.18), 0.05, 0.6)

    def at_height(h):
        return w10 * (h / 10.0) ** alpha

    # thermodynamics for VPD
    es = 0.6108 * np.exp(17.27 * t2 / (t2 + 237.3))
    ea = es * rh / 100.0
    vpd = np.clip(es - ea, 0, None)

    snow = np.where(t2 < 1.0, precip * 0.7, 0.0)
    albedo = np.where(~np.isnan(alb_raw), alb_raw,
                      np.where(snow > 0.2, 0.55, 0.20))

    out = pd.DataFrame({"Datetime": nasa.index})
    fill = {
        "shortwave_radiation": ghi,
        "direct_radiation": np.clip(ghi - dhi, 0, None),
        "diffuse_radiation": dhi,
        "direct_normal_irradiance": dni,
        "global_tilted_irradiance": np.clip(ghi * 1.05, 0, None),
        "terrestrial_radiation": toa,
        "temperature_2m": t2,
        "temperature_80m": t2 - 0.0065 * (80 - 2),
        "temperature_120m": t2 - 0.0065 * (120 - 2),
        "temperature_180m": t2 - 0.0065 * (180 - 2),
        "apparent_temperature": t2,
        "dew_point_2m": dew,
        "soil_temperature_0cm": ts,
        "soil_temperature_6cm": 0.9 * ts + 0.1 * np.mean(t2),
        "soil_temperature_18cm": 0.7 * ts + 0.3 * np.mean(t2),
        "soil_temperature_54cm": 0.4 * ts + 0.6 * np.mean(t2),
        "relative_humidity_2m": rh,
        "vapour_pressure_deficit": vpd,
        "evapotranspiration": np.clip(0.02 * ghi / 100.0 + 0.1 * vpd, 0, None),
        "et0_fao_evapotranspiration": np.clip(0.022 * ghi / 100.0 + 0.11 * vpd, 0, None),
        "soil_moisture_0_to_1cm": np.full(n, 0.25),
        "soil_moisture_1_to_3cm": np.full(n, 0.25),
        "soil_moisture_3_to_9cm": np.full(n, 0.26),
        "soil_moisture_9_to_27cm": np.full(n, 0.27),
        "wind_speed_10m": w10,
        "wind_speed_80m": at_height(80),
        "wind_speed_120m": at_height(120),
        "wind_speed_180m": at_height(180),
        "wind_direction_10m": wd10,
        "wind_direction_80m": wd50,
        "wind_direction_120m": wd50,
        "wind_direction_180m": wd50,
        "wind_gusts_10m": w10 * 1.4,
        "pressure_msl": ps_hpa + 0.12 * (elevation or 0.0),
        "surface_pressure": ps_hpa,
        "cloud_cover": np.clip(100.0 * (1.0 - np.divide(ghi, np.clip(g("CLRSKY_SFC_SW_DWN", ghi), 1, None),
                               out=np.ones(n), where=g("CLRSKY_SFC_SW_DWN", ghi) > 1)), 0, 100),
        "cloud_cover_low": np.nan, "cloud_cover_mid": np.nan, "cloud_cover_high": np.nan,
        "precipitation": precip,
        "rain": np.where(t2 >= 1.0, precip, 0.0),
        "snowfall": snow,
        # aerosols are not in POWER hourly -> climatological placeholders
        "aerosol_optical_depth": np.full(n, 0.3),
        "dust": np.full(n, 50.0),
        "pm2_5": np.full(n, 40.0),
        "pm10": np.full(n, 80.0),
        "uv_index": np.clip(ghi / 100.0, 0, 13),
        "ozone": np.full(n, 50.0),
        "carbon_monoxide": np.full(n, 300.0),
        "nitrogen_dioxide": np.full(n, 20.0),
        "sulphur_dioxide": np.full(n, 10.0),
        "surface_albedo": albedo,
        "visibility": np.clip(40000 - 200 * np.clip(ghi * 0 + 30, 0, 100), 200, 50000),
        "is_day": (ghi > 0).astype(float),
        "sunshine_duration": np.where(ghi > 120, 3600.0, 0.0),
        "freezing_level_height": np.clip(4000 + 120 * t2, 0, 6000),
        "soil_moisture_27_to_81cm": np.full(n, 0.25),
    }
    for col, val in fill.items():
        out[col] = val

    out.attrs["elevation"] = elevation if elevation is not None else 200.0
    out.attrs["latitude"] = latitude
    out.attrs["longitude"] = longitude
    out.attrs["timezone"] = "LST (NASA POWER)"
    out.attrs["source"] = "NASA POWER (hourly, RE community — government)"
    return _clean_raw(out)


# ===========================================================================
# 2. OFFLINE synthetic fallback (identical schema)
# ===========================================================================
def generate_synthetic_raw(
    latitude: float = config.DEFAULT_LATITUDE,
    longitude: float = config.DEFAULT_LONGITUDE,
    elevation: float | None = config.DEFAULT_ELEVATION,
    n_days: int = 90,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Physically-plausible hourly raw data with the SAME 57 columns as the live
    fetch.  Latitude controls the solar geometry so the diurnal/seasonal cycle
    is location-aware; elevation sets the baseline pressure.
    """
    rng = np.random.default_rng(seed)
    elev = float(elevation) if elevation is not None else 200.0
    periods = int(n_days) * 24
    end = pd.Timestamp(datetime.now()).floor("h")
    idx = pd.date_range(end=end, periods=periods, freq="h")

    hour = idx.hour.to_numpy().astype(float)
    doy = idx.dayofyear.to_numpy().astype(float)

    # Solar geometry (simplified) so irradiance is latitude/season aware.
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284 + doy) / 365.0))
    lat_r = np.deg2rad(latitude)
    decl_r = np.deg2rad(decl)
    hour_angle = np.deg2rad(15.0 * (hour - 12.0))
    cos_zen = (np.sin(lat_r) * np.sin(decl_r)
               + np.cos(lat_r) * np.cos(decl_r) * np.cos(hour_angle))
    cos_zen = np.clip(cos_zen, 0.0, 1.0)            # night -> 0

    seasonal = np.sin(2 * np.pi * (doy - 80) / 365.0)

    # Clouds: skewed toward clearer skies.
    cloud = np.clip(rng.beta(2.0, 4.0, periods) * 100.0, 0, 100)
    cloud_low = np.clip(cloud * rng.uniform(0.3, 0.7, periods), 0, 100)
    cloud_mid = np.clip(cloud * rng.uniform(0.2, 0.5, periods), 0, 100)
    cloud_high = np.clip(cloud * rng.uniform(0.1, 0.4, periods), 0, 100)

    I0 = 1361.0                                     # solar constant
    toa = I0 * cos_zen                              # extraterrestrial on horizontal
    clear_ghi = toa * 0.75
    ghi = np.clip(clear_ghi * (1 - 0.75 * cloud / 100.0)
                  + rng.normal(0, 15, periods), 0, 1200)
    ghi[cos_zen <= 0] = 0.0
    diffuse = np.clip(ghi * (0.3 + 0.5 * cloud / 100.0), 0, ghi)
    direct = np.clip(ghi - diffuse, 0, None)
    dni = np.where(cos_zen > 0.05, direct / np.clip(cos_zen, 0.05, 1), 0.0)
    dni = np.clip(dni, 0, 1100)
    tilt_gain = 1.05
    gti = np.clip(ghi * tilt_gain, 0, 1300)
    sunshine = np.where(ghi > 120, 3600.0, 0.0)

    # Temperature with elevation lapse, season and a diurnal swing.
    base_t = 27.0 - 0.0065 * elev
    temp2 = (base_t + 8 * seasonal
             + 6 * np.sin(2 * np.pi * (hour - 9) / 24.0)
             + 0.004 * ghi
             + rng.normal(0, 1.2, periods))
    temp80 = temp2 - 0.5 + rng.normal(0, 0.3, periods)
    temp120 = temp2 - 0.8 + rng.normal(0, 0.3, periods)
    temp180 = temp2 - 1.2 + rng.normal(0, 0.3, periods)
    apparent = temp2 + rng.normal(0, 0.6, periods)

    humidity = np.clip(92 - 1.7 * (temp2 - 20) - 0.01 * ghi
                       + rng.normal(0, 6, periods), 8, 100)
    es = 0.6108 * np.exp(17.27 * temp2 / (temp2 + 237.3))     # kPa
    ea = es * humidity / 100.0
    dew = (237.3 * np.log(ea / 0.6108)) / (17.27 - np.log(ea / 0.6108))
    vpd = np.clip(es - ea, 0, None)

    # Wind: Weibull base, sheared up with height, gusty.
    w10 = np.clip(rng.weibull(2.0, periods) * 5.5
                  + 0.8 * np.sin(2 * np.pi * (hour - 15) / 24.0), 0, 28)
    alpha = rng.uniform(0.12, 0.30, periods)
    w80 = w10 * (80.0 / 10.0) ** alpha
    w120 = w10 * (120.0 / 10.0) ** alpha
    w180 = w10 * (180.0 / 10.0) ** alpha
    gusts = np.clip(w10 * rng.uniform(1.2, 1.8, periods), 0, 40)
    wd10 = rng.uniform(0, 360, periods)
    wd80 = (wd10 + rng.normal(0, 8, periods)) % 360
    wd120 = (wd10 + rng.normal(0, 10, periods)) % 360
    wd180 = (wd10 + rng.normal(0, 12, periods)) % 360

    # Pressure from barometric formula + synoptic wobble.
    p_surface = (1013.25 * (1 - 2.25577e-5 * elev) ** 5.25588
                 + 6 * np.sin(2 * np.pi * (doy - 30) / 365.0)
                 + rng.normal(0, 2.5, periods))
    p_msl = p_surface + 0.12 * elev + rng.normal(0, 1.0, periods)

    # Precipitation: mostly dry, heavier under thick cloud.
    rain_event = rng.random(periods) < (cloud / 100.0) ** 3
    rain = np.where(rain_event, rng.gamma(2.0, 1.4, periods), 0.0)
    showers = np.where(rain_event & (rng.random(periods) < 0.3),
                       rng.gamma(1.5, 1.0, periods), 0.0)
    precip = rain + showers
    snow = np.where(temp2 < 1.0, rain * 0.7, 0.0)
    snow_depth = np.clip(np.cumsum(snow) * 0.0, 0, None)
    precip_prob = np.clip(cloud * 0.8 + rng.normal(0, 8, periods), 0, 100)

    # Soil temperature damped/lagged vs air; soil moisture responds to rain.
    soil0 = temp2 * 0.8 + base_t * 0.2 + rng.normal(0, 0.5, periods)
    soil6 = soil0 * 0.9 + base_t * 0.1
    soil18 = soil0 * 0.7 + base_t * 0.3
    soil54 = soil0 * 0.4 + base_t * 0.6
    sm1 = np.clip(0.25 + 0.02 * np.cumsum(precip) / max(periods, 1)
                  + rng.normal(0, 0.02, periods), 0.05, 0.55)
    sm3 = np.clip(sm1 + rng.normal(0, 0.01, periods), 0.05, 0.55)
    sm9 = np.clip(sm1 * 0.95 + 0.02, 0.05, 0.55)
    sm27 = np.clip(sm1 * 0.9 + 0.03, 0.05, 0.55)
    sm81 = np.clip(sm1 * 0.85 + 0.05, 0.05, 0.55)

    et = np.clip(0.02 * ghi / 100.0 + 0.1 * vpd, 0, None)
    et0 = np.clip(et * 1.1 + rng.normal(0, 0.02, periods), 0, None)

    visibility = np.clip(40000 - 250 * cloud - 1500 * (precip > 0)
                         + rng.normal(0, 1500, periods), 200, 50000)
    freezing = np.clip(4000 + 120 * temp2 + rng.normal(0, 200, periods), 0, 6000)
    is_day = (cos_zen > 0).astype(float)
    wcode = np.where(precip > 5, 65, np.where(precip > 0, 61,
                     np.where(cloud > 70, 3, np.where(cloud > 30, 2, 0)))).astype(float)

    # --- Aerosols / air quality / surface albedo -------------------------
    # Coupled to weather: higher under dry, calm, polluted conditions; rain
    # and wind clear them.  This gives the soiling/POA physics real structure
    # so AOD / dust / PM / albedo show up in the impact analysis.
    dryness = np.clip((70.0 - humidity) / 60.0, 0, 1)        # dry-air proxy
    wash = np.where(precip > 0.5, 0.5, 1.0)                  # rain clears aerosols
    calm = np.clip(1.0 - w10 / 12.0, 0, 1)
    aod = np.clip((0.15 + 0.5 * dryness + 0.2 * calm) * wash
                  + rng.normal(0, 0.05, periods), 0.02, 2.5)
    pm10v = np.clip((40 + 120 * dryness + 60 * calm) * wash
                    + rng.normal(0, 12, periods), 2, 400)
    pm25v = np.clip(pm10v * rng.uniform(0.45, 0.65, periods)
                    + rng.normal(0, 6, periods), 1, 300)
    dustv = np.clip((30 + 180 * dryness) * wash * calm
                    + rng.normal(0, 15, periods), 0, 400)
    ozonev = np.clip(40 + 0.03 * ghi + rng.normal(0, 8, periods), 5, 200)
    cov = np.clip(200 + 400 * dryness * calm + rng.normal(0, 60, periods), 50, 1500)
    no2v = np.clip(15 + 40 * calm + rng.normal(0, 8, periods), 1, 120)
    so2v = np.clip(8 + 20 * calm + rng.normal(0, 4, periods), 0.5, 80)
    uvv = np.clip(ghi / 100.0 + rng.normal(0, 0.4, periods), 0, 13)
    albedo = np.where(snow > 0.2,
                      np.clip(0.55 + rng.normal(0, 0.05, periods), 0.30, 0.85),
                      np.clip(0.20 + rng.normal(0, 0.03, periods), 0.10, 0.40))

    df = pd.DataFrame({
        "Datetime": idx,
        "shortwave_radiation": ghi,
        "direct_radiation": direct,
        "diffuse_radiation": diffuse,
        "direct_normal_irradiance": dni,
        "global_tilted_irradiance": gti,
        "terrestrial_radiation": toa,
        "shortwave_radiation_instant": ghi,
        "direct_radiation_instant": direct,
        "diffuse_radiation_instant": diffuse,
        "direct_normal_irradiance_instant": dni,
        "global_tilted_irradiance_instant": gti,
        "terrestrial_radiation_instant": toa,
        "temperature_2m": temp2,
        "temperature_80m": temp80,
        "temperature_120m": temp120,
        "temperature_180m": temp180,
        "apparent_temperature": apparent,
        "dew_point_2m": dew,
        "soil_temperature_0cm": soil0,
        "soil_temperature_6cm": soil6,
        "soil_temperature_18cm": soil18,
        "soil_temperature_54cm": soil54,
        "relative_humidity_2m": humidity,
        "vapour_pressure_deficit": vpd,
        "evapotranspiration": et,
        "et0_fao_evapotranspiration": et0,
        "soil_moisture_0_to_1cm": sm1,
        "soil_moisture_1_to_3cm": sm3,
        "soil_moisture_3_to_9cm": sm9,
        "soil_moisture_9_to_27cm": sm27,
        "wind_speed_10m": w10,
        "wind_speed_80m": w80,
        "wind_speed_120m": w120,
        "wind_speed_180m": w180,
        "wind_direction_10m": wd10,
        "wind_direction_80m": wd80,
        "wind_direction_120m": wd120,
        "wind_direction_180m": wd180,
        "wind_gusts_10m": gusts,
        "pressure_msl": p_msl,
        "surface_pressure": p_surface,
        "cloud_cover": cloud,
        "cloud_cover_low": cloud_low,
        "cloud_cover_mid": cloud_mid,
        "cloud_cover_high": cloud_high,
        "precipitation": precip,
        "rain": rain,
        "showers": showers,
        "snowfall": snow,
        "snow_depth": snow_depth,
        "precipitation_probability": precip_prob,
        "weather_code": wcode,
        "aerosol_optical_depth": aod,
        "dust": dustv,
        "pm2_5": pm25v,
        "pm10": pm10v,
        "uv_index": uvv,
        "ozone": ozonev,
        "carbon_monoxide": cov,
        "nitrogen_dioxide": no2v,
        "sulphur_dioxide": so2v,
        "surface_albedo": albedo,
        "visibility": visibility,
        "is_day": is_day,
        "sunshine_duration": sunshine,
        "freezing_level_height": freezing,
        "soil_moisture_27_to_81cm": sm81,
    })

    df.attrs["elevation"] = elev
    df.attrs["latitude"] = latitude
    df.attrs["longitude"] = longitude
    df.attrs["timezone"] = "synthetic"
    df.attrs["source"] = "Offline synthetic (physically-plausible)"
    return _clean_raw(df)


# ===========================================================================
# Shared cleaning
# ===========================================================================
def _clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every raw column exists, fill gaps, drop all-night-leading NaNs."""
    attrs = dict(df.attrs)
    df = df.sort_values("Datetime").reset_index(drop=True)

    for col in config.RAW_HOURLY_VARS:
        if col not in df.columns:
            df[col] = np.nan

    # Interpolate short gaps, then forward/back fill, then any remaining -> 0.
    num_cols = config.RAW_HOURLY_VARS
    df[num_cols] = (
        df[num_cols]
        .interpolate(method="linear", limit=6, limit_direction="both")
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    # Radiation can never be negative.
    rad = [c for c in num_cols if "radiation" in c or "irradiance" in c]
    df[rad] = df[rad].clip(lower=0.0)

    df = df[["Datetime"] + config.RAW_HOURLY_VARS]
    df.attrs.update(attrs)
    return df


# ===========================================================================
# Unified loader used by the app / CLI
# ===========================================================================
def load_raw_data(
    latitude: float,
    longitude: float,
    elevation: float | None = None,
    past_days: int = config.DEFAULT_PAST_DAYS,
    forecast_days: int = config.DEFAULT_FORECAST_DAYS,
    prefer: str = "openmeteo",
) -> tuple[pd.DataFrame, str]:
    """
    Fetch the canonical hourly table from the requested government data source,
    falling back to the offline synthetic generator on any failure.

    prefer:
        "nasa"      -> NASA POWER (NASA / U.S. government, near-real-time)
        "openmeteo" -> Open-Meteo live (ECMWF/GFS models) + air quality
                       (alias: "realtime")
        "synthetic" -> offline physically-plausible generator
    Returns (df, status_message).
    """
    prefer = (prefer or "openmeteo").lower()
    if prefer in ("realtime", "open-meteo"):
        prefer = "openmeteo"

    if prefer == "synthetic":
        df = generate_synthetic_raw(latitude, longitude, elevation,
                                    n_days=max(past_days, 30))
        return df, f"🧪 Offline synthetic data · {len(df):,} rows"

    if prefer == "nasa":
        try:
            df = fetch_nasa_power(latitude, longitude, elevation, past_days, forecast_days)
            return df, f"✅ Live data from {df.attrs.get('source')} · {len(df):,} hourly rows"
        except Exception as exc:
            df = generate_synthetic_raw(latitude, longitude, elevation,
                                        n_days=max(past_days, 30))
            return df, (f"⚠️ Could not reach NASA POWER ({exc}). "
                        f"Using offline synthetic data · {len(df):,} rows.")

    # default: Open-Meteo
    try:
        df = fetch_realtime_data(latitude, longitude, elevation, past_days, forecast_days)
        return df, f"✅ Live data from {df.attrs.get('source')} · {len(df):,} hourly rows"
    except Exception as exc:
        df = generate_synthetic_raw(latitude, longitude, elevation,
                                    n_days=max(past_days, 30))
        return df, (f"⚠️ Could not reach Open-Meteo ({exc}). "
                    f"Using offline synthetic data · {len(df):,} rows.")
