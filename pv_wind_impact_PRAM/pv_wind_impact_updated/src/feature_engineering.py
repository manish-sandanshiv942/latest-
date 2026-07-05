"""
src/feature_engineering.py
==========================
Turn the 57 RAW atmospheric variables (from src/data_fetcher) into the full
100-parameter model input space by engineering 43 physically-grounded and
temporal features.

The single public entry point is

    build_feature_table(df_raw, latitude, longitude, elevation) -> DataFrame

which returns the raw columns plus exactly the 43 features listed in
``config.DERIVED_FEATURES`` (asserted), giving 100 input columns in total.

All formulas are standard meteorology / solar-engineering relations:

  * Solar position (declination, hour angle, zenith/elevation/azimuth, air mass,
    extraterrestrial irradiance) -- Cooper / Spencer style equations.
  * Air density from the ideal-gas law; wind shear (power law) to extrapolate
    wind to hub height; wind power density 0.5*rho*v^3.
  * Magnus formula for saturation vapour pressure; specific humidity / mixing
    ratio; Stull (2011) wet-bulb approximation; NWS heat index.
  * Cyclical encodings of hour / day-of-year / month, and short rolling means
    plus a pressure tendency to give the models temporal context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

SOLAR_CONSTANT = 1361.0   # W/m^2


# ===========================================================================
# Solar geometry
# ===========================================================================
def _solar_geometry(dt: pd.Series, latitude: float, longitude: float) -> dict:
    """Vectorised solar position for a naive local-time hourly index."""
    doy = dt.dt.dayofyear.to_numpy().astype(float)
    hour = (dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy() / 60.0).astype(float)

    # Declination (Cooper).
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))      # deg

    # Equation of time (Spencer) for a slightly better hour angle.
    b = np.deg2rad(360.0 * (doy - 1) / 365.0)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(b) - 0.032077 * np.sin(b)
                    - 0.014615 * np.cos(2 * b) - 0.040849 * np.sin(2 * b))  # minutes

    # Approximate solar time from local clock time (timezone already applied by
    # the data source -> assume local standard meridian near the longitude).
    solar_time = hour + eot / 60.0
    hra = 15.0 * (solar_time - 12.0)                                       # deg

    lat_r = np.deg2rad(latitude)
    decl_r = np.deg2rad(decl)
    hra_r = np.deg2rad(hra)

    cos_zen = (np.sin(lat_r) * np.sin(decl_r)
               + np.cos(lat_r) * np.cos(decl_r) * np.cos(hra_r))
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    zenith = np.rad2deg(np.arccos(cos_zen))
    elevation_ang = 90.0 - zenith

    # Azimuth (0 = North, clockwise).
    sin_az = -np.cos(decl_r) * np.sin(hra_r) / np.clip(np.cos(np.deg2rad(elevation_ang)), 1e-3, None)
    cos_az = ((np.sin(decl_r) - np.sin(lat_r) * np.sin(np.deg2rad(elevation_ang)))
              / np.clip(np.cos(lat_r) * np.cos(np.deg2rad(elevation_ang)), 1e-3, None))
    azimuth = np.rad2deg(np.arctan2(sin_az, cos_az)) % 360.0

    # Kasten-Young relative air mass (large when sun is low; cap at night).
    elev_safe = np.clip(elevation_ang, 0.0, 90.0)
    air_mass = np.where(
        elev_safe > 0.0,
        1.0 / (np.cos(np.deg2rad(zenith)) + 0.50572 * (96.07995 - zenith) ** -1.6364),
        0.0,
    )
    air_mass = np.clip(np.nan_to_num(air_mass, nan=0.0, posinf=40.0), 0.0, 40.0)

    # Extraterrestrial irradiance on a horizontal surface (eccentricity corr.).
    ecc = 1.0 + 0.033 * np.cos(np.deg2rad(360.0 * doy / 365.0))
    extra = np.clip(SOLAR_CONSTANT * ecc * np.clip(cos_zen, 0.0, 1.0), 0.0, None)

    # Day length from the sunrise hour angle.
    cos_ws = -np.tan(lat_r) * np.tan(decl_r)
    cos_ws = np.clip(cos_ws, -1.0, 1.0)
    day_length = 2.0 / 15.0 * np.rad2deg(np.arccos(cos_ws))                # hours

    return {
        "solar_zenith_angle": zenith,
        "solar_elevation_angle": elevation_ang,
        "solar_azimuth_angle": azimuth,
        "air_mass": air_mass,
        "extraterrestrial_irradiance": extra,
        "declination_angle": decl,
        "hour_angle": hra,
        "day_length_hours": day_length,
    }


# ===========================================================================
# Thermodynamics / moisture helpers
# ===========================================================================
def _saturation_vapor_pressure(temp_c: np.ndarray) -> np.ndarray:
    """Magnus formula, kPa."""
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def _wet_bulb_stull(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Stull (2011) empirical wet-bulb temperature, deg C (valid ~ -20..50 C)."""
    rh = np.clip(rh_pct, 1.0, 100.0)
    tw = (temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
          + np.arctan(temp_c + rh)
          - np.arctan(rh - 1.676331)
          + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
          - 4.686035)
    return tw


def _heat_index(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """NWS heat index (Rothfusz), returned in deg C."""
    t = temp_c * 9.0 / 5.0 + 32.0                       # to Fahrenheit
    rh = np.clip(rh_pct, 0.0, 100.0)
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh
          - 0.22475541 * t * rh - 6.83783e-3 * t ** 2
          - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t ** 2 * rh
          + 8.5282e-4 * t * rh ** 2 - 1.99e-6 * t ** 2 * rh ** 2)
    hi = np.where(t < 80.0, t, hi)                      # formula only valid when hot
    return (hi - 32.0) * 5.0 / 9.0                      # back to Celsius


# ===========================================================================
# Main builder
# ===========================================================================
def build_feature_table(
    df_raw: pd.DataFrame,
    latitude: float,
    longitude: float,
    elevation: float | None = None,
) -> pd.DataFrame:
    """Return raw + 43 engineered features (100 input columns total)."""
    df = df_raw.copy()

    # Force every raw numeric column to float.  Slider overrides in the
    # forecaster (and some API payloads) can arrive as int, which makes
    # np.zeros_like() integer-typed and breaks in-place np.divide(out=...).
    for _c in config.RAW_HOURLY_VARS:
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce").astype(float)

    elev = float(elevation) if elevation is not None else float(df.attrs.get("elevation", 200.0) or 200.0)

    # --- pull commonly-used raw arrays ---------------------------------
    t2 = df["temperature_2m"].to_numpy()
    t120 = df["temperature_120m"].to_numpy()
    rh = df["relative_humidity_2m"].to_numpy()
    psfc_hpa = df["surface_pressure"].to_numpy()
    psfc_pa = psfc_hpa * 100.0
    ghi = df["shortwave_radiation"].to_numpy()
    dni = df["direct_normal_irradiance"].to_numpy()
    dhi = df["diffuse_radiation"].to_numpy()
    w10 = df["wind_speed_10m"].to_numpy()
    w120 = df["wind_speed_120m"].to_numpy()
    gusts = df["wind_gusts_10m"].to_numpy()
    soil0 = df["soil_temperature_0cm"].to_numpy()
    cloud = df["cloud_cover"].to_numpy()
    precip = df["precipitation"].to_numpy()

    # ===================================================================
    # 1. Solar geometry (8)
    # ===================================================================
    geo = _solar_geometry(df["Datetime"], latitude, longitude)
    for k, v in geo.items():
        df[k] = v
    extra = geo["extraterrestrial_irradiance"]

    # ===================================================================
    # 2. Solar / PV derived (6)
    # ===================================================================
    df["clearness_index"] = np.clip(
        np.divide(ghi, extra, out=np.zeros_like(ghi), where=extra > 5.0), 0.0, 1.0)
    total = ghi.copy()
    df["diffuse_fraction"] = np.clip(
        np.divide(dhi, total, out=np.zeros_like(dhi), where=total > 5.0), 0.0, 1.0)

    # Effective plane-of-array irradiance: beam (incidence-weighted) + diffuse +
    # ground-reflected (albedo) term  G_r = albedo * GHI * (1 - cos β)/2.
    n = len(df)
    cos_aoi = np.clip(np.cos(np.deg2rad(np.clip(geo["solar_zenith_angle"]
                                                - config.PV_TILT_DEG, 0, 90))), 0, 1)
    albedo = (df["surface_albedo"].to_numpy() if "surface_albedo" in df
              else np.full(n, 0.20))
    cos_tilt = np.cos(np.deg2rad(config.PV_TILT_DEG))
    ground_poa = ghi * np.clip(albedo, 0.0, 0.9) * (1.0 - cos_tilt) / 2.0
    eff_irr = np.clip(dhi + dni * cos_aoi + ground_poa, 0.0, 1400.0)
    df["effective_irradiance"] = eff_irr

    # Cell temperature (NOCT model) and the resulting thermal derate.
    cell_t = t2 + (config.PV_NOCT_C - 20.0) / 800.0 * ghi
    df["cell_temperature"] = cell_t
    df["pv_thermal_derate"] = np.clip(1.0 - config.PV_TEMP_COEFF * (cell_t - 25.0),
                                      0.70, 1.05)

    # Soiling index (PV loss): grows with aerosol load (AOD), airborne dust and
    # PM10, and with humidity; rain washes the modules and partially resets it.
    # This is what makes AOD / dust / PM genuinely affect PV output and rank in
    # the impact analysis (mirrors the deck's exponential soiling model).
    aod = (df["aerosol_optical_depth"].to_numpy() if "aerosol_optical_depth" in df
           else np.zeros(n))
    dust = df["dust"].to_numpy() if "dust" in df else np.zeros(n)
    pm10 = df["pm10"].to_numpy() if "pm10" in df else np.zeros(n)
    aerosol_load = (np.clip(aod / 1.0, 0, 1)
                    + np.clip(dust / 120.0, 0, 1)
                    + np.clip(pm10 / 150.0, 0, 1)) / 3.0
    soiling = 1.0 - 0.05 * np.clip((rh - 50.0) / 50.0, 0, 1) - 0.10 * aerosol_load
    soiling = np.where(precip > 0.5, np.minimum(soiling + 0.08, 1.0), soiling)
    df["soiling_index"] = np.clip(soiling, 0.78, 1.0)

    # ===================================================================
    # 3. Wind derived (9)
    # ===================================================================
    t2_k = t2 + 273.15
    rho = psfc_pa / (287.05 * t2_k)                     # ideal-gas air density
    df["air_density"] = rho
    df["air_density_ratio"] = rho / 1.225

    # Air density at hub height using temp_120m and a pressure lapse.
    p_hub_pa = psfc_pa * (1.0 - 2.25577e-5 * config.WIND_HUB_HEIGHT_M) ** 5.25588
    t_hub_k = t120 + 273.15
    rho_hub = p_hub_pa / (287.05 * np.clip(t_hub_k, 200, None))
    df["air_density_hub"] = rho_hub

    # Wind shear exponent alpha from the 10 m and 120 m speeds (power law).
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(w120, w10, out=np.full_like(w10, np.nan), where=w10 > 0.3)
        alpha = np.log(np.clip(ratio, 1e-3, None)) / np.log(120.0 / 10.0)
    alpha = np.clip(np.nan_to_num(alpha, nan=0.20), 0.0, 0.6)
    df["wind_shear_exponent"] = alpha

    # Wind speed at hub height (extrapolate the 10 m wind with that exponent).
    w_hub = w10 * (config.WIND_HUB_HEIGHT_M / 10.0) ** alpha
    df["wind_speed_hub"] = np.clip(w_hub, 0.0, 45.0)

    df["wind_power_density_10m"] = 0.5 * rho * w10 ** 3
    df["wind_power_density_hub"] = 0.5 * rho_hub * df["wind_speed_hub"].to_numpy() ** 3

    df["gust_factor"] = np.where(w10 > 0.3, np.clip(gusts / w10, 1.0, 4.0), 1.0)
    df["turbulence_intensity"] = np.where(
        w10 > 0.3, np.clip((gusts - w10) / w10, 0.0, 2.0), 0.0)

    # ===================================================================
    # 4. Thermodynamics / moisture (10)
    # ===================================================================
    es = _saturation_vapor_pressure(t2)                 # kPa
    ea = es * np.clip(rh, 0, 100) / 100.0
    df["saturation_vapor_pressure"] = es
    df["actual_vapor_pressure"] = ea

    # Specific humidity & mixing ratio (g/kg) from vapour pressure & pressure.
    p_kpa = psfc_hpa / 10.0
    w_mix = 0.622 * ea / np.clip(p_kpa - ea, 1e-3, None)            # kg/kg
    df["mixing_ratio"] = np.clip(w_mix * 1000.0, 0, None)          # g/kg
    df["specific_humidity"] = np.clip(w_mix / (1.0 + w_mix) * 1000.0, 0, None)

    df["wet_bulb_temperature"] = _wet_bulb_stull(t2, rh)
    df["heat_index"] = _heat_index(t2, rh)
    df["dew_point_depression"] = t2 - df["dew_point_2m"].to_numpy()
    df["temp_air_soil_diff"] = t2 - soil0
    df["temp_lapse_rate"] = t2 - t120

    # Potential temperature (K) referenced to 1000 hPa.
    df["potential_temperature"] = t2_k * (1000.0 / np.clip(psfc_hpa, 1, None)) ** 0.286

    # ===================================================================
    # 5. Temporal cyclical encodings (6)
    # ===================================================================
    hour = df["Datetime"].dt.hour.to_numpy().astype(float)
    doy = df["Datetime"].dt.dayofyear.to_numpy().astype(float)
    month = df["Datetime"].dt.month.to_numpy().astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # ===================================================================
    # 6. Temporal dynamics: rolling means + pressure tendency (4)
    # ===================================================================
    df["ghi_roll3_mean"] = (df["shortwave_radiation"]
                            .rolling(3, min_periods=1).mean().to_numpy())
    df["wind10_roll3_mean"] = (df["wind_speed_10m"]
                               .rolling(3, min_periods=1).mean().to_numpy())
    df["temp_roll3_mean"] = (df["temperature_2m"]
                             .rolling(3, min_periods=1).mean().to_numpy())
    df["pressure_tendency_3h"] = (df["surface_pressure"]
                                  - df["surface_pressure"].shift(3)).fillna(0.0).to_numpy()

    # --- finalise -------------------------------------------------------
    missing = [f for f in config.DERIVED_FEATURES if f not in df.columns]
    assert not missing, f"feature engineering missing: {missing}"

    # Replace any inf/NaN that slipped through.
    df[config.FEATURES] = (df[config.FEATURES]
                           .replace([np.inf, -np.inf], np.nan)
                           .fillna(0.0))
    return df


def attach_power_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: add PV/Wind power columns via the physical models."""
    from src import power_models
    df = df.copy()
    df[config.TARGET_PV] = power_models.pv_power_from_features(df)
    df[config.TARGET_WIND] = power_models.wind_power_from_features(df)
    return df
