"""
src/power_models.py
===================
Calibrated *physical* models that convert the real atmospheric conditions into
plant power output (the two ML targets).  Because Open-Meteo provides weather,
not SCADA readings, the power output is produced by transparent engineering
models that are driven entirely by the real fetched/derived data:

    pv_power_from_features(df)   -> PV_Power_kW   (irradiance + thermal + soiling)
    wind_power_from_features(df) -> Wind_Power_kW (turbine power curve + air density)

These are the same physics used in the original single-file app, upgraded to
consume the richer 100-parameter feature table (effective POA irradiance, cell
temperature, hub-height wind, hub air-density ratio, soiling index, ...).

A small multiplicative measurement noise (~5 %) is added so the ML models have a
realistic, non-degenerate signal to learn (and so R^2 is not a trivial 1.0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


# ===========================================================================
# Photovoltaic
# ===========================================================================
def pv_power_from_features(df: pd.DataFrame, add_noise: bool = True,
                           seed: int = config.RANDOM_STATE) -> np.ndarray:
    """
    PV AC power (kW), clipped to inverter capacity.

    DC power = effective POA irradiance / 1000  * area * efficiency
               * thermal_derate * soiling_index
    then clipped at the AC capacity.
    """
    rng = np.random.default_rng(seed)

    # Prefer engineered effective irradiance; fall back to tilted/GHI.
    if "effective_irradiance" in df:
        poa = df["effective_irradiance"].to_numpy()
    elif "global_tilted_irradiance" in df:
        poa = df["global_tilted_irradiance"].to_numpy()
    else:
        poa = df["shortwave_radiation"].to_numpy()

    derate_t = (df["pv_thermal_derate"].to_numpy() if "pv_thermal_derate" in df
                else np.ones(len(df)))
    soiling = (df["soiling_index"].to_numpy() if "soiling_index" in df
               else np.ones(len(df)))

    dc_kw = (poa / 1000.0) * config.PV_AREA_M2 * config.PV_EFFICIENCY * derate_t * soiling
    if add_noise:
        dc_kw = dc_kw * rng.normal(1.0, 0.05, len(df))
    return np.clip(dc_kw, 0.0, config.PV_CAPACITY_KW)


# ===========================================================================
# Wind
# ===========================================================================
def _turbine_curve(v: np.ndarray) -> np.ndarray:
    """Ideal turbine power (kW) from the hub-height wind speed (cubic ramp)."""
    power = np.zeros_like(v)
    ramp = (v >= config.WIND_CUT_IN) & (v < config.WIND_RATED_SPEED)
    rated = (v >= config.WIND_RATED_SPEED) & (v <= config.WIND_CUT_OUT)
    power[ramp] = config.WIND_RATED_KW * (
        (v[ramp] ** 3 - config.WIND_CUT_IN ** 3)
        / (config.WIND_RATED_SPEED ** 3 - config.WIND_CUT_IN ** 3)
    )
    power[rated] = config.WIND_RATED_KW
    return power


def wind_power_from_features(df: pd.DataFrame, add_noise: bool = True,
                             seed: int = config.RANDOM_STATE + 1) -> np.ndarray:
    """
    Wind electrical power (kW): ideal power curve at hub-height wind speed,
    scaled by the hub-height air-density ratio (rho / 1.225).
    """
    rng = np.random.default_rng(seed)

    v_hub = (df["wind_speed_hub"].to_numpy() if "wind_speed_hub" in df
             else df["wind_speed_10m"].to_numpy())

    if "air_density_hub" in df:
        density_ratio = df["air_density_hub"].to_numpy() / 1.225
    elif "air_density_ratio" in df:
        density_ratio = df["air_density_ratio"].to_numpy()
    else:
        density_ratio = np.ones(len(df))

    power = _turbine_curve(v_hub) * np.clip(density_ratio, 0.6, 1.3)
    if add_noise:
        power = power * rng.normal(1.0, 0.05, len(df))
    return np.clip(power, 0.0, config.WIND_RATED_KW)


# ===========================================================================
# Single-point prediction helper (used by the live "now" snapshot / forecaster)
# ===========================================================================
def physical_point_estimate(row: pd.Series) -> dict:
    """Physics-only PV & wind estimate for one feature row (no ML, no noise)."""
    single = pd.DataFrame([row])
    pv = float(pv_power_from_features(single, add_noise=False)[0])
    wd = float(wind_power_from_features(single, add_noise=False)[0])
    return {"pv_kw": pv, "wind_kw": wd, "total_kw": pv + wd}


# ===========================================================================
# PV YIELD ESTIMATION  (the standard solar-resource metrics)
# ===========================================================================
def pv_yield_metrics(feat: pd.DataFrame) -> dict:
    """
    Standard photovoltaic yield-estimation metrics for the rooftop PV system,
    computed from the simulated hourly PV energy and the plane-of-array
    insolation that drives it.

        specific yield        kWh per kWp over the observed window
        annual specific yield kWh/kWp/year  (window extrapolated to 365 days)
                              -- THE headline PV yield figure
        performance ratio     final yield / reference yield (system-loss factor)
        peak sun hours        equivalent full-sun (1 kW/m^2) hours per day
        annual energy         kWh/year for the whole array

    The DC nameplate used as the kWp basis is PV_AREA_M2 * PV_EFFICIENCY
    (the array's output at 1000 W/m^2 STC).
    """
    n = len(feat)
    days = max(n / 24.0, 1e-6)
    annual_factor = 365.0 / days

    pv_kwh = float(feat[config.TARGET_PV].sum())
    pdc_kwp = config.PV_AREA_M2 * config.PV_EFFICIENCY      # DC nameplate (kWp)

    # Plane-of-array insolation driving the model (kWh/m^2 over the window).
    if "effective_irradiance" in feat:
        poa = feat["effective_irradiance"]
    elif "global_tilted_irradiance" in feat:
        poa = feat["global_tilted_irradiance"]
    else:
        poa = feat["shortwave_radiation"]
    poa_kwh_m2 = float(poa.sum()) / 1000.0                  # W/m^2 * 1 h -> kWh/m^2

    yf_period = pv_kwh / pdc_kwp if pdc_kwp > 0 else 0.0    # final yield (kWh/kWp)
    yr_period = poa_kwh_m2                                  # reference yield (h)
    pr = yf_period / yr_period if yr_period > 0 else 0.0

    return {
        "days": days,
        "pdc_kwp": pdc_kwp,
        "pv_kwh_period": pv_kwh,
        "specific_yield_period": yf_period,
        "annual_specific_yield": yf_period * annual_factor,   # kWh/kWp/year
        "annual_energy_kwh": pv_kwh * annual_factor,
        "performance_ratio": pr,
        "poa_kwh_m2_period": poa_kwh_m2,
        "peak_sun_hours": poa_kwh_m2 / days,                  # hours/day
    }
