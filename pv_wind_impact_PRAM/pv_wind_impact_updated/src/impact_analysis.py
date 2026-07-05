"""
src/impact_analysis.py
======================
The scientific core: *which atmospheric parameters drive PV vs wind output?*

Provided analyses:

  permutation_impact(result)        model-agnostic permutation importance over
                                    all 100 features (the headline ranking).

  grouped_impact(importance_df)     roll the 100 features up into physical
                                    categories (radiation, temperature, wind,
                                    humidity, cloud, ...) for an interpretable
                                    high-level picture.

  secondary_impact(df, kind, model) the two-level method from the original app:
                                    normalise out the first-order driver
                                    (irradiance for PV, hub wind for wind) and
                                    rank the atmospheric modulators of conversion
                                    *efficiency* -- the subtle secondary effects
                                    a raw output ranking hides.

  correlation_impact(df, target)    |Pearson r| of each feature with the target.

  shap_impact(result)               mean |SHAP value| per feature (only if the
                                    `shap` package is installed; otherwise None).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

import config
from src import ml_models

RS = config.RANDOM_STATE


# ===========================================================================
# Category map for grouped importance
# ===========================================================================
def _category(feature: str) -> str:
    f = feature
    if "radiation" in f or "irradiance" in f or f in {
        "clearness_index", "diffuse_fraction", "effective_irradiance",
        "solar_zenith_angle", "solar_elevation_angle", "solar_azimuth_angle",
        "air_mass", "declination_angle", "hour_angle", "day_length_hours",
        "sunshine_duration",
    }:
        return "Solar / radiation"
    if f.startswith("wind") or f in {
        "gust_factor", "turbulence_intensity", "air_density", "air_density_hub",
        "air_density_ratio", "wind_shear_exponent",
    }:
        return "Wind / air-density"
    if "temperature" in f or f in {
        "apparent_temperature", "cell_temperature", "pv_thermal_derate",
        "temp_air_soil_diff", "temp_lapse_rate", "potential_temperature",
        "heat_index", "wet_bulb_temperature", "freezing_level_height",
    } or f.startswith("soil_temperature") or f.startswith("temp_roll"):
        return "Temperature"
    if f in {
        "relative_humidity_2m", "dew_point_2m", "dew_point_depression",
        "vapour_pressure_deficit", "saturation_vapor_pressure",
        "actual_vapor_pressure", "specific_humidity", "mixing_ratio",
        "evapotranspiration", "et0_fao_evapotranspiration", "soiling_index",
    } or f.startswith("soil_moisture"):
        return "Humidity / moisture"
    if "cloud" in f:
        return "Cloud cover"
    if f in {"precipitation", "rain", "showers", "snowfall", "snow_depth",
             "precipitation_probability", "weather_code", "visibility"}:
        return "Precipitation / sky"
    if f in {"pressure_msl", "surface_pressure", "pressure_tendency_3h"}:
        return "Pressure"
    if f.endswith("_sin") or f.endswith("_cos") or f == "is_day":
        return "Temporal"
    return "Other"


# ===========================================================================
# Permutation importance (headline)
# ===========================================================================
def permutation_impact(result: "ml_models.TrainResult", n_repeats: int = 8) -> pd.DataFrame:
    """Permutation importance on the held-out test set; sorted descending."""
    res = permutation_importance(
        result.model, result.X_test, result.y_test,
        n_repeats=n_repeats, random_state=RS, n_jobs=1,
    )
    imp = pd.DataFrame({
        "Feature": config.FEATURES,
        "Importance": np.clip(res.importances_mean, 0, None),
        "Std": res.importances_std,
    })
    total = imp["Importance"].sum()
    imp["Share"] = imp["Importance"] / total if total > 0 else 0.0
    imp["Category"] = imp["Feature"].map(_category)
    imp["Label"] = imp["Feature"].map(config.pretty)
    return imp.sort_values("Importance", ascending=False).reset_index(drop=True)


def grouped_impact(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a feature-level importance table into physical categories."""
    g = (importance_df.groupby("Category")["Importance"].sum()
         .sort_values(ascending=False).reset_index())
    total = g["Importance"].sum()
    g["Share"] = g["Importance"] / total if total > 0 else 0.0
    return g


# ===========================================================================
# Correlation
# ===========================================================================
def correlation_impact(df: pd.DataFrame, target: str, top: int = 20) -> pd.DataFrame:
    """|Pearson r| of each feature with the target."""
    corr = df[config.FEATURES].corrwith(df[target]).abs()
    out = (corr.rename("AbsCorr").reset_index()
           .rename(columns={"index": "Feature"})
           .sort_values("AbsCorr", ascending=False))
    out["Label"] = out["Feature"].map(config.pretty)
    out["Category"] = out["Feature"].map(_category)
    return out.head(top).reset_index(drop=True)


# ===========================================================================
# Secondary (efficiency) impact -- the two-level method
# ===========================================================================
def secondary_impact(df: pd.DataFrame, kind: str, model_name: str,
                     n_repeats: int = 8) -> pd.DataFrame:
    """
    Rank the atmospheric modulators of *conversion efficiency* once the
    first-order driver is normalised out.

    kind = "pv"   -> efficiency = PV power / ideal irradiance-limited power,
                     primary driver (shortwave_radiation) excluded.
    kind = "wind" -> efficiency = wind power / ideal power-curve power,
                     primary driver (wind_speed_hub) excluded.
    """
    if kind == "pv":
        # Efficiency relative to the plane-of-array (POA) irradiance-limited
        # ideal, so the diffuse/geometry effects are already in the denominator
        # and the residual reflects the true efficiency modulators (temperature,
        # soiling from aerosols/dust, albedo, humidity, wind cooling).
        ideal = (df["effective_irradiance"] / 1000.0) * config.PV_AREA_M2 * config.PV_EFFICIENCY
        mask = ideal > 5.0
        sub = df[mask].copy()
        sub["_eff"] = np.clip(sub[config.TARGET_PV] / ideal[mask], 0, 1.2)
        drop = {
            # available-light proxies (already captured by the POA ideal)
            "shortwave_radiation", "direct_radiation", "diffuse_radiation",
            "direct_normal_irradiance", "global_tilted_irradiance",
            "terrestrial_radiation", "effective_irradiance", "ghi_roll3_mean",
            "clearness_index", "diffuse_fraction", "extraterrestrial_irradiance",
            "solar_zenith_angle", "solar_elevation_angle", "solar_azimuth_angle",
            "air_mass", "uv_index",
            # derived efficiency intermediates -> we want the raw atmospheric causes
            "cell_temperature", "pv_thermal_derate", "soiling_index",
            # evapotranspiration is a radiation/humidity-derived output, not a
            # mechanistic PV-efficiency modulator -> exclude so the real ones show
            "evapotranspiration", "et0_fao_evapotranspiration",
        }
    else:
        from src.power_models import _turbine_curve
        v_hub = df["wind_speed_hub"].to_numpy()
        ideal = _turbine_curve(v_hub)
        mask = ideal > 5.0
        sub = df[mask].copy()
        sub["_eff"] = np.clip(sub[config.TARGET_WIND].to_numpy() / ideal[mask], 0, 1.5)
        drop = {"wind_speed_hub", "wind_speed_10m", "wind_speed_80m",
                "wind_speed_120m", "wind_speed_180m", "wind_power_density_10m",
                "wind_power_density_hub", "wind_gusts_10m", "wind10_roll3_mean"}

    feats = [f for f in config.FEATURES if f not in drop]
    if len(sub) < 50:
        # Not enough productive samples -> return empty-ish frame gracefully.
        return pd.DataFrame({"Feature": feats, "Importance": 0.0,
                             "Share": 0.0, "Label": [config.pretty(f) for f in feats],
                             "Category": [_category(f) for f in feats]})

    X, y = sub[feats], sub["_eff"]
    # Chronological split (no shuffle): this is hourly time-series data, so a
    # shuffled split would leak temporally adjacent rows between train and test.
    n_te = max(1, int(round(len(X) * 0.2)))
    X_tr, X_te = X.iloc[:-n_te], X.iloc[-n_te:]
    y_tr, y_te = y.iloc[:-n_te], y.iloc[-n_te:]
    model = ml_models.MODEL_CATALOGUE[model_name]()
    model.fit(X_tr, y_tr)
    res = permutation_importance(model, X_te, y_te, n_repeats=n_repeats,
                                 random_state=RS, n_jobs=1)
    imp = pd.DataFrame({"Feature": feats,
                        "Importance": np.clip(res.importances_mean, 0, None)})
    total = imp["Importance"].sum()
    imp["Share"] = imp["Importance"] / total if total > 0 else 0.0
    imp["Label"] = imp["Feature"].map(config.pretty)
    imp["Category"] = imp["Feature"].map(_category)
    return imp.sort_values("Importance", ascending=False).reset_index(drop=True)


# ===========================================================================
# Optional SHAP (explainable AI)
# ===========================================================================
def shap_impact(result: "ml_models.TrainResult", max_samples: int = 400):
    """
    Mean |SHAP value| per feature for tree models, if the `shap` package is
    available.  Returns a DataFrame or None.
    """
    try:
        import shap
    except Exception:
        return None

    model = result.model
    # Only attempt the fast TreeExplainer on bare tree estimators.
    tree_like = model.__class__.__name__ in {
        "RandomForestRegressor", "ExtraTreesRegressor",
        "GradientBoostingRegressor", "HistGradientBoostingRegressor",
        "DecisionTreeRegressor", "XGBRegressor", "LGBMRegressor",
    }
    if not tree_like:
        return None

    X = result.X_test
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=RS)
    try:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        mean_abs = np.abs(vals).mean(axis=0)
    except Exception:
        return None

    imp = pd.DataFrame({"Feature": config.FEATURES, "MeanAbsSHAP": mean_abs})
    total = imp["MeanAbsSHAP"].sum()
    imp["Share"] = imp["MeanAbsSHAP"] / total if total > 0 else 0.0
    imp["Label"] = imp["Feature"].map(config.pretty)
    imp["Category"] = imp["Feature"].map(_category)
    return imp.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
