"""
config.py
=========
Central configuration for the project:

    "Impact Analysis of Atmospheric Parameters on Photovoltaic and Wind Power
     Output Using Data-Driven Modelling and Machine Learning"

Everything that defines the *shape* of the problem lives here:

* the location defaults (latitude / longitude / elevation),
* the representative PV plant and wind turbine specifications,
* the EXACT list of the 100 atmospheric / derived parameters used as model
  inputs, split into

      RAW_HOURLY_VARS  -> 57 variables fetched LIVE from Open-Meteo
      DERIVED_FEATURES -> 43 physics / temporal features engineered locally
                          (see src/feature_engineering.py)

  57 + 43 = 100 input parameters.  The two targets (PV_Power_kW and
  Wind_Power_kW) are computed by calibrated physical models that are *driven by
  the real fetched data* (see src/power_models.py).

Nothing here imports streamlit, pandas or numpy, so it is safe to import from
anywhere (tests, CLI, the app).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Default location  (Government College of Engineering, Karad -- Block M hostel)
# Vidyanagar / Saidapur, Karad, Maharashtra.  At the weather-grid resolution
# (~2-11 km) every block on the ~40-acre campus shares the same atmospheric and
# solar data, so any hostel block uses the campus coordinates below.
# ---------------------------------------------------------------------------
DEFAULT_LATITUDE = 17.2877
DEFAULT_LONGITUDE = 74.1818
DEFAULT_ELEVATION = 565.0          # metres above sea level; None -> auto-detect

# Named site presets selectable in the app sidebar / CLI (--site).
# Each: (latitude, longitude, elevation_m, description).  Coordinates can be
# taken from Google Maps (right-click a point -> copy the lat, lon).
SITES = {
    "GCE Karad — Block M Hostel": (17.2877, 74.1818, 565.0,
        "Govt. College of Engineering, Karad · Vidyanagar · default site"),
    "GCE Karad — Main Campus": (17.2880, 74.1822, 565.0,
        "GCEK academic campus, Vidyanagar, Karad"),
    # --- All-India reference locations (one per region/climate) ---------
    "Mumbai (West coast)": (19.0760, 72.8777, 14.0, "Coastal, humid, Maharashtra"),
    "Pune (Deccan)": (18.5204, 73.8567, 560.0, "Inland plateau, Maharashtra"),
    "New Delhi (North plains)": (28.6139, 77.2090, 216.0, "Indo-Gangetic plain"),
    "Jodhpur (Thar desert)": (26.2389, 73.0243, 231.0, "Arid, very high solar, Rajasthan"),
    "Leh (Himalaya)": (34.1526, 77.5771, 3500.0, "High-altitude cold desert, Ladakh"),
    "Bengaluru (South plateau)": (12.9716, 77.5946, 920.0, "Mild, elevated, Karnataka"),
    "Chennai (East coast)": (13.0827, 80.2707, 6.0, "Coastal, humid, Tamil Nadu"),
    "Kolkata (East)": (22.5726, 88.3639, 9.0, "Humid subtropical, West Bengal"),
    "Hyderabad (Central)": (17.3850, 78.4867, 542.0, "Semi-arid, Telangana"),
    "Ahmedabad (West)": (23.0225, 72.5714, 53.0, "Hot semi-arid, Gujarat"),
    "Jaisalmer (Solar park)": (26.9157, 70.9083, 225.0, "Among India's best solar sites"),
    "Guwahati (North-east)": (26.1445, 91.7362, 55.0, "High rainfall, Assam"),
    "Thiruvananthapuram (SW coast)": (8.5241, 76.9366, 16.0, "Tropical, Kerala"),
}
DEFAULT_SITE = "GCE Karad — Block M Hostel"

# How much real data to pull for training / analysis.
# The Open-Meteo *forecast* endpoint serves recent past + live + near forecast.
DEFAULT_PAST_DAYS = 92             # max 92 for the forecast endpoint
DEFAULT_FORECAST_DAYS = 2          # near-term forecast appended to the history

# ---------------------------------------------------------------------------
# Representative plant specifications
# (used to convert real atmospheric conditions -> plant power output)
# ---------------------------------------------------------------------------
# -- Photovoltaic array --
PV_CAPACITY_KW = 300.0             # AC capacity / inverter clipping limit (kW)
PV_AREA_M2 = 1500.0                # total module area (m^2)
PV_EFFICIENCY = 0.19               # module efficiency at STC
PV_TEMP_COEFF = 0.0040             # power loss per degC above 25 C (~ -0.40 %/C)
PV_NOCT_C = 45.0                   # nominal operating cell temperature
PV_TILT_DEG = 20.0                 # panel tilt (used for tilted irradiance proxy)

# -- Wind turbine --
WIND_RATED_KW = 500.0              # rated electrical power (kW)
WIND_CUT_IN = 3.0                  # m/s, turbine starts producing
WIND_RATED_SPEED = 12.0            # m/s, rated power reached
WIND_CUT_OUT = 25.0               # m/s, safety shutdown
WIND_HUB_HEIGHT_M = 100.0          # hub height (m) -> wind extrapolated to this

# ---------------------------------------------------------------------------
# The 57 RAW hourly variables requested LIVE from Open-Meteo.
# Names are the exact Open-Meteo `hourly=` API identifiers.
# ---------------------------------------------------------------------------
RAW_HOURLY_VARS = [
    # --- Solar radiation (6) : the primary PV drivers --------------------
    "shortwave_radiation",               # GHI  (W/m^2)
    "direct_radiation",                  # beam on horizontal
    "diffuse_radiation",                 # diffuse on horizontal (DHI)
    "direct_normal_irradiance",          # DNI
    "global_tilted_irradiance",          # GTI on the configured tilt
    "terrestrial_radiation",             # top-of-atmosphere
    # --- Air temperature at several heights (6) -------------------------
    "temperature_2m",
    "temperature_80m",
    "temperature_120m",
    "temperature_180m",
    "apparent_temperature",
    "dew_point_2m",
    # --- Soil temperature profile (4) -----------------------------------
    "soil_temperature_0cm",
    "soil_temperature_6cm",
    "soil_temperature_18cm",
    "soil_temperature_54cm",
    # --- Humidity / moisture / evapotranspiration (8) -------------------
    "relative_humidity_2m",
    "vapour_pressure_deficit",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
    "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm",
    # --- Wind at several heights (9) : the primary wind drivers ----------
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_direction_180m",
    "wind_gusts_10m",
    # --- Pressure (2) ---------------------------------------------------
    "pressure_msl",
    "surface_pressure",
    # --- Cloud cover (4) ------------------------------------------------
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    # --- Precipitation family (3) ---------------------------------------
    "precipitation",
    "rain",
    "snowfall",
    # --- Aerosols / air quality / optical (10) : critical PV soiling -----
    # (live from the Open-Meteo Air-Quality API; the deck flags AOD, dust,
    #  PM, UV and albedo as among the 15 most critical parameters)
    "aerosol_optical_depth",             # AOD 550 nm (Beer-Lambert attenuation)
    "dust",                              # surface dust concentration (µg/m^3)
    "pm2_5",                             # fine particulate (µg/m^3)
    "pm10",                              # coarse particulate (µg/m^3) -> soiling
    "uv_index",                          # surface UV index
    "ozone",                             # column / surface ozone
    "carbon_monoxide",                   # CO (µg/m^3)
    "nitrogen_dioxide",                  # NO2 (µg/m^3)
    "sulphur_dioxide",                   # SO2 (µg/m^3)
    "surface_albedo",                    # ground reflectivity (bifacial / POA)
    # --- Other meteorological state (5) ---------------------------------
    "visibility",
    "is_day",
    "sunshine_duration",
    "freezing_level_height",
    "soil_moisture_27_to_81cm",
]
assert len(RAW_HOURLY_VARS) == 57, f"expected 57 raw vars, got {len(RAW_HOURLY_VARS)}"

# ---------------------------------------------------------------------------
# The 43 DERIVED features engineered in src/feature_engineering.py.
# Order here is purely documentary; the engineering module is the source of
# truth and is asserted against this list at import time.
# ---------------------------------------------------------------------------
DERIVED_FEATURES = [
    # --- Solar geometry (8) ---------------------------------------------
    "solar_zenith_angle",
    "solar_elevation_angle",
    "solar_azimuth_angle",
    "air_mass",
    "extraterrestrial_irradiance",
    "declination_angle",
    "hour_angle",
    "day_length_hours",
    # --- Solar / PV derived (6) -----------------------------------------
    "clearness_index",
    "diffuse_fraction",
    "effective_irradiance",
    "cell_temperature",
    "pv_thermal_derate",
    "soiling_index",
    # --- Wind derived (9) -----------------------------------------------
    "air_density",
    "air_density_ratio",
    "air_density_hub",
    "wind_shear_exponent",
    "wind_speed_hub",
    "wind_power_density_10m",
    "wind_power_density_hub",
    "gust_factor",
    "turbulence_intensity",
    # --- Thermodynamics / moisture (10) ---------------------------------
    "saturation_vapor_pressure",
    "actual_vapor_pressure",
    "specific_humidity",
    "mixing_ratio",
    "wet_bulb_temperature",
    "heat_index",
    "dew_point_depression",
    "temp_air_soil_diff",
    "temp_lapse_rate",
    "potential_temperature",
    # --- Temporal cyclical encodings (6) --------------------------------
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    # --- Temporal dynamics: rolling means + pressure tendency (4) --------
    "ghi_roll3_mean",
    "wind10_roll3_mean",
    "temp_roll3_mean",
    "pressure_tendency_3h",
]
assert len(DERIVED_FEATURES) == 43, f"expected 43 derived, got {len(DERIVED_FEATURES)}"

# The full 100-parameter model input space.
FEATURES = RAW_HOURLY_VARS + DERIVED_FEATURES
assert len(FEATURES) == 100, f"expected 100 features, got {len(FEATURES)}"

# Targets (NOT part of the 100 inputs; produced by the physical models).
TARGET_PV = "PV_Power_kW"
TARGET_WIND = "Wind_Power_kW"
TARGETS = [TARGET_PV, TARGET_WIND]

# The first-order driver of each target -- excluded when we look for the
# *secondary* atmospheric modulators of conversion efficiency.
PV_PRIMARY_DRIVER = "shortwave_radiation"
WIND_PRIMARY_DRIVER = "wind_speed_hub"

# ---------------------------------------------------------------------------
# The 15 MOST CRITICAL atmospheric parameters for PV + wind generation.
# A focused subset of the 100, used for the "15 critical parameters" view.
# Edit this list to match the parameters in your reference PPT/paper.
# Every name here must be one of the 100 in FEATURES.
# ---------------------------------------------------------------------------
CRITICAL_PARAMETERS = [
    "shortwave_radiation",        # 1  Solar irradiance (GHI/DNI/DHI)   [SOLAR]
    "temperature_2m",             # 2  Ambient temperature              [THERMAL]
    "relative_humidity_2m",       # 3  Relative humidity                [MOISTURE]
    "wind_speed_hub",             # 4  Wind speed (hub height)          [WIND]
    "wind_direction_10m",         # 5  Wind direction                   [WIND]
    "air_density",                # 6  Air density                      [THERMO]
    "surface_pressure",           # 7  Atmospheric pressure             [PRESSURE]
    "cloud_cover",                # 8  Cloud cover fraction             [CLOUD]
    "rain",                       # 9  Rainfall amount                  [PRECIP]
    "aerosol_optical_depth",      # 10 Aerosol optical depth (AOD)      [AEROSOL]
    "dust",                       # 11 Dust concentration               [AEROSOL]
    "visibility",                 # 12 Visibility                       [OPTICAL]
    "dew_point_2m",               # 13 Dew-point temperature            [MOISTURE]
    "turbulence_intensity",       # 14 Turbulence intensity             [WIND]
    "surface_albedo",             # 15 Albedo (surface reflectivity)    [SURFACE]
]
assert all(p in FEATURES for p in CRITICAL_PARAMETERS), \
    "every critical parameter must be one of the 100 features"
assert len(CRITICAL_PARAMETERS) == 15, "expected 15 critical parameters"

# Category tag for each critical parameter (mirrors the reference deck).
CRITICAL_CATEGORY = {
    "shortwave_radiation": "SOLAR",
    "temperature_2m": "THERMAL",
    "relative_humidity_2m": "MOISTURE",
    "wind_speed_hub": "WIND",
    "wind_direction_10m": "WIND",
    "air_density": "THERMO",
    "surface_pressure": "PRESSURE",
    "cloud_cover": "CLOUD",
    "rain": "PRECIP",
    "aerosol_optical_depth": "AEROSOL",
    "dust": "AEROSOL",
    "visibility": "OPTICAL",
    "dew_point_2m": "MOISTURE",
    "turbulence_intensity": "WIND",
    "surface_albedo": "SURFACE",
}

# ---------------------------------------------------------------------------
# Human-readable labels for plots / tables (best-effort; unknown -> raw name)
# ---------------------------------------------------------------------------
PRETTY = {
    "shortwave_radiation": "Solar irradiance GHI (W/m²)",
    "direct_radiation": "Direct radiation (W/m²)",
    "diffuse_radiation": "Diffuse radiation (W/m²)",
    "direct_normal_irradiance": "DNI (W/m²)",
    "global_tilted_irradiance": "Tilted irradiance POA (W/m²)",
    "terrestrial_radiation": "Top-of-atmosphere irradiance (W/m²)",
    "temperature_2m": "Air temperature 2 m (°C)",
    "temperature_80m": "Air temperature 80 m (°C)",
    "temperature_120m": "Air temperature 120 m (°C)",
    "temperature_180m": "Air temperature 180 m (°C)",
    "apparent_temperature": "Apparent temperature (°C)",
    "dew_point_2m": "Dew point (°C)",
    "relative_humidity_2m": "Relative humidity (%)",
    "vapour_pressure_deficit": "Vapour-pressure deficit (kPa)",
    "wind_speed_10m": "Wind speed 10 m (m/s)",
    "wind_speed_80m": "Wind speed 80 m (m/s)",
    "wind_speed_120m": "Wind speed 120 m (m/s)",
    "wind_speed_180m": "Wind speed 180 m (m/s)",
    "wind_speed_hub": "Wind speed @ hub (m/s)",
    "wind_gusts_10m": "Wind gusts 10 m (m/s)",
    "wind_direction_10m": "Wind direction 10 m (°)",
    "pressure_msl": "Mean-sea-level pressure (hPa)",
    "surface_pressure": "Surface pressure (hPa)",
    "cloud_cover": "Total cloud cover (%)",
    "cloud_cover_low": "Low cloud cover (%)",
    "cloud_cover_mid": "Mid cloud cover (%)",
    "cloud_cover_high": "High cloud cover (%)",
    "precipitation": "Precipitation (mm)",
    "rain": "Rain (mm)",
    "snowfall": "Snowfall (cm)",
    "aerosol_optical_depth": "Aerosol optical depth AOD (-)",
    "dust": "Dust concentration (µg/m³)",
    "pm2_5": "PM2.5 (µg/m³)",
    "pm10": "PM10 (µg/m³)",
    "uv_index": "UV index (-)",
    "ozone": "Ozone (µg/m³)",
    "carbon_monoxide": "Carbon monoxide (µg/m³)",
    "nitrogen_dioxide": "Nitrogen dioxide (µg/m³)",
    "sulphur_dioxide": "Sulphur dioxide (µg/m³)",
    "surface_albedo": "Surface albedo (-)",
    "visibility": "Visibility (m)",
    "air_density": "Air density (kg/m³)",
    "air_density_ratio": "Air-density ratio (-)",
    "air_density_hub": "Air density @ hub (kg/m³)",
    "wind_shear_exponent": "Wind-shear exponent α (-)",
    "wind_power_density_10m": "Wind power density 10 m (W/m²)",
    "wind_power_density_hub": "Wind power density @ hub (W/m²)",
    "gust_factor": "Gust factor (-)",
    "turbulence_intensity": "Turbulence intensity (-)",
    "clearness_index": "Clearness index Kt (-)",
    "diffuse_fraction": "Diffuse fraction (-)",
    "effective_irradiance": "Effective POA irradiance (W/m²)",
    "cell_temperature": "PV cell temperature (°C)",
    "pv_thermal_derate": "PV thermal derate (-)",
    "soiling_index": "Soiling index (-)",
    "solar_zenith_angle": "Solar zenith angle (°)",
    "solar_elevation_angle": "Solar elevation angle (°)",
    "solar_azimuth_angle": "Solar azimuth angle (°)",
    "air_mass": "Relative air mass (-)",
    "extraterrestrial_irradiance": "Extraterrestrial irradiance (W/m²)",
    "saturation_vapor_pressure": "Saturation vapour pressure (kPa)",
    "actual_vapor_pressure": "Actual vapour pressure (kPa)",
    "specific_humidity": "Specific humidity (g/kg)",
    "mixing_ratio": "Mixing ratio (g/kg)",
    "wet_bulb_temperature": "Wet-bulb temperature (°C)",
    "heat_index": "Heat index (°C)",
    "dew_point_depression": "Dew-point depression (°C)",
    "temp_air_soil_diff": "Air–soil temp. difference (°C)",
    "temp_lapse_rate": "Temp. lapse 2→120 m (°C)",
    "potential_temperature": "Potential temperature (K)",
    "ghi_roll3_mean": "GHI 3 h rolling mean (W/m²)",
    "wind10_roll3_mean": "Wind 3 h rolling mean (m/s)",
    "temp_roll3_mean": "Temp. 3 h rolling mean (°C)",
    "pressure_tendency_3h": "3 h pressure tendency (hPa)",
    "PV_Power_kW": "PV power output (kW)",
    "Wind_Power_kW": "Wind power output (kW)",
}


def pretty(name: str) -> str:
    """Human-readable label for a feature, falling back to a tidied raw name."""
    return PRETTY.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
PV_COLOR = "#FDB813"
WIND_COLOR = "#2E86AB"
ACCENT = "#16A085"
