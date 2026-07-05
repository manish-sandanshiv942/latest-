"""
src/dkasc_parser.py
===================
Helper module to parse DKASC datasets, extract available meteorological variables,
resample to hourly, and synthesize missing 57 raw parameters so they can be
passed through the standard `feature_engineering.build_feature_table()`.
"""

import pandas as pd
import numpy as np
import config

# Common DKASC column names and their Open-Meteo equivalents
DKASC_MAPPING = {
    "Weather_Temperature_Celsius": "temperature_2m",
    "Weather_Relative_Humidity": "relative_humidity_2m",
    "Global_Horizontal_Radiation": "shortwave_radiation",
    "Diffuse_Horizontal_Radiation": "diffuse_radiation",
    "Wind_Speed": "wind_speed_10m",
    "Wind_Direction": "wind_direction_10m",
    "Weather_Daily_Rainfall": "precipitation",
    "Active_Power": "DKASC_Active_Power",  # To keep separate from our modeled target
}

def parse_dkasc_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parses a raw DKASC CSV dataframe:
    1. Identifies time column.
    2. Maps known DKASC variables to Open-Meteo standard names.
    3. Resamples to hourly.
    4. Backfills the remainder of the 57 raw vars with reasonable approximations.
    """
    df = df_raw.copy()
    
    # 1. Identify Time Column
    time_col = None
    for col in df.columns:
        if "timestamp" in col.lower() or "date" in col.lower() or "time" in col.lower():
            time_col = col
            break
            
    if time_col is None:
        raise ValueError("Could not find a valid timestamp column in DKASC data.")
        
    df["Datetime"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
    
    # 2. Map columns
    for dkasc_col, om_col in DKASC_MAPPING.items():
        if dkasc_col in df.columns:
            df[om_col] = pd.to_numeric(df[dkasc_col], errors="coerce")
            
    # Look for alternative power column names if "Active_Power" wasn't exact
    if "DKASC_Active_Power" not in df.columns:
        for col in df.columns:
            if "power" in col.lower() and ("active" in col.lower() or "kw" in col.lower()):
                df["DKASC_Active_Power"] = pd.to_numeric(df[col], errors="coerce")
                break
                
    if "DKASC_Active_Power" not in df.columns:
        # Sometimes it's labelled "Energy" if the data is 5-min intervals
        # But we'll try to calculate average power during the hour
        for col in df.columns:
            if "energy" in col.lower():
                df["DKASC_Active_Power"] = pd.to_numeric(df[col], errors="coerce") * (60 / 5) # Rough approximation if 5-min kWh
                break

    # 3. Resample to hourly
    df.set_index("Datetime", inplace=True)
    
    # We take the mean for environmental variables, sum for precip
    resample_dict = {}
    for col in df.columns:
        if col == "precipitation":
            resample_dict[col] = "sum"
        elif pd.api.types.is_numeric_dtype(df[col]):
            resample_dict[col] = "mean"
            
    if resample_dict:
        df_hourly = df.resample("1h").agg(resample_dict)
    else:
        df_hourly = df.resample("1h").mean(numeric_only=True)
        
    df_hourly = df_hourly.reset_index()
    
    # Forward fill small gaps, then drop remaining NAs
    df_hourly = df_hourly.ffill(limit=2).dropna(subset=["DKASC_Active_Power"])
    
    # 4. Fill missing 57 Open-Meteo parameters
    for var in config.RAW_HOURLY_VARS:
        if var not in df_hourly.columns:
            # Approximate or default to 0
            if var == "surface_pressure" or var == "pressure_msl":
                df_hourly[var] = 1013.25
            elif "temperature" in var and var != "apparent_temperature":
                # Default other temperatures to T2M if available, else 25
                df_hourly[var] = df_hourly.get("temperature_2m", 25.0)
            elif var == "dew_point_2m":
                # Rough approximation: Td ≈ T - (100 - RH)/5
                t2m = df_hourly.get("temperature_2m", 25.0)
                rh = df_hourly.get("relative_humidity_2m", 50.0)
                df_hourly[var] = t2m - (100 - rh) / 5.0
            elif var == "surface_albedo":
                df_hourly[var] = 0.20
            elif "wind_speed" in var or var == "wind_gusts_10m":
                df_hourly[var] = df_hourly.get("wind_speed_10m", 3.0)
            elif "wind_direction" in var:
                df_hourly[var] = df_hourly.get("wind_direction_10m", 0.0)
            elif var == "cloud_cover" or var == "cloud_cover_low" or var == "cloud_cover_mid" or var == "cloud_cover_high":
                df_hourly[var] = 0.0
            elif var == "direct_normal_irradiance":
                # Very rough approximation if not available
                df_hourly[var] = df_hourly.get("shortwave_radiation", 0.0) * 0.8
            else:
                # Safest fallback
                df_hourly[var] = 0.0
                
    return df_hourly
