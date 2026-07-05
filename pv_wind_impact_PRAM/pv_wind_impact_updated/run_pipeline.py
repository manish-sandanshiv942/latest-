#!/usr/bin/env python3
"""
run_pipeline.py
===============
Command-line runner for the *Impact Analysis of Atmospheric Parameters on
Photovoltaic and Wind Power Output* project.

It executes the SAME pipeline the Streamlit app uses, but headless, so you can
verify the whole project from a terminal (no browser needed) and get the two
headline answers the project is about:

    1. impact analysis of the atmospheric parameters, and
    2. the energy generated,

for any latitude / longitude / elevation you give it.

Examples
--------
    # Karad, India (project default), live data, Random Forest
    python run_pipeline.py

    # Specific coordinates, 60 days of history, Gradient Boosting
    python run_pipeline.py --lat 28.61 --lon 77.21 --elev 216 \
                           --past-days 60 --model "Gradient Boosting"

    # Force offline synthetic data (no internet required)
    python run_pipeline.py --synthetic

    # Benchmark every ML algorithm for both targets
    python run_pipeline.py --benchmark
"""
from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError with emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import config
from src import (data_fetcher, feature_engineering, ml_models,
                 impact_analysis, power_models)


# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Headless PV/Wind atmospheric impact-analysis pipeline.")
    p.add_argument("--site", default=None,
                   help="Named site preset (overrides --lat/--lon/--elev). "
                        "Use --list-sites to see options. "
                        f"Default coordinates are '{config.DEFAULT_SITE}'.")
    p.add_argument("--lat", type=float, default=config.DEFAULT_LATITUDE,
                   help=f"Latitude  (default {config.DEFAULT_LATITUDE})")
    p.add_argument("--lon", type=float, default=config.DEFAULT_LONGITUDE,
                   help=f"Longitude (default {config.DEFAULT_LONGITUDE})")
    p.add_argument("--elev", type=float, default=config.DEFAULT_ELEVATION,
                   help=f"Elevation in metres (default {config.DEFAULT_ELEVATION})")
    p.add_argument("--past-days", type=int, default=config.DEFAULT_PAST_DAYS,
                   help=f"Days of history to pull (default {config.DEFAULT_PAST_DAYS})")
    p.add_argument("--forecast-days", type=int, default=config.DEFAULT_FORECAST_DAYS,
                   help=f"Forecast days (default {config.DEFAULT_FORECAST_DAYS})")
    p.add_argument("--model", default="Random Forest",
                   help="ML algorithm to train (default 'Random Forest'). "
                        "Use --list-models to see all options.")
    p.add_argument("--synthetic", action="store_true",
                   help="Skip the live API and use offline synthetic data "
                        "(alias for --source synthetic).")
    p.add_argument("--source", default="openmeteo",
                   choices=["nasa", "openmeteo", "synthetic"],
                   help="Data source: nasa = NASA POWER (government, near-real-time), "
                        "openmeteo = Open-Meteo live + air quality (default), "
                        "synthetic = offline.")
    p.add_argument("--benchmark", action="store_true",
                   help="Benchmark every available algorithm on both targets.")
    p.add_argument("--list-models", action="store_true",
                   help="Print the available ML algorithms and exit.")
    p.add_argument("--list-sites", action="store_true",
                   help="Print the available site presets and exit.")
    args = p.parse_args()

    if args.list_models:
        print("Available ML algorithms:")
        for m in ml_models.MODEL_NAMES:
            print("  -", m)
        return

    if args.list_sites:
        print("Available site presets (use with --site):")
        for name, (la, lo, el, desc) in config.SITES.items():
            print(f"  - {name:28s} ({la}, {lo}, {el} m)  — {desc}")
        return

    # Resolve a named preset, if given.
    site_label = "custom coordinates"
    if args.site:
        match = next((k for k in config.SITES
                      if k.lower() == args.site.lower()
                      or args.site.lower() in k.lower()), None)
        if match is None:
            print(f"Unknown site '{args.site}'. Use --list-sites to see options.")
            return
        args.lat, args.lon, args.elev, _ = config.SITES[match]
        site_label = match

    # -------------------------------------------------------------------
    # 1. DATA  (real-time, location-synchronised — with offline fallback)
    # -------------------------------------------------------------------
    _rule("1) FETCHING LOCATION-SYNCHRONISED ATMOSPHERIC DATA")
    print(f"   Site     : {site_label}")
    print(f"   Location : lat {args.lat}, lon {args.lon}, elev {args.elev} m")
    prefer = "synthetic" if args.synthetic else args.source
    raw, status = data_fetcher.load_raw_data(
        args.lat, args.lon, args.elev,
        past_days=args.past_days, forecast_days=args.forecast_days,
        prefer=prefer,
    )
    used_elev = raw.attrs.get("elevation", args.elev)
    print(f"   Status   : {status}")
    print(f"   Source   : {raw.attrs.get('source', 'n/a')}"
          f"   Timezone: {raw.attrs.get('timezone', 'n/a')}")
    print(f"   Rows     : {len(raw)} hourly records  "
          f"({len(config.RAW_HOURLY_VARS)} raw variables each)")

    # -------------------------------------------------------------------
    # 2. FEATURE ENGINEERING  -> 100 parameters
    # -------------------------------------------------------------------
    _rule("2) ENGINEERING THE 100-PARAMETER FEATURE TABLE")
    feat = feature_engineering.build_feature_table(
        raw, args.lat, args.lon, used_elev)
    feat = feature_engineering.attach_power_targets(feat)
    print(f"   {len(config.RAW_HOURLY_VARS)} raw + "
          f"{len(config.DERIVED_FEATURES)} engineered = "
          f"{len(config.FEATURES)} input parameters")
    print(f"   Targets  : {config.TARGET_PV}, {config.TARGET_WIND} "
          f"(from transparent physical plant models)")

    # -------------------------------------------------------------------
    # 3. ENERGY GENERATED
    # -------------------------------------------------------------------
    _rule("3) ENERGY GENERATED OVER THE PERIOD")
    hours = len(feat)
    print(f"   Window               : {hours} h (~{hours/24:.0f} days)")

    # Physical model energy.
    print("\n   --- Physical model energy (engineering equations) ---")
    pv_mwh = feat[config.TARGET_PV].sum() / 1000.0
    wd_mwh = feat[config.TARGET_WIND].sum() / 1000.0
    pv_cf = feat[config.TARGET_PV].mean() / config.PV_CAPACITY_KW
    wd_cf = feat[config.TARGET_WIND].mean() / config.WIND_RATED_KW
    print(f"   PV    energy         : {pv_mwh:8.2f} MWh   "
          f"(capacity factor {_fmt_pct(pv_cf)})")
    print(f"   Wind  energy         : {wd_mwh:8.2f} MWh   "
          f"(capacity factor {_fmt_pct(wd_cf)})")
    print(f"   Combined energy      : {pv_mwh + wd_mwh:8.2f} MWh")

    # ML-predicted energy (trained on all 100 parameters).
    import numpy as _np
    print(f"\n   --- ML-predicted energy ({args.model} — all 100 parameters) ---")
    pv_res_e = ml_models.train_model(feat, config.TARGET_PV, args.model)
    wd_res_e = ml_models.train_model(feat, config.TARGET_WIND, args.model)
    X_all = feat[config.FEATURES]
    ml_pv = _np.clip(pv_res_e.model.predict(X_all), 0, config.PV_CAPACITY_KW)
    ml_wd = _np.clip(wd_res_e.model.predict(X_all), 0, config.WIND_RATED_KW)
    ml_pv_mwh = float(_np.sum(ml_pv)) / 1000.0
    ml_wd_mwh = float(_np.sum(ml_wd)) / 1000.0
    ml_pv_cf = float(_np.mean(ml_pv)) / config.PV_CAPACITY_KW
    ml_wd_cf = float(_np.mean(ml_wd)) / config.WIND_RATED_KW
    print(f"   PV    energy (ML)    : {ml_pv_mwh:8.2f} MWh   "
          f"(capacity factor {_fmt_pct(ml_pv_cf)})")
    print(f"   Wind  energy (ML)    : {ml_wd_mwh:8.2f} MWh   "
          f"(capacity factor {_fmt_pct(ml_wd_cf)})")
    print(f"   Combined energy (ML) : {ml_pv_mwh + ml_wd_mwh:8.2f} MWh")
    print(f"   ML model R² — PV    : {_fmt_pct(pv_res_e.metrics['r2'])}")
    print(f"   ML model R² — Wind  : {_fmt_pct(wd_res_e.metrics['r2'])}")

    # PV yield estimation using ML predictions.
    feat_ml = feat.copy()
    feat_ml[config.TARGET_PV] = ml_pv
    yld = power_models.pv_yield_metrics(feat_ml)
    print()
    print(f"   --- PV yield estimation (ML-based, {args.model}) ---")
    print(f"   Array DC nameplate   : {yld['pdc_kwp']:8.0f} kWp")
    print(f"   Specific yield (win) : {yld['specific_yield_period']:8.0f} kWh/kWp"
          f"  over {yld['days']:.0f} days")
    print(f"   Annual specific yield: {yld['annual_specific_yield']:8.0f} kWh/kWp/yr"
          f"  (extrapolated)")
    print(f"   Est. annual energy   : {yld['annual_energy_kwh']/1000:8.2f} MWh/yr")
    print(f"   Performance ratio    : {_fmt_pct(yld['performance_ratio'])}")
    print(f"   Peak sun hours       : {yld['peak_sun_hours']:8.2f} h/day")

    # -------------------------------------------------------------------
    # 4. BENCHMARK  (optional)
    # -------------------------------------------------------------------
    if args.benchmark:
        for target, label in [(config.TARGET_PV, "PV"),
                              (config.TARGET_WIND, "WIND")]:
            _rule(f"4) ALL-ALGORITHM BENCHMARK — {label}")
            lb = ml_models.benchmark_models(feat, target)
            print(lb.to_string(index=False,
                               columns=["Model", "R2_pct", "MAE", "RMSE"]))
        return

    # -------------------------------------------------------------------
    # 4. TRAIN + IMPACT ANALYSIS
    # -------------------------------------------------------------------
    _rule(f"4) TRAINING '{args.model}' AND ANALYSING PARAMETER IMPACT")
    pv_res = ml_models.train_model(feat, config.TARGET_PV, args.model)
    wd_res = ml_models.train_model(feat, config.TARGET_WIND, args.model)
    print(f"   PV   model accuracy  : R2 {_fmt_pct(pv_res.metrics['r2'])}  "
          f"MAE {pv_res.metrics['mae']:.2f} kW")
    print(f"   Wind model accuracy  : R2 {_fmt_pct(wd_res.metrics['r2'])}  "
          f"MAE {wd_res.metrics['mae']:.2f} kW")

    pv_imp = impact_analysis.permutation_impact(pv_res)
    wd_imp = impact_analysis.permutation_impact(wd_res)

    print("\n   --- PV: top 8 driving parameters ---")
    for _, r in pv_imp.head(8).iterrows():
        print(f"     {_fmt_pct(r['Share'])}  {r['Label']}")
    print("\n   --- WIND: top 8 driving parameters ---")
    for _, r in wd_imp.head(8).iterrows():
        print(f"     {_fmt_pct(r['Share'])}  {r['Label']}")

    print("\n   --- PV: impact grouped by physical category ---")
    for _, r in impact_analysis.grouped_impact(pv_imp).iterrows():
        print(f"     {_fmt_pct(r['Share'])}  {r['Category']}")
    print("\n   --- WIND: impact grouped by physical category ---")
    for _, r in impact_analysis.grouped_impact(wd_imp).iterrows():
        print(f"     {_fmt_pct(r['Share'])}  {r['Category']}")

    # secondary (efficiency) modulators -- the two-level method
    pv_sec = impact_analysis.secondary_impact(feat, "pv", args.model)
    wd_sec = impact_analysis.secondary_impact(feat, "wind", args.model)
    if len(pv_sec) and len(wd_sec):
        _rule("5) SECONDARY (EFFICIENCY) MODULATORS — TWO-LEVEL METHOD")
        print("   Once the first-order driver is normalised out, conversion")
        print("   efficiency is most sensitive to:")
        print(f"     PV   : {pv_sec.iloc[0]['Label']}, {pv_sec.iloc[1]['Label']}")
        print(f"     Wind : {wd_sec.iloc[0]['Label']}, {wd_sec.iloc[1]['Label']}")

    _rule("DONE")
    print("Launch the full interactive app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
