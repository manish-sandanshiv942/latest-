# Impact Analysis of Atmospheric Parameters on Photovoltaic and Wind Power Output Using Data-Driven Modelling and Machine Learning

A complete, runnable multi-file project that takes a **latitude / longitude /
elevation**, pulls **real-time, location-synchronised** atmospheric data,
builds a **100-parameter** feature table, trains **12 machine-learning
algorithms**, and reports:

1. **the impact analysis of the atmospheric parameters** on PV and wind output,
2. **a PV energy-yield estimation** (specific yield in kWh/kWp, performance ratio, annual energy), and
3. **the energy generated** (MWh and capacity factors).

The project ships with **named site presets** — the default is
**GCE Karad — Block M hostel** (Vidyanagar, Karad: 17.2877° N, 74.1818° E,
565 m) — so it runs out-of-the-box for that exact location. At the weather-grid
resolution (~2–11 km) every block on the ~40-acre campus shares the same
atmospheric and solar data, so any hostel block uses the campus coordinates.
You can also type any latitude/longitude/elevation, or search a place by name.

---

## What you give it → what it gives back

```
            ┌──────────────────────────────────────────────────────┐
 latitude   │  1. Real-time fetch (NASA POWER or Open-Meteo —       │
 longitude ─┼─►    government weather models, by coordinates)       │─► Impact analysis
 elevation  │  2. 57 raw + 43 engineered = 100 parameters           │   + Energy generated
 (anywhere  │  3. Physical plant models → PV & wind power           │   + PV yield estimation
  in India) │  4. 12 ML models + permutation / secondary impact     │   + 15 critical params
            └──────────────────────────────────────────────────────┘
```

Coordinates can be taken straight from **Google Maps** (right-click any point →
click the `lat, lon` to copy → paste into the app). Works for **any location in
India** (15 built-in presets across regions, from Karad to Leh, Jaisalmer and
Thiruvananthapuram).

---

## The 100 parameters

| Block | Count | Examples |
|-------|------:|----------|
| **Raw — fetched live** (NASA POWER / Open-Meteo + air quality) | **57** | shortwave / direct / diffuse / tilted irradiance, multi-height temperature (2–180 m), multi-height wind speed & direction, soil temperature & moisture, humidity, dew point, pressure (surface & MSL), cloud cover (low/mid/high), precipitation, **aerosol optical depth (AOD), dust, PM2.5, PM10, UV index, ozone, CO, NO₂, SO₂, surface albedo**, visibility, … |
| **Derived — engineered here** | **43** | solar geometry (zenith, azimuth, air mass, declination, day length), clearness index, diffuse fraction, **effective POA irradiance (with albedo ground-reflection)**, cell temperature & thermal derate, **soiling index (driven by AOD / dust / PM)**, air density (surface & hub), wind-shear exponent, hub wind speed, wind-power density, turbulence, wet-bulb & heat index, specific humidity, mixing ratio, cyclical hour/day/month encodings, 3-hour rolling means, pressure tendency, … |
| **Total input features** | **100** | (asserted in `config.py`) |

The two **targets** — `PV_Power_kW` and `Wind_Power_kW` — are *not* counted in
the 100; they are what the ML models predict.

### The 15 critical parameters

A focused subset of the 100, aligned with the reference study's *Top 15*:
solar irradiance (GHI/DNI/DHI), ambient temperature, relative humidity, wind
speed, wind direction, air density, atmospheric pressure, cloud cover, rainfall,
**aerosol optical depth**, **dust concentration**, visibility, dew point,
turbulence intensity and **albedo**. The Impact-Analysis tab ranks these 15 (for
both PV and wind) and shows where each sits among all 100. Aerosols and albedo
are wired into the PV physics (Beer-Lambert attenuation, soiling, ground
reflection), so they genuinely affect output and surface in the analysis. Edit
`config.CRITICAL_PARAMETERS` to use a different set.

---

## The 12 machine-learning algorithms

Random Forest · Extra Trees · Gradient Boosting · Hist Gradient Boosting ·
Decision Tree · Linear Regression · Ridge · Lasso · Elastic Net ·
K-Nearest Neighbors · Support Vector Regression · Neural Network (MLP).

Scale-sensitive models are wrapped in a `StandardScaler` pipeline
automatically. If `xgboost` and/or `lightgbm` are installed, **XGBoost** and
**LightGBM** are added to the list and the benchmark automatically — the app
runs fine without them.

A dedicated **“📈 ML Accuracy” tab** trains every algorithm on the current
location's data and shows a single chart comparing their accuracy (R² %) for PV
vs wind, plus a sortable/downloadable table of R², MAE and RMSE and the
best-performing model for each target.

---

## Impact-analysis methods

* **Permutation importance** — the headline, model-agnostic ranking of all 100
  parameters on the held-out test set.
* **Grouped impact** — the 100 parameters rolled up into physical categories
  (solar/radiation, wind/air-density, temperature, humidity, cloud, …).
* **Secondary (two-level) impact** — the first-order driver (irradiance for PV,
  hub wind for wind) is normalised out so the subtle modulators of *conversion
  efficiency* surface — the secondary effects a raw output ranking hides.
* **SHAP** (optional) — mean |SHAP| per feature for tree models, shown only if
  the `shap` package is installed.

---

## Data sources & an important honesty note

* **Atmospheric data is real, from government weather models.** Pick the source
  in the sidebar (or `--source` on the CLI):
  * **🛰️ NASA POWER** — NASA / U.S.-government Prediction Of Worldwide Energy
    Resources, the source the reference study recommends for an *India M.Tech /
    PhD dissertation*. Free, global, no API key, near-real-time (a few days'
    latency). Supplies the core solar + meteorological variables; the rest of
    the canonical schema is reconstructed from them with standard atmospheric
    relations (lapse-rate temperatures, power-law wind shear, etc.).
  * **🌐 Open-Meteo** — built on government numerical-weather models (ECMWF,
    NOAA GFS, DWD ICON); gives the current hour plus a short forecast and adds
    **aerosols / air quality** (AOD, dust, PM, ozone, …) from its Air-Quality
    API. No API key required.
  * **🧪 Offline synthetic** — if the network is unavailable the project falls
    back to a physically-plausible generator with the *same 57-column schema*
    (latitude drives the solar geometry, elevation sets the baseline pressure)
    so the whole pipeline still runs offline for development/testing.
* **PV and wind power are modelled, not measured.** Because a generic site has
  no SCADA/meter feed, the two target series are produced by **transparent
  physical plant models** (an effective-POA-irradiance PV model with a NOCT
  thermal derate, albedo ground-reflection and an AOD/dust-driven soiling loss,
  and a turbine power-curve wind model corrected for air density) driven by the
  *real* fetched weather, with a small noise term so the ML task is non-trivial.
  Edit the plant specifications in `config.py` (`PV_CAPACITY_KW`,
  `WIND_RATED_KW`, hub height, efficiency, …) to match a real installation, or
  replace the targets with measured generation if you have it.

---

## Project structure

```
solar_wind_impact/
├── app.py                     # Streamlit UI (5 tabs)
├── run_pipeline.py            # headless CLI runner (no browser needed)
├── config.py                  # all constants: 100-feature definition, plant specs, defaults
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── data_fetcher.py        # Open-Meteo live fetch + offline synthetic fallback + geocoding
    ├── feature_engineering.py # the 43 derived features (solar geometry, thermodynamics, wind physics, temporal)
    ├── power_models.py        # physical PV & wind plant models → the two targets
    ├── ml_models.py           # the 12-algorithm catalogue, training, benchmark leaderboard
    ├── impact_analysis.py     # permutation / grouped / secondary / correlation / SHAP
    └── plotting.py            # matplotlib charts used by the app
```

---

## Quick start

```bash
# 1. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3a. run the interactive app
streamlit run app.py

# 3b. …or run headless from the terminal
# 3b. …or run headless from the terminal (defaults to GCE Karad — Block M hostel)
python run_pipeline.py
```

### The Streamlit app (`streamlit run app.py`)

In the sidebar: pick a **site preset** (defaults to *GCE Karad — Block M
hostel*), or type lat/lon/elevation, or search a place name, **or paste
coordinates straight from Google Maps**; choose an ML model and the history
window; choose the **data source** (NASA POWER / Open-Meteo / synthetic); press
**Run**. Six tabs:

0. **Energy & Yield** — energy generated (MWh), capacity factors, and a **PV yield-estimation** panel (annual specific yield in kWh/kWp·yr, performance ratio, peak-sun-hours, estimated annual energy), plus current-hour conditions and a power timeseries.
1. **Impact Analysis** — dominant parameter, **the 15 critical parameters** (ranked, with each one's place among all 100), top-driver bars, category donuts, secondary modulators, optional SHAP, full 100-parameter table.
2. **ML Accuracy** — the dedicated chart comparing **all 12 algorithms** (R² % for PV vs wind) + sortable/downloadable accuracy table.
3. **Data Explorer** — the dataset, summary statistics, parameter-vs-power scatter plots.
4. **Model Performance** — R² / MAE / RMSE and predicted-vs-actual for the selected model.
5. **Forecaster** — override key conditions and get a what-if ML prediction with a physical-model cross-check.

### The CLI runner (`run_pipeline.py`)

```bash
python run_pipeline.py                                   # default site (GCE Karad Block M), Open-Meteo
python run_pipeline.py --source nasa                     # NASA POWER (government) data
python run_pipeline.py --site "Jaisalmer"                # any India preset (partial match works)
python run_pipeline.py --list-sites                      # show all 15 India site presets
python run_pipeline.py --lat 28.61 --lon 77.21 --elev 216 --past-days 92
python run_pipeline.py --model "Gradient Boosting"       # pick an algorithm
python run_pipeline.py --synthetic                       # force offline data
python run_pipeline.py --benchmark                       # accuracy of all algorithms
python run_pipeline.py --list-models                     # show available algorithms
```

It prints the location, the energy generated, the **PV yield estimation**, and
the per-parameter impact ranking (overall, by category, and the secondary
efficiency modulators).

> **Note on yield extrapolation.** The annual specific yield is the observed
> window scaled to 365 days. For a representative figure use the full
> `--past-days 92` window (a wider seasonal sample); a short window biased to
> one season will over- or under-estimate the annual number. On synthetic data
> the yield runs optimistic — real Open-Meteo data gives realistic values
> (good sites in this region ≈ 1500–1700 kWh/kWp·yr).

---

## Reproducibility

All randomised steps use a fixed seed (`RANDOM_STATE = 42` in `config.py`),
so model scores and rankings are reproducible across runs.

## Attribution

Weather & solar data: **NASA POWER** (NASA Langley Research Center; free for use
with attribution — *“These data were obtained from the NASA Langley Research
Center POWER Project funded through the NASA Earth Science Directorate Applied
Science Program.”*) and **Open-Meteo** (incl. its Air-Quality API), licensed
CC-BY 4.0. Atmospheric-parameter taxonomy and the Top-15 critical-parameter
selection follow the author's seminar study (CS2208, under Dr. S. A. Thorat).

---

## PVGIS independent cross-validation (new)

A dedicated **🛰️ PVGIS Validation** tab cross-checks the project's computed PV
energy for the *same coordinate* against **PVGIS** (the European Commission's
Photovoltaic Geographical Information System) — an independent, satellite-based
estimator. Because PVGIS is coordinate-driven (like NASA POWER and Open-Meteo),
it fits the live design and gives a citable external reference.

The tab shows: what PVGIS is and how it acquires its data; the **annual
specific-yield comparison** (your model vs PVGIS, with a % difference and an
adaptive verdict — *approximately equal / reasonable / check settings*); a
**92-day matched-period comparison** (PVGIS's typical energy for the same
calendar days your live window covers); a **monthly energy profile** (project
vs PVGIS); and an honest "why they differ" note.

Expect agreement within roughly **5–15%** on real data; a longer (≈1-year)
window tightens it. Both figures are *modelled*, so this is cross-verification
of yield magnitude against an independent estimator — not validation against
metered generation.

## Recent fixes & changes

- **Bug fix (data fetch):** Open-Meteo requires an `azimuth` parameter whenever
  `global_tilted_irradiance` is requested. It was missing, which made the live
  Open-Meteo call error out and silently fall back to synthetic data. Added
  `azimuth=0` (south-facing). Live Open-Meteo now works.
- **Robustness (NASA POWER):** `ALLSKY_SRF_ALB` (surface albedo) is requested in
  its own isolated call, so if it isn't offered at hourly resolution it can no
  longer take down the precipitation / wind batch with it.
- **UI:** professional themed header with dissertation credits
  (Submitted by · Guide: Dr. S. A. Thorat · Government College of Engineering,
  Karad · M.Tech CSE · A.Y. 2026–2027), a `.streamlit/config.toml` theme, and
  restyled metric cards / tabs.


## Soft-computing layer (FACL)

`src/facl.py` adds a **physics-residual-gated fuzzy attribution-confidence layer** on top of PRAM: it fuzzifies PRAM's residual diagnostics into a smooth 0-100 Attribution-Confidence Index with a linguistic verdict, an interpretable rule-firing trace, and two physics gates (leakage suppression + aerosol sign-consistency). It renders automatically under **Fuzzy attribution confidence** on the PRAM tabs. This is an *applied/framework* contribution, not a new fuzzy algorithm. Run `python run_facl_validation.py` to reproduce the real-DKASC checks. Full write-up (including honest validation findings): `docs/FACL_soft_computing.md`.


## Uncertainty layer (CRISP)

`src/crisp.py` adds **CRISP** — *Conformal Regime-Indexed Split-conformal Prediction intervals* — a distribution-free **uncertainty-quantification** layer that turns the physics/ML *point* prediction into a **calibrated prediction interval** with a finite-sample coverage guarantee, and keeps that guarantee **conditionally within each irradiance regime** (Mondrian conformal). It is model-agnostic (wraps the physics model, the PRAM-corrected series, or any of the 12 ML models) and is validated on the **real metered DKASC** data. On the whole-site Alice Springs meter it holds ≈90% marginal coverage with a near-diagonal reliability curve, and — the headline — lifts the high-irradiance regime that a single global band under-covers (86% → ≈90%), cutting the worst-regime coverage gap. This is an *applied/framework* contribution (a distinct axis from FACL: predictive uncertainty vs attribution confidence), **not** a new conformal algorithm. It renders under the **🎯 CRISP Intervals** tab. Run `python run_crisp_validation.py --monthly` to reproduce the real-DKASC coverage checks. Full write-up (including the honest short-window limitation): `docs/CRISP_conformal.md`.
