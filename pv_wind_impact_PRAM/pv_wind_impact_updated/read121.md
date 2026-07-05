# read121.md — Complete Project Reference

**Project title:** *Impact Analysis of Atmospheric Parameters on Photovoltaic and Wind Power Output Using Data-Driven Modelling and Machine Learning*

**Context:** M.Tech (CSE) dissertation project · Government College of Engineering, Karad · Guide: Dr. S. A. Thorat · A.Y. 2026–2027.

This document is the single, self-contained reference for the whole project: what it is, why it exists, how the code flows end-to-end, the complete mathematical modelling of every physical and statistical formula, and the purpose and functioning of every Streamlit tab. It is written so that a reader who has never opened the code can understand the entire system, and so that a viva/paper reviewer can trace every number back to an equation.

> **Rendering note.** Equations use `$...$` / `$$...$$` (KaTeX/MathJax). All math is written with plain parentheses and standard macros so it renders on GitHub, VS Code, Obsidian and Jupyter without KaTeX "unknown group" errors.

---

## Table of contents

1. [Purpose of the project](#1-purpose-of-the-project)
2. [High-level architecture & data flow](#2-high-level-architecture--data-flow)
3. [Repository / file map](#3-repository--file-map)
4. [The 100-parameter feature space](#4-the-100-parameter-feature-space)
5. [Code workflow — step by step](#5-code-workflow--step-by-step)
6. [Mathematical modelling — complete](#6-mathematical-modelling--complete)
7. [The tabs — purpose and functioning](#7-the-tabs--purpose-and-functioning)
8. [Data sources & the honesty note](#8-data-sources--the-honesty-note)
9. [Running the project](#9-running-the-project)
10. [Configuration reference](#10-configuration-reference)
11. [Reproducibility, limitations & notation](#11-reproducibility-limitations--notation)

---

## 1. Purpose of the project

The project answers three questions for **any location in India** (or anywhere, given coordinates), using only a **latitude / longitude / elevation** as input:

1. **Impact analysis** — *which atmospheric parameters most influence PV and wind power output*, ranked, and rolled up into physical categories.
2. **PV energy-yield estimation** — specific yield (kWh/kWp·yr), performance ratio, peak-sun-hours, and estimated annual energy.
3. **Energy generated** — MWh produced and capacity factors for a representative plant.

**Why it exists.** A generic site has no on-site SCADA/meter feed, so the project pulls *real* government weather-model data by coordinate, reconstructs a canonical 100-parameter atmospheric feature table, drives *transparent physical plant models* to produce PV and wind power series, and then trains a catalogue of ML regressors to (a) learn the mapping and (b) expose which inputs matter via model-agnostic interpretability. On top of this sit two research contributions:

- **PRAM** (Physics-Residual Attribution Model) — learns the *residual* between a reference power series and the physics model, then attributes that residual to atmospheric drivers. Validated against **real metered DKASC (Alice Springs)** generation.
- **FACL** (Physics-Residual-Gated Fuzzy Attribution-Confidence Layer) — a transparent Mamdani fuzzy system that turns PRAM's residual diagnostics into a smooth 0–100 Attribution-Confidence Index with a linguistic verdict and two physics "gates".
- **CRISP** (Conformal Regime-Indexed Split-conformal Prediction intervals) — a distribution-free **uncertainty-quantification** layer that wraps a calibrated, guaranteed-coverage prediction interval around the point prediction, with **regime-conditional (Mondrian)** validity, verified against **real metered DKASC** power. A distinct axis from FACL: predictive uncertainty vs attribution confidence.

**Design principle — honesty.** Atmospheric inputs are *real*. Plant power is *modelled* (not metered) for a generic site, and this is stated everywhere. The DKASC tab is the one place real *measured generation* is used, so it is the scientific anchor.

---

## 2. High-level architecture & data flow

```
             ┌───────────────────────────────────────────────────────────────┐
 latitude    │  1. FETCH        data_fetcher.load_raw_data()                  │
 longitude ──┼─►                 NASA POWER │ Open-Meteo │ synthetic fallback │──► 57 raw vars
 elevation   │  2. ENGINEER     feature_engineering.build_feature_table()     │──► +43 derived = 100
 (or a       │  3. TARGETS      power_models.pv/wind_power_from_features()     │──► PV_Power_kW, Wind_Power_kW
  site name/ │  4. TRAIN        ml_models.train_model() (chronological split)  │──► fitted model + R²/MAE/RMSE
  Google-Maps│  5. IMPACT       impact_analysis.permutation/secondary/…       │──► ranked parameters
  paste)     │  6. ATTRIBUTE    pram.run_pram() -> facl.run_facl() / crisp    │──► residual attribution + ACI + intervals
             │  7. CROSS-CHECK  pvgis.pvgis_yield(), dksac.* (measured)        │──► independent yield / validation
             └───────────────────────────────────────────────────────────────┘
                        │
                        ▼
        app.py (Streamlit UI, 13 tabs)  ·  run_pipeline.py (headless CLI)  ·  run_facl_validation.py
```

**Two entry points share the same core:**
- `app.py` — interactive Streamlit UI with 13 tabs.
- `run_pipeline.py` — headless CLI printing the same headline answers.
- `run_facl_validation.py` — an offline harness that runs PRAM+FACL over monthly windows of the real DKASC CSV.
- `run_crisp_validation.py` — an offline harness that runs CRISP (conformal intervals) over the real DKASC CSV and reports empirical marginal + conditional coverage, interval width, Winkler score, and the reliability curve.

Everything below `config.py` is import-safe (no Streamlit/pandas in `config`), so the pipeline can run from tests, CLI, or the app.

---

## 3. Repository / file map

```
pv_wind_impact_updated/
├── app.py                     # Streamlit UI (13 tabs)
├── run_pipeline.py            # headless CLI runner
├── run_facl_validation.py     # FACL empirical validation on real DKASC data
├── config.py                  # ALL constants: 100-feature definition, plant specs, 15 India presets
├── requirements.txt
├── README.md                  # short public readme
├── read121.md                 # THIS document
├── docs/
│   ├── FACL_soft_computing.md # FACL write-up + honest validation findings
│   └── CRISP_conformal.md     # CRISP write-up + real-DKASC coverage validation
├── .streamlit/config.toml     # UI theme
└── src/
    ├── __init__.py
    ├── data_fetcher.py        # live fetch (NASA POWER / Open-Meteo) + synthetic fallback + geocoding
    ├── feature_engineering.py # the 43 derived features (solar geometry, thermo, wind, temporal)
    ├── power_models.py        # physical PV & wind models -> the 2 targets + yield metrics
    ├── ml_models.py           # algorithm catalogue, train_model, benchmark leaderboard
    ├── impact_analysis.py     # permutation / grouped / secondary / correlation / SHAP
    ├── pram.py                # Physics-Residual Attribution Model
    ├── facl.py                # Fuzzy Attribution-Confidence Layer (Mamdani)
    ├── crisp.py               # Conformal prediction intervals (regime-indexed / Mondrian)
    ├── dksac.py               # real DKASC measured-generation validation
    ├── dkasc_parser.py        # parse a raw DKASC CSV into the canonical schema
    ├── pvgis.py               # independent PVGIS (EU JRC) yield cross-check (hardened transport)
    ├── plotting.py            # matplotlib chart factories
    └── pdf_report.py          # 7-page publication-style PDF report generator
```

A representative **DKASC Alice Springs 2018 CSV** (`datadkasc_alice_springs.csv`, ~105k rows, 1-minute→hourly) ships alongside for the measured-validation and PRAM/FACL tabs.

---

## 4. The 100-parameter feature space

Defined and asserted in `config.py`. **57 raw + 43 derived = 100 inputs.** The two targets (`PV_Power_kW`, `Wind_Power_kW`) are **not** counted in the 100.

**Raw (57), fetched live** — grouped: solar radiation (6), air temperature at heights (6), soil temperature (4), humidity/moisture/ET (8), wind at heights (9), pressure (2), cloud cover (4), precipitation (3), aerosols/air-quality/optical (10: AOD, dust, PM2.5, PM10, UV, ozone, CO, NO₂, SO₂, albedo), other meteorological state (5).

**Derived (43), engineered locally** — solar geometry (8), solar/PV derived (6), wind derived (9), thermodynamics/moisture (10), temporal cyclical (6), temporal dynamics (4). Each is defined mathematically in §6.

**15 critical parameters** — a focused subset aligned with the reference study's "Top 15": GHI, ambient temperature, relative humidity, hub wind speed, wind direction, air density, surface pressure, cloud cover, rain, AOD, dust, visibility, dew point, turbulence intensity, albedo. Each is tagged with a category in `config.CRITICAL_CATEGORY`.

---

## 5. Code workflow — step by step

**Step 1 — Fetch (`data_fetcher.load_raw_data(lat, lon, elev, past_days, source)`)**
- Dispatches to one of three sources: `nasa` (NASA POWER), `openmeteo` (Open-Meteo forecast + air-quality), `synthetic` (offline generator).
- NASA POWER / Open-Meteo supply the core variables; the remaining canonical columns are reconstructed with standard atmospheric relations (lapse-rate temperatures, power-law wind shear, etc.).
- `_clean_raw()` guarantees all 57 `RAW_HOURLY_VARS` exist, interpolates → ffills → bfills → 0, clips radiation ≥ 0, and reindexes to canonical column order.
- On any network exception, falls back to synthetic data (same 57-column schema) so the pipeline always runs. The chosen source and any fallback are reported in `df.attrs`/`meta["status"]`.

**Step 2 — Engineer (`feature_engineering.build_feature_table(df_raw, lat, lon, elev)`)**
- Coerces all raw columns to float, then computes the 43 derived features (§6.1–6.5), asserting the exact `config.DERIVED_FEATURES` set is produced (100 columns total). Replaces any inf/NaN with 0.

**Step 3 — Targets (`feature_engineering.attach_power_targets` → `power_models`)**
- `pv_power_from_features()` and `wind_power_from_features()` convert the engineered features into `PV_Power_kW` and `Wind_Power_kW` via transparent physics (§6.6–6.7), adding ~5 % multiplicative noise so ML is non-trivial.

**Step 4 — Train (`ml_models.train_model(df, target, model_name)`)**
- Builds `X = df[FEATURES]`, `y = df[target]`, does a **chronological** train/test split (earliest 80 % train, latest 20 % test — no shuffle, to avoid time-series leakage), fits the chosen algorithm, and returns a `TrainResult` with R²/MAE/RMSE.

**Step 5 — Impact (`impact_analysis.*`)**
- `permutation_impact` (headline), `grouped_impact` (categories), `secondary_impact` (efficiency modulators after normalising out the first-order driver), `correlation_impact` (|Pearson r|), `shap_impact` (optional).

**Step 6 — Attribute (`pram.run_pram` → `facl.run_facl`)**
- PRAM learns `residual = reference - physics`, fits a gradient-boosting residual model on a chronological split, computes improvement/bias signals, attributes drivers (SHAP → tree → permutation), and classifies a tier. FACL turns those diagnostics into a fuzzy Attribution-Confidence Index (§6.11–6.12).

**Step 7 — Cross-check (`pvgis`, `dksac`)**
- `pvgis.pvgis_yield()` gives an independent coordinate-driven annual/monthly yield. `dksac.*` validates the PV physics against **real metered** DKASC generation.

---

## 6. Mathematical modelling — complete

Notation: angles in degrees unless a variable name ends in `_r` (radians). `deg2rad(x) = pi*x/180`. `clip(x, lo, hi)` bounds `x` to `[lo, hi]`. All formulas are exactly as implemented; function names in each heading point to the source.

### 6.1 Solar geometry (8 features)
Source: `feature_engineering._solar_geometry()`. Inputs: day-of-year `n`, local decimal hour `h`, latitude `phi`.

**Declination (Cooper):**
$$\delta = 23.45 \cdot \sin\left(\frac{360(284+n)}{365}\right) \quad [\deg]$$

**Equation of time (Spencer):** with $B = \dfrac{360(n-1)}{365}$ (radians),
$$E_t = 229.18\,(0.000075 + 0.001868\cos B - 0.032077\sin B - 0.014615\cos 2B - 0.040849\sin 2B) \quad [\min]$$

**Solar time & hour angle:** $t_{sol} = h + E_t/60$, and $\omega = 15\,(t_{sol}-12)$ [deg].

**Zenith / elevation:**
$$\cos\theta_z = \sin\varphi\sin\delta + \cos\varphi\cos\delta\cos\omega, \qquad \theta_z = \arccos(\cos\theta_z), \qquad \alpha = 90 - \theta_z$$

**Azimuth (0 = North, clockwise):**
$$\sin A = \frac{-\cos\delta\,\sin\omega}{\cos\alpha}, \qquad \cos A = \frac{\sin\delta - \sin\varphi\,\sin\alpha}{\cos\varphi\,\cos\alpha}, \qquad A = \operatorname{atan2}(\sin A, \cos A)\ \bmod\ 360$$
(denominators clamped to a minimum of $10^{-3}$ for numerical safety.)

**Kasten–Young relative air mass** (0 at night):
$$m = \frac{1}{\cos\theta_z + 0.50572\,(96.07995 - \theta_z)^{-1.6364}}, \qquad m \in [0, 40]$$

**Extraterrestrial horizontal irradiance** with eccentricity correction $\varepsilon = 1 + 0.033\cos\left(\frac{360 n}{365}\right)$:
$$I_0 = G_{sc}\,\varepsilon\,\max(\cos\theta_z, 0), \qquad G_{sc} = 1361\ \mathrm{W/m^2}$$

**Day length** from the sunrise hour angle $\cos\omega_s = -\tan\varphi\tan\delta$:
$$L = \frac{2}{15}\arccos(\cos\omega_s) \quad [\text{hours}]$$

Outputs: `solar_zenith_angle, solar_elevation_angle, solar_azimuth_angle, air_mass, extraterrestrial_irradiance, declination_angle, hour_angle, day_length_hours`.

### 6.2 Solar / PV derived (6 features)

**Clearness index:** $K_t = \operatorname{clip}(\mathrm{GHI}/I_0,\ 0,\ 1)$ (only where $I_0 > 5$).

**Diffuse fraction:** $F_d = \operatorname{clip}(\mathrm{DHI}/\mathrm{GHI},\ 0,\ 1)$ (only where GHI > 5).

**Effective plane-of-array (POA) irradiance** — beam (incidence-weighted) + diffuse + ground-reflected. With tilt $\beta =$ `PV_TILT_DEG` and albedo $\rho_{alb}$ (default 0.20, clipped to ≤ 0.9):
$$\cos(\mathrm{AOI}) = \cos(\theta_z - \beta), \qquad G_{ground} = \mathrm{GHI}\cdot\rho_{alb}\cdot\frac{1 - \cos\beta}{2}$$
$$G_{POA} = \operatorname{clip}(\mathrm{DHI} + \mathrm{DNI}\cdot\cos(\mathrm{AOI}) + G_{ground},\ 0,\ 1400)$$
Output: `effective_irradiance`.

**Cell temperature (NOCT model):**
$$T_{cell} = T_{2m} + \frac{\mathrm{NOCT} - 20}{800}\,\mathrm{GHI}, \qquad \mathrm{NOCT} = 45\,^\circ\mathrm{C}$$

**PV thermal derate:**
$$f_T = \operatorname{clip}(1 - \gamma\,(T_{cell} - 25),\ 0.70,\ 1.05), \qquad \gamma = \mathrm{PV\_TEMP\_COEFF} = 0.0040\ \mathrm{K^{-1}}$$

**Soiling index** (aerosol/humidity-driven PV loss; rain resets it) — normalised aerosol load:
$$L_{aero} = \frac{1}{3}\left(\operatorname{clip}\left(\frac{\mathrm{AOD}}{1},0,1\right) + \operatorname{clip}\left(\frac{\mathrm{dust}}{120},0,1\right) + \operatorname{clip}\left(\frac{\mathrm{PM10}}{150},0,1\right)\right)$$
$$s = 1 - 0.05\cdot\operatorname{clip}\left(\frac{\mathrm{RH}-50}{50},0,1\right) - 0.10\,L_{aero}, \qquad s \leftarrow \min(s+0.08,\ 1)\ \text{ if precip} > 0.5$$
$$\mathrm{soiling\_index} = \operatorname{clip}(s,\ 0.78,\ 1.0)$$
This is what makes AOD/dust/PM genuinely affect PV output and surface in the impact ranking.

### 6.3 Wind derived (9 features)

**Air density (ideal gas):** with $T_{2K} = T_{2m} + 273.15$ and surface pressure $p_{sfc}$ [Pa]:
$$\rho = \frac{p_{sfc}}{287.05\,T_{2K}}, \qquad \mathrm{air\_density\_ratio} = \rho / 1.225$$

**Hub-height air density** (barometric pressure lapse to hub height $H = 100$ m, temperature from 120 m):
$$p_{hub} = p_{sfc}\,(1 - 2.25577\times10^{-5}\,H)^{5.25588}, \qquad \rho_{hub} = \frac{p_{hub}}{287.05\,\max(T_{120K},\ 200)}$$

**Wind-shear exponent (power law)** from the 10 m and 120 m speeds:
$$\alpha = \frac{\ln(v_{120}/v_{10})}{\ln(120/10)}, \qquad \alpha \in [0,\ 0.6]\ \ (\text{default } 0.20)$$

**Hub-height wind speed:**
$$v_{hub} = v_{10}\left(\frac{H}{10}\right)^{\alpha}, \qquad v_{hub} \in [0,\ 45]\ \mathrm{m/s}$$

**Wind power density:** $\mathrm{WPD}_{10} = \tfrac{1}{2}\rho\,v_{10}^3$ and $\mathrm{WPD}_{hub} = \tfrac{1}{2}\rho_{hub}\,v_{hub}^3$.

**Gust factor:** $G = \operatorname{clip}(v_{gust}/v_{10},\ 1,\ 4)$. **Turbulence intensity:** $I = \operatorname{clip}((v_{gust} - v_{10})/v_{10},\ 0,\ 2)$.

### 6.4 Thermodynamics / moisture (10 features)

**Saturation vapour pressure (Magnus, kPa):** $e_s = 0.6108\exp\left(\dfrac{17.27\,T}{T + 237.3}\right)$.
**Actual vapour pressure:** $e_a = e_s\cdot\mathrm{RH}/100$.

**Mixing ratio & specific humidity** (g/kg), with $p$ in kPa:
$$w = \frac{0.622\,e_a}{p - e_a}\ \ [\mathrm{kg/kg}], \qquad \mathrm{mixing\_ratio} = 1000\,w, \qquad \mathrm{specific\_humidity} = \frac{1000\,w}{1 + w}$$

**Wet-bulb temperature (Stull 2011):**
$$T_w = T\arctan(0.151977\sqrt{\mathrm{RH} + 8.313659}) + \arctan(T + \mathrm{RH}) - \arctan(\mathrm{RH} - 1.676331) + 0.00391838\,\mathrm{RH}^{1.5}\arctan(0.023101\,\mathrm{RH}) - 4.686035$$

**Heat index (NWS Rothfusz):** computed in °F with $T_F = \tfrac{9}{5}T + 32$:
$$\mathrm{HI} = -42.379 + 2.04901523\,T_F + 10.14333127\,\mathrm{RH} - 0.22475541\,T_F\,\mathrm{RH} - 6.83783\times10^{-3}\,T_F^2 - 5.481717\times10^{-2}\,\mathrm{RH}^2 + \dots$$
returned in °C; for $T_F < 80$, $\mathrm{HI} \equiv T_F$ (the formula is valid only when hot).

**Others:** dew-point depression $= T_{2m} - T_{dew}$; air–soil temp diff $= T_{2m} - T_{soil,0}$; lapse rate $= T_{2m} - T_{120m}$; potential temperature $\theta = T_{2K}\,(1000/p_{hPa})^{0.286}$.

### 6.5 Temporal encodings (6 + 4 features)

**Cyclical (6):** for hour $h$, day-of-year $d$, month $m$:
$$\left(\sin,\cos\right)\frac{2\pi h}{24}, \qquad \left(\sin,\cos\right)\frac{2\pi d}{365}, \qquad \left(\sin,\cos\right)\frac{2\pi m}{12}$$

**Dynamics (4):** 3-hour rolling means of GHI, 10 m wind, and 2 m temperature; and pressure tendency $\Delta p_{3h} = p_t - p_{t-3}$.

### 6.6 PV power plant model (target 1)
Source: `power_models.pv_power_from_features()`.
$$P_{DC} = \frac{G_{POA}}{1000}\cdot A\cdot\eta\cdot f_T\cdot s, \qquad A = 1500\ \mathrm{m^2},\ \ \eta = 0.19$$
with $f_T$ = thermal derate (§6.2) and $s$ = soiling index (§6.2). Optional ~5 % multiplicative noise, then clipped to the inverter AC capacity:
$$P_{PV} = \operatorname{clip}(P_{DC}\cdot\mathcal{N}(1, 0.05),\ 0,\ \mathrm{PV\_CAPACITY\_KW} = 300) \quad [\mathrm{kW}]$$

### 6.7 Wind power plant model (target 2)
Source: `power_models._turbine_curve()` + `wind_power_from_features()`. Cubic-ramp power curve with cut-in $v_{ci} = 3$, rated $v_r = 12$, cut-out $v_{co} = 25$ m/s, rated power $P_r = 500$ kW:
$$P_{curve}(v) = \begin{cases} 0 & v < v_{ci}\ \text{ or }\ v > v_{co} \\ P_r\,\dfrac{v^3 - v_{ci}^3}{v_r^3 - v_{ci}^3} & v_{ci} \le v < v_r \\ P_r & v_r \le v \le v_{co} \end{cases}$$
Corrected for hub air-density ratio and noise:
$$P_{wind} = \operatorname{clip}\left(P_{curve}(v_{hub})\cdot\operatorname{clip}\left(\frac{\rho_{hub}}{1.225},\ 0.6,\ 1.3\right)\cdot\mathcal{N}(1, 0.05),\ 0,\ P_r\right)$$

### 6.8 PV yield-estimation metrics
Source: `power_models.pv_yield_metrics()`. Over a window of `n` hourly rows (`days = n/24`, `annual_factor = 365/days`), with DC nameplate $P_{dc}^{kWp} = A\eta$ and POA insolation $H_{POA} = \sum G_{POA}/1000$ [kWh/m²]:

| Metric | Formula |
|---|---|
| PV energy (window) | $E = \sum P_{PV}$ [kWh] |
| Final yield $Y_f$ | $E / P_{dc}^{kWp}$ [kWh/kWp] |
| Reference yield $Y_r$ | $H_{POA}$ [h] (implicit $G_{STC} = 1$ kW/m²) |
| **Performance ratio** | $\mathrm{PR} = Y_f / Y_r$ |
| **Annual specific yield** | $Y_f \cdot \mathrm{annual\_factor}$ [kWh/kWp·yr] — *the headline figure* |
| Annual energy | $E \cdot \mathrm{annual\_factor}$ [kWh/yr] |
| Peak-sun-hours | $H_{POA} / \mathrm{days}$ [h/day] |

> Annual figures are a linear 365-day extrapolation of the observed window; use the full 92-day window for a representative number. Good sites in this region ≈ 1500–1700 kWh/kWp·yr on real data.

### 6.9 Machine-learning layer
Source: `ml_models.py`. Each model predicts one target from the 100 features.

**Core scikit-learn catalogue (always available):** Random Forest, Extra Trees, Gradient Boosting, Hist Gradient Boosting, Decision Tree, Linear Regression, Ridge, Lasso, Elastic Net, K-Nearest Neighbors, Support Vector Regression, Neural Net (MLP), Deep Neural Net (MLP), Wide Neural Net (MLP), plus a numpy Self-Organizing Map.

**Optional (auto-detected):** XGBoost, LightGBM; and — if `torch` is installed — a suite of deep nets (1D CNN, RNN, LSTM, GRU, BiLSTM, Transformer, RBF Network, Capsule Network, Autoencoder, GAN, Deep Belief Network). *torch is optional: the module imports and runs even when torch is absent* (deep models are simply not registered).

**Scaling:** scale-sensitive estimators (linear family, KNN, SVR, MLP, torch nets) are wrapped in a `StandardScaler` pipeline; the scaler is fit on the **train split only**.

**Split (leakage-safe):** `train_model` uses a **chronological** split by default — train on the earliest $(1 - t)$ fraction, test on the most recent $t = 0.2$. This is correct for hourly time-series data that contains lag / rolling-window features; a shuffled split would leak temporally adjacent rows. `chronological=False` restores the old shuffled behaviour if needed.

**Metrics** (`_metrics`): with residual $e_i = \hat{y}_i - y_i$,
$$R^2 = 1 - \frac{\sum e_i^2}{\sum (y_i - \bar{y})^2}, \qquad \mathrm{MAE} = \frac{1}{N}\sum |e_i|, \qquad \mathrm{RMSE} = \sqrt{\frac{1}{N}\sum e_i^2}$$

**Benchmark** (`benchmark_models`): trains every selected algorithm on a target and returns a leaderboard sorted by R² descending; failures become NaN rows rather than crashing.

### 6.10 Impact-analysis methods
Source: `impact_analysis.py`.

**Permutation importance (headline).** For a fitted model on the held-out test set, importance of feature $j$ is the mean drop in score when column $j$ is randomly permuted (`n_repeats=8`); reported as `Importance`, `Std`, and normalised `Share`. Negative means are clipped to 0.

**Grouped impact.** Feature importances summed into physical categories (Solar/radiation, Wind/air-density, Temperature, Humidity/moisture, Cloud, Precipitation/sky, Pressure, Temporal, Other) via `_category()`.

**Correlation impact.** $|\text{Pearson } r|$ of each feature with the target (top-N).

**Secondary (two-level) impact.** Normalises out the first-order driver so the modulators of *conversion efficiency* surface:
- **PV:** ideal $= (G_{POA}/1000)\,A\,\eta$; efficiency $\varepsilon = \operatorname{clip}(P_{PV}/\mathrm{ideal},\ 0,\ 1.2)$ over productive hours (ideal > 5). All radiation/geometry proxies and derived PV intermediates are dropped from the feature set so the *raw atmospheric causes* (temperature, humidity, aerosols/dust, albedo, wind cooling) rank.
- **Wind:** ideal $= P_{curve}(v_{hub})$; efficiency $\varepsilon = \operatorname{clip}(P_{wind}/\mathrm{ideal},\ 0,\ 1.5)$; all wind-speed/density proxies dropped.

Permutation importance is then computed on a chronological split of $(\text{features} \to \varepsilon)$.

**SHAP (optional).** Mean $|\text{SHAP}|$ per feature for bare tree models, if the `shap` package is installed; otherwise `None`.

### 6.11 PRAM — Physics-Residual Attribution Model
Source: `pram.py`. **Goal:** attribute the gap between a *reference* power series and the *physics model* to atmospheric drivers.

**Two modes.**
- **measured** — reference = real metered DKASC power, model = project physics on the same on-site weather. $\mathrm{residual} = P_{meas} - P_{phys}$ is the scientifically valid quantity (what the physics misses vs. reality).
- **crosssrc** — reference = NASA-driven physics PV, model = Open-Meteo-driven physics PV. A *data-source consistency* residual (no metered ground truth).

**Baseline calibration (measured mode, `calibrate_baseline`).** A single fixed-tilt physics model has the wrong daily *shape* for a meter aggregating tracking/off-axis arrays. A per-hour-of-day multiplicative gain is fit on the **training portion only** (chronological, no leakage):
$$g_h = \operatorname{clip}\left(\operatorname{median}_{t \in \text{train},\ \mathrm{hour}=h}\frac{P_{ref}(t)}{P_{mod}(t)},\ 0.2,\ 5.0\right), \qquad \hat{P}_{mod}(t) = P_{mod}(t)\,g_{\mathrm{hour}(t)}$$
This turns the dominant multiplicative error into a small additive residual.

**Residual feature matrix (`build_residual_features`).** Recognised atmospheric columns (POA, GHI, DHI, DNI, temp, RH, wind, rain, pressure, AOD, PM, dust) + temporal encodings + physical derivatives:
$$K_{cs} = \operatorname{clip}\left(\frac{\mathrm{GHI}}{\max(1361\cos\theta_z^{proxy},\ 1)},\ 0,\ 1.5\right), \qquad \cos\theta_z^{proxy} = \operatorname{clip}(\cos\delta\cos\omega,\ 0,\ 1)$$
plus `ghi_sq`, `ghi_x_elev`, `poa_over_ghi`, `ghi_lag1`, `ghi_roll3`, `temp_x_wind`, and `temp_excess = max(T - 25, 0)`.

**Residual model (`fit_residual_model`).** Chronological split (test = held-out future), train-only 1st–99th-percentile outlier trim, then a `HistGradientBoostingRegressor` (fallback RandomForest). Reports residual $R^2$, RMSE, MAE on the test window.

**Improvement signals (`improvement_signals`).** On the test period, does physics + learned residual beat physics alone? With $\text{corrected} = P_{mod}^{te} + \hat{r}$:
$$\mathrm{improvement\%} = \left(1 - \frac{\mathrm{RMSE}(P_{ref}^{te},\ \text{corrected})}{\mathrm{RMSE}(P_{ref}^{te},\ P_{mod}^{te})}\right)\times 100$$
plus mean bias error before/after ($\mathrm{MBE} = \overline{P_{mod} - P_{ref}}$).

**Attribution (`attribution`).** SHAP mean $|\text{value}|$ if available → native tree importance → permutation importance (with signed direction from the sign of each feature's correlation with the residual).

**Aerosol signature (`aerosol_signature`).** *Strong* = an aerosol feature (AOD/PM10/PM2.5/dust) ranks in the top-3 **and** its signed effect is negative (physics over-predicts in dusty air → the omitted soiling loss).

**Classification (`classify`).**
$$\mathrm{tier} = \begin{cases} \text{strong} & R^2_{res} \ge 0.40\ \text{ or }\ \mathrm{impr} \ge 8\% \\ \text{moderate} & R^2_{res} \ge 0.18\ \text{ or }\ \mathrm{impr} \ge 3\% \\ \text{weak} & \text{otherwise} \end{cases}$$
with a demotion guard and a leakage flag $R^2_{res} > 0.85$ (suspiciously perfect → investigate).

### 6.12 FACL — Fuzzy Attribution-Confidence Layer
Source: `facl.py`. A textbook **Mamdani** fuzzy inference system (triangular/trapezoidal memberships, min-inference, max-aggregation, centroid defuzzification) whose *inputs are PRAM's residual diagnostics*, not raw weather. Output: a smooth 0–100 **Attribution-Confidence Index (ACI)**.

**Four crisp inputs** (from `facl_inputs`): `improvement` (%), `residual_r2`, `aerosol` (signed consistency in [−1, 1]), `bias` (|MBE_phys| − |MBE_corr|).

**Fuzzy terms** (breakpoints tied to PRAM's crisp thresholds so FACL is a *strict generalisation*):
- improvement: low / medium / high (around 3 %, 8 %)
- residual_r2: low / medium / high / **suspicious** (> ~0.85 = leakage region)
- aerosol: contradictory / neutral / supportive
- bias: worsened / neutral / improved
- output confidence: very_low / low / moderate / high / very_high

**Aerosol consistency score** (`aerosol_consistency_score`): rank weight $w = \operatorname{clip}(1.2/\mathrm{rank},\ 0,\ 1)$; $+w$ if physically negative (supportive), $-0.7w$ if wrong-sign (contradictory), $0.3w$ if sign unknown.

**Inference.** Rule strength = min of antecedent memberships × weight; each output term is clipped at the max strength of rules concluding it; aggregate by max; defuzzify by centroid over a universe $u \in [0, 100]$ (501 points):
$$\mathrm{ACI}_{raw} = \frac{\sum_u u\cdot\mu_{agg}(u)}{\sum_u \mu_{agg}(u)}$$

**Two physics gates.**
- **GATE 1 — leakage suppression** (applied as an authoritative convex guard, not a competing rule): with $\mu_{susp}$ = membership of `residual_r2` in "suspicious" and anchor = 28,
$$\mathrm{ACI} = \mathrm{ACI}_{raw} + \mu_{susp}\,(\mathrm{LEAK\_GUARD\_ANCHOR} - \mathrm{ACI}_{raw})$$
A suspiciously-perfect residual pulls confidence *down* regardless of apparent gain.
- **GATE 2 — aerosol sign-consistency** (encoded as rules): a physically-negative aerosol driver *raises* confidence; a positive-sign "signature" is implausible and *lowers* it.

**Verdict** (`verdict_for`): ACI ≥ 80 very high, ≥ 60 high, ≥ 45 moderate, ≥ 25 low, else very low — each with a linguistic label and CSS class. The system also returns a ranked rule-firing **trace** and the aggregate output MF for plotting the defuzzification and the no-cliff ACI surface (`aci_surface`).

> **Honest scope (from `docs/FACL_soft_computing.md`):** FACL is an *applied/framework-integration* contribution — a smoother, more conservative, leakage-aware, interpretable verdict layer. It is **not** a new fuzzy algorithm and **not** a proven predictor of model accuracy (on real DKASC data the Spearman of ACI vs the independent test-period correlation was slightly negative). The defensible claim is the leakage gate correctly demoting an over-fit window that crisp thresholds rated "strong".

### 6.13 DKASC measured-data validation
Source: `dksac.py`. This is the one place **real metered generation** anchors the physics. Steps: locate the DKASC CSV → detect data-bearing power channels (excluding reactive/apparent/power-factor/energy) → resample hourly → estimate rated power from the 99.5th percentile → apply the project's PV physics to on-site weather → score.

**Model PV from DKASC weather:**
$$T_{cell} = T + \frac{\mathrm{NOCT} - 20}{800}\,\mathrm{GHI}, \qquad f_T = \operatorname{clip}(1 - \gamma(T_{cell} - 25),\ 0.70,\ 1.05)$$
$$\hat{P} = \operatorname{clip}\left(\mathrm{rated}\cdot\frac{G_{POA}}{1000}\cdot f_T,\ 0,\ \mathrm{rated}\right)$$
(soiling fixed at 1.0 — DKASC has no aerosol channel, stated as a limitation).

**Metrics (operating hours only, meas > 2 % rated):** MAE, RMSE, MBE, nRMSE (% of mean measured), $R^2$, Pearson corr. **Verdict:** nRMSE ≤ 10 % strong / ≤ 20 % reasonable / else notable difference.

### 6.14 PVGIS independent cross-validation
Source: `pvgis.py`. Queries **PVGIS** (EU JRC Photovoltaic Geographical Information System) `PVcalc` for the *same coordinate* — an independent, satellite-based estimator using its own SARAH/ERA5 databases. Returns annual energy $E_y$, in-plane irradiation, specific yield $E_y / P_{kWp}$, and a 12-row monthly profile.

**Transport is hardened** (the version now in `pvgis.py`): browser-like User-Agent, endpoint fallback (v5_2 → v5_3 → legacy), exponential-backoff retries, and a `PVGISError` that *classifies* failures (TCP reset / timeout / DNS) — so a firewall reset is reported as a reset, not "no internet". Expect agreement within ~5–15 % on real data; both figures are *modelled*, so this is yield-magnitude cross-verification, not validation against metered generation.

### 6.15 CRISP — Conformal prediction intervals
Source: `crisp.py`. **Goal:** turn a *point* power prediction `ŷ` into a
**prediction interval** with a distribution-free, finite-sample coverage
guarantee, and keep that guarantee **conditionally within each irradiance/wind
regime**. Model-agnostic: `ŷ` may be the physics model, the PRAM-corrected
series, or any of the 12 ML models.

**Nonconformity score** (operating hours only): `s_i = |y_i − ŷ_i|`.

**Split-conformal quantile** (`conformal_quantile`). For a calibration set of
size `n` and target miscoverage `α`, take the `k`-th smallest score with
$k = \lceil (n+1)(1-\alpha) \rceil$ (return $+\infty$ if $k > n$):
$$Q = s_{(k)}, \qquad \text{interval} = \big[\,\hat{y} - Q,\ \hat{y} + Q\,\big]\ \text{clipped to } [0,\ \text{rated}].$$

**Guarantee (marginal).** With calibration/test scores exchangeable,
$\;\mathbb{P}\!\left(y_{\text{test}} \in [\hat{y}-Q,\ \hat{y}+Q]\right) \ge 1-\alpha\;$ — no assumption on the error distribution.

**Mondrian (regime-conditional).** Partition operating hours into `n_bins`
quantile regimes of the driver (GHI/POA for PV; hub wind for wind) and compute a
separate $Q_k$ per regime, giving $\mathbb{P}(y \in \text{interval}\mid \text{regime}=k) \ge 1-\alpha$ for each `k` — the conditional validity a single global `Q` lacks under heteroscedastic PV errors. A `normalized` variant instead scales the width by a per-regime mean-absolute residual $\hat{\sigma}$ for locally-adaptive bands.

**Split.** Chronological (earlier = calibration, later = held-out test) — no
shuffle, leakage-safe, matching the rest of the project; this deliberately
stresses exchangeability with real drift.

**Metrics** (`coverage_report`): empirical **marginal coverage**; per-regime
**conditional coverage**; **worst-regime gap** $\max_k |\mathrm{cov}_k - (1-\alpha)|$; **MPIW** (mean width, kW) and **PINAW** (MPIW ÷ power range); and the mean **Winkler / interval score** $W = (u-l) + \tfrac{2}{\alpha}(l-y)\mathbf{1}[y<l] + \tfrac{2}{\alpha}(y-u)\mathbf{1}[y>u]$. `reliability()` sweeps α to produce the nominal-vs-empirical calibration curve.

> **Honest scope (from `docs/CRISP_conformal.md`):** conformal prediction is an
> established framework — this is an *applied/framework-integration* contribution,
> not a new algorithm. On real DKASC data the pooled full-period intervals hold
> near-nominal marginal coverage (≈90–93%) with a near-diagonal reliability curve,
> and Mondrian lifts the under-covered high-irradiance regime (≈86% → ≈90%) that a
> global band misses. Short single-month windows under-cover (~81% mean) because
> small, drifting calibration sets break exchangeability — reported, not hidden.

---

## 7. The tabs — purpose and functioning

`app.py` renders 13 tabs (labels and dispatch in `main()`). The sidebar selects a site preset (or lat/lon/elev, place-name search, or a **Google-Maps coordinate paste**), an ML model, the history window (`past_days`), and the data source. `main()` fetches once, trains PV + wind, shows a KPI band, then dispatches each tab.

| # | Tab | Purpose & functioning |
|---|-----|-----------------------|
| 0 | **🇦🇺 Australian DKASC** | The scientific anchor. Validates the PV physics against **real metered** DKASC Alice Springs generation: loads the CSV, picks a power channel, applies §6.13, and shows MAE/RMSE/nRMSE/R²/corr with a verdict — plus a step-by-step math walkthrough and the **PRAM + FACL** attribution block for the measured residual. |
| 1 | **🧪 PRAM (Live)** | Runs PRAM in *cross-source* mode on live data (NASA vs Open-Meteo physics PV), attributes the residual to atmospheric drivers, and renders the FACL fuzzy-attribution-confidence panel (fuzzified diagnostics, fired-rules trace, no-cliff ACI surface). |
| 2 | **🎯 CRISP Intervals** | Conformal prediction intervals (§6.15) on the real DKASC data: a channel/confidence/regime selector, a fan chart of the calibrated band vs measured power over the held-out test period, the per-regime **Mondrian-vs-Global conditional-coverage** table, and a reliability-curve expander. The project's uncertainty-quantification layer. |
| 3 | **⚡ Energy & Yield** | The headline answers: energy generated (MWh), capacity factors, and the **PV yield-estimation** panel (annual specific yield kWh/kWp·yr, PR, peak-sun-hours, annual energy from §6.8), current-hour conditions, and a power timeseries. |
| 4 | **🔬 Impact Analysis** | Which parameters drive PV vs wind: dominant parameter, the **15 critical parameters** ranked with their place among all 100, top-driver bars, category donuts, **secondary** efficiency modulators, optional SHAP, and the full 100-parameter table. |
| 5 | **📈 ML Accuracy** | Trains every algorithm on the current location's data and shows one chart comparing accuracy (R² %) for PV vs wind, plus a sortable/downloadable table of R²/MAE/RMSE and the best model per target. |
| 6 | **📊 Data Explorer** | The dataset itself: summary statistics and parameter-vs-power scatter plots for exploring relationships. |
| 7 | **⚙️ Model Performance** | For the selected model: R²/MAE/RMSE and a predicted-vs-actual chart on the held-out test set. |
| 8 | **🔮 Forecaster** | What-if tool: override key conditions (irradiance, temperature, wind, …) and get an ML prediction with a physical-model cross-check. |
| 9 | **📋 Dataset** | The full engineered feature table for the location/window, with metadata (source, coordinates, elevation) and download. |
| 10 | **🧮 Algorithm Evaluation** | Deep per-algorithm evaluation with per-model step-by-step "math walkthrough" helpers explaining how each family makes a prediction. |
| 11 | **🌐 Real-Time Dataset & Evaluation** | Trains on the **57 raw real-time parameters only** (chronological split) and shows the per-model real-time math — isolating what the live API alone can predict. |
| 12 | **🛰️ PVGIS Validation** | Independent cross-check (§6.14): what PVGIS is, the annual specific-yield comparison (your model vs PVGIS, % difference + adaptive verdict), a matched-period comparison, a monthly energy profile, and an honest "why they differ" note. |

Supporting helpers: cached heavy steps (`get_feature_table`, `get_trained`, `get_permutation`, `get_secondary`, `get_benchmark`, `get_pvgis`, `get_site_pv`), the landing map picker (`_landing_map_picker`), CSS/theme injection, and the header/footer with dissertation credits. Caching uses `@st.cache_data` for data and `@st.cache_resource` for the fitted model; large frames are passed as `_feat` (leading underscore) so Streamlit keys on the cheap `feat_key` instead of hashing the frame.

---

## 8. Data sources & the honesty note

- **Atmospheric data is real, from government weather models.**
  - **NASA POWER** — free, global, no API key, near-real-time; supplies core solar + meteorological variables, the rest reconstructed with standard atmospheric relations.
  - **Open-Meteo** — built on ECMWF / NOAA GFS / DWD ICON; current hour + short forecast, plus **aerosols/air-quality** (AOD, dust, PM, ozone, …) from its Air-Quality API. No API key.
  - **Offline synthetic** — if the network is unavailable, a physically-plausible generator with the *same 57-column schema* keeps the whole pipeline runnable.
- **PV and wind power are modelled, not measured.** For a generic site there is no SCADA/meter feed, so the two targets come from the transparent physical plant models (§6.6–6.7) driven by the *real* fetched weather, with a small noise term. Edit the plant specs in `config.py`, or replace the targets with measured generation if you have it. The **DKASC tab** is the exception — it uses real metered generation.

---

## 9. Running the project

```bash
# 1. (optional) virtual environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3a. interactive app
streamlit run app.py

# 3b. headless CLI (defaults to GCE Karad — Block M hostel, Open-Meteo)
python run_pipeline.py
python run_pipeline.py --source nasa                 # NASA POWER data
python run_pipeline.py --site "Jaisalmer"            # any India preset (partial match)
python run_pipeline.py --lat 28.61 --lon 77.21 --elev 216 --past-days 92
python run_pipeline.py --model "Gradient Boosting"   # pick an algorithm
python run_pipeline.py --benchmark                   # accuracy of all algorithms
python run_pipeline.py --list-sites / --list-models

# 3c. FACL validation on the real DKASC CSV
python run_facl_validation.py --csv datadkasc_alice_springs.csv \
       --channel-index 2 --max-windows 10 --min-hours 250
```

**Dependencies:** required — `streamlit, pandas, numpy, scikit-learn, matplotlib`; optional (auto-detected) — `xgboost, lightgbm, torch, shap, folium, streamlit-folium`.

---

## 10. Configuration reference

Key constants in `config.py`:

| Constant | Value | Meaning |
|---|---|---|
| `RANDOM_STATE` | 42 | global seed (reproducibility) |
| `DEFAULT_LATITUDE/LONGITUDE/ELEVATION` | 17.2877, 74.1818, 565 m | GCE Karad — Block M hostel |
| `SITES` | 15 presets | Karad + all-India reference locations |
| `DEFAULT_PAST_DAYS / FORECAST_DAYS` | 92 / 2 | Open-Meteo forecast-endpoint window |
| `PV_CAPACITY_KW` | 300 | AC / inverter clipping limit |
| `PV_AREA_M2 / PV_EFFICIENCY` | 1500 / 0.19 | module area / STC efficiency |
| `PV_TEMP_COEFF` | 0.0040 K⁻¹ | ≈ −0.40 %/°C |
| `PV_NOCT_C / PV_TILT_DEG` | 45 / 20° | NOCT cell temp / panel tilt |
| `WIND_RATED_KW` | 500 | turbine rated power |
| `WIND_CUT_IN / RATED_SPEED / CUT_OUT` | 3 / 12 / 25 m/s | power-curve breakpoints |
| `WIND_HUB_HEIGHT_M` | 100 | hub height for wind extrapolation |
| `RAW_HOURLY_VARS / DERIVED_FEATURES / FEATURES` | 57 / 43 / 100 | the feature space (asserted) |
| `CRITICAL_PARAMETERS` | 15 | the focused critical subset |
| `TARGET_PV / TARGET_WIND` | PV_Power_kW / Wind_Power_kW | the two modelled targets |

---

## 11. Reproducibility, limitations & notation

- **Reproducibility:** all randomised steps use `RANDOM_STATE = 42`; model scores and rankings are reproducible across runs (noise realizations are seeded).
- **Chronological evaluation:** the main ML split, the PRAM residual fit, and the secondary-impact split are all **time-ordered (no shuffle)** — the honest way to evaluate time-series data and what to report in the paper.
- **What is real vs modelled:** atmospheric inputs = real; PV/wind targets = modelled (except the DKASC tab = real metered). PVGIS and DKASC are the two independent external references.
- **Known modelling simplifications** (documented for honesty): the POA transposition uses $\cos(\theta_z - \beta)$ (in-plane approximation, azimuth-agnostic) and full DHI (no explicit sky-view weighting); cell temperature uses GHI rather than POA; annual yield is a linear 365-day extrapolation; the plant models add ~5 % noise so ML is non-trivial. These bias PV magnitudes slightly and are stated so a reader can calibrate expectations.
- **Attribution honesty:** PRAM's `crosssrc` mode is a model-consistency residual (no ground truth); FACL is an interpretable, conservative verdict layer, not a proven accuracy predictor. The DKASC `measured` mode is the scientifically valid attribution.

**Attribution / data licences:** Weather & solar — **NASA POWER** (NASA Langley Research Center, free with attribution) and **Open-Meteo** incl. its Air-Quality API (CC-BY 4.0). Independent yield — **PVGIS** (EU Joint Research Centre). Measured generation — **DKASC** (Desert Knowledge Australia Solar Centre), Alice Springs.

---

*End of read121.md — this document reflects the code as implemented in `config.py`, `src/`, and `app.py`.*
