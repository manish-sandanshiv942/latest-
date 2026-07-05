# working.md — Complete Workflow, Mathematical Modelling & Tab-by-Tab Functioning

**Project:** *Physics-Residual-Gated Fuzzy Attribution Confidence for Photovoltaic
Power: A Formally Characterised Soft-Computing Layer Validated on Metered Generation*
**App:** `app.py` (Streamlit) · **Pipeline:** `src/` · **Config:** `config.py`

This document explains, end to end: (A) the overall execution workflow, (B) every
mathematical model implemented in the code, and (C) the detailed functioning of
each of the 13 tabs in the Streamlit application.

---

## Part A — Overall Execution Workflow

```
 USER INPUT (sidebar)                          STAGE 1  src/data_fetcher.py
 lat, lon, elevation,                          ────────────────────────────
 data source, history window        ┌────────► fetch hourly weather
        │                           │          NASA POWER ─or─ Open-Meteo (+air quality)
        ▼                           │          ─or─ synthetic fallback (same schema)
 🚀 Run analysis ───────────────────┘                     │  57 raw variables
                                                          ▼
                                               STAGE 2  src/feature_engineering.py
                                               ────────────────────────────
                                               +43 derived physics/temporal features
                                               = canonical 100-parameter table
                                                          │
                                                          ▼
                                               STAGE 3  src/power_models.py
                                               ────────────────────────────
                                               physical PV model  → target `pv_power_kw`
                                               physical wind model→ target `wind_power_kw`
                                                          │
                                                          ▼
                                               STAGE 4  src/ml_models.py
                                               ────────────────────────────
                                               12+ ML models, chronological 80/20 split
                                               metrics: R², RMSE, MAE, MAPE
                                                          │
                    ┌─────────────────────────────────────┼─────────────────────────┐
                    ▼                                     ▼                         ▼
        STAGE 5  impact_analysis.py            STAGE 6  pram.py            STAGE 7  external
        ────────────────────────────           ─────────────────           ─────────────────
        permutation importance,                residual = ref − physics    pvgis.py  (modelled
        grouped & SECONDARY ranking            attribution + diagnostics    cross-check)
        (efficiency-level drivers)                       │                 dksac.py  (metered
                    │                                    ▼                  DKASC anchor)
                    │                          STAGE 8a  facl.py
                    │                          Mamdani fuzzy → ACI (0–100)
                    │                                    │
                    │                          STAGE 8b  acgc.py
                    │                          w = g(ACI) gates the correction,
                    │                          conformal-wrapped (coverage preserved)
                    │                                    │
                    │                          STAGE 8c  crisp.py
                    │                          Mondrian split-conformal intervals
                    │                                    │
                    │                          STAGE 8d  rca.py
                    │                          per-regime attribution on the SAME
                    │                          Mondrian bins → AII → s(AII)·ACI
                    ▼                                    ▼
                ┌───────────────────────────────────────────────┐
                │        STREAMLIT UI — 13 TABS (Part C)        │
                └───────────────────────────────────────────────┘
```

**Caching:** `get_feature_table()` (line 61, `@st.cache_data`) keys the fetched
feature table on `(lat, lon, elev, past_days, source)`; `get_trained()` (line 80,
`@st.cache_resource`) keys trained models on `(feat_key, target, model_name)`.
Changing any sidebar input invalidates only what depends on it.

**Entry points:**
- `streamlit run app.py` — full UI.
- `python run_pipeline.py` — headless CLI (fetch → features → train → report).
- `python run_facl_validation.py` — PRAM + FACL on the real DKASC CSV.
- `python run_crisp_validation.py` — CRISP conformal coverage on DKASC.
- `python validate_formal_properties.py` — numerical verification of P1–P13.

---

## Part B — Mathematical Modelling (as implemented)

### B.1 Solar geometry (`feature_engineering.py`)

Declination (Cooper): $$\delta = 23.45°\sin\!\left(\frac{360(284+n)}{365}\right)$$

Hour angle: $$h = 15°(t_{solar} - 12)$$

Solar elevation: $$\sin\alpha_s = \sin\phi\sin\delta + \cos\phi\cos\delta\cos h,\qquad \theta_z = 90° - \alpha_s$$

Air mass (Kasten–Young): $$AM = \frac{1}{\cos\theta_z + 0.50572\,(96.07995-\theta_z)^{-1.6364}}$$

Clearness index: $$k_t = \frac{GHI}{GHI_{clearsky}}$$

### B.2 Plane-of-array irradiance
For tilt $$\beta$$ (isotropic sky):

$$POA = DNI\cos\theta_i + DHI\,\frac{1+\cos\beta}{2} + GHI\,\rho_{alb}\,\frac{1-\cos\beta}{2}$$

### B.3 PV thermal & power model (`power_models.py`)
Cell temperature (NOCT): $$T_{cell} = T_{amb} + \frac{NOCT-20}{800}\,POA$$

Temperature derate: $$f_T = 1 + \gamma\,(T_{cell}-25),\quad \gamma \approx -0.0035\,/°C$$

Soiling index from aerosols (AOD / PM / dust) reduces the effective irradiance;
DC→AC via system derate $$\eta_{sys}$$ and inverter clipping at rated AC power:

$$P_{PV} = \min\!\big(P_{dc0}\cdot\frac{POA_{eff}}{1000}\cdot f_T\cdot\eta_{sys},\; P_{AC,rated}\big)$$

### B.4 Wind model (`power_models.py`)
Hub-height wind (power/log law): $$v_{hub} = v_{10}\left(\frac{z_{hub}}{z_{10}}\right)^{\alpha}$$

Air density from pressure/temperature/elevation:
$$\rho = \frac{p}{R_{d} T}\quad\text{with lapse-rate correction } T(z) = T_0 - 0.0065\,z$$

Piecewise power curve:
$$P_W = \begin{cases}0 & v < v_{ci}\ \text{or}\ v > v_{co}\\ P_r\,\dfrac{v^3 - v_{ci}^3}{v_r^3 - v_{ci}^3}\cdot\dfrac{\rho}{\rho_0} & v_{ci}\le v < v_r\\ P_r\cdot\rho/\rho_0 & v_r \le v \le v_{co}\end{cases}$$

### B.5 ML training & metrics (`ml_models.py`)
Chronological (leakage-safe) 80/20 split; metrics on held-out block:

$$R^2 = 1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2},\quad RMSE=\sqrt{\tfrac1n\sum(y-\hat y)^2},\quad MAE=\tfrac1n\sum|y-\hat y|$$

### B.6 Impact analysis (`impact_analysis.py`)
Permutation importance: $$I_j = \mathbb{E}\big[\mathcal{L}(y,\hat f(X^{perm(j)}))\big] - \mathcal{L}(y,\hat f(X))$$

**Secondary (two-level) ranking:** define conversion efficiency
$$\varepsilon = P / P_{ideal}$$ over productive hours, drop all first-order
resource proxies (radiation/geometry for PV, wind speed for wind), and rank the
remaining features' importance for $$\varepsilon$$ — exposing temperature derate,
humidity, aerosol/soiling, and albedo effects hidden by the dominant driver.

### B.7 PRAM (`src/pram.py`)
Residual: $$r(t) = y_{ref}(t) - P_{phys}(t)$$

Train-only per-hour-of-day baseline calibration gains
$$g_h = \frac{\sum_{t\in train, hod(t)=h} y_{ref}(t)}{\sum_{t\in train, hod(t)=h} P_{phys}(t)}$$
remove the multiplicative shape mismatch before the residual is formed. A
gradient-boosting learner fits $$\hat r(X)$$ on a chronological split; attribution
via SHAP → tree importance → permutation. Meta-diagnostics: RMSE improvement %,
residual $$R^2$$ (with a leakage-suspicion flag above ≈0.85), signed
aerosol-consistency, bias reduction; classified into weak/moderate/strong tiers.

### B.8 FACL (`src/facl.py`)
Mamdani inference over the four PRAM diagnostics: triangular/trapezoidal
memberships $$\mu(x)$$, rule strength by min-conjunction, max-aggregation,
centroid defuzzification:

$$ACI = \frac{\int z\,\mu_{agg}(z)\,dz}{\int \mu_{agg}(z)\,dz} \in [0,100]$$

**Gate 1 (leakage):** membership in `residual_r2 = suspicious` drives an explicit
convex guard $$ACI' = (1-\lambda)\,ACI + \lambda\,ACI_{anchor}$$ with
$$\lambda = \mu_{susp}(R^2)$$, pulling confidence toward the low anchor. A graded
"suspicious → low" support rule keeps the rule base non-empty in the deep
suspicious region (continuity, P2).
**Gate 2 (aerosol sign):** an aerosol driver raises confidence only when its
high-value effect on the residual is negative (physically consistent soiling).

### B.9 CRISP (`src/crisp.py`)
Split conformal: with calibration scores $$s_i = |y_i - \hat y_i|$$, the interval
half-width is the $$\lceil (n+1)(1-\alpha)\rceil / n$$ empirical quantile
$$\hat Q_\alpha$$; then $$P\big(y \in [\hat y \pm \hat Q_\alpha]\big) \ge 1-\alpha$$
under exchangeability. **Mondrian:** a separate $$\hat Q_{\alpha,b}$$ per
irradiance/wind regime bin $$b$$ restores conditional coverage under
heteroscedastic PV errors. Reported: coverage, MPIW, PINAW, Winkler score,
per-regime reliability.

### B.10 ACGC (`src/acgc.py`)
Three chronological blocks: **gate → calibration → test**. FACL runs on the gate
block, yielding a gate ACI; the smootherstep gain

$$w = g(ACI) = 3t^2 - 2t^3,\qquad t=\mathrm{clip}\!\left(\frac{ACI-25}{80-25},0,1\right)$$

(C¹, Lipschitz $$L\approx 0.0273$$) scales the correction:
$$\hat y_w = P_{phys} + w\,\hat r$$. Conformal calibration then wraps
$$\hat y_w$$ (B.9). Since $$w$$ depends only on the gate block, the predictor is
fixed before calibration and coverage is provably preserved (P7). Baselines
compared on the same test block: $$w=0$$ (physics-only), $$w=1$$
(always-correct), crisp hard-switch.

### B.11 RCA (`src/rca.py`)
**Regime-conditional attribution.** Per-regime permutation importance is
computed on the held-out test rows inside the **same Mondrian quantile bins**
CRISP uses (attribution and uncertainty share one taxonomy). Cross-regime
disagreement is compressed into the bounded **Attribution Instability Index**:

$$\text{AII} = \frac{1-\bar\tau_w}{2} \in [0,1],\qquad \bar\tau_w = \text{mean pairwise top-weighted Kendall }\tau$$

(0 = ranking identical in every regime; 1 = systematically reversed). A
smootherstep maps AII to a confidence retention factor
$$s(\text{AII}) \in [0.40, 1]$$, and $$\text{ACI}_{ra} = s\cdot\text{ACI}$$ —
regime-unstable global attributions lose confidence continuously (the same
convex-guard idiom as FACL Gate 1).

### B.12 Formal properties (verified by `validate_formal_properties.py`)
P1 boundedness · P2 continuity (finite modulus even across the leakage gate) ·
P3 crisp-consistency at the extremes · P4 Gate-1 dominance · P5 sectionwise
monotonicity · P6 bounded threshold sensitivity · P7 coverage preservation ·
P8 explicit Lipschitz gate · P9 physics fallback ($$w\equiv 0$$ for ACI ≤ 25) ·
P10 full-trust limit ($$w\equiv 1$$ for ACI ≥ 80) · P11 AII boundedness &
composition safety · P12 stability-map monotone + Lipschitz · P13 permutation
invariance. Current status: **15/15 pass**.

---

## Part C — The Streamlit App, Tab by Tab

### C.0 Landing flow & sidebar (before any tab)
- **Sidebar — Location:** 15 India site presets (default GCE Karad, Block M
  Hostel), free-text "lat, lon" entry with range validation, and keyed number
  inputs for latitude, longitude, elevation (elevation affects air density and
  pressure features).
- **Sidebar — Modelling:** ML model selector (12+ algorithms), data source radio
  (NASA POWER / Open-Meteo + air quality / offline synthetic), history-window
  slider (14–92 days).
- **Landing page (before 🚀 Run analysis):** an interactive **folium map picker**
  (`_landing_map_picker`, line 2167) — click anywhere to set coordinates — plus a
  preview expander listing all 100 parameters (`_show_parameter_overview`).
- **On Run:** Stage 1–4 execute with spinners; a status strip shows the data
  source actually used; a 5-KPI headline band shows PV MWh, wind MWh, annualised
  specific yield (kWh/kWp·yr), PV model R², and the 100-parameter count.

### C.1 🇦🇺 Australian DKASC — `_dksac_validation_section()` (line 2836)
The **metered-truth anchor**. Loads the real DKASC (Alice Springs) generation
CSV via `src/dkasc_parser.py`, maps its weather channels to the pipeline schema,
runs the physics model against the **measured** power, and reports agreement
metrics (correlation, nRMSE, energy ratio) per array/channel. Includes a
step-by-step math walkthrough (`_dksac_math_walkthrough`, line 3084) showing each
equation with the actual numbers substituted. This tab is what makes every
"validated on metered generation" claim true.

### C.2 🧪 PRAM (Live) — `_tab_pram_live()` (line 2576)
Runs PRAM (B.7) on the live-fetched location data in **crosssrc** mode
(NASA-driven physics vs Open-Meteo-driven physics — honestly labelled as a
data-source-consistency residual, not ground truth). Shows: the calibrated
baseline vs reference series, residual attribution ranking, the four
meta-diagnostics, the crisp tier, and the FACL ACI with its rule-firing trace
and gate status.

### C.3 🎯 CRISP Intervals — `_tab_crisp()` (line 2651)
Runs CRISP (B.9) on the DKASC metered series: user picks target coverage
(e.g. 90 %), sees global vs Mondrian intervals plotted over the test block, the
reliability curve (nominal vs empirical coverage), per-regime coverage bars
(exposing the high-irradiance under-coverage that Mondrian repairs), and
MPIW/Winkler summaries.

### C.4 ⚡ Energy & Yield — `_tab_energy()` (line 538)
"Energy generated at this location": daily/hourly PV and wind energy profiles,
capacity factors, PV specific yield and performance-ratio metrics
(`power_models.pv_yield_metrics`), monthly aggregation, and a downloadable PDF
report (`src/pdf_report.py`).

### C.5 🔬 Impact Analysis — `_tab_impact()` (line 711)
"Which atmospheric parameters drive the output?" Permutation-importance rankings
for PV and wind (B.6), the **grouped** view (radiation / thermodynamic / wind /
aerosol / temporal families), and the **secondary two-level ranking** that
normalises out the first-order driver and ranks conversion-**efficiency**
modulators (temperature derate, humidity, aerosols/dust, albedo, wind cooling).
SHAP summaries where the model supports them.

### C.6 📈 ML Accuracy — `_tab_accuracy()` (line 826)
Trains **every** available algorithm on the same chronological split and shows a
sortable leaderboard of R², RMSE, MAE, MAPE for both targets — the evidence for
"12+ ML algorithms compared".

### C.7 📊 Data Explorer — `_tab_explorer()` (line 868)
Raw dataset browser plus an interactive **parameter-vs-power scatter** — pick any
of the 100 parameters against PV or wind output, with hourly colouring, to
visually inspect the physical relationships the rankings report.

### C.8 ⚙️ Model Performance — `_tab_performance()` (line 896)
Deep-dive on the **selected** model's held-out test block (20 %): predicted vs
actual overlay, residual-over-time and residual-vs-prediction plots,
error-distribution histogram, and the metric table.

### C.9 🔮 Forecaster — `_tab_forecaster()` (line 936)
**What-if forecaster:** sliders for the key atmospheric inputs (GHI, temperature,
wind speed, humidity, AOD, …); the app rebuilds the derived features consistently
and runs the trained models to predict instantaneous PV and wind power for that
hypothetical atmosphere — the interactive sensitivity demo.

### C.10 📋 Dataset — `_tab_dataset()` (line 1473)
The full 100-parameter hourly table with provenance labels (which of the 57 came
from which API, which of the 43 are derived and by which formula), column-family
filters, and CSV download for reproducibility.

### C.11 🧮 Algorithm Evaluation — `_tab_algorithm_eval()` (line 1561)
**Step-by-step evaluation** of the selected algorithm: shows the actual
train/test split, walks through the metric computations with real substituted
numbers (the "show your work" tab for the viva), and explains each
hyperparameter choice.

### C.12 🌐 Real-Time Dataset & Evaluation — `_tab_realtime_only()` (line 999)
Fetch-focused view: the live API payload summary, the real-time 100-parameter
table for the most recent hours, and an immediate PV evaluation on the freshest
data — demonstrating the coordinate-only, real-time reproducibility claim.

### C.13 🛰️ PVGIS Validation — `_tab_pvgis()` (line 3197)
Independent **modelled** cross-check (EU JRC PVGIS, `src/pvgis.py`): queries
PVGIS for the same coordinates and plant spec, compares monthly/annual yield
against this pipeline's estimate, and reports the agreement percentage
(~5–15 % expected). Explicitly labelled as a modelled cross-check — only DKASC
(C.1) is metered truth.

---

## Cross-references

| Topic | File |
|---|---|
| Novelty claims & honest tiering | `novelty.md` |
| Q2 gap analysis & contributions C1–C7 | `NOVELTY.md` |
| Full equation derivations & symbol table | `WORKFLOW_AND_MATHEMATICS.md` |
| Formal property suite (P1–P13) | `validate_formal_properties.py` |
| Regime-conditional attribution module | `src/rca.py` |
| Headless pipeline | `run_pipeline.py` |
| Real-data harnesses | `run_facl_validation.py`, `run_crisp_validation.py` |
