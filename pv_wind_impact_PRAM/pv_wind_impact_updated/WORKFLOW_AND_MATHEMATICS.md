# WORKFLOW_AND_MATHEMATICS.md

## Complete Working Workflow and Mathematical Modelling
### Physics-Residual-Gated Fuzzy Attribution Confidence for Photovoltaic Power

This document maps every stage of the pipeline to its source file and gives the full mathematical model implemented in the code.

---

## Part A — End-to-End Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 0  Site & configuration            config.py                     │
│   lat, lon, elevation, tilt, azimuth; plant specs (300 kW PV,          │
│   500 kW wind); the 100-parameter feature definition                   │
└──────────────────────────┬─────────────────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1  Weather acquisition             src/data_fetcher.py           │
│   NASA POWER / Open-Meteo hourly archives (57 raw variables);          │
│   synthetic clear-sky fallback when offline                            │
└──────────────────────────┬─────────────────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2  Feature engineering (100 params) src/feature_engineering.py   │
│   solar geometry, clearness index, POA transposition, cell temp,       │
│   thermal derate, soiling index, hub wind, air density, humidity       │
│   metrics, temporal encodings                                          │
└──────────────────────────┬─────────────────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 3  Physical plant models            src/power_models.py          │
│   PV: irradiance→DC→thermal/soiling derates→inverter clip              │
│   Wind: power-law shear→hub speed→power curve→density scaling          │
│   Yield metrics: specific yield, PR, peak-sun-hours                    │
└──────────────┬───────────────────────────────────┬─────────────────────┘
               ▼                                   ▼
┌───────────────────────────────┐   ┌───────────────────────────────────┐
│ STAGE 4a  ML benchmark        │   │ STAGE 4b  Metered ground truth    │
│ src/ml_models.py              │   │ src/dksac.py, dkasc_parser.py     │
│ 12+ regressors, chrono splits │   │ Real DKASC 5-min→hourly power     │
│ impact_analysis.py: permut.   │   │ aligned with on-site weather      │
│ importance, SHAP, groups      │   │                                   │
└──────────────┬────────────────┘   └───────────────┬───────────────────┘
               └──────────────────┬─────────────────┘
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 5  PRAM — physics-residual attribution   src/pram.py             │
│   residual r(t) = P_measured − P_physics_calibrated                    │
│   train-only per-hour gain calibration → HistGB residual learner →     │
│   SHAP attribution → 4 meta-diagnostics                                │
└──────────────────────────┬─────────────────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 6  FACL — fuzzy attribution confidence   src/facl.py             │
│   Mamdani inference over the 4 meta-diagnostics → ACI (0–100),         │
│   linguistic verdict, rule trace; Gate 1 (leakage), Gate 2 (aerosol)   │
└──────────────────────────┬─────────────────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 7  CRISP — conformal intervals           src/crisp.py            │
│   Mondrian (regime-indexed) split-conformal intervals on the           │
│   corrected predictor; marginal + conditional coverage, Winkler        │
└──────────────────────────┬─────────────────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 8  Reporting                        app.py, src/plotting.py,     │
│   Streamlit tabs, PVGIS cross-check (src/pvgis.py), PDF report         │
└────────────────────────────────────────────────────────────────────────┘
```

Headless equivalents: `run_pipeline.py` (stages 1–4a), `validate_crisp_dkasc.py`, `validate_facl_dkasc.py` (stages 4b–7 on metered data).

---

## Part B — Mathematical Modelling

### B.1 Solar geometry (`src/feature_engineering.py :: _solar_geometry`)

Declination (Cooper's equation), day of year $$n$$:

$$\delta = 23.45° \cdot \sin\!\left(\frac{2\pi(284+n)}{365}\right)$$

Hour angle from local solar time $$t_s$$:

$$\omega = 15°\,(t_s - 12)$$

Solar zenith angle at latitude $$\phi$$:

$$\cos\theta_z = \sin\phi\sin\delta + \cos\phi\cos\delta\cos\omega, \qquad \theta_z = \arccos(\cos\theta_z),\qquad \alpha_s = 90° - \theta_z$$

Relative air mass (Kasten–Young 1989):

$$AM = \frac{1}{\cos\theta_z + 0.50572\,(96.07995 - \theta_z)^{-1.6364}}$$

Extraterrestrial horizontal irradiance with solar constant $$G_{sc}=1361\ \text{W/m}^2$$:

$$G_0 = G_{sc}\cdot\max(\cos\theta_z, 0)$$

### B.2 Clearness index and plane-of-array irradiance

$$k_t = \operatorname{clip}\!\left(\frac{GHI}{G_0},\,0,\,1.2\right)$$

Angle of incidence on a plane tilted at $$\beta$$: the beam component is transposed with $$\cos(\text{AOI})$$, and the effective (POA) irradiance combines beam, isotropic diffuse and ground-reflected terms:

$$G_{POA} = DNI\cos(\text{AOI}) + DHI\,\frac{1+\cos\beta}{2} + GHI\,\rho_g\,\frac{1-\cos\beta}{2}$$

### B.3 PV thermal model (NOCT) and derates

Cell temperature from ambient $$T_a$$ and GHI (NOCT model):

$$T_{cell} = T_a + \frac{NOCT - 20}{800}\cdot GHI$$

Thermal derate with power temperature coefficient $$\gamma$$ (per °C, from `config.PV_TEMP_COEFF`):

$$\eta_T = \operatorname{clip}\big(1 - \gamma\,(T_{cell} - 25),\ \eta_{min},\ 1\big)$$

Soiling index from relative humidity, aerosol load $$A$$ and precipitation-driven cleaning:

$$S = \operatorname{clip}\Big(1 - 0.05\,\operatorname{clip}\!\big(\tfrac{RH-50}{50},0,1\big) - 0.10\,A + 0.08\cdot\mathbb{1}[P > 0.5\,\text{mm}],\ 0.78,\ 1\Big)$$

### B.4 PV power model (`src/power_models.py :: pv_power_from_features`)

$$P_{PV} = \operatorname{clip}\!\left(\frac{G_{eff}}{1000}\cdot A_{PV}\cdot \eta_{PV}\cdot \eta_T \cdot S \cdot \varepsilon,\ 0,\ P_{AC}^{cap}\right)$$

where $$G_{eff}$$ is the effective POA irradiance, $$A_{PV}$$ the array area, $$\eta_{PV}$$ the STC efficiency, $$P_{AC}^{cap}$$ the inverter capacity, and $$\varepsilon \sim \mathcal{N}(1, 0.05)$$ a multiplicative measurement-noise term (disabled for point physics estimates).

**Yield metrics** (window of $$D$$ days): DC nameplate $$P_{kWp} = A_{PV}\,\eta_{PV}$$; final yield $$Y_f = E_{PV}/P_{kWp}$$; reference yield $$Y_r = H_{POA}$$ (kWh/m² ≡ peak-sun-hours); performance ratio:

$$PR = \frac{Y_f}{Y_r}, \qquad \text{annual specific yield} = Y_f\cdot\frac{365}{D}\ \ \text{kWh/kWp/yr}$$

### B.5 Wind resource model

Wind-shear exponent from the two measured levels (power law):

$$\alpha = \frac{\ln(v_{120}/v_{10})}{\ln(120/10)}, \qquad v_{hub} = v_{10}\left(\frac{h_{hub}}{10}\right)^{\alpha}$$

Air density at hub height (barometric lapse + ideal gas, $$R = 287.05$$ J·kg⁻¹·K⁻¹):

$$p_{hub} = p_{sfc}\,(1 - 2.25577\times10^{-5}\,h_{hub})^{5.25588}, \qquad \rho_{hub} = \frac{p_{hub}}{R\,T_{hub}}$$

Turbine power curve (cut-in $$v_{ci}$$, rated $$v_r$$, cut-out $$v_{co}$$):

$$P_{W}(v) = \begin{cases} 0 & v < v_{ci} \\ P_r\,\dfrac{v^3 - v_{ci}^3}{v_r^3 - v_{ci}^3} & v_{ci} \le v < v_r \\ P_r & v_r \le v \le v_{co} \\ 0 & v > v_{co} \end{cases} \qquad P_{Wind} = P_W(v_{hub})\cdot\operatorname{clip}\!\left(\frac{\rho_{hub}}{1.225},\,0.6,\,1.3\right)$$

Wind power density: $$WPD = \tfrac{1}{2}\rho v^3$$.

### B.6 ML benchmark and impact analysis (`src/ml_models.py`, `src/impact_analysis.py`)

Each regressor $$f_m$$ is trained on a chronological split (no shuffling). Scores on the held-out block:

$$R^2 = 1 - \frac{\sum_t (y_t - \hat y_t)^2}{\sum_t (y_t - \bar y)^2}, \quad RMSE = \sqrt{\tfrac{1}{N}\sum_t (y_t-\hat y_t)^2}, \quad MAE = \tfrac{1}{N}\sum_t |y_t - \hat y_t|$$

Permutation importance of feature $$j$$ (mean over $$K$$ repeats, $$\pi$$ a random permutation of column $$j$$):

$$PI_j = \frac{1}{K}\sum_{k=1}^{K}\Big[\mathcal{L}\big(y, f(X^{\pi_k(j)})\big) - \mathcal{L}\big(y, f(X)\big)\Big]$$

SHAP values decompose each prediction additively: $$\hat y_t = \phi_0 + \sum_j \phi_{j,t}$$, with global importance $$\overline{|\phi_j|} = \tfrac{1}{N}\sum_t |\phi_{j,t}|$$ and signed mean $$\bar\phi_j = \tfrac{1}{N}\sum_t \phi_{j,t}$$ (used for sign-consistency in B.8).

---

### B.7 PRAM — Physics-Residual Attribution Model (`src/pram.py`)

**Operating mask** (measured mode): $$\mathcal{O} = \{t : P^{meas}_t > 0.02\,P_{rated}\}$$.

**Step 1 — Baseline calibration (train-only, leakage-safe).** Over the first $$1-\tau$$ fraction ($$\tau = 0.2$$) of operating hours, a per-hour-of-day multiplicative gain corrects the shape mismatch between a single fixed-tilt physics model and a whole-site meter:

$$g_h = \operatorname{clip}\!\Big(\operatorname{median}_{t\in \mathcal{T}_{train},\, hour(t)=h} \frac{P^{meas}_t}{P^{phys}_t},\ 0.2,\ 5\Big), \qquad \tilde P^{phys}_t = g_{hour(t)}\,P^{phys}_t$$

**Step 2 — Residual definition.**

$$r_t = P^{meas}_t - \tilde P^{phys}_t$$

**Step 3 — Residual feature space** $$X_t$$: atmospheric drivers (GHI, POA, DHI, DNI, T, RH, wind, rain, pressure, AOD/PM), temporal encodings $$\sin/\cos(2\pi h/24)$$, $$\sin/\cos(2\pi n/365)$$, and physics-derived terms: clear-sky index $$GHI/G_0$$, $$GHI^2$$, $$GHI\cdot\cos\theta_z$$, $$POA/GHI$$, lag-1 and 3-h rolling GHI, $$T\cdot v$$ (wind cooling), $$\max(T-25,0)$$ (thermal excess).

**Step 4 — Residual learner.** Histogram gradient boosting $$\hat r = g(X)$$ fitted on the chronological train block (train residual outliers trimmed at the 1st/99th percentile; test untouched). Out-of-sample residual explainability:

$$R^2_{res} = 1 - \frac{\sum_{t\in test}(r_t - \hat r_t)^2}{\sum_{t\in test}(r_t - \bar r)^2}$$

**Step 5 — The four meta-diagnostics** (the FACL input vector):

$$I = 100\left(1 - \frac{RMSE(P^{meas}, \tilde P^{phys} + \hat r)}{RMSE(P^{meas}, \tilde P^{phys})}\right) \quad \text{(residual-correction gain, \%, test block)}$$

$$R^2_{res} \quad\text{(step 4)}, \qquad \Delta_{bias} = |MBE_{phys}| - |MBE_{corr}|, \qquad MBE = \tfrac{1}{N}\sum_t(\hat y_t - y_t)$$

Aerosol sign-consistency score from the top-ranked aerosol feature (rank $$\rho$$, signed SHAP mean $$\bar\phi$$):

$$s_{aero} = \begin{cases} \operatorname{clip}(1.2/\rho,0,1) & \bar\phi < 0 \ \text{(physically consistent soiling)}\\ -0.7\,\operatorname{clip}(1.2/\rho,0,1) & \bar\phi > 0 \ \text{(wrong sign)}\\ 0.3\,\operatorname{clip}(1.2/\rho,0,1) & \text{sign unknown} \end{cases}$$

---

### B.8 FACL — Fuzzy Attribution-Confidence Layer (`src/facl.py`)

**Membership functions.** Triangular and trapezoidal:

$$\mu_{tri}(x; a,b,c) = \max\!\Big(\min\!\Big(\frac{x-a}{b-a}, \frac{c-x}{c-b}\Big), 0\Big), \qquad \mu_{trap}(x; a,b,c,d) = \max\!\Big(\min\!\Big(\frac{x-a}{b-a}, 1, \frac{d-x}{d-c}\Big), 0\Big)$$

Input partitions (breakpoints tied to the legacy crisp thresholds so FACL is a strict generalisation):

| Variable | Terms (parameters) |
|---|---|
| improvement $$I$$ (%) | low: trap(−∞,−∞,1,5) · medium: tri(2, 5.5, 9) · high: trap(6,10,∞,∞) |
| residual R² | low: trap(−1,−1,0.10,0.20) · medium: tri(0.14,0.30,0.46) · high: trap(0.34,0.52,0.80,0.84) · **suspicious: trap(0.80,0.90,1,1)** |
| aerosol $$s_{aero}$$ | contradictory: trap(−1,−1,−0.5,0) · neutral: tri(−0.3,0,0.3) · supportive: trap(0,0.5,1,1) |
| bias $$\Delta_{bias}$$ (kW) | worsened: trap(−∞,−∞,−0.02,0) · neutral: tri(−0.05,0,0.05) · improved: trap(0,0.02,∞,∞) |

Output confidence universe $$u \in [0,100]$$: very_low, low, moderate, high, very_high (tri/trap partitions).

**Inference (Mamdani).** For rule $$k$$ with antecedent terms $$(v_i, T_i)$$ and weight $$w_k$$:

$$\text{firing strength: } \sigma_k = w_k \cdot \min_i \mu_{T_i}(x_{v_i})$$

Max-aggregation per output term $$T$$: $$\Sigma_T = \max_{k:\,out(k)=T} \sigma_k$$. Aggregated output MF and **centroid defuzzification**:

$$\mu_{agg}(u) = \max_T \min\big(\mu_T(u), \Sigma_T\big), \qquad \text{ACI}_{raw} = \frac{\int_0^{100} u\,\mu_{agg}(u)\,du}{\int_0^{100} \mu_{agg}(u)\,du}$$

**Rule base** (14 rules): a 3×3 improvement × explainability core grid (very_high … very_low), two aerosol-gate rules (supportive → boost; contradictory → suppress), and two bias-corroboration rules.

**Gate 1 — leakage guard** (applied as a convex operator, provably dominant over max-aggregation). With suspicion membership $$\mu_{susp} = \mu_{trap}(R^2_{res};0.80,0.90,1,1)$$ and anchor $$A = 28$$:

$$\boxed{\ \text{ACI} = \text{ACI}_{raw} + \mu_{susp}\,\big(A - \text{ACI}_{raw}\big)\ }$$

so $$\mu_{susp}=1 \Rightarrow \text{ACI} = A$$ regardless of every other signal (property P4), and $$\mu_{susp}=0 \Rightarrow \text{ACI}=\text{ACI}_{raw}$$.

**Gate 2 — aerosol sign-consistency** operates through $$s_{aero}$$ (B.7): a supportive (negative-signed, high-rank) aerosol driver fires the boost rules; a contradictory one fires the suppression rule.

**Verdict mapping:** ACI ≥ 80 very_high · ≥ 60 high · ≥ 45 moderate · ≥ 25 low · else very_low, each with a human-readable label and the ranked rule-firing trace.

**Formal properties (paper §4):**
- **P1 Boundedness:** $$\text{ACI} \in [0,100]$$ — centroid of a non-negative function supported on [0,100]; the gate is a convex combination with $$A\in[0,100]$$.
- **P2 Continuity:** all MFs are piecewise-linear (continuous); min, max, product and the centroid $$\Sigma \mapsto \int u\,\mu_{agg}/\int \mu_{agg}$$ are continuous wherever $$\int\mu_{agg} > 0$$; the rule base guarantees at least one rule fires everywhere on the closed input domain.
- **P3 Crisp-consistency:** in MF saturation regions the fired-term set is a singleton and the ACI verdict reproduces the legacy tiers of `pram.classify`.
- **P4 Gate dominance:** boxed equation above.
- **P5 Sectionwise monotonicity** (in $$I$$ and $$R^2_{res}$$ outside the suspicious region) and **P6 bounded breakpoint sensitivity** — verified numerically on a dense grid (`aci_surface`).

---

### B.9 CRISP — Regime-Indexed Split-Conformal Intervals (`src/crisp.py`)

**Nonconformity score** on the calibration block (earliest $$c$$ fraction of operating rows, chronological): $$s_t = |y_t - \hat y_t|$$.

**Finite-sample conformal quantile** (Lei et al. 2018) at miscoverage $$\alpha$$ with $$n$$ calibration scores:

$$\hat Q = s_{(k)}, \qquad k = \lceil (n+1)(1-\alpha) \rceil$$

giving the marginal guarantee $$\Pr\big(y \in [\hat y - \hat Q,\ \hat y + \hat Q]\big) \ge 1 - \alpha$$ under exchangeability.

**Mondrian (regime-conditional) variant.** The driver (GHI/POA) is partitioned into $$B$$ quantile regimes with edges $$e_1 < \dots < e_{B-1}$$; each regime gets its own quantile $$\hat Q_b$$ (falling back to the global $$\hat Q$$ if $$n_b < n_{min}$$), restoring per-regime conditional validity:

$$\Pr\big(y \in \hat y \pm \hat Q_{b(x)} \,\big|\, x \in \text{regime } b\big) \ge 1-\alpha$$

**Normalized variant:** locally-adaptive width $$\hat y \pm \hat Q^{norm}\,\hat\sigma_b$$ with per-regime difficulty $$\hat\sigma_b = \text{mean}|s|_b$$ and a single quantile of the normalized scores $$s_t/\hat\sigma_{b(t)}$$.

Bounds are clipped to the physical range $$[0, P_{rated}]$$ (which can only raise coverage).

**Evaluation metrics** on the held-out later test block:

$$\text{coverage} = \tfrac{1}{N}\sum_t \mathbb{1}[l_t \le y_t \le u_t], \qquad MPIW = \tfrac{1}{N}\sum_t (u_t - l_t), \qquad PINAW = \frac{MPIW}{\max y - \min y}$$

Winkler / interval score (proper scoring rule; lower better):

$$W_t = (u_t - l_t) + \frac{2}{\alpha}(l_t - y_t)\,\mathbb{1}[y_t < l_t] + \frac{2}{\alpha}(y_t - u_t)\,\mathbb{1}[y_t > u_t]$$

**Headline conditional-validity metric** — the worst-regime coverage gap:

$$\Gamma = \max_b \big|\,\text{coverage}_b - (1-\alpha)\,\big|$$

with the empirical finding that Mondrian shrinks $$\Gamma$$ versus the global calibrator on real DKASC data. A reliability curve (nominal $$1-\alpha$$ vs empirical coverage across $$\alpha \in \{0.5, 0.2, 0.1, 0.05\}$$) verifies calibration at every level.

---

### B.10 Validation protocol (honesty rules used everywhere)

1. **Chronological splits only** — train/calibration is always an *earlier* block; test always a *later* block; never shuffled.
2. **Train-only calibration** — the per-hour physics gains (B.7 step 1) and outlier trimming touch the training block only.
3. **Dual-mode labelling** — `measured` (DKASC metered ground truth) vs `crosssrc` (NASA-POWER vs Open-Meteo model-consistency) results are never conflated.
4. **Leakage screening** — $$R^2_{res} > 0.85$$ triggers Gate 1 suppression rather than being reported as success.
5. **Operating-hours masking** — all PV metrics computed on $$P > 2\%$$ of rated only (no trivial night-time inflation of coverage or R²).

---

## Part C — Symbol Table

| Symbol | Meaning | Where |
|---|---|---|
| $$\theta_z,\ \delta,\ \omega,\ AM$$ | zenith, declination, hour angle, air mass | B.1 |
| $$G_0,\ k_t,\ G_{POA},\ G_{eff}$$ | extraterrestrial, clearness, POA, effective irradiance | B.1–B.2 |
| $$T_{cell},\ \eta_T,\ S$$ | cell temperature, thermal derate, soiling index | B.3 |
| $$P_{PV},\ P_{Wind},\ P_{rated}$$ | PV / wind power, rated capacity | B.4–B.5 |
| $$Y_f,\ Y_r,\ PR$$ | final/reference yield, performance ratio | B.4 |
| $$\alpha,\ v_{hub},\ \rho_{hub}$$ | shear exponent, hub wind, hub air density | B.5 |
| $$PI_j,\ \phi_{j,t}$$ | permutation importance, SHAP value | B.6 |
| $$g_h,\ r_t,\ \hat r_t,\ R^2_{res}$$ | hourly gain, residual, learned residual, residual R² | B.7 |
| $$I,\ \Delta_{bias},\ s_{aero}$$ | RMSE gain %, bias reduction, aerosol consistency | B.7 |
| $$\mu,\ \sigma_k,\ \Sigma_T,\ \text{ACI}$$ | membership, rule strength, term aggregate, confidence index | B.8 |
| $$\mu_{susp},\ A$$ | leakage suspicion, guard anchor (28) | B.8 |
| $$\hat Q,\ \hat Q_b,\ \Gamma$$ | conformal quantile, per-regime quantile, worst-regime gap | B.9 |
| $$MPIW,\ PINAW,\ W$$ | interval width metrics, Winkler score | B.9 |
