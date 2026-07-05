# novelty.md — The Novel Contributions of this Project

**Project:** *Impact Analysis of Atmospheric Parameters on Photovoltaic and Wind
Power Output Using Data-Driven Modelling and Machine Learning*
**Context:** M.Tech (CSE) dissertation · Government College of Engineering, Karad
· Guide: Dr. S. A. Thorat · A.Y. 2026–2027

This document is the single place that states **every** novel contribution in the
project, **honestly classifies** each one (so nothing is over-claimed in the
paper or viva), and gives the defensible claims and the claims to avoid for each.
It is written to be read *before* writing the dissertation's "Contributions"
chapter.

> **Guiding principle — honesty of tier.** None of the contributions below claim
> a new core algorithm. They are **applied / framework-integration** novelties:
> the value is in *what is combined, on what data, and validated how*. This is a
> legitimate and defensible contribution tier for an M.Tech dissertation — and
> stating the tier plainly is exactly what makes it credible to a reviewer.

---

## 0. The contribution stack at a glance

```
                                   ┌─ FACL   →  attribution-confidence index   (how much we trust the "why")
 real/live weather                 │           (fuzzy, physics-gated)
      │                            │
      ▼                            │
 physics plant model ─► PRAM ──────┤
   (transparent)      (residual    │
                       attribution)└─ CRISP  →  prediction interval            (how uncertain the "how much")
                                                (conformal, regime-indexed)
```

| # | Name | One line | Novelty tier | Validated on |
|---|------|----------|--------------|--------------|
| 1 | **PRAM** | Attributes the *physics-model error* to atmospheric drivers | applied/framework | real metered DKASC |
| 2 | **FACL** | Fuzzifies PRAM's residual diagnostics into a gated 0–100 confidence index | applied/framework (soft-computing) | real metered DKASC |
| 3 | **CRISP** | Distribution-free, regime-conditional prediction intervals around the prediction | applied/framework (uncertainty quantification) | real metered DKASC |
| 4 | **Secondary (two-level) impact analysis** | Ranks *conversion-efficiency* modulators after normalising out the first-order driver | methodological | live + modelled |
| 5 | **Canonical 100-parameter live pipeline** | Coordinate-driven reconstruction of a 100-feature atmospheric schema from government models | integration/engineering | live NASA POWER / Open-Meteo |
| 6 | **Dual independent external validation** | The same coordinate cross-checked against **two** independent references (PVGIS + DKASC) | methodological | real |

Contributions 1–3 are the **research-tier** novelties (each has its own write-up
and its own real-data validation harness). Contributions 4–6 are **supporting**
methodological / integration novelties that strengthen the thesis but should be
positioned as engineering contributions, not headline research claims.

---

## 1. PRAM — Physics-Residual Attribution Model
`src/pram.py` · harness `run_facl_validation.py` (also exercises PRAM) · tabs 🇦🇺 / 🧪

### What it is
Instead of asking "which weather variable predicts power" (the saturated
question), PRAM asks **"which weather variable explains what my physics model
*gets wrong*."** It forms `residual = reference − physics_model`, fits a
gradient-boosting model on that residual over a **leakage-safe chronological
split**, and attributes the residual to interpretable atmospheric + temporal
drivers (SHAP → tree importance → permutation, in that order of availability).

Two honest modes:
- **measured** — reference = real metered DKASC power; `residual = P_measured −
  P_physics` is the *scientifically valid* quantity (what the physics misses vs
  reality). This is the anchor.
- **crosssrc** — reference = NASA-driven physics vs Open-Meteo-driven physics; a
  *data-source consistency* residual with **no** ground truth (labelled as such).

### Why the integration is non-generic (the reportable part)
1. **The learning target is the physics error, not the power.** The residual only
   exists because a transparent physical plant model produced a prediction first,
   so the attribution is downstream of physics — not the usual "correlate GHI
   with output."
2. **Baseline calibration turns a multiplicative shape-error into an additive
   residual.** A single fixed-tilt physics curve has the wrong daily *shape* for a
   meter that aggregates tracking / off-axis arrays. A per-hour-of-day gain,
   fitted **on the training portion only**, removes that gross bias so the
   residual reflects real physics gaps, not geometry mismatch.
3. **An aerosol/soiling signature detector.** PRAM flags when an aerosol feature
   (AOD/PM/dust) both ranks highly *and* pushes the residual **negative** at high
   values — the physical fingerprint of the soiling loss the physics omits.
4. **A leakage-aware verdict.** A suspiciously perfect residual (R² > 0.85) is
   flagged as a probable leakage / mis-scaled-baseline artefact rather than a win.

### Defensible claims
- "We attribute the *residual error* of a transparent PV physics model to
  atmospheric drivers, validated against real metered DKASC generation."
- "A train-only per-hour baseline calibration isolates the genuine physical gap
  from the fixed-tilt-vs-whole-site shape mismatch."

### Claims to avoid
- That the residual learner (gradient boosting / SHAP) is itself novel.
- Treating `crosssrc` mode as validation — it has no metered truth.

---

## 2. FACL — Physics-Residual-Gated Fuzzy Attribution-Confidence Layer
`src/facl.py` · harness `run_facl_validation.py` · docs `docs/FACL_soft_computing.md`

### What it is
A textbook **Mamdani** fuzzy inference system (triangular/trapezoidal
memberships, min-inference, max-aggregation, centroid defuzzification) whose
**inputs are PRAM's residual diagnostics, not raw weather**. It consumes
`RMSE-improvement`, `residual R²`, a signed `aerosol-consistency` score and
`bias-reduction`, and emits a smooth **0–100 Attribution-Confidence Index (ACI)**
with a linguistic verdict, a human-readable **rule-firing trace**, and two
**physics gates**.

### Why the integration is non-generic (the reportable part)
1. **The input space is physics-residual *explainability*, not meteorology.**
   "Fuzzy logic for PV/wind" is saturated (forecasting, MPPT, fault detection) and
   would be rejected on sight. Fuzzifying *attribution diagnostics that only exist
   because PRAM ran first* is what makes this downstream and specific.
2. **Two domain constraints encoded as graded logic ("gates"):**
   - **Gate 1 — leakage suppression:** a residual R² in the suspicious region
     (> ~0.85) pulls confidence *down* toward a low anchor regardless of apparent
     gain (a convex guard, because one Mamdani rule cannot override
     max-aggregation).
   - **Gate 2 — aerosol sign-consistency:** an aerosol driver *raises* confidence
     only when its effect is physically negative at high AOD; a positive-sign
     aerosol "signature" is implausible and *lowers* it.
3. **A strict generalisation of the crisp tiers.** Membership breakpoints are tied
   to PRAM's existing thresholds, so at the extremes FACL reproduces
   weak/moderate/strong, and near the old hard cut-offs it degrades *smoothly*
   instead of flipping (the no-cliff ACI surface).

### What the real-data validation actually showed (do not overclaim)
- **Gate 1 fired on real data.** Oct 2018 (Array M10): residual R² = 0.883
  (suspicious), apparent improvement +72 %, **crisp tier = "strong"** — FACL's
  leakage guard demoted the ACI to ≈39 ("low"). This is the headline: FACL is
  *correctly more conservative exactly where the crisp logic is most easily
  fooled.*
- **No validated accuracy-monotonicity.** Spearman between ACI and the
  *independent* test-period correlation `r` (not a FACL input) was slightly
  **negative (≈ −0.2)** on that channel. So FACL is a smoother, leakage-aware,
  interpretable *verdict* layer — **not** a proven predictor of model accuracy.

### Defensible claims
- "We replace PRAM's crisp verdict thresholds with a physics-residual-gated fuzzy
  inference layer yielding graded, uncertainty-aware, interpretable verdicts."
- "On measured DKASC data the leakage gate correctly demoted an over-fit window
  that crisp thresholds rated 'strong'."

### Claims to avoid
- Any statement that the fuzzy engine is algorithmically novel.
- "FACL improves forecasting accuracy" / "ACI predicts accuracy" — contradicted by
  the harness's own output.

---

## 3. CRISP — Conformal Regime-Indexed Split-conformal Prediction intervals
`src/crisp.py` · harness `run_crisp_validation.py` · docs `docs/CRISP_conformal.md` · tab 🎯

### What it is
A distribution-free **uncertainty-quantification** layer. It turns a *point*
prediction into a **prediction interval** with a finite-sample coverage
guarantee, and — the reportable twist — keeps that guarantee **conditionally
within each irradiance/wind regime** (Mondrian conformal). It is **model-agnostic**
(wraps the physics model, the PRAM-corrected series, or any of the 12 ML models),
which is why the core is pure `numpy`/`pandas`.

**Method:** split conformal (Vovk; Lei et al. 2018) — record the predictor's
absolute errors on an earlier *calibration* block; the `⌈(n+1)(1−α)⌉/n` empirical
quantile of those errors is the interval half-width on the later held-out *test*
block. With exchangeable data this **guarantees** marginal coverage ≥ 1−α with no
distributional assumption. **Mondrian** computes a separate quantile per regime
to restore *conditional* validity that a single global width cannot provide under
heteroscedastic PV errors.

### Why the integration is non-generic (the reportable part)
1. **Uncertainty on a physics / physics-residual predictor**, so the interval
   inherits the interpretability of the model underneath it.
2. **The Mondrian regimes are the atmospheric drivers the impact analysis ranks**
   (GHI/POA for PV, hub wind for wind) — the conditional guarantee is expressed
   in a physically meaningful coordinate.
3. **Validated for *conditional* coverage on real metered data**, not just "we
   drew a band."
4. **A distinct axis from FACL** — predictive uncertainty vs attribution
   confidence; they compose on top of PRAM.

### What the real-data validation actually showed
On the DKASC Alice Springs 2018 whole-site meter (2083 calib → 2084 test, target
90 %):
- **Marginal coverage holds:** Mondrian **93.3 %**, Global 91.7 %.
- **Reliability curve near-diagonal:** nominal 0.50/0.80/0.90/0.95 → empirical
  0.509/0.775/0.933/0.964.
- **The headline (conditional):** a single global band **under-covers the
  high-irradiance (peak-power) regime at 86.3 %** while over-covering the dark
  regime at 97.5 %; **Mondrian evens every regime to 89.8–93.6 %**, cutting the
  worst-regime coverage gap.
- **Single-array channel:** 89.4 % marginal at a 90 % target — near-perfect.
- **Honest limitation:** short single-month windows under-cover (~81 % mean)
  because small, drifting calibration sets break exchangeability — reported, not
  hidden; motivates adaptive/online conformal as future work.

### Defensible claims
- "We add distribution-free, finite-sample-valid prediction intervals to the PV
  predictor, with near-nominal marginal coverage on real metered DKASC data."
- "Regime-indexed (Mondrian) conformal restores conditional coverage: on real
  data it lifts the under-covered high-irradiance regime from ≈86 % to ≈90 %."

### Claims to avoid
- That the conformal method itself is algorithmically novel.
- "Valid on any window" — short drifting windows under-cover.
- "CRISP improves point accuracy" — it quantifies uncertainty, it does not change
  the point prediction.

---

## 4. Secondary (two-level) impact analysis  *(supporting novelty)*
`src/impact_analysis.py`

Standard impact analysis ranks the *first-order* driver (irradiance for PV, wind
speed for wind) and stops — which is uninformative because everyone already knows
sun drives PV. The **secondary** method normalises the first-order driver out
(computing a conversion **efficiency** target `ε = P / P_ideal` over productive
hours and dropping all radiation/geometry or wind-speed proxies), then ranks what
modulates *efficiency*: cell-temperature derate, humidity, **aerosols/dust**,
albedo, wind cooling. This surfaces the subtle conversion-efficiency effects a
raw output ranking hides.

**Defensible claim:** "A two-level analysis that isolates conversion-efficiency
modulators from the dominant resource driver." **Avoid:** calling the underlying
permutation importance novel.

---

## 5. Canonical 100-parameter coordinate-driven live pipeline  *(supporting novelty)*
`src/data_fetcher.py`, `src/feature_engineering.py`, `config.py`

From only a **latitude / longitude / elevation**, the pipeline fetches real
government-model weather (NASA POWER / Open-Meteo + air-quality) and reconstructs
a **canonical 57 raw + 43 derived = 100-parameter** atmospheric feature schema
(asserted in `config.py`), including solar geometry, thermodynamics, wind physics,
an **AOD/dust/PM-driven soiling index**, and albedo ground-reflection — with a
same-schema synthetic fallback so the whole system runs offline. The novelty is
the **standardised, reproducible, coordinate-only reconstruction** of a fixed
100-feature space across heterogeneous data sources, not any single formula.

**Defensible claim:** "A reproducible coordinate-driven pipeline that reconstructs
one canonical 100-parameter atmospheric feature space from multiple government
sources with a physically-plausible offline fallback." **Avoid:** implying the
individual atmospheric relations (Cooper declination, Kasten–Young air mass, etc.)
are new — they are standard and cited.

---

## 6. Dual independent external validation  *(supporting novelty)*
`src/pvgis.py` (PVGIS, EU JRC) · `src/dksac.py` (measured DKASC)

The same coordinate/plant is cross-checked against **two independent references**:
**PVGIS** (satellite-based yield estimator, a coordinate-driven *modelled*
cross-check, ~5–15 % agreement expected) and **DKASC** (real *metered* generation,
the scientific anchor). Using both a modelled independent estimator *and* a
measured anchor — and being explicit about which is which — is a stronger
validation design than either alone.

**Defensible claim:** "Yield magnitude is cross-verified against an independent
modelled estimator (PVGIS) and, separately, the physics is validated against real
metered generation (DKASC)." **Avoid:** calling PVGIS agreement "validation
against measurement" — only DKASC is metered truth.

---

## 7. How to position this in the dissertation / viva

- **Lead with the three research contributions (PRAM → FACL → CRISP)** as one
  coherent stack: *explain the physics error → grade trust in that explanation →
  bound the prediction's uncertainty.* The narrative arc is itself a selling
  point: three small, honest, individually-defensible layers on one transparent
  physical base.
- **State the tier out loud.** "These are applied/framework-integration
  contributions validated on real metered data" is a *strength*, not a hedge — it
  pre-empts the reviewer's "is the algorithm new?" objection.
- **Let the honest negative findings stand.** FACL's near-zero independent
  Spearman and CRISP's short-window under-coverage are *evidence of rigour*. A
  viva panel trusts a candidate who reports them.
- **Each contribution has a reproducible harness on the real DKASC CSV**
  (`run_facl_validation.py`, `run_crisp_validation.py`) — run them live if asked;
  every number in this document is traceable to that code.

## 8. Pointers

| Contribution | Code | Write-up | Math (read121.md) |
|---|---|---|---|
| PRAM | `src/pram.py` | (read121 §6.11) | §6.11 |
| FACL | `src/facl.py` | `docs/FACL_soft_computing.md` | §6.12 |
| CRISP | `src/crisp.py` | `docs/CRISP_conformal.md` | §6.15 |
| Secondary impact | `src/impact_analysis.py` | (read121 §6.10) | §6.10 |
| 100-param pipeline | `src/data_fetcher.py`, `src/feature_engineering.py` | (read121 §4, §6) | §6.1–6.5 |
| Dual validation | `src/pvgis.py`, `src/dksac.py` | (read121 §6.13–6.14) | §6.13–6.14 |
