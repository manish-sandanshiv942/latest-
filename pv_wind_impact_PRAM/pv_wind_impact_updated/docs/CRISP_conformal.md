# CRISP — Conformal Prediction Intervals for PV/Wind Power

**Conformal Regime-Indexed Split-conformal Prediction intervals**
Module: `src/crisp.py` · Harness: `run_crisp_validation.py` · UI: 🎯 CRISP Intervals tab in `app.py`

---

## 1. What was added, in one sentence

A distribution-free **uncertainty-quantification layer** that turns the
project's *point* PV/wind predictions into **calibrated prediction intervals**
with a finite-sample coverage guarantee, and — the reportable part — keeps that
guarantee **conditionally within each irradiance/wind regime** (Mondrian
conformal), validated on the **real metered DKASC** generation.

## 2. Honest novelty classification — read this before writing the paper

This is an **applied / framework-integration** contribution, in the *same tier*
as PRAM's and FACL's own novelty. State it that way. It is deliberately **NOT**:

- a new conformal-prediction algorithm (split conformal, Mondrian/class-
  conditional conformal, and normalized nonconformity scores are all textbook —
  Vovk et al. 2005; Lei et al. 2018; Romano et al. 2019);
- "machine learning forecasts PV/wind" as a novel claim — that space is
  saturated and any such claim will be rejected on sight.

What makes the *integration* non-generic and reportable:

1. **It quantifies uncertainty on a physics / physics-residual predictor, not a
   black box.** CRISP is model-agnostic: it consumes point predictions
   (`ŷ`) plus truth (`y`) and a **regime driver**, so it wraps the transparent
   physics PV model (or the PRAM-corrected series, or any of the 12 ML models)
   in a guaranteed-coverage band. The interval inherits the interpretability of
   the predictor underneath it.

2. **The Mondrian taxonomy is the atmospheric driver the impact analysis
   ranks.** Regimes are irradiance (GHI/POA) bins for PV and hub-wind bins for
   wind — the very first-order drivers PRAM and the impact tab identify. So the
   conditional-coverage guarantee is expressed in the physically meaningful
   coordinate, not an arbitrary partition.

3. **The validation is conditional coverage on real metered data.** The headline
   is not "we drew a band" but "on real DKASC generation a single global band
   silently under-covers the high-power regime, and the regime-indexed band
   fixes it" — a concrete, measurable, honest result.

4. **A distinct axis from FACL.** FACL scores *confidence in the attribution*
   ("do we trust which drivers explain the residual"). CRISP scores *predictive
   uncertainty* ("how wide must the band be to contain tomorrow's power"). They
   answer different questions and compose: PRAM → {FACL attribution confidence,
   CRISP predictive interval}.

## 3. Architecture

```
point predictor (physics / PRAM-corrected / any ML)
   │  ŷ on operating hours, time-ordered
   ▼
chronological split ──►  calibration block (earlier)      test block (later, held-out)
                              │                                   │
                    nonconformity scores s = |y - ŷ|              │
                              │                                   │
             ┌────────────────┴─────────────────┐                │
   regime driver (GHI/POA)  →  quantile bins (Mondrian taxonomy)  │
             │                                                    │
     per-regime conformal quantile  Q_k = ⌈(n_k+1)(1-α)⌉/n_k      │
             │                                                    ▼
             └───────────────►  interval  [ŷ - Q_k , ŷ + Q_k]  ──► coverage report
                                                                    (marginal + per-regime,
                                                                     MPIW/PINAW, Winkler,
                                                                     reliability curve)
```

Three interchangeable methods (`fit_conformal(method=...)`):

| method       | half-width at a test point            | guarantees |
|--------------|----------------------------------------|-----------|
| `global`     | one `Q` for all rows                   | marginal coverage only |
| `mondrian`   | `Q_k` for the point's regime `k`       | **marginal + conditional** |
| `normalized` | `Q·σ̂(regime)`, `σ̂` = per-regime MAD  | marginal, locally-adaptive width |

## 4. The mathematics (exactly as implemented)

**Nonconformity score** (operating hours only, so night zeros do not inflate
coverage): `s_i = |y_i − ŷ_i|`.

**Split-conformal quantile** (`conformal_quantile`). For a calibration set of
size `n` and target miscoverage `α`, take the `k`-th smallest score with
`k = ⌈(n+1)(1−α)⌉`; return `+∞` if `k > n` (calibration too small for the
level). This is the finite-sample-valid quantile of Lei et al. (2018).

**Interval.** `[ŷ − Q, ŷ + Q]`, clipped to the physical range `[0, rated]`
(clipping can only *raise* coverage, so the guarantee is preserved).

**Guarantee (marginal).** If the calibration and test scores are exchangeable,
`P(y_{test} ∈ interval) ≥ 1 − α`, with no assumption on the error distribution.

**Mondrian (conditional).** Partition by regime `k` and compute `Q_k` on that
regime's calibration scores; within-regime exchangeability then gives
`P(y ∈ interval | regime = k) ≥ 1 − α` for each `k` — the property a single
global `Q` does **not** provide under heteroscedastic PV errors.

**Metrics** (`coverage_report`): empirical **marginal coverage**; per-regime
**conditional coverage**; the **worst-regime coverage gap**
`max_k |cov_k − (1−α)|`; **MPIW** (mean interval width, kW) and **PINAW** (MPIW ÷
observed power range); and the mean **Winkler / interval score**
`W = (u−l) + (2/α)(l−y)·1[y<l] + (2/α)(y−u)·1[y>u]` — a proper scoring rule that
cannot be gamed by trivially wide or narrow bands.

## 5. What the real-data validation actually showed (do not overclaim)

`run_crisp_validation.py` runs CRISP over the **real** DKASC Alice Springs 2018
CSV, using the project's physics as the point predictor (per-hour centre
calibration first, leakage-safe), with a chronological calib→test split.

**Full-period pooled, whole-site Master Meter 1 (~185 kW; 2083 calib → 2084
test), target 90%:**

- **Marginal coverage holds:** Mondrian **93.3%**, Global 91.7% (both ≥ 90%).
- **Excellent reliability curve** (nominal → empirical): 0.50→0.509, 0.80→0.775,
  0.90→0.933, 0.95→0.964 — essentially on the diagonal.
- **The conditional story (the headline).** A single global width **under-covers
  the high-irradiance regime** (POA 964–1200 W/m²) at **86.3%** while *over*-
  covering the low regime (28–249 W/m²) at 97.5%. Mondrian evens every regime to
  **89.8–93.6%**. So the global band looks "90% correct" on average yet misses
  ~14% of peak-power hours; the regime-indexed band does not.
- **The cost.** Mondrian's mean width is larger (MPIW 42.7 vs 33.9 kW) and its
  Winkler slightly worse (50.5 vs 40.8): conditional validity is bought with
  some marginal efficiency. Report the trade-off, don't hide it.

**Full-period pooled, single Array M10-BC (~16 kW; 1978 calib → 1978 test):**
marginal coverage **89.4%** (Mondrian) / 89.7% (Global) against the 90% target —
near-perfect calibration, both methods uniform across regimes (a single fixed
array is closer to homoscedastic than the mixed-orientation whole-site meter).

**Per-month windows (the honest limitation).** Splitting *within* a single month
gives small (~150-hour) calibration blocks laced with seasonal drift, which
breaks exchangeability. Mean marginal coverage drops to **~81%** (well-behaved
months Jul/Oct/Dec land 92–96%; drift-heavy Jan/Sep/Nov fall to 54–67%). This is
a property of applying conformal prediction to **short, non-stationary** windows,
not a defect of the method — and it motivates using a large, seasonally-
representative calibration set (or adaptive/online conformal) rather than a
one-month slice.

### Defensible claims for the paper / viva
- "We add distribution-free, finite-sample-valid prediction intervals to the
  physics PV predictor and demonstrate near-nominal marginal coverage on real
  metered DKASC generation."
- "A regime-indexed (Mondrian) conformal calibration restores *conditional*
  coverage: on real data it lifts the under-covered high-irradiance regime from
  86% back to ≈90%, cutting the worst-regime coverage gap left by a single
  global interval."
- "The reliability curve sits on the diagonal across the 50–95% levels."
- "The interval is model-agnostic and inherits the interpretability of whatever
  predictor it wraps (physics, PRAM-corrected, or ML)."

### Claims to avoid
- Any statement that the conformal method itself is algorithmically novel.
- "Intervals are valid on any window" — the monthly harness shows short,
  drifting windows under-cover; state the exchangeability requirement.
- "CRISP improves point-forecast accuracy" — it quantifies uncertainty around a
  given predictor; it does not change the point prediction.

## 6. How to reproduce

```bash
# with the real DKASC CSV present (auto-located, or pass --csv):
python run_crisp_validation.py --csv datadkasc_alice_springs.csv \
       --channel-index 0 --alpha 0.10 --n-bins 5 --monthly
```

Prints the full-period pooled coverage (Mondrian vs Global), the per-regime
conditional-coverage table, the reliability curve across 50/80/90/95%, and the
per-month windows with their honest coverage shortfall under drift.

The interactive layer appears under the **🎯 CRISP Intervals** tab in `app.py`:
a channel/confidence/regime selector, a fan chart of the calibrated band against
the real measured power over the held-out test period, the per-regime
Mondrian-vs-Global conditional-coverage table, and the reliability-curve
expander.

> Note: the DKASC CSV is large and is **excluded** from the distributed zip by
> design (as with PRAM/FACL). Drop your downloaded copy into the project root or
> a `data/` folder; both the app and the harness auto-locate it.

## 7. Where CRISP sits in the contribution stack

```
                         ┌─ FACL  → attribution-confidence index (how much we trust the "why")
physics → PRAM residual ─┤
                         └─ CRISP → prediction interval          (how uncertain the "how much")
```

PRAM explains the physics gap; FACL grades the trustworthiness of that
explanation; CRISP puts a guaranteed-coverage band around the prediction itself.
Three small, honest, individually-defensible applied contributions on one
transparent physical base.
