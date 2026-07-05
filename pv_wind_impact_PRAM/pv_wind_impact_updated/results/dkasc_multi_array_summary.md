# DKASC Multi-Array Validation Summary

Full stack (PRAM → FACL → ACGC → RCA) executed independently on **four
Alice Springs arrays**, each with 25,861 hourly rows spanning
2013-04-23 → 2016-10-21 (3.5 years, all seasons). Source: DKASC raw
exports via public GitHub mirror (`fetch_dkasc_mirror.py`); per-array
detail tables in `dkasc_results_<ARRAY>.md`.

> Array technology labels: 1A is documented as Trina mono-Si on a
> dual-axis tracker. Confirm 1C / 3A / 4A module technologies against the
> official DKASC site list (dkasolarcentre.com.au) before publication —
> the differing rated powers (1.6 / 1.6 / 4.7 / 5.1 kW) confirm they are
> physically distinct installations.

## Table 1 — Physics surrogate & PRAM diagnostics

| Array | Rated (kW) | RMSE phys (kW) | Corrected Δ | Residual R² | Tier | Global ACI | Local ACI | Effective ACI (min) | w |
|-------|-----------|----------------|-------------|-------------|------|-----------|-----------|--------------------:|---|
| 1A | 1.6 | 0.237 | −2.9% | −0.119 | weak | 9.2 | 55.0 | 9.2 (very_low) | 0.000 |
| 1C | 1.6 | 0.196 | −2.0% | −0.085 | weak | 9.2 | 92.6 | 9.2 (very_low) | 0.000 |
| 3A | 4.7 | 0.686 | +0.8% | −0.070 | weak | 9.2 | 76.0 | 9.2 (very_low) | 0.000 |
| 4A | 5.1 | 0.678 | +1.3% | −0.043 | weak | 9.4 | 92.6 | 9.4 (very_low) | 0.000 |

**Key finding (Gate 2 justification):** on *every* array the local
(gate-block) ACI is high (55–93) while the global ACI is very low (~9) —
the residual model consistently looks skilful immediately after the
training cut and decays over the multi-year horizon. Without the
min t-norm temporal-consistency guard, ACGC would have applied a
correction that degrades RMSE on 2 of 4 arrays. The guard correctly
forces the physics fallback (w = 0) on all four.

## Table 2 — Conformal intervals (rolling Mondrian, α = 0.10)

| Array | Variant | Coverage | MPIW (kW) | Winkler | Worst-regime gap | Fixed-split cov |
|-------|---------|---------:|----------:|--------:|-----------------:|----------------:|
| 1A | acgc/physics/hard | 90.7% | 0.34 | 0.42 | 10.0p | 90.6% |
| 1A | full (w=1) | 90.9% | 0.25 | 0.29 | 10.0p | 89.2% |
| 1C | acgc/physics/hard | 85.6% | 0.26 | 0.37 | 12.5p | 86.1% |
| 1C | full (w=1) | 88.5% | 0.17 | 0.27 | 10.0p | 87.5% |
| 3A | acgc/physics/hard | 96.1% | 0.90 | 0.95 | 10.0p | 96.8% |
| 3A | full (w=1) | 92.8% | 0.58 | 0.73 | 10.0p | 89.0% |
| 4A | acgc/physics/hard | 92.5% | 0.88 | 1.04 | 10.0p | 94.4% |
| 4A | full (w=1) | 85.1% | 0.43 | 0.68 | 13.0p | 81.0% |

**Reading:** ACGC (= physics fallback here, since w = 0 everywhere)
achieves nominal-or-better coverage on 3 of 4 arrays; the always-correct
baseline (w = 1) undercovers on 4A (85.1%, fixed-split 81.0%) — exactly
the failure mode the evidence-gated w is designed to avoid. On 1C all
variants undercover slightly (85.6–88.5%), worth a note in the paper
(higher regime drift on that array).

## Table 3 — RCA: regime-conditional attribution

| Array | AII | s(AII) | ACI → regime-aware | Top driver by regime (low → high irradiance) |
|-------|----:|-------:|-------------------:|----------------------------------------------|
| 1A | 0.327 | 0.962 | 9.2 → 8.9 | cos_zenith_proxy → hour_cos → doy_cos → doy_cos |
| 1C | 0.307 | 0.979 | 9.2 → 9.0 | ghi_lag1 → hour → hour → clear_sky_index |
| 3A | 0.279 | 0.994 | 9.2 → 9.2 | doy_sin → clear_sky_index → clear_sky_index → doy_cos |
| 4A | 0.346 | 0.942 | 9.4 → 8.9 | rh_roll24 → clear_sky_index → clear_sky_index → doy_sin |

**Reading:** every array exhibits a genuine regime flip (AII 0.28–0.35 —
moderate instability, consistent across technologies), confirming that a
single global attribution table hides real regime structure. The
mid-irradiance regimes are clear-sky-index dominated while the extremes
are driven by geometry/seasonal terms — physically sensible and
consistent across all four installations.

## Reproduction

```
python fetch_dkasc_mirror.py --array 1C
python run_dkasc_experiments.py --csv data/dkasc_1C.csv --tag 1C
```
