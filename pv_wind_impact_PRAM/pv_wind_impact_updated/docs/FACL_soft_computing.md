# FACL — Soft-Computing Integration for PRAM

**Physics-Residual-Gated Fuzzy Attribution-Confidence Layer**
Module: `src/facl.py` · Harness: `run_facl_validation.py` · UI: PRAM tabs in `app.py`

---

## 1. What was added, in one sentence

A small Mamdani fuzzy inference system that consumes **PRAM's residual
diagnostics** (not raw weather) and produces a smooth 0–100 **Attribution-
Confidence Index (ACI)** with a linguistic verdict, a human-readable rule-firing
trace, and two **physics gates** encoded as fuzzy logic.

## 2. Honest novelty classification — read this before writing the paper

This is an **applied / framework-integration** contribution, in the *same tier*
as PRAM's own novelty. State it that way. It is deliberately **NOT**:

- a new fuzzy algorithm (the engine is textbook: triangular/trapezoidal
  memberships, min-inference, max-aggregation, centroid defuzzification);
- "fuzzy logic applied to PV/wind" as a novel claim — that space is saturated
  (fuzzy/ANFIS for forecasting, MPPT, fault detection) and any such claim will
  be rejected on sight.

What makes the *integration* non-generic and reportable:

1. **The input space is the physics-residual explainability, not meteorology.**
   FACL fuzzifies `residual_R²`, `RMSE_improvement`, `aerosol_sign_consistency`
   and `bias_reduction` — quantities that exist only because a physics-residual
   attribution model produced them first. This is genuinely downstream of PRAM,
   not the usual "fuzzify GHI/temperature/wind."

2. **Two domain constraints are encoded as graded logic ("gates"):**
   - **Gate 1 — leakage suppression.** A residual R² in the suspicious region
     (> ~0.85) pulls confidence *down* toward a low anchor, whatever the other
     signals say. A suspiciously perfect residual is the leakage / mis-scaled-
     baseline fingerprint, so a large apparent "improvement" is a reason for
     *less* trust, not more. Implemented as an authoritative convex guard
     (`LEAK_GUARD_ANCHOR`), because a single competing Mamdani rule cannot
     override max-aggregation.
   - **Gate 2 — aerosol sign-consistency.** An aerosol driver raises confidence
     only when its effect is physically *negative* at high AOD (physics over-
     predicts in dusty air → the omitted soiling loss). A positive-sign aerosol
     "signature" is physically implausible and instead lowers confidence.

3. **Strict generalisation of the existing crisp tiers.** FACL's membership
   breakpoints are tied to PRAM's current thresholds (improvement 3 %/8 %,
   residual R² 0.18–0.20 / 0.40, leakage 0.85). At the extremes it reproduces
   weak/moderate/strong; near the old hard cut-offs it degrades *smoothly*
   instead of flipping. `facl.aci_surface()` + the app's expander plot show the
   crisp step at R²=0.40 next to the continuous ACI surface.

## 3. Architecture

```
PRAM result dict
   │  facl.facl_inputs()
   ▼
four crisp inputs ──► fuzzify (linguistic terms) ──► Mamdani rule base (15 rules)
   │                                                        │ max-aggregation
   │                                                        ▼
   │                                              centroid defuzzification → ACI_raw
   │  Gate 1 (leakage guard, driven by μ_suspicious)        │
   └────────────────────────────────────────────────────────► ACI  → verdict + trace
```

Inputs → fuzzy variable terms:

| PRAM diagnostic (crisp)            | Fuzzy terms                                   |
|------------------------------------|-----------------------------------------------|
| RMSE improvement %                 | low / medium / high                           |
| residual R²                        | low / medium / high / **suspicious**          |
| aerosol sign-consistency [-1,1]    | contradictory / neutral / supportive          |
| bias reduction (kW)                | worsened / neutral / improved                 |
| **output:** confidence (0–100)     | very_low / low / moderate / high / very_high  |

`aerosol_consistency_score()` maps PRAM's `aerosol_signature()` dict to the
signed [-1,1] input (+1 = top-ranked & physically negative; −ve = wrong sign).

## 4. What the real-data validation actually showed (do not overclaim)

`run_facl_validation.py` runs PRAM + FACL over successive monthly windows of the
**real** DKASC Alice Springs 2018 CSV and prints the relationships. Findings on
a single-array channel (Array M10, 10 monthly windows):

- **Gate 1 fired on real data.** October 2018: residual R² = 0.883 (suspicious),
  apparent improvement +72 %, **crisp tier = "strong"** — FACL's leakage guard
  triggered (suspicion 0.83) and suppressed ACI to ~39 ("low"). This is the
  headline demonstration: FACL is correctly more conservative than the crisp
  logic exactly where the crisp logic is most likely to be fooled.
- **FACL demotes unexplained "gains."** Windows with a positive RMSE improvement
  but a non-explainable residual (July/Aug 2018: residual R² ≈ 0.1 / −0.03,
  crisp = "moderate") are demoted to "low" — the gain came from baseline
  calibration, not from an explainable atmospheric residual.
- **No validated accuracy-monotonicity.** Spearman between ACI and the
  *independent* test-period correlation `r` (which is **not** a FACL input) was
  **slightly negative (≈ −0.2)** on this channel. Cause: a few windows have good
  shape-correlation but a blown-up scale/RMSE (or vice-versa), and `r` ignores
  scale while RMSE does not. **Conclusion: FACL is a smoother, more conservative,
  leakage-aware, interpretable verdict layer — NOT a proven predictor of model
  accuracy.** Claiming the latter is contradicted by this harness's own output.

### Defensible claims for the paper / viva
- "We replace PRAM's crisp verdict thresholds with a physics-residual-gated
  fuzzy inference layer that yields graded, uncertainty-aware, interpretable
  attribution verdicts."
- "Two physics constraints (leakage suppression, aerosol sign-consistency) are
  encoded as fuzzy gates; on measured DKASC data the leakage gate correctly
  demoted an over-fit window that crisp thresholds rated 'strong'."
- "The layer is a strict generalisation of the crisp tiers (identical at the
  extremes, smooth across the old cut-offs)."

### Claims to avoid
- Any statement that fuzzy logic here is algorithmically novel.
- "FACL improves forecasting accuracy" / "ACI predicts model accuracy" — not
  supported by the validation.

## 5. How to reproduce

```bash
# with the real DKASC CSV present (auto-located, or pass --csv):
python run_facl_validation.py --csv datadkasc_alice_springs.csv \
       --channel-index 2 --max-windows 10 --min-hours 250
```

The interactive layer appears automatically under **🌫️ Fuzzy attribution
confidence** on both the 🇦🇺 measured-DKASC tab and the live-API PRAM tab in
`app.py`, including the fuzzified-diagnostics table, the fired-rules trace, and
the no-cliff ACI-surface expander.

> Note: the DKASC CSV (~200 MB) is **excluded** from the distributed zip by
> design (as in the prior package). Drop your downloaded copy into the project
> root or a `data/` folder; both the app and the harness auto-locate it.
