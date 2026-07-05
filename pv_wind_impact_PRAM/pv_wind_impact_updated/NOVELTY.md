# NOVELTY.md

## Physics-Residual-Gated Fuzzy Attribution Confidence for Photovoltaic Power: A Formally Characterised Soft-Computing Layer Validated on Metered Generation

**Target: Q2-level journal publication**
(e.g. *Applied Soft Computing* (Q1/Q2), *Renewable Energy* (Q1), *Energy Reports* (Q2), *Sustainable Energy Technologies and Assessments* (Q2), *Engineering Applications of Artificial Intelligence* (Q1/Q2), *Solar Energy* (Q1/Q2), *IEEE Access* (Q1/Q2))

---

## 1. Evolution of the Work (why the title changed)

| Stage | Title | Contribution level | Problem |
|---|---|---|---|
| Title 1 | *Impact Analysis of Atmospheric Parameters on PV & Wind Power Output Using Data-Driven Modelling and ML* | Descriptive impact study | 12 ML models + SHAP ranking of weather features is **saturated** — hundreds of such papers exist; reviewers reject as incremental |
| Title 2 | *Explainable Atmospheric-Driver Attribution for PV and Wind Power: A Physics-Based Surrogate Framework Validated Against Metered Generation* | Applied framework (PRAM) | Physics-residual learning + SHAP attribution is stronger, but hybrid physics+ML residual correction is now common; the attribution itself is still a standard XAI ranking |
| **Title 3 (this paper)** | *Physics-Residual-Gated Fuzzy Attribution Confidence for PV Power: A Formally Characterised Soft-Computing Layer Validated on Metered Generation* | **Methodological + applied** | Moves the contribution one level up: not *what drives the power*, but *how much can the attribution itself be trusted* — a question nobody in the PV-XAI literature currently answers |

---

## 2. Research Gaps Identified from the Literature (2019–2026)

### Gap G1 — XAI for PV is saturated at the feature level; **no meta-level attribution confidence exists**
SHAP/permutation-importance studies for PV forecasting (2020–2026) all answer *"which weather variable matters most?"*. Recent work (2024–2025) extends this to seasonal variation of feature influence and SHAP-guided failure-mode diagnosis (e.g. cloud-enhancement events). **None of them quantify the trustworthiness of the attribution itself.** An importance ranking computed on a leaky, mis-scaled, or noise-dominated residual looks identical to a genuine one. The literature has no mechanism that says *"this SHAP ranking is 78/100 trustworthy"* — and no mechanism that *suppresses* confidence when the diagnostics are suspicious.

### Gap G2 — Fuzzy logic in PV research fuzzifies **raw weather**, never **attribution diagnostics**
The fuzzy/ANFIS PV literature (including 2025 ANFIS–LSTM and PSO-ANFIS hybrids) universally places fuzzy membership functions over irradiance, temperature, humidity and cloud cover to predict power. That input space is exhausted. **No published work places a fuzzy inference system over the meta-diagnostic space** — residual explainability (R²), residual-correction gain (ΔRMSE %), physical sign-consistency of a driver, and bias reduction — quantities that only exist *after* a physics-residual attribution has been performed.

### Gap G3 — Hybrid physics+ML residual models exist, but **without epistemic safeguards**
Physics-informed and residual-learning hybrids (2023–2026) report that "physics + learned residual beats physics alone", but they treat a high residual-model R² as unambiguously good. In reality a *suspiciously* high residual R² (> ~0.85) is the classic fingerprint of **data leakage or a mis-scaled baseline**. No published PV framework encodes this anti-leakage knowledge as a formal, graded rule that actively **reduces** reported confidence.

### Gap G4 — Attribution claims are almost never checked for **physical sign-consistency**
When aerosol/soiling variables rank highly in a PV importance list, papers report the rank and stop. Physically, an aerosol driver is only credible if its effect is **negative at high load** (physics over-predicts in dusty air because the omitted soiling loss). A high-ranked aerosol feature with the *wrong sign* is an artefact — yet no existing attribution pipeline penalises it.

### Gap H5 — Crisp thresholds create **cliff effects** in diagnostic classification
Every published diagnostic tiering ("strong / moderate / weak evidence") uses hard cut-offs: R² = 0.399 → "moderate", R² = 0.401 → "strong". This is scientifically indefensible at the boundary and unstable under resampling. Graceful (fuzzy) degradation of evidence tiers has not been applied to attribution quality in the energy domain.

### Gap G6 — Applied fuzzy layers in energy are **never formally characterised**
Soft-computing papers in energy typically present a rule base and empirical results only. Reviewers of Q1/Q2 soft-computing venues increasingly demand **provable properties**: boundedness, continuity, monotonicity, and consistency with the crisp system being generalised. No PV-domain fuzzy layer has been published with such a formal characterisation.

### Gap G7 — Validation on **real metered generation** is rare; conditional coverage is rarer
Most PV-XAI and fuzzy-PV papers validate on simulated or reanalysis-driven power. Conformal prediction has reached PV point-forecasting (2024–2026), but **regime-conditional (Mondrian) coverage on real metered data, coupled to the attribution layer**, has not been reported.

### Gap G8 — Attribution confidence is never made **operational**
Even where XAI quality metrics exist (rank-stability scores, faithfulness metrics), they are reported and then ignored: **no published PV framework feeds a trust-in-attribution signal back into the prediction pipeline as a control variable.** The ML correction is always applied fully or not at all (a crisp switch); no framework applies *as much correction as the attribution evidence warrants*, and none does so while preserving a finite-sample uncertainty guarantee.

### Gap G9 — Attribution is always **global**, never regime-conditional
Published PV importance rankings are averaged over the whole evaluation period, silently mixing physically distinct operating regimes: clear-sky hours are thermal-derate dominated, overcast hours diffuse-fraction dominated, transition hours mixed. Recent work (2024–2025) examines *seasonal* variation of SHAP values, but **no framework conditions the attribution on the same operating-regime taxonomy used for conditional uncertainty**, quantifies the cross-regime instability of the ranking in a bounded index, or lets that instability moderate the confidence placed in a "global" attribution.

---

## 3. The Novelty (what this paper contributes)

> **Core novelty statement (one sentence):**
> *We introduce FACL — the first fuzzy inference layer whose input space is the residual-attribution diagnostic space of a physics-residual PV model (not raw weather), with two physics-motivated epistemic gates (leakage suppression and aerosol sign-consistency) encoded as graded logic, formally characterised by provable mathematical properties, and validated for both attribution fidelity and regime-conditional uncertainty coverage on real metered PV generation (DKASC).*

### Contribution C1 — A new input space for fuzzy inference (addresses G1, G2)
FACL fuzzifies four **meta-diagnostics** produced by the PRAM physics-residual stage:

- `improvement` — % RMSE reduction of (physics + learned residual) over physics alone on a chronological held-out test block
- `residual_r2` — out-of-sample explainability of the residual
- `aerosol` — a signed sign-consistency evidence score in [−1, 1]
- `bias` — |MBE_physics| − |MBE_corrected| (bias moved toward zero?)

These quantities **do not exist** without the physics-residual stage — this is what makes the fuzzy layer non-generic and distinguishes it from every ANFIS-on-weather paper. Output: a smooth 0–100 **Attribution-Confidence Index (ACI)** with a linguistic verdict and a fully auditable rule-firing trace.

### Contribution C2 — Physics-epistemic gates as graded logic (addresses G3, G4)
Two domain constraints are encoded **inside** the inference, not as post-hoc filters:

- **Gate 1 (leakage suppression):** membership of `residual_r2` in the "suspicious" region (≳ 0.85) convexly pulls the ACI toward a low anchor, *regardless* of apparent RMSE gain. A "too-perfect" residual lowers trust instead of raising it — the first graded encoding of anti-leakage reasoning in a PV attribution pipeline.
- **Gate 2 (aerosol sign-consistency):** an aerosol driver raises confidence **only** when its SHAP effect is physically negative at high aerosol load (the omitted-soiling signature); a wrong-signed aerosol "signature" actively lowers confidence.

### Contribution C3 — Formal characterisation (addresses G5, G6) — **this is what lifts the paper to Q2**
The paper states and proves the following properties of the FACL operator
$$\text{ACI} = F(\text{improvement}, R^2_{res}, s_{aero}, \Delta_{bias}) : \mathbb{R}^4 \to [0,100]$$

- **P1 (Boundedness):** $$0 \le F(\mathbf{x}) \le 100$$ for all inputs (centroid of a non-negative aggregate over [0,100]).
- **P2 (Continuity):** F is continuous in each input (piecewise-linear memberships, min/max operators and centroid defuzzification preserve continuity) — hence **no cliff effects**, unlike the crisp tiering it replaces.
- **P3 (Crisp-consistency):** in the saturation regions of the membership functions, F reproduces the legacy crisp tiers (weak/moderate/strong); FACL is a **strict generalisation**, demonstrated by the ACI surface plot with the crisp decision boundaries overlaid.
- **P4 (Gate-1 dominance):** for full leakage suspicion ($$\mu_{susp}=1$$), $$F \equiv \text{ACI}_{anchor} = 28$$ independent of all other inputs — the guard provably overrides max-aggregation (proved directly from the convex-combination form $$\text{ACI} = \text{ACI}_{raw} + \mu_{susp}(A - \text{ACI}_{raw})$$).
- **P5 (Sectionwise monotonicity):** outside the suspicious region, F is non-decreasing in `improvement` and in `residual_r2` (verified analytically per rule-activation cell and numerically on a dense grid) — more genuine evidence never reduces confidence.
- **P6 (Threshold-perturbation robustness):** a sensitivity analysis showing $$|\partial \text{ACI} / \partial \theta|$$ is bounded for every membership breakpoint θ, versus the unbounded (discontinuous) sensitivity of the crisp classifier.

*(P1–P4 are already true of the implemented system in `src/facl.py`; P5–P6 require only a numerical verification script — see §5.)*

### Contribution C4 — Confidence-coupled conditional uncertainty (addresses G7)
The CRISP layer (Mondrian split-conformal intervals, regimes indexed by the very irradiance driver the attribution ranks) is reported **jointly** with FACL: the paper shows that periods/regimes with high ACI coincide with regimes where the physics-residual predictor achieves the tightest calibrated intervals, and that Mondrian binning shrinks the **worst-regime coverage gap** versus a global conformal quantile on real metered data. This closes the loop: *attribution → confidence in attribution → guaranteed uncertainty on the corrected prediction.*

### Contribution C6 — **ACGC: the confidence signal becomes a control signal** (addresses G8) — *the headline architectural novelty*
Implemented in `src/acgc.py`. The ACI is no longer descriptive metadata — it operationally decides **how much of the learned residual correction is applied**:

$$\hat{y}_w = P_{phys} + w \cdot \hat{r}, \qquad w = g(\text{ACI}) \in [0,1]$$

where $$g$$ is a C¹ smootherstep with an explicit Lipschitz constant (P8), $$g \equiv 0$$ for ACI ≤ 25 (physics fallback, P9) and $$g \equiv 1$$ for ACI ≥ 80 (full trust, P10). The gated predictor is wrapped in Mondrian split-conformal intervals using a **three-block chronological split (gate → calibration → test)**: the gate weight is a measurable function of the gate block only, so the finite-sample coverage guarantee provably survives the gating (P7 — *coverage preservation*, the paper's one theorem-level result). Evaluated against three baselines on the same held-out block: physics-only (w=0), always-correct (w=1), and the crisp hard-switch. **No published PV framework couples a fuzzy attribution-trust signal to a conformal prediction wrapper as a convex correction gain.**

### Contribution C7 — **RCA: regime-conditional attribution with a bounded instability index** (addresses G9)
Implemented in `src/rca.py`. Per-regime permutation importance is computed on the held-out test rows **inside the exact Mondrian quantile bins CRISP uses for conditional coverage** — attribution and uncertainty share one physical taxonomy. Cross-regime disagreement is compressed into the **Attribution Instability Index**:

$$\text{AII} = \frac{1 - \bar{\tau}_w}{2} \in [0, 1], \qquad \bar{\tau}_w = \text{mean pairwise top-weighted Kendall } \tau \text{ between regime importance vectors}$$

AII = 0 → the ranking is regime-invariant; AII → 1 → systematically reversed. A smootherstep maps AII to a confidence retention factor $$s(\text{AII}) \in [0.40, 1]$$ applied multiplicatively to the ACI ($$\text{ACI}_{ra} = s \cdot \text{ACI}$$) — a regime-unstable global attribution *continuously loses confidence*, the same convex-guard idiom as Gate 1. Three verified formal properties: **P11** (AII bounded with correct ordering on identical/uncorrelated/reversed rankings; the composed ACI stays in [0,100]), **P12** (s monotone non-increasing with an exact advertised Lipschitz modulus), **P13** (AII invariant to feature relabelling). The synthetic harness reproduces the physics — with a temperature-dependent high-irradiance derate in the truth, the recovered top driver flips from GHI to temperature in the top regime. **No published PV framework conditions attribution on the conformal regime taxonomy or feeds attribution instability back into a confidence layer.**

### Contribution C5 — Honest dual-mode validation on metered generation
- **Measured mode:** residual = real DKASC metered power − physics model on on-site weather (genuine ground truth), chronological train/test split, train-only baseline calibration (no leakage).
- **Cross-source mode:** NASA-POWER-driven vs Open-Meteo-driven physics (model-consistency residual) — used transparently as a *secondary* robustness check, never conflated with measured validation.
- The paper explicitly scopes each claim to its evidence mode — the honesty framing itself is a differentiator against overclaiming XAI papers.

---

## 4. Novelty Positioning Matrix (for the paper's Introduction)

| Capability | ANFIS / fuzzy-PV (2020–25) | SHAP-PV XAI (2020–26) | Physics+ML residual hybrids (2023–26) | Conformal-PV (2024–26) | **This work** |
|---|---|---|---|---|---|
| Fuzzifies raw weather | ✅ | — | — | — | — (deliberately not) |
| Ranks atmospheric drivers | — | ✅ | partial | — | ✅ |
| Quantifies confidence *in the attribution* | ❌ | ❌ | ❌ | ❌ | ✅ **ACI (0–100)** |
| Anti-leakage encoded in inference | ❌ | ❌ | ❌ | ❌ | ✅ **Gate 1** |
| Physical sign-consistency check | ❌ | ❌ | ❌ | ❌ | ✅ **Gate 2** |
| Formal properties proved | ❌ | ❌ | ❌ | partial (coverage) | ✅ **P1–P13, runnable suite** |
| Confidence-gated correction w/ coverage guarantee | ❌ | ❌ | ❌ | ❌ | ✅ **ACGC** |
| Regime-conditional attribution + instability index | ❌ | seasonal only | ❌ | ❌ | ✅ **RCA / AII** |
| Metered-generation validation | rare | rare | some | some | ✅ DKASC |
| Regime-conditional coverage | ❌ | ❌ | ❌ | partial | ✅ Mondrian CRISP |
| Auditable rule trace | partial | ❌ | ❌ | ❌ | ✅ |

---

## 5. What to Add to Reach Q2 (concrete action list)

The codebase already implements C1, C2, C4, C5, C6, C7. To make the paper referee-proof:

1. **Formal-properties script — DONE** (`validate_formal_properties.py`): numerically verifies all thirteen properties P1–P13 (15/15 checks pass). Report as a properties table. **Bonus result for the paper:** the suite *caught two real implementation defects* — (i) an empty-rule-base region above R² ≈ 0.84 where the ACI cliff-dropped to 0, fixed with a graded "suspicious → low" support rule; (ii) a shoulder-trapezoid membership function that returned 0 instead of 1 at the exact universe boundary. Reporting this ("the formal characterisation found and fixed defects a purely empirical evaluation would have missed") is itself a strong argument for Gap G6.
2. **Ablation study:** ACI with (a) no gates, (b) Gate 1 only, (c) Gate 2 only, (d) full FACL — on the same DKASC runs. Show gates change verdicts in exactly the physically-correct cases.
3. **Baseline comparisons:** (i) legacy crisp classifier, (ii) a logistic-regression "confidence" trained on the same four diagnostics (shows fuzzy transparency ≠ accuracy loss), (iii) raw SHAP rank stability under bootstrap (shows why a confidence layer is needed at all).
4. **Bootstrap robustness:** resample the test block (block bootstrap, preserves autocorrelation) → distribution of ACI; show the crisp tier flips across bootstrap replicates while the ACI varies smoothly. *This is the killer figure for G5.*
5. **Multiple DKASC systems / periods:** run ≥ 3 different array technologies (poly-Si, mono-Si, CdTe) and ≥ 2 seasons to demonstrate generality.
6. **Statistical significance:** Diebold–Mariano test on physics vs physics+residual test errors; report p-values.
7. **Scope discipline:** frame the paper as **PV-only** (as the title already does). Keep wind as "framework extensibility" in one paragraph — trying to validate both weakens the metered-validation story since DKASC is PV.
8. **Run RCA and ACGC on the DKASC metered data — HARNESS DONE, awaiting real CSV** (`run_dkasc_experiments.py`): a headless runner that executes the full PRAM → FACL → ACGC → RCA stack on a downloaded DKASC export and writes the paper's headline tables to `results/dkasc_results.md` (plus `rca_regime_importance.csv` and `acgc_intervals.csv`). Verified end-to-end on a DKASC-format smoke CSV; the only remaining step is downloading a real Alice Springs export (free, https://dkasolarcentre.com.au/download) into `data/` and running `python run_dkasc_experiments.py` — then repeating with `--channel` for ≥ 3 array technologies (item 5).

---

## 6. Claims to Avoid (reviewer traps)

- ❌ "A new fuzzy algorithm" — the Mamdani engine is textbook; the novelty is the **input space, the gates, and the formal characterisation**. Say so explicitly (the code's docstrings already do — keep that honesty in the paper).
- ❌ "Better forecasting accuracy than deep learning" — this is not a forecasting paper; it is an **attribution-trust** paper.
- ❌ Conflating cross-source residuals with measured validation — always label the mode.
- ❌ Claiming SHAP itself is novel — it is the *consumer* of SHAP outputs that is new.

---

## 7. One-Paragraph Abstract Skeleton

> Explainable-AI rankings of atmospheric drivers for photovoltaic power are now routine, yet no existing method quantifies how much an attribution itself can be trusted: a driver ranking computed on a leaky or noise-dominated residual is indistinguishable from a genuine one. We propose FACL, a physics-residual-gated fuzzy attribution-confidence layer. A calibrated physics model of a real PV plant is subtracted from metered generation; the residual is learned by a gradient-boosted model and attributed to atmospheric drivers via SHAP. Four resulting meta-diagnostics — residual explainability, residual-correction gain, aerosol sign-consistency, and bias reduction — form the input space of a transparent Mamdani system that outputs a 0–100 Attribution-Confidence Index (ACI) with an auditable rule trace. Two physics-epistemic gates are embedded in the inference: a leakage-suppression gate that provably pulls confidence to a low anchor when the residual is suspiciously explainable, and a sign-consistency gate that rewards aerosol drivers only when their effect matches soiling physics. We formally characterise the layer (boundedness, continuity, crisp-consistency, gate dominance, sectionwise monotonicity, bounded threshold sensitivity), and validate on metered generation from the Desert Knowledge Australia Solar Centre with chronological leakage-safe splits, block-bootstrap robustness, and Mondrian split-conformal intervals whose worst-regime coverage gap shrinks versus a global calibrator. FACL strictly generalises crisp evidence tiers, eliminating their cliff effects while adding epistemic safeguards absent from the PV-XAI literature.

---

## 8. Suggested Paper Structure

1. Introduction (gaps G1–G7 → contributions C1–C5)
2. Related work (fuzzy-PV; PV-XAI; physics-residual hybrids; conformal-PV) — use the positioning matrix
3. Physics-residual attribution stage (PRAM): plant physics, baseline calibration, residual learner, SHAP
4. FACL: membership functions, rule base, gates, defuzzification — **with the formal properties P1–P6 and proofs**
5. CRISP: regime-indexed conformal intervals and conditional-coverage guarantees
6. Data & experimental design (DKASC, chronological splits, dual-mode honesty)
7. Results: attribution results → ACI results → ablations → bootstrap robustness → conditional coverage → baselines
8. Discussion, limitations (single site family, hourly resolution, calibration transferability), conclusion
