"""
FACL — Physics-Residual-Gated Fuzzy Attribution-Confidence Layer
================================================================

What this is (and, honestly, what it is *not*)
----------------------------------------------
FACL replaces PRAM's crisp ``classify()`` thresholds with a small, fully
transparent **Mamdani fuzzy inference system** that turns PRAM's *residual
diagnostics* into a smooth 0–100 **Attribution-Confidence Index (ACI)**, a
linguistic verdict, and a human-readable rule-firing trace.

The contribution is an **applied / framework-integration** one, in the exact
same tier as PRAM's own novelty. It is deliberately NOT a new fuzzy algorithm:
the engine below is textbook (triangular/trapezoidal memberships, min-inference,
max-aggregation, centroid defuzzification). Two things make the *integration*
non-generic and worth reporting:

1.  **The input space is PRAM's residual-explainability, not raw weather.**
    Everyone fuzzifies GHI / temperature / wind for PV forecasting — that space
    is saturated. FACL instead fuzzifies ``residual_R2``, ``RMSE_improvement``,
    ``aerosol_sign_consistency`` and ``bias_reduction`` — quantities that only
    exist because a physics-residual attribution model produced them first.

2.  **Two physics constraints are encoded as fuzzy rules ("gates"):**
      * *Leakage-suppression gate* — a residual R² in the "suspicious" region
        (> ~0.85) actively pulls confidence DOWN (a suspiciously perfect
        residual usually means leakage or a mis-scaled baseline), instead of
        rewarding it. This encodes PRAM's anti-leakage heuristic as graded logic
        rather than a hard ``if``.
      * *Aerosol sign-consistency gate* — an aerosol driver only RAISES
        confidence when its effect is physically negative at high AOD (physics
        over-predicts in dusty air → the omitted soiling loss). A positive-sign
        aerosol "signature" is physically wrong and instead lowers confidence.

FACL is a strict *generalisation* of the existing crisp tiers: at the extremes
it reproduces weak/moderate/strong, but near the old hard cut-offs it degrades
gracefully instead of flipping. ``crisp_reference()`` is provided so the app /
paper can show the two side by side (and plot the smooth ACI surface with no
cliff at R²=0.40).

Pure Python + NumPy. No ``scikit-fuzzy`` dependency, so the whole rule base is
auditable in one file — which is the point for a defensible research artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ===========================================================================
# 1.  A minimal, transparent Mamdani fuzzy engine
# ===========================================================================
def tri(a: float, b: float, c: float) -> Callable[[float], float]:
    """Triangular membership with corners a<=b<=c (μ=1 at b)."""
    def mu(x: float) -> float:
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        return (x - a) / (b - a) if x < b else (c - x) / (c - b)
    return mu


def trap(a: float, b: float, c: float, d: float) -> Callable[[float], float]:
    """Trapezoidal membership with corners a<=b<=c<=d (μ=1 on [b,c])."""
    def mu(x: float) -> float:
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        return (x - a) / (b - a) if x < b else (d - x) / (d - c)
    return mu


@dataclass
class FuzzyVar:
    """A linguistic variable: named membership functions over a universe."""
    name: str
    terms: dict[str, Callable[[float], float]]

    def fuzzify(self, x: float) -> dict[str, float]:
        return {t: float(np.clip(mf(x), 0.0, 1.0)) for t, mf in self.terms.items()}


@dataclass
class Rule:
    """IF (var==term AND ...) THEN out==term, with an optional weight."""
    antecedent: list[tuple[str, str]]      # [(var_name, term), ...]  (AND-joined)
    out_term: str
    weight: float = 1.0
    label: str = ""                        # human-readable description

    def strength(self, memberships: dict[str, dict[str, float]]) -> float:
        # AND == min (Mamdani); scaled by the rule weight.
        vals = [memberships[v].get(t, 0.0) for (v, t) in self.antecedent]
        return self.weight * (min(vals) if vals else 0.0)


@dataclass
class MamdaniSystem:
    inputs: dict[str, FuzzyVar]
    output: FuzzyVar
    rules: list[Rule]
    universe: np.ndarray = field(default_factory=lambda: np.linspace(0, 100, 501))

    def infer(self, crisp: dict[str, float]) -> dict:
        """Run inference. Returns ACI, memberships, per-term aggregates, trace."""
        memberships = {name: var.fuzzify(crisp[name])
                       for name, var in self.inputs.items()}

        # Clip each output term's MF at the max firing strength of the rules
        # that conclude it (max-aggregation across the rule base).
        term_strength: dict[str, float] = {t: 0.0 for t in self.output.terms}
        trace: list[dict] = []
        for r in self.rules:
            s = r.strength(memberships)
            if s > term_strength[r.out_term]:
                term_strength[r.out_term] = s
            if s > 1e-6:
                trace.append({"rule": r.label or self._auto_label(r),
                              "then": r.out_term, "strength": round(s, 3)})

        # Aggregate the clipped output MFs (max) over the universe, defuzzify
        # by centroid.
        agg = np.zeros_like(self.universe)
        for t, mf in self.output.terms.items():
            clip = term_strength[t]
            if clip <= 0:
                continue
            curve = np.minimum(np.array([mf(u) for u in self.universe]), clip)
            agg = np.maximum(agg, curve)

        area = float(agg.sum())
        aci = float((self.universe * agg).sum() / area) if area > 1e-9 else 0.0

        trace.sort(key=lambda d: d["strength"], reverse=True)
        return {"aci": aci, "memberships": memberships,
                "term_strength": term_strength, "trace": trace,
                "universe": self.universe, "aggregate": agg}

    @staticmethod
    def _auto_label(r: Rule) -> str:
        ante = " AND ".join(f"{v} is {t}" for v, t in r.antecedent)
        return f"IF {ante} THEN confidence is {r.out_term}"


# ===========================================================================
# 2.  FACL specification
#     Breakpoints are tied to PRAM's existing crisp thresholds so FACL is a
#     smooth superset of the current logic:
#       improvement: moderate≈3 %, strong≈8 %   (see pram.classify)
#       residual R2: moderate≈0.18–0.20, strong≈0.40, suspicious/leak≈0.85
# ===========================================================================
def _build_system() -> MamdaniSystem:
    improvement = FuzzyVar("improvement", {
        # % RMSE reduction from adding the learned residual (can be negative)
        "low":    trap(-100.0, -100.0, 1.0, 5.0),
        "medium": tri(2.0, 5.5, 9.0),
        "high":   trap(6.0, 10.0, 100.0, 100.0),
    })

    residual_r2 = FuzzyVar("residual_r2", {
        "low":        trap(-1.0, -1.0, 0.10, 0.20),
        "medium":     tri(0.14, 0.30, 0.46),
        "high":       trap(0.34, 0.52, 0.80, 0.84),   # genuine explainability
        "suspicious": trap(0.80, 0.90, 1.0, 1.0),     # leakage region
    })

    # Aerosol sign-consistency, a signed evidence score in [-1, 1] built by
    # aerosol_consistency_score() below:  +1 = top-ranked & physically negative
    # (real soiling signature);  -1 = strong but wrong sign (physically implausible).
    aerosol = FuzzyVar("aerosol", {
        "contradictory": trap(-1.0, -1.0, -0.5, 0.0),
        "neutral":       tri(-0.3, 0.0, 0.3),
        "supportive":    trap(0.0, 0.5, 1.0, 1.0),
    })

    # Bias reduction: |MBE_physics| - |MBE_corrected|  (kW). Positive = better.
    bias = FuzzyVar("bias", {
        "worsened": trap(-1e6, -1e6, -0.02, 0.0),
        "neutral":  tri(-0.05, 0.0, 0.05),
        "improved": trap(0.0, 0.02, 1e6, 1e6),
    })

    confidence = FuzzyVar("confidence", {
        "very_low":  trap(0.0, 0.0, 10.0, 25.0),
        "low":       tri(15.0, 32.0, 50.0),
        "moderate":  tri(40.0, 55.0, 70.0),
        "high":      tri(60.0, 75.0, 90.0),
        "very_high": trap(80.0, 92.0, 100.0, 100.0),
    })

    R = Rule
    rules = [
        # --- core improvement × explainability grid ------------------------
        R([("improvement", "high"), ("residual_r2", "high")], "very_high", 1.0,
          "Large RMSE gain AND highly explainable residual → very high"),
        R([("improvement", "high"), ("residual_r2", "medium")], "high", 1.0,
          "Large RMSE gain AND moderately explainable → high"),
        R([("improvement", "medium"), ("residual_r2", "high")], "high", 1.0,
          "Moderate RMSE gain AND highly explainable → high"),
        R([("improvement", "medium"), ("residual_r2", "medium")], "moderate", 1.0,
          "Moderate gain AND moderate explainability → moderate"),
        R([("improvement", "high"), ("residual_r2", "low")], "moderate", 0.9,
          "RMSE improves but residual not explainable → only moderate"),
        R([("improvement", "low"), ("residual_r2", "high")], "moderate", 0.9,
          "Explainable residual but little RMSE gain → moderate"),
        R([("improvement", "medium"), ("residual_r2", "low")], "low", 1.0,
          "Modest gain, unexplainable residual → low"),
        R([("improvement", "low"), ("residual_r2", "medium")], "low", 1.0,
          "Explainable-ish but no real gain → low"),
        R([("improvement", "low"), ("residual_r2", "low")], "very_low", 1.0,
          "No gain AND no structure → very low (mostly noise)"),

        # --- GATE 1 (leakage) is applied as an explicit convex guard in
        #     run_facl(), driven by membership in residual_r2="suspicious".
        #     A single competing rule cannot override max-aggregation, so the
        #     guard is the authoritative suppressor (see LEAK_GUARD_ANCHOR).

        # --- GATE 2: aerosol sign-consistency ------------------------------
        R([("aerosol", "supportive"), ("residual_r2", "medium")], "high", 0.9,
          "GATE: physically-negative aerosol driver + explainable → boost to high"),
        R([("aerosol", "supportive"), ("improvement", "medium")], "high", 0.8,
          "GATE: real soiling signature reinforces a moderate gain → high"),
        R([("aerosol", "contradictory")], "low", 0.85,
          "GATE: aerosol driver has wrong (positive) sign → physically implausible → low"),

        # --- bias-reduction corroboration ----------------------------------
        R([("bias", "worsened"), ("improvement", "medium")], "low", 0.7,
          "Bias got worse despite RMSE gain → distrust → pull to low"),
        R([("bias", "improved"), ("improvement", "high")], "very_high", 0.7,
          "RMSE gain corroborated by bias moving toward zero → very high"),
    ]
    return MamdaniSystem(inputs={"improvement": improvement,
                                 "residual_r2": residual_r2,
                                 "aerosol": aerosol,
                                 "bias": bias},
                         output=confidence, rules=rules)


_SYSTEM = _build_system()


# ===========================================================================
# 3.  Bridging PRAM's diagnostics -> FACL crisp inputs
# ===========================================================================
def aerosol_consistency_score(aero: dict) -> float:
    """Map PRAM's aerosol_signature() dict to a signed evidence score in [-1,1].

    +ve  = an aerosol feature is highly ranked AND its effect is physically
           negative (physics over-predicts in dusty air = real soiling loss).
    -ve  = highly ranked but positive sign (physically implausible for soiling).
     0   = no aerosol channel / no signal / sign unknown & low rank.
    """
    if not aero or not aero.get("available"):
        return 0.0
    rank = aero.get("rank") or 99
    # rank weight: rank1 -> 1.0, rank2 -> ~0.7, rank3 -> ~0.5, then decays fast
    rank_w = float(np.clip(1.2 / rank, 0.0, 1.0))
    neg = aero.get("negative")
    if neg is True:
        return rank_w                      # supportive
    if neg is False:
        return -0.7 * rank_w               # contradictory (wrong sign)
    # sign unknown (no SHAP): mild positive if top-ranked, else ~neutral
    return 0.3 * rank_w


def facl_inputs(res: dict) -> dict:
    """Extract the four FACL crisp inputs from a pram.run_pram() result dict."""
    impr = res["impr"]
    fit = res["fit"]
    aero = res.get("aerosol", {})
    bias_drop = abs(impr["mbe_phys"]) - abs(impr["mbe_corr"])
    return {
        "improvement": float(impr["improvement_pct"]),
        "residual_r2": float(fit["resid_r2"]),
        "aerosol": aerosol_consistency_score(aero),
        "bias": float(bias_drop),
    }


# ===========================================================================
# 4.  Public API
# ===========================================================================
_VERDICTS = [
    (80.0, "very_high", "🟢 Very high confidence — measured-grounded, explainable, physically consistent", "good"),
    (60.0, "high",      "🟢 High confidence — a real, well-supported attribution", "good"),
    (45.0, "moderate",  "🟡 Moderate confidence — a genuine but modest signal", "warn"),
    (25.0, "low",       "🟠 Low confidence — weak or partly contradictory evidence", "warn"),
    (0.0,  "very_low",  "🔴 Very low confidence — mostly noise; do not force a claim", "bad"),
]


def verdict_for(aci: float) -> tuple[str, str, str]:
    """(term, human_label, css_class) for an ACI in [0,100]."""
    for thresh, term, label, css in _VERDICTS:
        if aci >= thresh:
            return term, label, css
    return "very_low", _VERDICTS[-1][2], "bad"


# When the residual R² sits fully in the "suspicious" (leakage) region, the
# attribution confidence is pulled convexly toward this LOW anchor, whatever
# the other signals say. A suspiciously-perfect residual is the fingerprint of
# leakage / a mis-scaled baseline, so a large apparent "improvement" is a reason
# for LESS trust, not more. This is GATE 1, applied here (not as a rule) so it
# authoritatively overrides max-aggregation.
LEAK_GUARD_ANCHOR = 28.0


def run_facl(res: dict) -> dict:
    """End-to-end: PRAM result dict -> fuzzy attribution confidence + trace.

    Returns
    -------
    dict with:
      aci            : float 0–100 Attribution-Confidence Index (after gates)
      aci_raw        : ACI before the leakage guard (for transparency)
      term/label/css : linguistic verdict + UI class
      inputs         : the four crisp FACL inputs
      memberships    : per-input fuzzified degrees (for display)
      trace          : ranked list of fired rules with strengths
      leak_suspicion : membership in residual_r2="suspicious" (0–1)
      crisp_tier     : the legacy crisp tier, for side-by-side comparison
      universe/aggregate : output MF (for plotting the defuzzification)
    """
    crisp = facl_inputs(res)
    out = _SYSTEM.infer(crisp)
    aci_raw = out["aci"]

    # GATE 1 — leakage guard: convex pull toward the low anchor by suspicion.
    mu_susp = out["memberships"]["residual_r2"].get("suspicious", 0.0)
    aci = aci_raw + mu_susp * (LEAK_GUARD_ANCHOR - aci_raw)

    trace = list(out["trace"])
    if mu_susp > 1e-3:
        trace.insert(0, {
            "rule": "GATE 1 (leakage guard): residual R² is suspiciously high → "
                    "confidence suppressed toward 'low' regardless of apparent gain",
            "then": "suppress", "strength": round(float(mu_susp), 3)})

    term, label, css = verdict_for(aci)
    return {
        "ok": True,
        "aci": aci, "aci_raw": aci_raw,
        "term": term, "label": label, "css": css,
        "inputs": crisp,
        "memberships": out["memberships"],
        "term_strength": out["term_strength"],
        "trace": trace,
        "leak_suspicion": float(mu_susp),
        "crisp_tier": res.get("classify", {}).get("tier"),
        "universe": out["universe"],
        "aggregate": out["aggregate"],
    }


def crisp_reference(improvement_pct: float, resid_r2: float) -> str:
    """Legacy crisp tiering (mirrors pram.classify's core) for comparison plots."""
    if resid_r2 >= 0.40 or improvement_pct >= 8:
        return "strong"
    if resid_r2 >= 0.18 or improvement_pct >= 3:
        return "moderate"
    return "weak"


def aci_surface(impr_grid: np.ndarray, r2_grid: np.ndarray,
                aerosol: float = 0.0, bias: float = 0.0) -> np.ndarray:
    """ACI over an (improvement %, residual R²) grid — for the no-cliff plot.

    Demonstrates FACL's graceful boundaries vs. the crisp tier's hard step at
    R²=0.40 / improvement=8 %.
    """
    Z = np.zeros((len(r2_grid), len(impr_grid)), dtype=float)
    for i, r2 in enumerate(r2_grid):
        for j, im in enumerate(impr_grid):
            o = _SYSTEM.infer({"improvement": float(im), "residual_r2": float(r2),
                               "aerosol": aerosol, "bias": bias})
            mu = o["memberships"]["residual_r2"].get("suspicious", 0.0)
            Z[i, j] = o["aci"] + mu * (LEAK_GUARD_ANCHOR - o["aci"])  # GATE 1
    return Z
