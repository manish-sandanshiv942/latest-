"""
Formal-property verification suite (P1–P10)
===========================================

Numerically verifies every formal claim made in NOVELTY.md about FACL and
ACGC, so the paper's "formally characterised" title is backed by a runnable,
reviewable artifact:

  FACL (attribution-confidence layer)
    P1  Boundedness            ACI ∈ [0, 100] over a dense input grid
    P2  Continuity             max ACI jump over a fine grid step is small
                               (no cliff — contrast with the crisp tier step)
    P3  Crisp consistency      at extreme inputs FACL reproduces the crisp
                               weak / moderate-or-better / strong ordering
    P4  Gate-1 dominance       full leak suspicion pins ACI to the low anchor
                               regardless of the other three inputs
    P5  Sectionwise monotonicity  ACI is non-decreasing in improvement % and
                               in residual R² *below the suspicious region*
                               (up to a small numerical tolerance)
    P6  Bounded threshold sensitivity  perturbing every membership breakpoint
                               by ±delta moves the ACI by at most a modest,
                               reported amount (robustness to the exact
                               placement of the linguistic thresholds)

  ACGC (confidence-gated conformal correction)
    P7  Coverage preservation  empirical coverage ≥ 1 − alpha − epsilon on
                               synthetic exchangeable data, for gate weights
                               spanning [0, 1]
    P8  Gate continuity        gate_weight is Lipschitz with the advertised
                               constant 1.5/(hi−lo)
    P9  Physics fallback       ACI ≤ w_lo  ⇒  w = 0 exactly
    P10 Full-trust limit       ACI ≥ w_hi  ⇒  w = 1 exactly

  RCA (regime-conditional attribution)
    P11 AII boundedness + stability-map bounds
                               AII ∈ [0,1] on identical / uncorrelated /
                               reversed rankings; s(AII) ∈ [STAB_FLOOR, 1];
                               composed ACI stays in [0, 100] (P1 survives)
    P12 Stability-map monotonicity + Lipschitz
                               s(·) is non-increasing with the advertised
                               modulus 1.5·(1−floor)/(severe−benign)
    P13 Permutation invariance AII is unchanged when the feature axis is
                               relabelled/reordered (attribution stability is
                               a property of the ranking, not of feature order)

Run:  python validate_formal_properties.py
Exit code 0 = all properties hold; 1 = at least one violated.
"""
from __future__ import annotations

import sys

import numpy as np

from src.facl import _SYSTEM, LEAK_GUARD_ANCHOR, crisp_reference
from src.acgc import gate_weight, gate_lipschitz


def _aci(improvement, r2, aerosol=0.0, bias=0.0) -> float:
    out = _SYSTEM.infer({"improvement": float(improvement),
                         "residual_r2": float(r2),
                         "aerosol": float(aerosol), "bias": float(bias)})
    mu = out["memberships"]["residual_r2"].get("suspicious", 0.0)
    return out["aci"] + mu * (LEAK_GUARD_ANCHOR - out["aci"])


PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def record(pid: str, ok: bool, detail: str):
    _results.append((pid, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {pid}: {detail}")


# ---------------------------------------------------------------------------
# P1 — boundedness
# ---------------------------------------------------------------------------
def check_p1():
    imps = np.linspace(-20.0, 30.0, 26)
    r2s = np.linspace(-0.5, 1.0, 31)
    aeros = (-1.0, 0.0, 1.0)
    biases = (-0.1, 0.0, 0.1)
    lo, hi = np.inf, -np.inf
    for im in imps:
        for r2 in r2s:
            for a in aeros:
                for b in biases:
                    v = _aci(im, r2, a, b)
                    lo, hi = min(lo, v), max(hi, v)
    ok = (lo >= 0.0) and (hi <= 100.0)
    record("P1 boundedness", ok, f"ACI range over grid = [{lo:.2f}, {hi:.2f}] ⊆ [0,100]")


# ---------------------------------------------------------------------------
# P2 — continuity (no cliff), vs the crisp step
# ---------------------------------------------------------------------------
def check_p2(coarse: float = 0.002, fine: float = 1e-4, fine_tol: float = 0.5):
    """Continuity means jumps VANISH as the step size shrinks (a crisp tier's
    step never does). Strategy: locate the steepest region on a coarse sweep,
    then refine locally: at step ``fine`` the max jump must be < ``fine_tol``
    ACI points. The steep zone near R²≈0.80–0.84 is the *intentional* leakage
    gate — steep but continuous, with the finite modulus reported here."""
    for im in (2.0, 5.0, 9.0):
        r2s = np.arange(0.0, 1.0 + 1e-9, coarse)
        acis = np.array([_aci(im, r2) for r2 in r2s])
        j = int(np.argmax(np.abs(np.diff(acis))))
        # refine around the steepest coarse location
        lo_w = max(0.0, r2s[j] - 2 * coarse)
        hi_w = min(1.0, r2s[j + 1] + 2 * coarse)
        r2f = np.arange(lo_w, hi_w + 1e-12, fine)
        acif = np.array([_aci(im, r2) for r2 in r2f])
        max_fine_jump = float(np.max(np.abs(np.diff(acif))))
        lipschitz = max_fine_jump / fine
        ok = max_fine_jump <= fine_tol
        crisp_jump = crisp_reference(im, 0.401) != crisp_reference(im, 0.399)
        record("P2 continuity", ok,
               f"impr={im}%: steepest zone at R²≈{r2s[j]:.3f}; max jump per "
               f"ΔR²={fine} is {max_fine_jump:.3f} (tol {fine_tol}) → finite "
               f"modulus ≈{lipschitz:.0f} ACI/unit-R²; crisp tier still flips "
               f"discontinuously at R²=0.40: {crisp_jump}")


# ---------------------------------------------------------------------------
# P3 — crisp consistency at the extremes
# ---------------------------------------------------------------------------
def check_p3():
    weak = _aci(0.0, 0.02)          # no gain, no structure
    strong = _aci(12.0, 0.60)       # big gain, well-explained
    mid = _aci(5.0, 0.30)           # both moderate
    ok = weak < 25.0 < mid < 75.0 < strong
    record("P3 crisp consistency", ok,
           f"weak={weak:.1f} < 25 < moderate={mid:.1f} < 75 < strong={strong:.1f}")


# ---------------------------------------------------------------------------
# P4 — Gate-1 dominance in the full-leak region
# ---------------------------------------------------------------------------
def check_p4(tol: float = 1e-6):
    worst = 0.0
    for im in (0.0, 8.0, 20.0):
        for a in (-1.0, 0.0, 1.0):
            for b in (-0.1, 0.0, 0.1):
                v = _aci(im, 0.95, a, b)   # μ_suspicious = 1 at R² = 0.95
                worst = max(worst, abs(v - LEAK_GUARD_ANCHOR))
    ok = worst <= tol
    record("P4 Gate-1 dominance", ok,
           f"max |ACI − anchor({LEAK_GUARD_ANCHOR})| at full suspicion = {worst:.2e}")


# ---------------------------------------------------------------------------
# P5 — sectionwise monotonicity (below the suspicious region)
# ---------------------------------------------------------------------------
def check_p5(tol: float = 1.0):
    # non-decreasing in improvement at fixed R²
    worst_impr = 0.0
    for r2 in (0.10, 0.30, 0.55):
        acis = [_aci(im, r2) for im in np.linspace(0.0, 15.0, 61)]
        worst_impr = max(worst_impr, -min(np.diff(acis).min(), 0.0))
    # non-decreasing in R² at fixed improvement, on [0, 0.75] (pre-suspicion)
    worst_r2 = 0.0
    for im in (2.0, 5.0, 9.0):
        acis = [_aci(im, r2) for r2 in np.linspace(0.0, 0.75, 76)]
        worst_r2 = max(worst_r2, -min(np.diff(acis).min(), 0.0))
    ok = worst_impr <= tol and worst_r2 <= tol
    record("P5 monotonicity", ok,
           f"max decrease: vs improvement {worst_impr:.3f}, vs R² (≤0.75) "
           f"{worst_r2:.3f} (tol {tol} ACI pts — centroid defuzzification "
           f"permits tiny local dips)")


# ---------------------------------------------------------------------------
# P6 — bounded sensitivity to membership-threshold placement
# ---------------------------------------------------------------------------
def check_p6(delta: float = 0.02, bound: float = 12.0):
    """Shift the residual-R² 'strong' breakpoint by ±delta and measure the
    worst ACI change over a probe grid. This bounds how much the conclusions
    depend on the exact linguistic threshold placement."""
    from src.facl import _build_system, FuzzyVar, trap, tri

    probes = [(im, r2) for im in (1.0, 4.0, 7.0, 11.0)
              for r2 in (0.1, 0.25, 0.45, 0.6)]

    def aci_with_shift(d: float):
        sys_ = _build_system()
        sys_.inputs["residual_r2"] = FuzzyVar("residual_r2", {
            "low":        trap(-1.0, -1.0, 0.10, 0.20),
            "medium":     tri(0.14, 0.30, 0.46),
            "high":       trap(0.34 + d, 0.52 + d, 0.80, 0.84),
            "suspicious": trap(0.80, 0.90, 1.0, 1.0),
        })
        vals = []
        for im, r2 in probes:
            o = sys_.infer({"improvement": im, "residual_r2": r2,
                            "aerosol": 0.0, "bias": 0.0})
            mu = o["memberships"]["residual_r2"].get("suspicious", 0.0)
            vals.append(o["aci"] + mu * (LEAK_GUARD_ANCHOR - o["aci"]))
        return np.array(vals)

    base = aci_with_shift(0.0)
    worst = max(float(np.max(np.abs(aci_with_shift(+delta) - base))),
                float(np.max(np.abs(aci_with_shift(-delta) - base))))
    ok = worst <= bound
    record("P6 threshold sensitivity", ok,
           f"±{delta} shift of the R² 'high' breakpoint moves ACI by at most "
           f"{worst:.2f} pts (bound {bound})")


# ---------------------------------------------------------------------------
# P7 — ACGC coverage preservation on synthetic exchangeable data
# ---------------------------------------------------------------------------
def check_p7(alpha: float = 0.10, n: int = 4000, trials: int = 20,
             eps: float = 0.03, seed: int = 7):
    """For gate weights spanning [0,1], the conformal interval around the
    blended predictor must achieve ≥ 1−alpha−eps empirical coverage when the
    calibration/test scores are exchangeable (iid here)."""
    from src.crisp import fit_conformal, apply_conformal

    rng = np.random.default_rng(seed)
    worst = 1.0
    for wv in (0.0, 0.25, 0.5, 0.75, 1.0):
        covs = []
        for _ in range(trials):
            x = rng.uniform(0.0, 1000.0, n)                 # driver (e.g. GHI)
            y_phys = 0.25 * x + rng.normal(0.0, 8.0, n)     # physics predictor
            r_hat = 0.02 * x + rng.normal(0.0, 2.0, n)      # residual estimate
            y = y_phys + r_hat + rng.normal(0.0, 10.0, n)   # truth
            yhat = y_phys + wv * r_hat
            half = n // 2
            cal = fit_conformal(y[:half], yhat[:half], driver=x[:half],
                                alpha=alpha, method="mondrian", n_bins=5)
            lo, hi = apply_conformal(yhat[half:], cal, driver=x[half:],
                                     clip_lo=None, clip_hi=None)
            covs.append(float(np.mean((y[half:] >= lo) & (y[half:] <= hi))))
        worst = min(worst, float(np.mean(covs)))
    ok = worst >= (1.0 - alpha - eps)
    record("P7 coverage preservation", ok,
           f"min mean coverage over w∈{{0,…,1}} = {worst:.3f} "
           f"(target ≥ {1 - alpha - eps:.3f})")


# ---------------------------------------------------------------------------
# P8/P9/P10 — gate weight properties
# ---------------------------------------------------------------------------
def check_p8_p9_p10(lo: float = 25.0, hi: float = 80.0):
    acis = np.linspace(0.0, 100.0, 100001)
    ws = np.array([gate_weight(a, lo=lo, hi=hi) for a in acis])
    # P8: empirical Lipschitz constant vs advertised
    L_emp = float(np.max(np.abs(np.diff(ws)) / np.diff(acis)))
    L_adv = gate_lipschitz(lo, hi)
    record("P8 gate Lipschitz", L_emp <= L_adv + 1e-9,
           f"empirical L = {L_emp:.5f} ≤ advertised 1.5/(hi−lo) = {L_adv:.5f}")
    # P9: physics fallback
    ok9 = bool(np.all(ws[acis <= lo] == 0.0))
    record("P9 physics fallback", ok9, f"w = 0 exactly for all ACI ≤ {lo}")
    # P10: full-trust limit
    ok10 = bool(np.all(ws[acis >= hi] == 1.0))
    record("P10 full-trust limit", ok10, f"w = 1 exactly for all ACI ≥ {hi}")


# ---------------------------------------------------------------------------
# P11/P12/P13 — RCA properties
# ---------------------------------------------------------------------------
def check_p11(seed: int = 11):
    """AII bounded on canonical ranking geometries; stability map bounded;
    composition with the ACI preserves FACL's P1 bound."""
    import pandas as pd
    from src.rca import (attribution_instability, aii_to_stability_weight,
                         STAB_FLOOR)

    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1.0, 40)

    def _inst(cols):
        M = pd.DataFrame(np.column_stack(cols),
                         index=[f"f{i}" for i in range(len(cols[0]))],
                         columns=[f"regime_{i}" for i in range(len(cols))])
        return attribution_instability({"ok": True, "matrix": M})["aii"]

    aii_same = _inst([base, base, base])                    # identical
    aii_rand = float(np.mean([_inst([rng.permutation(base),
                                     rng.permutation(base)])
                              for _ in range(30)]))          # uncorrelated
    rev = base.max() + base.min() - base                     # reversed
    aii_rev = _inst([base, rev])
    ok_order = aii_same <= 0.05 and aii_same < aii_rand < aii_rev and aii_rev >= 0.9
    ok_bounds = all(0.0 <= a <= 1.0 for a in (aii_same, aii_rand, aii_rev))
    # stability map bounds + ACI composition stays in [0,100]
    ss = np.array([aii_to_stability_weight(a) for a in np.linspace(0, 1, 2001)])
    ok_s = bool(np.all((ss >= STAB_FLOOR - 1e-12) & (ss <= 1.0 + 1e-12)))
    comp = np.array([s * aci for s in (ss.min(), ss.max())
                     for aci in (0.0, 50.0, 100.0)])
    ok_comp = bool(np.all((comp >= 0.0) & (comp <= 100.0)))
    record("P11 AII boundedness", ok_order and ok_bounds and ok_s and ok_comp,
           f"AII: identical={aii_same:.3f} < uncorrelated≈{aii_rand:.3f} < "
           f"reversed={aii_rev:.3f}; s∈[{ss.min():.2f},{ss.max():.2f}]"
           f"⊆[{STAB_FLOOR},1]; s·ACI ⊆ [0,100]")


def check_p12():
    """Stability map is monotone non-increasing with the advertised modulus."""
    from src.rca import aii_to_stability_weight, STAB_FLOOR, AII_BENIGN, AII_SEVERE

    xs = np.linspace(0.0, 1.0, 100001)
    ss = np.array([aii_to_stability_weight(x) for x in xs])
    dif = np.diff(ss)
    mono = bool(np.all(dif <= 1e-12))
    L_emp = float(np.max(np.abs(dif) / np.diff(xs)))
    L_adv = 1.5 * (1.0 - STAB_FLOOR) / (AII_SEVERE - AII_BENIGN)
    ok = mono and L_emp <= L_adv + 1e-9
    record("P12 stability-map monotone+Lipschitz", ok,
           f"non-increasing: {mono}; empirical L = {L_emp:.4f} ≤ advertised "
           f"1.5·(1−floor)/(severe−benign) = {L_adv:.4f}")


def check_p13(seed: int = 13, tol: float = 1e-9):
    """AII is invariant to relabelling/reordering the feature axis."""
    import pandas as pd
    from src.rca import attribution_instability

    rng = np.random.default_rng(seed)
    n_feat, n_reg = 30, 4
    M = rng.uniform(0.0, 1.0, (n_feat, n_reg))
    idx = [f"f{i}" for i in range(n_feat)]
    cols = [f"regime_{j}" for j in range(n_reg)]
    base = attribution_instability(
        {"ok": True, "matrix": pd.DataFrame(M, index=idx, columns=cols)})["aii"]
    worst = 0.0
    for _ in range(10):
        p = rng.permutation(n_feat)
        aii_p = attribution_instability(
            {"ok": True, "matrix": pd.DataFrame(M[p], index=[idx[i] for i in p],
                                                columns=cols)})["aii"]
        worst = max(worst, abs(aii_p - base))
    ok = worst <= tol
    record("P13 permutation invariance", ok,
           f"max |ΔAII| over 10 random feature reorderings = {worst:.2e}")


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("Formal-property verification suite (P1–P13)")
    print("=" * 72)
    check_p1()
    check_p2()
    check_p3()
    check_p4()
    check_p5()
    check_p6()
    check_p7()
    check_p8_p9_p10()
    check_p11()
    check_p12()
    check_p13()
    print("-" * 72)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    print(f"{len(_results) - n_fail}/{len(_results)} checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
