"""
ACGC — Attribution-Confidence-Gated Conformal correction
========================================================

THE NOVEL CLOSED LOOP (and why it is more than PRAM + FACL + CRISP)
-------------------------------------------------------------------
PRAM produces a learned residual correction ``r̂`` for the physics predictor.
FACL produces a fuzzy Attribution-Confidence Index (ACI ∈ [0,100]) that says
how much the *attribution behind that correction* should be trusted. CRISP
produces distribution-free intervals for any point predictor.

Until now those three were **parallel reports**. ACGC couples them into a
single decision loop:

    ACI  →  gate weight  w = g(ACI) ∈ [0,1]
    ŷ_w  =  P_phys + w · r̂          (confidence-gated predictor)
    intervals = Mondrian split-conformal around ŷ_w

i.e. the fuzzy attribution-confidence is no longer descriptive metadata — it
*operationally decides how much of the ML correction is applied*, and the
uncertainty band adapts accordingly. Low-trust attributions automatically
fall back toward pure physics; high-trust attributions apply the full
correction; everything in between is a smooth convex blend (no cliff).

Why the guarantee survives (Property P7 — the provable bit)
-----------------------------------------------------------
Split conformal requires the point predictor to be FIXED before the
calibration scores are computed. A gate weight estimated on the calibration
data would break exchangeability. ACGC therefore uses a **three-block
chronological split of the post-training rows**:

    [ GATE block ]  [ CALIBRATION block ]  [ TEST block ]
       (earliest)                              (latest)

*   ``w`` is a measurable function of the GATE block only
    (its local improvement %, local residual R², plus the global aerosol /
    bias evidence — all frozen before calibration).
*   The conformal quantiles are then computed on the CALIBRATION block for the
    now-fixed predictor ``ŷ_w``.
*   All reported coverage/width numbers come from the untouched TEST block.

Proposition (coverage preservation).  If the calibration and test
nonconformity scores of ``ŷ_w`` are exchangeable, then for any gate weight
``w`` chosen as a function of the gate block alone,

    P( y ∈ [ŷ_w − Q_α ,  ŷ_w + Q_α] ) ≥ 1 − α ,

with ``Q_α`` the ceil((n+1)(1−α))/n calibration quantile — the standard split-
conformal result (Vovk; Lei et al. 2018) applied to the fixed composite
predictor ``P_phys + w·r̂``. Gating cannot destroy validity; it can only move
the point predictor (and hence the interval CENTRE) to a better location.

Additional formal properties (verified numerically in
``validate_facl_properties.py``):

  P8  (continuity)     w = g(ACI) is Lipschitz-continuous, so ŷ_w is
                       continuous in the ACI — no prediction cliff when the
                       evidence sits near a linguistic boundary.
  P9  (fallback)       ACI → 0  ⇒  ŷ_w → P_phys  (pure physics; the system
                       never applies a correction it does not trust).
  P10 (full trust)     ACI → 100 ⇒ ŷ_w → P_phys + r̂ (full PRAM correction).

Honest scope
------------
The conformal machinery is textbook and the fuzzy engine is textbook; the
contribution is the *architecture*: an attribution-confidence signal, itself
physics-residual-gated, used as the convex gain of a residual correction
inside a conformal wrapper — with the split designed so the finite-sample
guarantee provably survives. Evaluated against three baselines (physics-only,
always-correct, crisp hard-switch) on the same held-out block.

Pure numpy/pandas. No Streamlit here; callers render.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import crisp as _crisp
from .facl import _SYSTEM, LEAK_GUARD_ANCHOR, verdict_for


# ---------------------------------------------------------------------------
# 1.  Gate weight from ACI
# ---------------------------------------------------------------------------
def gate_weight(aci: float, *, lo: float = 25.0, hi: float = 80.0) -> float:
    """Smooth monotone map ACI∈[0,100] → w∈[0,1] (smootherstep).

    * ACI ≤ ``lo``  → w = 0 (pure physics fallback; matches FACL's
      'low'/'very_low' verdict boundary at 25).
    * ACI ≥ ``hi``  → w = 1 (full correction; matches 'very_high' at 80).
    * In between    → C¹ smootherstep 3t²−2t³, Lipschitz with constant
      1.5/(hi−lo) — this is Property P8's explicit modulus.
    """
    t = (float(aci) - lo) / (hi - lo)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def gate_lipschitz(lo: float = 25.0, hi: float = 80.0) -> float:
    """Exact Lipschitz constant of gate_weight (max slope of smootherstep)."""
    return 1.5 / (hi - lo)


# ---------------------------------------------------------------------------
# 2.  Local FACL evaluation on the GATE block only
# ---------------------------------------------------------------------------
def _local_facl_aci(y_true_g: np.ndarray, y_phys_g: np.ndarray,
                    r_hat_g: np.ndarray, *, aerosol: float, ) -> dict:
    """Recompute the FACL inputs *on the gate block alone* and infer the ACI.

    This is what makes the gate legitimate: every data-dependent quantity in
    ``w`` is a function of gate-block rows only.
    """
    resid_g = y_true_g - y_phys_g                    # true physics residual
    corr_g = y_phys_g + r_hat_g

    def _rmse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")

    rmse_phys = _rmse(y_true_g, y_phys_g)
    rmse_corr = _rmse(y_true_g, corr_g)
    improvement = (1.0 - rmse_corr / rmse_phys) * 100.0 \
        if np.isfinite(rmse_phys) and rmse_phys > 0 else 0.0

    # local residual R²: how much of the gate-block residual r̂ explains
    m = np.isfinite(resid_g) & np.isfinite(r_hat_g)
    if m.sum() >= 10 and np.var(resid_g[m]) > 1e-12:
        ss_res = float(np.sum((resid_g[m] - r_hat_g[m]) ** 2))
        ss_tot = float(np.sum((resid_g[m] - np.mean(resid_g[m])) ** 2))
        resid_r2 = 1.0 - ss_res / ss_tot
    else:
        resid_r2 = 0.0

    def _mbe(a, b):
        mm = np.isfinite(a) & np.isfinite(b)
        return float(np.mean(b[mm] - a[mm])) if mm.any() else 0.0

    bias_drop = abs(_mbe(y_true_g, y_phys_g)) - abs(_mbe(y_true_g, corr_g))

    inputs = {"improvement": improvement,
              "residual_r2": float(np.clip(resid_r2, -1.0, 1.0)),
              "aerosol": float(aerosol),
              "bias": float(bias_drop)}
    out = _SYSTEM.infer(inputs)
    mu_susp = out["memberships"]["residual_r2"].get("suspicious", 0.0)
    aci = out["aci"] + mu_susp * (LEAK_GUARD_ANCHOR - out["aci"])   # GATE 1
    term, label, css = verdict_for(aci)
    return {"aci": float(aci), "aci_raw": float(out["aci"]),
            "inputs": inputs, "leak_suspicion": float(mu_susp),
            "term": term, "label": label, "css": css,
            "trace": out["trace"]}


# ---------------------------------------------------------------------------
# 3.  End-to-end runner
# ---------------------------------------------------------------------------
def run_acgc(pram_res: dict, driver, *,
             alpha: float = 0.10, rated_kw: float | None = None,
             gate_frac: float = 0.30, calib_frac: float = 0.35,
             n_bins: int = 5, min_bin: int = 20,
             w_lo: float = 25.0, w_hi: float = 80.0) -> dict:
    """Attribution-confidence-gated conformal correction, end to end.

    Parameters
    ----------
    pram_res : dict
        A successful ``pram.run_pram()`` result. ACGC uses its post-training
        rows: index ``fit['index'][cut:]``, truth ``ref``, calibrated physics
        ``model``, and the residual model's honest predictions ``fit['pred']``.
    driver : pd.Series or array aligned to the weather index
        Regime variable for Mondrian binning (GHI / POA for PV).
    alpha : target miscoverage (0.10 → 90 % intervals).
    gate_frac, calib_frac : chronological fractions of the post-training rows
        for the GATE and CALIBRATION blocks (remainder = TEST).

    Returns
    -------
    dict with the gated result plus three baselines (physics-only w=0,
    always-correct w=1, crisp hard-switch), each with a full CRISP
    coverage report on the SAME test rows.
    """
    if not pram_res.get("ok"):
        return {"ok": False, "reason": "PRAM result not ok"}
    fit = pram_res["fit"]
    cut = fit["cut"]
    idx_post = fit["index"][cut:]                       # honest, unseen by training
    n = len(idx_post)
    if n < 120:
        return {"ok": False,
                "reason": f"Only {n} post-training rows (need ≥ 120 for a "
                          f"gate/calibration/test split)."}

    y_true = pram_res["ref"].reindex(idx_post).to_numpy(dtype=float)
    y_phys = pram_res["model"].reindex(idx_post).to_numpy(dtype=float)
    r_hat = np.asarray(fit["pred"], dtype=float)
    drv = pd.Series(driver).reindex(idx_post).to_numpy(dtype=float)

    okm = np.isfinite(y_true) & np.isfinite(y_phys) & np.isfinite(r_hat) & np.isfinite(drv)
    y_true, y_phys, r_hat, drv = y_true[okm], y_phys[okm], r_hat[okm], drv[okm]
    n = y_true.size
    g_end = int(n * gate_frac)
    c_end = g_end + int(n * calib_frac)
    if g_end < 30 or (c_end - g_end) < 40 or (n - c_end) < 30:
        return {"ok": False,
                "reason": f"Blocks too small (gate={g_end}, "
                          f"calib={c_end - g_end}, test={n - c_end})."}

    # ---- GATE block: local FACL → ACI → w (frozen before calibration) -----
    from .facl import aerosol_consistency_score
    aero_score = aerosol_consistency_score(pram_res.get("aerosol", {}))
    gate = _local_facl_aci(y_true[:g_end], y_phys[:g_end], r_hat[:g_end],
                           aerosol=aero_score)
    w = gate_weight(gate["aci"], lo=w_lo, hi=w_hi)

    # crisp hard-switch baseline: apply full correction iff crisp tier ≠ weak
    crisp_tier = (pram_res.get("classify") or {}).get("tier", "weak")
    w_hard = 1.0 if crisp_tier in ("moderate", "strong") else 0.0

    variants = {
        "acgc":     w,        # the novel gated predictor
        "physics":  0.0,      # baseline 1: never correct
        "full":     1.0,      # baseline 2: always correct
        "hard":     w_hard,   # baseline 3: crisp all-or-nothing switch
    }

    yt_cal, yt_te = y_true[g_end:c_end], y_true[c_end:]
    yp_cal_phys, yp_te_phys = y_phys[g_end:c_end], y_phys[c_end:]
    rh_cal, rh_te = r_hat[g_end:c_end], r_hat[c_end:]
    dv_cal, dv_te = drv[g_end:c_end], drv[c_end:]
    y_range = float(np.max(yt_te) - np.min(yt_te)) or 1.0

    results = {}
    for name, wv in variants.items():
        yhat_cal = yp_cal_phys + wv * rh_cal
        yhat_te = yp_te_phys + wv * rh_te
        cal = _crisp.fit_conformal(yt_cal, yhat_cal, driver=dv_cal, alpha=alpha,
                                   method="mondrian", n_bins=n_bins,
                                   min_bin=min_bin)
        lo, hi = _crisp.apply_conformal(yhat_te, cal, driver=dv_te,
                                        clip_lo=0.0, clip_hi=rated_kw)
        rep = _crisp.coverage_report(yt_te, lo, hi, driver=dv_te,
                                     edges=cal["edges"], alpha=alpha,
                                     y_range=y_range)
        m = np.isfinite(yt_te) & np.isfinite(yhat_te)
        rmse = float(np.sqrt(np.mean((yt_te[m] - yhat_te[m]) ** 2))) if m.any() else float("nan")
        results[name] = {"w": float(wv), "report": rep, "rmse": rmse,
                         "lower": lo, "upper": hi, "yhat": yhat_te,
                         "calibrator": cal}

    return {
        "ok": True, "alpha": float(alpha),
        "gate": gate, "w": float(w), "w_hard": float(w_hard),
        "gate_bounds": (float(w_lo), float(w_hi)),
        "lipschitz": gate_lipschitz(w_lo, w_hi),
        "n_gate": int(g_end), "n_calib": int(c_end - g_end),
        "n_test": int(n - c_end),
        "y_test": yt_te, "driver_test": dv_te,
        "variants": results,
    }


def summarize(res: dict) -> str:
    """Compact human-readable comparison table for a run_acgc result."""
    if not res.get("ok"):
        return f"ACGC: {res.get('reason', 'failed')}"
    g = res["gate"]
    lines = [
        f"ACGC (alpha={res['alpha']:.2f}, blocks: gate={res['n_gate']} / "
        f"calib={res['n_calib']} / test={res['n_test']})",
        f"  gate-block ACI  : {g['aci']:.1f}/100 ({g['term']})  →  w = {res['w']:.3f}",
        f"  crisp switch    : w_hard = {res['w_hard']:.0f} (tier-based)",
        f"  {'variant':<9} {'w':>5}  {'RMSE':>8}  {'coverage':>9}  "
        f"{'MPIW':>8}  {'Winkler':>9}  {'worst-gap':>9}",
    ]
    for name in ("acgc", "physics", "full", "hard"):
        v = res["variants"][name]
        r = v["report"]
        lines.append(
            f"  {name:<9} {v['w']:>5.2f}  {v['rmse']:>8.2f}  "
            f"{r['marginal_coverage']*100:>8.1f}%  {r['mpiw']:>8.2f}  "
            f"{r['winkler']:>9.2f}  "
            f"{r.get('worst_slab_gap', float('nan'))*100:>8.1f}p")
    return "\n".join(lines)
