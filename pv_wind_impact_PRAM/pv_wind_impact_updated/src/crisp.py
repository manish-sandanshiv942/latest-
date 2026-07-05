"""
CRISP — Conformal Regime-Indexed Split-conformal Prediction intervals
=====================================================================

A distribution-free **uncertainty-quantification layer** for the project's PV /
wind power predictions. Where the ML leaderboard reports a single point score
(R²/MAE/RMSE) and PRAM/FACL answer *"which drivers matter"* and *"how much do we
trust the attribution"*, CRISP answers a different, complementary question:

    "Given this predictor, produce an interval that contains the true power
     with a guaranteed probability — and prove the guarantee holds on real
     metered data, even conditionally within each operating regime."

Design philosophy
-----------------
* **Model-agnostic.** CRISP never owns a model. It consumes *point predictions*
  ``y_pred`` (physics model, PRAM-corrected series, or any of the 12 ML
  regressors) plus the matching truth ``y_true`` and a **regime driver** (GHI /
  POA for PV, hub wind for wind), and turns them into calibrated intervals.
  This is why the module is pure ``numpy`` / ``pandas`` — no sklearn needed.

* **Split conformal (Vovk; Lei et al. 2018).** With an exchangeable calibration
  set of size ``n`` and target miscoverage ``alpha``, the interval
  ``ŷ ± Q`` where ``Q`` is the ``ceil((n+1)(1-alpha))/n`` empirical quantile of
  the absolute residuals has **finite-sample marginal coverage ≥ 1-alpha**. No
  distributional assumption on the errors.

* **Mondrian / regime-conditional (the reportable integration).** Marginal
  coverage can hide gross *conditional* miscoverage — a solar interval that is
  far too wide at night (trivially covered) and too narrow at midday still looks
  "90% covered" on average. CRISP computes a **separate conformal quantile per
  irradiance/wind regime**, restoring *conditional* validity. The headline
  empirical finding is that Mondrian shrinks the **worst-regime coverage gap**
  versus a single global quantile, on real DKASC data.

* **Chronological, leakage-safe.** Calibration is an *earlier* block and the
  test set a *later* held-out block (no shuffling), matching the rest of the
  project. This deliberately stresses the exchangeability assumption with real
  temporal drift, so the reported coverage is honest rather than optimistic.

Honest novelty scope (read before writing the paper)
----------------------------------------------------
Conformal prediction is an established framework; this is **not** a new
conformal algorithm. The contribution is the *applied integration* — regime-
indexed split-conformal intervals wrapped around a physics/physics-residual PV
predictor, with the regimes defined by the very atmospheric drivers the impact
analysis ranks, **validated for conditional coverage on real metered DKASC
generation**. Same "applied / framework-integration" tier as PRAM and FACL.

This module is pure (no Streamlit); callers handle rendering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core conformal primitive
# ---------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile of nonconformity ``scores``.

    Returns the smallest threshold ``Q`` such that at least a fraction
    ``(1-alpha)`` of a *future* exchangeable score falls at or below it, using
    the standard small-sample correction: the ``ceil((n+1)(1-alpha)) / n``
    empirical quantile (Lei et al. 2018). Returns ``+inf`` when the calibration
    set is too small to guarantee the level (``n < ceil((n+1)(1-alpha))``).
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:                       # not enough calibration points for this level
        return float("inf")
    return float(np.sort(s)[k - 1])


# ---------------------------------------------------------------------------
# Regime binning (the "Mondrian taxonomy")
# ---------------------------------------------------------------------------
def regime_edges(driver: np.ndarray, n_bins: int = 5,
                 method: str = "quantile") -> np.ndarray:
    """Interior bin edges that partition ``driver`` into ``n_bins`` regimes.

    ``quantile`` (default) gives roughly equal-population regimes — the robust
    choice for skewed solar irradiance / wind distributions. ``uniform`` gives
    equal-width regimes.
    """
    d = np.asarray(driver, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or n_bins < 2:
        return np.array([])
    if method == "uniform":
        lo, hi = float(np.min(d)), float(np.max(d))
        if hi <= lo:
            return np.array([])
        return np.linspace(lo, hi, n_bins + 1)[1:-1]
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(d, qs))
    return edges


def assign_regime(driver: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map each value to a regime index ``0..len(edges)`` via ``np.digitize``."""
    d = np.asarray(driver, dtype=float)
    if edges is None or len(edges) == 0:
        return np.zeros(d.shape, dtype=int)
    return np.digitize(d, edges).astype(int)


# ---------------------------------------------------------------------------
# Calibrator: fit on the calibration split
# ---------------------------------------------------------------------------
def fit_conformal(y_true, y_pred, *, driver=None, alpha: float = 0.10,
                  method: str = "mondrian", n_bins: int = 5,
                  min_bin: int = 20, bin_method: str = "quantile") -> dict:
    """Fit a split-conformal calibrator on a calibration set.

    Parameters
    ----------
    y_true, y_pred : array-like
        Measured power and the predictor's point prediction on the calibration
        rows (operating hours only — pass the masked arrays).
    driver : array-like or None
        The regime variable (e.g. GHI / POA for PV). Required for ``mondrian``
        and ``normalized`` methods.
    alpha : float
        Target miscoverage (0.10 → 90% intervals).
    method : {"global", "mondrian", "normalized"}
        * ``global``     — one quantile ``Q`` for all rows: ``ŷ ± Q``.
        * ``mondrian``   — a separate ``Q_k`` per regime (conditional validity).
        * ``normalized`` — locally-adaptive width ``ŷ ± Q·σ̂(driver)`` where the
          difficulty ``σ̂`` is the per-regime mean absolute residual; a single
          global quantile is taken on the *normalized* scores.
    min_bin : int
        Regimes with fewer calibration points fall back to the global quantile.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scores = np.abs(y_true - y_pred)
    m = np.isfinite(scores)
    scores, yt, yp = scores[m], y_true[m], y_pred[m]
    drv = None if driver is None else np.asarray(driver, dtype=float)[m]

    q_global = conformal_quantile(scores, alpha)
    cal = {"method": method, "alpha": float(alpha), "q_global": q_global,
           "n_calib": int(scores.size), "edges": np.array([]),
           "q_by_bin": {}, "sigma_by_bin": {}, "q_norm": None,
           "n_by_bin": {}, "min_bin": min_bin}

    if method == "global" or drv is None:
        cal["method"] = "global"
        return cal

    edges = regime_edges(drv, n_bins=n_bins, method=bin_method)
    bins = assign_regime(drv, edges)
    cal["edges"] = edges

    if method == "mondrian":
        for k in np.unique(bins):
            sk = scores[bins == k]
            cal["n_by_bin"][int(k)] = int(sk.size)
            cal["q_by_bin"][int(k)] = (conformal_quantile(sk, alpha)
                                       if sk.size >= min_bin else q_global)
        return cal

    if method == "normalized":
        # difficulty σ̂ per regime = mean absolute residual in that regime
        sigma_by_bin = {}
        for k in np.unique(bins):
            sk = scores[bins == k]
            sig = float(np.mean(sk)) if sk.size >= min_bin else float(np.mean(scores))
            sigma_by_bin[int(k)] = max(sig, 1e-6)
        sigma_vec = np.array([sigma_by_bin[int(b)] for b in bins])
        cal["sigma_by_bin"] = sigma_by_bin
        cal["q_norm"] = conformal_quantile(scores / sigma_vec, alpha)
        cal["sigma_global"] = max(float(np.mean(scores)), 1e-6)
        return cal

    raise ValueError(f"unknown method {method!r}")


def apply_conformal(y_pred, cal: dict, *, driver=None,
                    clip_lo: float | None = 0.0,
                    clip_hi: float | None = None):
    """Produce ``(lower, upper)`` intervals for test predictions.

    Widths are looked up from the fitted calibrator; bounds are optionally
    clipped to the physical range ``[clip_lo, clip_hi]`` (e.g. 0 → rated kW),
    which is physically correct and never breaks the coverage guarantee (it can
    only *raise* coverage by widening feasible mass onto the boundary).
    """
    yp = np.asarray(y_pred, dtype=float)
    method = cal["method"]

    if method == "global" or driver is None:
        half = np.full(yp.shape, cal["q_global"])
    elif method == "mondrian":
        bins = assign_regime(driver, cal["edges"])
        half = np.array([cal["q_by_bin"].get(int(b), cal["q_global"]) for b in bins])
    elif method == "normalized":
        bins = assign_regime(driver, cal["edges"])
        sig = np.array([cal["sigma_by_bin"].get(int(b), cal.get("sigma_global", 1.0))
                        for b in bins])
        half = cal["q_norm"] * sig
    else:
        raise ValueError(f"unknown method {method!r}")

    lower = yp - half
    upper = yp + half
    if clip_lo is not None:
        lower = np.maximum(lower, clip_lo)
    if clip_hi is not None:
        upper = np.minimum(upper, clip_hi)
    return lower, upper


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def winkler_score(y_true, lower, upper, alpha: float) -> float:
    """Mean Winkler / interval score (a proper scoring rule; lower is better).

    ``W = (u-l) + (2/alpha)(l-y)·1[y<l] + (2/alpha)(y-u)·1[y>u]`` — rewards
    narrow intervals but penalises misses in proportion to how far outside they
    fall, so it cannot be gamed by trivially wide or trivially narrow bands.
    """
    y = np.asarray(y_true, float); l = np.asarray(lower, float); u = np.asarray(upper, float)
    m = np.isfinite(y) & np.isfinite(l) & np.isfinite(u)
    y, l, u = y[m], l[m], u[m]
    if y.size == 0:
        return float("nan")
    width = u - l
    below = (l - y) * (y < l)
    above = (y - u) * (y > u)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def coverage_report(y_true, lower, upper, *, driver=None, edges=None,
                    alpha: float = 0.10, y_range: float | None = None) -> dict:
    """Marginal + conditional coverage, interval width, and Winkler score.

    Returns the headline numbers for the paper: empirical marginal coverage
    (target ``1-alpha``), per-regime conditional coverage, the **worst-regime
    coverage gap**, mean interval width (MPIW, kW) and PINAW (width normalised
    by the observed power range), and the mean Winkler score.
    """
    y = np.asarray(y_true, float); l = np.asarray(lower, float); u = np.asarray(upper, float)
    m = np.isfinite(y) & np.isfinite(l) & np.isfinite(u)
    y, l, u = y[m], l[m], u[m]
    n = y.size
    if n == 0:
        return {"ok": False, "reason": "no finite test rows"}

    covered = (y >= l) & (y <= u)
    width = u - l
    if y_range is None:
        y_range = float(np.max(y) - np.min(y)) or 1.0

    out = {
        "ok": True, "alpha": float(alpha), "target": 1.0 - float(alpha),
        "n_test": int(n),
        "marginal_coverage": float(np.mean(covered)),
        "mpiw": float(np.mean(width)),                 # mean interval width, kW
        "pinaw": float(np.mean(width) / y_range),      # normalised avg width
        "winkler": winkler_score(y, l, u, alpha),
        "y_range": float(y_range),
        "bins": [],
    }

    if driver is not None and edges is not None and len(edges) > 0:
        drv = np.asarray(driver, float)[m]
        bins = assign_regime(drv, edges)
        gaps = []
        for k in np.unique(bins):
            sel = bins == k
            if sel.sum() == 0:
                continue
            cov_k = float(np.mean(covered[sel]))
            gaps.append(abs(cov_k - out["target"]))
            out["bins"].append({
                "regime": int(k), "n": int(sel.sum()),
                "driver_lo": float(np.min(drv[sel])),
                "driver_hi": float(np.max(drv[sel])),
                "coverage": cov_k,
                "mpiw": float(np.mean(width[sel])),
            })
        out["worst_slab_gap"] = float(max(gaps)) if gaps else float("nan")
        out["mean_slab_gap"] = float(np.mean(gaps)) if gaps else float("nan")
    return out


# ---------------------------------------------------------------------------
# End-to-end convenience runner
# ---------------------------------------------------------------------------
def run_crisp(y_true, y_pred, driver, *, alpha: float = 0.10,
              n_bins: int = 5, method: str = "mondrian",
              calib_frac: float = 0.5, operating_mask=None,
              min_bin: int = 20, rated_kw: float | None = None,
              compare_global: bool = True) -> dict:
    """Full pipeline: chronological calib/test split → fit → intervals → report.

    The inputs are assumed to be **time-ordered** (as the DKASC hourly frame is).
    The earliest ``calib_frac`` of operating rows is the conformal calibration
    set; the latest ``1-calib_frac`` is the held-out test set that all coverage
    numbers are computed on — a deliberately honest, leakage-safe evaluation.

    When ``compare_global`` is set, a plain global-quantile calibrator is also
    evaluated on the same test rows so the caller can show how much Mondrian
    improves *conditional* coverage.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    driver = np.asarray(driver, dtype=float)

    ok = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(driver)
    if operating_mask is not None:
        ok &= np.asarray(operating_mask, dtype=bool)
    yt, yp, dv = y_true[ok], y_pred[ok], driver[ok]

    n = yt.size
    if n < 80:
        return {"ok": False,
                "reason": f"Only {n} operating rows after cleaning (need ≥ 80)."}

    cut = int(n * float(calib_frac))
    if cut < 40 or (n - cut) < 30:
        return {"ok": False,
                "reason": f"Calib/test split too small (calib={cut}, test={n-cut})."}

    yt_cal, yp_cal, dv_cal = yt[:cut], yp[:cut], dv[:cut]
    yt_te,  yp_te,  dv_te  = yt[cut:], yp[cut:], dv[cut:]

    clip_hi = rated_kw
    y_range = float(np.max(yt_te) - np.min(yt_te)) or 1.0

    cal = fit_conformal(yt_cal, yp_cal, driver=dv_cal, alpha=alpha,
                        method=method, n_bins=n_bins, min_bin=min_bin)
    lo, hi = apply_conformal(yp_te, cal, driver=dv_te, clip_lo=0.0, clip_hi=clip_hi)
    rep = coverage_report(yt_te, lo, hi, driver=dv_te, edges=cal["edges"],
                          alpha=alpha, y_range=y_range)

    result = {"ok": True, "method": method, "alpha": float(alpha),
              "n_calib": int(cut), "n_test": int(n - cut),
              "calibrator": cal, "report": rep,
              "lower": lo, "upper": hi, "y_test": yt_te, "yhat_test": yp_te,
              "driver_test": dv_te, "edges": cal["edges"]}

    if compare_global and method != "global":
        cal_g = fit_conformal(yt_cal, yp_cal, alpha=alpha, method="global")
        lo_g, hi_g = apply_conformal(yp_te, cal_g, clip_lo=0.0, clip_hi=clip_hi)
        # score the global intervals with the SAME regime edges for a fair
        # conditional-coverage comparison
        rep_g = coverage_report(yt_te, lo_g, hi_g, driver=dv_te,
                                edges=cal["edges"], alpha=alpha, y_range=y_range)
        result["report_global"] = rep_g
        result["lower_global"] = lo_g
        result["upper_global"] = hi_g

    return result


def reliability(y_true, y_pred, driver, *, alphas=(0.5, 0.2, 0.1, 0.05),
                method: str = "mondrian", n_bins: int = 5,
                calib_frac: float = 0.5, operating_mask=None,
                rated_kw: float | None = None) -> pd.DataFrame:
    """Nominal-vs-empirical coverage across confidence levels (a calibration
    curve). A well-calibrated interval sits on the diagonal: empirical coverage
    ≈ nominal ``1-alpha`` at every level.
    """
    recs = []
    for a in alphas:
        r = run_crisp(y_true, y_pred, driver, alpha=a, n_bins=n_bins,
                      method=method, calib_frac=calib_frac,
                      operating_mask=operating_mask, rated_kw=rated_kw,
                      compare_global=False)
        if not r["ok"]:
            continue
        rep = r["report"]
        recs.append({
            "nominal": round(1.0 - a, 3),
            "empirical": round(rep["marginal_coverage"], 3),
            "worst_slab_gap": round(rep.get("worst_slab_gap", float("nan")), 3),
            "mpiw_kw": round(rep["mpiw"], 2),
            "pinaw": round(rep["pinaw"], 3),
            "winkler": round(rep["winkler"], 2),
            "n_test": rep["n_test"],
        })
    return pd.DataFrame(recs)


def summarize(result: dict) -> str:
    """One-block human-readable summary of a ``run_crisp`` result."""
    if not result.get("ok"):
        return f"CRISP: {result.get('reason', 'failed')}"
    rep = result["report"]
    lines = [
        f"CRISP ({result['method']}, alpha={result['alpha']:.2f} → "
        f"target {rep['target']*100:.0f}% coverage)",
        f"  calib rows        : {result['n_calib']}",
        f"  test rows         : {rep['n_test']}",
        f"  marginal coverage : {rep['marginal_coverage']*100:5.1f}%  "
        f"(target {rep['target']*100:.0f}%)",
        f"  mean width (MPIW) : {rep['mpiw']:.2f} kW   PINAW {rep['pinaw']:.3f}",
        f"  Winkler score     : {rep['winkler']:.2f}",
    ]
    if "worst_slab_gap" in rep:
        lines.append(f"  worst-regime gap  : {rep['worst_slab_gap']*100:4.1f} pts")
    if "report_global" in result:
        g = result["report_global"]
        lines.append(f"  [global baseline] marginal {g['marginal_coverage']*100:5.1f}%  "
                     f"worst-regime gap {g.get('worst_slab_gap', float('nan'))*100:4.1f} pts")
    return "\n".join(lines)
