"""
RCA — Regime-Conditional Attribution
====================================

The new research-tier layer: **attribution that changes with the sky.**

Everywhere else in the stack, feature-impact rankings are *global* — one
permutation-importance table averaged over the whole evaluation period. But the
true driver ranking is regime-dependent: clear-sky hours are
temperature/thermal-derate dominated, overcast hours are diffuse-fraction
dominated, and transition hours mix both. A single global ranking silently
averages over these physically distinct operating regimes, and no published
PV-XAI framework conditions the attribution itself on the same regime taxonomy
used for conditional uncertainty.

What RCA does
-------------
1. **Reuses the CRISP Mondrian regime bins** (quantile edges on the regime
   driver — GHI/POA for PV, hub wind for wind) so that attribution and
   uncertainty are conditioned on the *same* physically meaningful taxonomy.
2. Computes **per-regime permutation importance** on the held-out test rows of
   an already-trained model (no refitting, no extra leakage surface).
3. Summarises cross-regime disagreement in a single bounded diagnostic, the
   **Attribution Instability Index (AII ∈ [0, 1])**: 0 = the driver ranking is
   identical in every regime (a global attribution is trustworthy), 1 = the
   rankings are maximally discordant (a global attribution is misleading).
4. Exposes AII as a **fifth FACL input**: `aii_to_stability_weight` maps AII to
   a multiplicative confidence retention factor in (0, 1] that callers apply to
   the ACI, so regime-unstable attributions lose confidence *continuously*
   (same convex-guard pattern as FACL Gate 1 — formal properties P11–P13).

Honest novelty scope
--------------------
Permutation importance, Kendall's tau, and quantile binning are established
tools. The contribution is (a) conditioning the attribution on the conformal
regime taxonomy so uncertainty and explanation share one lens, (b) the bounded
AII diagnostic with verified formal properties, and (c) closing the loop into
the fuzzy confidence layer. Applied/framework-integration tier, same as
PRAM/FACL/CRISP — but the *combination* is unreported in the PV literature.

This module is pure (numpy/pandas/sklearn); callers handle rendering.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.crisp import assign_regime, regime_edges

RS = 42

# --- AII → confidence retention factor (the FACL coupling) -----------------
# Convex-guard anchors, mirroring FACL Gate 1: a perfectly stable attribution
# retains full confidence; a maximally unstable one retains STAB_FLOOR.
STAB_FLOOR = 0.40          # retention at AII = 1 (never hard-zero: the global
                           # ranking is degraded, not meaningless)
AII_BENIGN = 0.25          # below this, no penalty at all (rank noise)
AII_SEVERE = 0.75          # above this, the full floor penalty applies


# ---------------------------------------------------------------------------
# Rank-agreement primitive
# ---------------------------------------------------------------------------
def _weighted_kendall(a: np.ndarray, b: np.ndarray) -> float:
    """Top-weighted Kendall's tau between two importance vectors.

    Uses scipy's hyperbolic weighted tau (rank-1 disagreements matter most —
    exactly right for "did the HEADLINE drivers change between regimes").
    Falls back to plain Spearman if scipy is unavailable. Returns a value in
    [-1, 1]; NaN-safe (returns 0.0 on degenerate input).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    try:
        from scipy.stats import weightedtau
        t = float(weightedtau(a, b).statistic)
    except Exception:                                    # pragma: no cover
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        ca = ra - ra.mean()
        cb = rb - rb.mean()
        denom = float(np.sqrt((ca ** 2).sum() * (cb ** 2).sum()))
        t = float((ca * cb).sum() / denom) if denom > 0 else 0.0
    return t if np.isfinite(t) else 0.0


# ---------------------------------------------------------------------------
# Per-regime permutation importance
# ---------------------------------------------------------------------------
def regime_importance(model, X_test: pd.DataFrame, y_test,
                      driver_test, *, n_bins: int = 4,
                      min_bin: int = 30, n_repeats: int = 5,
                      features: Sequence[str] | None = None,
                      edges: np.ndarray | None = None) -> dict:
    """Permutation importance computed separately inside each operating regime.

    Parameters
    ----------
    model : fitted sklearn-compatible regressor (NOT refitted here)
    X_test, y_test : held-out test rows (time-ordered, as elsewhere)
    driver_test : the regime variable aligned to the test rows (GHI/POA or
        hub wind) — the same driver CRISP bins on
    n_bins : number of quantile regimes (kept small so each bin stays populated)
    min_bin : regimes with fewer test rows are merged into their neighbour
    edges : optional pre-computed interior edges (pass the CRISP calibrator's
        ``edges`` to condition attribution and uncertainty on the *identical*
        taxonomy — the recommended usage)

    Returns dict with ``edges``, ``bins`` (list of per-regime records each
    holding a sorted importance DataFrame), and ``matrix`` (features × regimes
    importance-share matrix used by the AII).
    """
    feats = list(features) if features is not None else list(X_test.columns)
    drv = np.asarray(driver_test, dtype=float)
    if edges is None:
        edges = regime_edges(drv, n_bins=n_bins, method="quantile")
    bins = assign_regime(drv, edges)

    # merge undersized regimes downward so every evaluated bin is meaningful
    labels = np.unique(bins)
    counts = {int(k): int((bins == k).sum()) for k in labels}
    remap = {}
    prev_ok = None
    for k in sorted(counts):
        if counts[k] >= min_bin:
            prev_ok = k
            remap[k] = k
        else:
            remap[k] = prev_ok if prev_ok is not None else k
    bins = np.array([remap[int(b)] for b in bins])

    records, cols = [], []
    for k in sorted(np.unique(bins)):
        sel = bins == k
        n_k = int(sel.sum())
        if n_k < min_bin:
            continue
        res = permutation_importance(
            model, X_test.iloc[sel], np.asarray(y_test)[sel],
            n_repeats=n_repeats, random_state=RS, n_jobs=1,
        )
        imp = np.clip(res.importances_mean, 0, None)
        total = float(imp.sum())
        share = imp / total if total > 0 else np.zeros_like(imp)
        df = (pd.DataFrame({"Feature": feats, "Importance": imp, "Share": share})
              .sort_values("Importance", ascending=False)
              .reset_index(drop=True))
        records.append({
            "regime": int(k), "n": n_k,
            "driver_lo": float(np.min(drv[sel])),
            "driver_hi": float(np.max(drv[sel])),
            "importance": df,
        })
        cols.append(share)

    matrix = (pd.DataFrame(np.column_stack(cols), index=feats,
                           columns=[f"regime_{r['regime']}" for r in records])
              if cols else pd.DataFrame(index=feats))
    return {"ok": len(records) >= 2, "edges": edges, "bins": records,
            "matrix": matrix,
            "reason": None if len(records) >= 2 else
            "fewer than 2 populated regimes — cannot assess stability"}


# ---------------------------------------------------------------------------
# Attribution Instability Index
# ---------------------------------------------------------------------------
def attribution_instability(reg: dict) -> dict:
    """The bounded cross-regime disagreement diagnostic.

    AII = (1 - mean pairwise top-weighted Kendall tau between the per-regime
    importance vectors) / 2, clipped to [0, 1]:

      * all regime rankings identical  → tau = 1 → AII = 0
      * uncorrelated rankings          → tau ≈ 0 → AII ≈ 0.5
      * systematically reversed        → tau = -1 → AII = 1

    Also reports the least-agreeing regime pair and each feature's cross-regime
    share range (which features actually drive the instability).
    """
    if not reg.get("ok"):
        return {"ok": False, "reason": reg.get("reason", "regime run failed")}
    M = reg["matrix"].to_numpy()
    names = list(reg["matrix"].columns)
    p = M.shape[1]
    taus, pairs = [], []
    for i in range(p):
        for j in range(i + 1, p):
            t = _weighted_kendall(M[:, i], M[:, j])
            taus.append(t)
            pairs.append({"pair": (names[i], names[j]), "tau": round(t, 4)})
    mean_tau = float(np.mean(taus))
    aii = float(np.clip((1.0 - mean_tau) / 2.0, 0.0, 1.0))
    worst = min(pairs, key=lambda r: r["tau"]) if pairs else None

    spread = (reg["matrix"].max(axis=1) - reg["matrix"].min(axis=1))
    unstable = (spread.sort_values(ascending=False).head(10)
                .rename("ShareRange").reset_index()
                .rename(columns={"index": "Feature"}))
    return {"ok": True, "aii": aii, "mean_tau": mean_tau,
            "pairwise": pairs, "worst_pair": worst,
            "unstable_features": unstable}


# ---------------------------------------------------------------------------
# FACL coupling: AII → confidence retention factor
# ---------------------------------------------------------------------------
def aii_to_stability_weight(aii: float) -> float:
    """Map AII ∈ [0,1] to a multiplicative ACI retention factor s ∈ [STAB_FLOOR, 1].

    Smootherstep between the benign and severe anchors (C¹, monotone
    non-increasing, explicitly Lipschitz — verified as P11–P13):

        s(aii) = 1                                  for aii <= AII_BENIGN
        s(aii) = STAB_FLOOR                         for aii >= AII_SEVERE
        s(aii) = 1 - (1-STAB_FLOOR)·(3t² - 2t³)     in between,
                 t = (aii - AII_BENIGN)/(AII_SEVERE - AII_BENIGN)

    Callers apply ``ACI_regime_aware = s(AII) · ACI`` — the same convex-guard
    idiom as FACL Gate 1, so all existing FACL bounds are preserved (a product
    of quantities in [0,1]·[0,100] stays in [0,100]: P1 survives).
    """
    a = float(np.clip(aii, 0.0, 1.0))
    if a <= AII_BENIGN:
        return 1.0
    if a >= AII_SEVERE:
        return STAB_FLOOR
    t = (a - AII_BENIGN) / (AII_SEVERE - AII_BENIGN)
    return 1.0 - (1.0 - STAB_FLOOR) * (3.0 * t ** 2 - 2.0 * t ** 3)


# ---------------------------------------------------------------------------
# End-to-end convenience runner
# ---------------------------------------------------------------------------
def run_rca(model, X_test: pd.DataFrame, y_test, driver_test, *,
            n_bins: int = 4, min_bin: int = 30, n_repeats: int = 5,
            edges: np.ndarray | None = None,
            aci: float | None = None) -> dict:
    """Full pipeline: regime importance → AII → stability weight (→ gated ACI).

    Pass ``edges`` from a fitted CRISP calibrator to share the taxonomy, and
    ``aci`` from a FACL run to obtain the regime-aware confidence directly.
    """
    reg = regime_importance(model, X_test, y_test, driver_test,
                            n_bins=n_bins, min_bin=min_bin,
                            n_repeats=n_repeats, edges=edges)
    if not reg["ok"]:
        return {"ok": False, "reason": reg["reason"]}
    inst = attribution_instability(reg)
    if not inst["ok"]:
        return {"ok": False, "reason": inst["reason"]}
    s = aii_to_stability_weight(inst["aii"])
    out = {"ok": True, "regimes": reg, "instability": inst,
           "stability_weight": s}
    if aci is not None:
        out["aci_input"] = float(aci)
        out["aci_regime_aware"] = float(np.clip(s * float(aci), 0.0, 100.0))
    return out


def summarize(result: dict) -> str:
    """One-block human-readable summary of a ``run_rca`` result."""
    if not result.get("ok"):
        return f"RCA: {result.get('reason', 'failed')}"
    inst = result["instability"]
    reg = result["regimes"]
    lines = [
        f"RCA (regime-conditional attribution, {len(reg['bins'])} regimes)",
        f"  mean pairwise weighted tau : {inst['mean_tau']:+.3f}",
        f"  AII (instability, 0-1)     : {inst['aii']:.3f}",
        f"  stability weight s(AII)    : {result['stability_weight']:.3f}",
    ]
    if inst.get("worst_pair"):
        wp = inst["worst_pair"]
        lines.append(f"  least-agreeing regimes     : {wp['pair'][0]} vs "
                     f"{wp['pair'][1]} (tau {wp['tau']:+.3f})")
    for r in reg["bins"]:
        top = r["importance"].iloc[0]
        lines.append(f"  regime {r['regime']} "
                     f"[{r['driver_lo']:.0f}–{r['driver_hi']:.0f}] "
                     f"n={r['n']}: top driver = {top['Feature']} "
                     f"(share {top['Share']*100:.1f}%)")
    if "aci_regime_aware" in result:
        lines.append(f"  ACI {result['aci_input']:.1f} → regime-aware "
                     f"{result['aci_regime_aware']:.1f}")
    return "\n".join(lines)
