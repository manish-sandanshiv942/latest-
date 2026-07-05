"""
run_statistical_analysis.py
===========================
REFEREE-PROOFING SUITE — completes NOVELTY.md action items 2, 3, 4 and 6 on
the same real DKASC arrays used by ``run_dkasc_experiments.py``:

  Item 2 — GATE ABLATION: ACI under (a) no gates, (b) Gate 1 only,
           (c) Gate 2 only, (d) full FACL — shows the gates change the
           verdict exactly in the physically-correct cases.
  Item 3 — BASELINES: (i) legacy crisp tier side-by-side; (ii) a logistic-
           regression "confidence" trained on the same four diagnostics;
           (iii) raw attribution rank stability under bootstrap (why a
           confidence layer is needed at all).
  Item 4 — BLOCK BOOTSTRAP: moving-block bootstrap of the test block →
           distribution of the ACI; shows the crisp tier FLIPS across
           replicates while the ACI varies smoothly (the killer G5 figure).
  Item 6 — DIEBOLD–MARIANO: physics vs physics+residual squared-error loss
           differential with the Harvey–Leybourne–Newbold small-sample
           correction; p-values per array.

Output: ``results/statistical_analysis.md`` (+ ``aci_bootstrap_<ARRAY>.csv``).

Usage
-----
    python run_statistical_analysis.py                  # all arrays found
    python run_statistical_analysis.py --arrays 1A 3A   # subset
    python run_statistical_analysis.py --n-boot 300     # bootstrap size
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from src import dksac
from src import pram
from src import facl
from src import acgc

RESULTS_DIR = "results"
DATA_DIR = "data"


# ---------------------------------------------------------------------------
# Shared per-array pipeline (mirrors run_dkasc_experiments.py stages 1–3)
# ---------------------------------------------------------------------------
def load_array(tag: str) -> dict | None:
    path = os.path.join(DATA_DIR, f"dkasc_{tag}.csv")
    if not os.path.exists(path):
        return None
    hourly, _wmap, power_cols = dksac.load_dksac_frame(path)
    chans = dksac.power_channels(hourly, power_cols)
    if not chans:
        return None
    pcol, plabel, _peak = chans[0]
    d = pd.DataFrame(index=hourly.index)
    d["P_meas"] = hourly[pcol]
    for k in ("ghi", "dhi", "poa", "temp", "rh", "wind", "rain"):
        d[k] = hourly[k] if k in hourly else np.nan
    d = d.dropna(subset=["P_meas", "ghi"])
    if len(d) < 500:
        return None
    rated = dksac.estimate_rated_kw(d["P_meas"])
    p_model = dksac.model_pv_from_dksac(d, rated)
    weather = d[["ghi", "dhi", "poa", "temp", "rh", "wind", "rain"]]
    res = pram.run_pram(weather, d["P_meas"], p_model, rated, mode="measured")
    if not res.get("ok"):
        return None
    return {"tag": tag, "label": plabel, "rated": rated, "d": d,
            "res": res}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")


def _mbe(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(b[m] - a[m])) if m.any() else 0.0


def test_block_arrays(res: dict) -> dict:
    """Extract aligned test-block arrays from a PRAM result dict."""
    fit = res["fit"]
    idx_te = fit["X_te"].index
    y_true = res["ref"].reindex(idx_te).to_numpy(dtype=float)
    y_phys = res["model"].reindex(idx_te).to_numpy(dtype=float)
    r_hat = np.asarray(fit["pred"], dtype=float)
    return {"idx": idx_te, "y_true": y_true, "y_phys": y_phys, "r_hat": r_hat}


# ---------------------------------------------------------------------------
# Item 6 — Diebold–Mariano with HLN small-sample correction (h = 1)
# ---------------------------------------------------------------------------
def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> dict:
    """DM test on squared-error loss, h=1, HLN correction.

    H0: equal predictive accuracy. Negative statistic → model 2 (corrected)
    is more accurate; positive → model 1 (physics) is more accurate.
    """
    from scipy import stats as sps
    m = np.isfinite(e1) & np.isfinite(e2)
    d = e1[m] ** 2 - e2[m] ** 2
    n = d.size
    if n < 30:
        return {"ok": False, "reason": f"n={n} too small"}
    dbar = float(np.mean(d))
    # h=1 → long-run variance = gamma_0 (no autocovariance terms needed),
    # but hourly loss differentials are autocorrelated, so use a
    # Newey–West window ~ 24 h to be honest about it.
    L = 24
    g0 = float(np.mean((d - dbar) ** 2))
    lrv = g0
    for k in range(1, L + 1):
        gk = float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
        lrv += 2.0 * (1.0 - k / (L + 1)) * gk
    if lrv <= 0:
        lrv = g0
    dm = dbar / np.sqrt(lrv / n)
    # HLN small-sample correction (h=1)
    hln = dm * np.sqrt((n + 1 - 2 * 1 + 1 * (1 - 1) / n) / n)
    p = 2.0 * float(sps.t.sf(abs(hln), df=n - 1))
    return {"ok": True, "n": n, "dm": float(dm), "hln": float(hln),
            "p": p, "mean_loss_diff": dbar,
            "better": "corrected" if dbar > 0 else "physics"}


# ---------------------------------------------------------------------------
# Item 2 — gate ablation
# ---------------------------------------------------------------------------
def gate_ablation(res: dict, gate_frac: float = 0.30) -> dict:
    """ACI and gate weight under the four gate configurations.

    The ablation is anchored on what ACGC would actually gate on. Without
    Gate 2 the gate weight comes from the LOCAL (gate-block) evidence alone —
    the naive implementation — so the local ACI is the "no temporal guard"
    axis, and Gate 1 (leakage guard) is the other axis:

    none : local raw Mamdani ACI               (no Gate 1, no Gate 2)
    g1   : local ACI after the leakage guard   (Gate 1 only — pre-Gate-2 ACGC)
    g2   : min(local raw, global raw)          (Gate 2 only)
    full : min(local post-G1, global post-G1)  (what ACGC uses)
    """
    tb = test_block_arrays(res)
    n = len(tb["y_true"])
    g_end = max(50, int(round(gate_frac * n)))
    aero = facl.aerosol_consistency_score(res.get("aerosol", {}))

    glob = facl.run_facl(res)
    loc = acgc._local_facl_aci(tb["y_true"][:g_end], tb["y_phys"][:g_end],
                               tb["r_hat"][:g_end], aerosol=aero)
    variants = {
        "none": loc["aci_raw"],
        "g1":   loc["aci"],
        "g2":   min(loc["aci_raw"], glob["aci_raw"]),
        "full": min(loc["aci"], glob["aci"]),
    }
    out = {}
    for k, aci in variants.items():
        term, _label, _css = facl.verdict_for(aci)
        out[k] = {"aci": float(aci), "term": term,
                  "w": float(acgc.gate_weight(aci))}
    out["_local"] = {"aci": loc["aci"], "aci_raw": loc["aci_raw"]}
    out["_global"] = {"aci": glob["aci"], "aci_raw": glob["aci_raw"]}
    return out


# ---------------------------------------------------------------------------
# Items 3(iii) + 4 — moving-block bootstrap of the test block
# ---------------------------------------------------------------------------
def block_bootstrap(res: dict, n_boot: int, block_h: int = 72,
                    seed: int = 42, rank_sub: int = 800,
                    rank_boot: int = 60) -> dict:
    """Moving-block bootstrap of the PRAM test block.

    Per replicate: recompute the four FACL inputs → ACI (post Gate 1) and
    the legacy crisp tier. For the first ``rank_boot`` replicates also
    recompute a cheap permutation importance (subsample, 1 repeat) to
    measure raw top-driver rank stability (item 3iii).
    """
    from sklearn.inspection import permutation_importance

    rng = np.random.default_rng(seed)
    tb = test_block_arrays(res)
    y_true, y_phys, r_hat = tb["y_true"], tb["y_phys"], tb["r_hat"]
    X_te, y_te = res["fit"]["X_te"], res["fit"]["y_te"]
    model = res["fit"]["model"]
    aero = facl.aerosol_consistency_score(res.get("aerosol", {}))
    n = len(y_true)
    n_blocks = int(np.ceil(n / block_h))

    acis, tiers, terms, top_drivers = [], [], [], []
    feat_names = list(X_te.columns)
    y_te_arr = np.asarray(y_te, dtype=float)

    for b in range(n_boot):
        starts = rng.integers(0, max(1, n - block_h), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_h, n))
                              for s in starts])[:n]
        yt, yp, rh = y_true[idx], y_phys[idx], r_hat[idx]
        corr = yp + rh
        rmse_p, rmse_c = _rmse(yt, yp), _rmse(yt, corr)
        impr = (1.0 - rmse_c / rmse_p) * 100.0 if rmse_p > 0 else 0.0
        resid = yt - yp
        m = np.isfinite(resid) & np.isfinite(rh)
        if m.sum() >= 10 and np.var(resid[m]) > 1e-12:
            r2 = 1.0 - np.sum((resid[m] - rh[m]) ** 2) / \
                       np.sum((resid[m] - np.mean(resid[m])) ** 2)
        else:
            r2 = 0.0
        bias = abs(_mbe(yt, yp)) - abs(_mbe(yt, corr))
        out = facl._SYSTEM.infer({"improvement": float(impr),
                                  "residual_r2": float(np.clip(r2, -1, 1)),
                                  "aerosol": aero, "bias": float(bias)})
        mu = out["memberships"]["residual_r2"].get("suspicious", 0.0)
        aci = out["aci"] + mu * (facl.LEAK_GUARD_ANCHOR - out["aci"])
        acis.append(float(aci))
        terms.append(facl.verdict_for(aci)[0])
        tiers.append(facl.crisp_reference(float(impr), float(r2)))

        if b < rank_boot:
            sub = rng.choice(idx, size=min(rank_sub, len(idx)), replace=False)
            try:
                pi = permutation_importance(
                    model, X_te.iloc[sub], y_te_arr[sub],
                    n_repeats=1, random_state=int(rng.integers(1 << 30)),
                    n_jobs=-1)
                top_drivers.append(feat_names[int(np.argmax(pi.importances_mean))])
            except Exception:
                pass

    acis = np.asarray(acis)
    tier_counts = pd.Series(tiers).value_counts().to_dict()
    term_counts = pd.Series(terms).value_counts().to_dict()
    top_counts = pd.Series(top_drivers).value_counts().to_dict() \
        if top_drivers else {}
    top_share = (max(top_counts.values()) / sum(top_counts.values())) \
        if top_counts else float("nan")
    return {
        "n_boot": n_boot, "block_h": block_h,
        "aci_mean": float(np.mean(acis)), "aci_std": float(np.std(acis)),
        "aci_q05": float(np.quantile(acis, 0.05)),
        "aci_q95": float(np.quantile(acis, 0.95)),
        "acis": acis,
        "crisp_tier_counts": tier_counts,
        "facl_term_counts": term_counts,
        "crisp_flip_rate": 1.0 - max(tier_counts.values()) / len(tiers),
        "facl_flip_rate": 1.0 - max(term_counts.values()) / len(terms),
        "top_driver_counts": top_counts,
        "top_driver_stability": float(top_share),
    }


# ---------------------------------------------------------------------------
# Item 3(ii) — logistic-regression confidence baseline
# ---------------------------------------------------------------------------
def logistic_baseline(res: dict, n_boot: int = 120, block_h: int = 72,
                      seed: int = 7) -> dict:
    """Train a logistic 'confidence' on the SAME four diagnostics.

    Per replicate: diagnostics computed on one bootstrap half, outcome
    ('did the correction actually reduce RMSE?') evaluated on the other
    half. Logistic regression then predicts the outcome from the
    diagnostics — the transparent-vs-blackbox accuracy comparison.
    FACL's implicit prediction: ACI ≥ 45 ('moderate'+) → correction helps.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    tb = test_block_arrays(res)
    y_true, y_phys, r_hat = tb["y_true"], tb["y_phys"], tb["r_hat"]
    aero = facl.aerosol_consistency_score(res.get("aerosol", {}))
    n = len(y_true)
    n_blocks = int(np.ceil(n / block_h))

    Xd, yd, facl_pred = [], [], []
    for _b in range(n_boot):
        starts = rng.integers(0, max(1, n - block_h), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_h, n))
                              for s in starts])[:n]
        half = len(idx) // 2
        ia, ib = idx[:half], idx[half:]
        # diagnostics on half A
        yt, yp, rh = y_true[ia], y_phys[ia], r_hat[ia]
        corr = yp + rh
        rmse_p, rmse_c = _rmse(yt, yp), _rmse(yt, corr)
        impr = (1.0 - rmse_c / rmse_p) * 100.0 if rmse_p > 0 else 0.0
        resid = yt - yp
        m = np.isfinite(resid) & np.isfinite(rh)
        r2 = (1.0 - np.sum((resid[m] - rh[m]) ** 2)
              / np.sum((resid[m] - np.mean(resid[m])) ** 2)) \
            if m.sum() >= 10 and np.var(resid[m]) > 1e-12 else 0.0
        bias = abs(_mbe(yt, yp)) - abs(_mbe(yt, corr))
        # outcome on half B
        ytb, ypb, rhb = y_true[ib], y_phys[ib], r_hat[ib]
        helps = _rmse(ytb, ypb + rhb) < _rmse(ytb, ypb)
        Xd.append([impr, np.clip(r2, -1, 1), aero, bias])
        yd.append(int(helps))
        out = facl._SYSTEM.infer({"improvement": float(impr),
                                  "residual_r2": float(np.clip(r2, -1, 1)),
                                  "aerosol": aero, "bias": float(bias)})
        mu = out["memberships"]["residual_r2"].get("suspicious", 0.0)
        aci = out["aci"] + mu * (facl.LEAK_GUARD_ANCHOR - out["aci"])
        facl_pred.append(int(aci >= 45.0))

    Xd = np.asarray(Xd)
    yd = np.asarray(yd)
    facl_pred = np.asarray(facl_pred)
    if len(np.unique(yd)) < 2:
        return {"ok": False,
                "reason": f"outcome is constant across replicates "
                          f"(helps={yd.mean():.0%}) — logistic fit degenerate",
                "base_rate": float(yd.mean()),
                "facl_acc": float(np.mean(facl_pred == yd))}
    # chronological-ish split over replicates
    cut = int(0.7 * len(yd))
    lr = LogisticRegression(max_iter=1000).fit(Xd[:cut], yd[:cut])
    return {"ok": True,
            "base_rate": float(yd.mean()),
            "logit_acc": float(lr.score(Xd[cut:], yd[cut:])),
            "facl_acc": float(np.mean(facl_pred[cut:] == yd[cut:])),
            "coef": dict(zip(["improvement", "residual_r2", "aerosol",
                              "bias"], lr.coef_[0].round(3).tolist()))}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Referee-proofing statistics "
                                             "(ablation, bootstrap, DM, baselines)")
    ap.add_argument("--arrays", nargs="+", default=["1A", "1C", "3A", "4A"])
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--block-h", type=int, default=72)
    args = ap.parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    L = ["# Statistical analysis — ablation, bootstrap, significance, baselines\n",
         "Auto-generated by `run_statistical_analysis.py` on the real DKASC "
         "arrays. Completes NOVELTY.md action items 2, 3, 4 and 6.\n"]

    for tag in args.arrays:
        print(f"===== ARRAY {tag} =====")
        ar = load_array(tag)
        if ar is None:
            print(f"  [skip] data/dkasc_{tag}.csv missing or unusable")
            L.append(f"## Array {tag}\n\n*(skipped — no usable data)*\n")
            continue
        res = ar["res"]
        tb = test_block_arrays(res)

        # --- Item 6: Diebold–Mariano ---------------------------------------
        e_phys = tb["y_true"] - tb["y_phys"]
        e_corr = tb["y_true"] - (tb["y_phys"] + tb["r_hat"])
        dm = diebold_mariano(e_phys, e_corr)
        print(f"  DM: {dm}")

        # --- Item 2: gate ablation ------------------------------------------
        ab = gate_ablation(res)
        print(f"  ablation: " + ", ".join(
            f"{k}={v['aci']:.1f}({v['term']},w={v['w']:.2f})"
            for k, v in ab.items() if not k.startswith("_")))

        # --- Items 3(iii)+4: block bootstrap --------------------------------
        bs = block_bootstrap(res, n_boot=args.n_boot, block_h=args.block_h)
        pd.DataFrame({"aci": bs["acis"]}).to_csv(
            os.path.join(RESULTS_DIR, f"aci_bootstrap_{tag}.csv"), index=False)
        print(f"  bootstrap ACI: {bs['aci_mean']:.1f} ± {bs['aci_std']:.1f} "
              f"[{bs['aci_q05']:.1f}, {bs['aci_q95']:.1f}]  "
              f"crisp flips {bs['crisp_flip_rate']:.0%} vs FACL flips "
              f"{bs['facl_flip_rate']:.0%}  top-driver stability "
              f"{bs['top_driver_stability']:.0%}")

        # --- Item 3(ii): logistic baseline ----------------------------------
        lb = logistic_baseline(res)
        print(f"  logistic baseline: {lb}")

        # --- report ----------------------------------------------------------
        L.append(f"## Array {tag} — {ar['label']} (rated ≈ {ar['rated']:.1f} kW)\n")

        L.append("### Item 6 — Diebold–Mariano (physics vs corrected, "
                 "squared loss, NW-24h, HLN)\n")
        if dm.get("ok"):
            L.append(f"- n = {dm['n']}, DM = {dm['dm']:+.2f}, "
                     f"HLN = {dm['hln']:+.2f}, **p = {dm['p']:.2e}** — "
                     f"{'significant' if dm['p'] < 0.05 else 'not significant'}; "
                     f"more accurate: **{dm['better']}**")
            L.append(f"- Interpretation: the correction "
                     f"{'significantly HURTS' if dm['better'] == 'physics' and dm['p'] < 0.05 else 'significantly helps' if dm['p'] < 0.05 else 'is statistically indistinguishable from physics-only'} "
                     f"on this array's held-out block — which is exactly what the "
                     f"gated w should respond to.\n")
        else:
            L.append(f"- failed: {dm.get('reason')}\n")

        L.append("### Item 2 — Gate ablation (ACI → w)\n")
        L.append("| Configuration | ACI | Verdict | Gate weight w |")
        L.append("|---|---|---|---|")
        names = {"none": "No gates (local raw Mamdani — naive gate)",
                 "g1": "Gate 1 only (leakage guard, local)",
                 "g2": "Gate 2 only (min t-norm vs global raw)",
                 "full": "Full FACL (both gates) — used by ACGC"}
        for k in ("none", "g1", "g2", "full"):
            v = ab[k]
            L.append(f"| {names[k]} | {v['aci']:.1f} | {v['term']} | "
                     f"{v['w']:.3f} |")
        L.append(f"\nLocal (gate-block) ACI raw/post-G1: "
                 f"{ab['_local']['aci_raw']:.1f} / {ab['_local']['aci']:.1f}; "
                 f"global: {ab['_global']['aci_raw']:.1f} / "
                 f"{ab['_global']['aci']:.1f}.\n")

        L.append(f"### Item 4 — Moving-block bootstrap of the test block "
                 f"(B = {bs['n_boot']}, block = {bs['block_h']} h)\n")
        L.append(f"- ACI: mean {bs['aci_mean']:.1f} ± {bs['aci_std']:.1f}, "
                 f"90% interval [{bs['aci_q05']:.1f}, {bs['aci_q95']:.1f}]")
        L.append(f"- Legacy **crisp tier flips in {bs['crisp_flip_rate']:.0%}** "
                 f"of replicates ({bs['crisp_tier_counts']}); FACL verdict "
                 f"flips in {bs['facl_flip_rate']:.0%} "
                 f"({bs['facl_term_counts']})")
        L.append(f"- Raw distribution: `aci_bootstrap_{tag}.csv`\n")

        L.append("### Item 3 — Baselines\n")
        L.append(f"- (i) Legacy crisp tier: "
                 f"**{res['classify']['tier']}** vs FACL "
                 f"{ab['g1']['aci']:.1f} ({ab['g1']['term']})")
        if lb.get("ok"):
            L.append(f"- (ii) Logistic confidence on the same 4 diagnostics: "
                     f"accuracy {lb['logit_acc']:.0%} vs FACL-threshold "
                     f"{lb['facl_acc']:.0%} (base rate "
                     f"{lb['base_rate']:.0%}); coefficients {lb['coef']}")
        else:
            L.append(f"- (ii) Logistic confidence: {lb.get('reason')} "
                     f"(base rate {lb.get('base_rate', float('nan')):.0%}, "
                     f"FACL-threshold accuracy "
                     f"{lb.get('facl_acc', float('nan')):.0%})")
        L.append(f"- (iii) Raw top-driver stability under bootstrap: "
                 f"the modal top driver holds in only "
                 f"**{bs['top_driver_stability']:.0%}** of replicates "
                 f"({bs['top_driver_counts']}) — attribution rankings are "
                 f"sampling-sensitive, which is precisely why a confidence "
                 f"layer over the attribution is needed (Gap G1).\n")

    out = os.path.join(RESULTS_DIR, "statistical_analysis.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
