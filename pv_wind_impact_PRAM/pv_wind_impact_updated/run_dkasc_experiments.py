"""
run_dkasc_experiments.py
========================
THE PRE-SUBMISSION EXPERIMENT: runs the full novel stack — PRAM → FACL →
ACGC → RCA — on **real metered DKASC generation** and writes the paper's
headline results tables.

This is the harness that upgrades ACGC (contribution C6) and RCA
(contribution C7) from "validated on synthetic harnesses" to "validated on
metered generation", which is the single biggest remaining blocker identified
in NOVELTY.md (action item 8).

What it produces (results/ directory)
-------------------------------------
1. ``dkasc_results.md``   — every headline table for the paper:
     * data provenance (file, channel, rows, period)
     * PRAM: physics vs corrected RMSE, residual R², aerosol signature, tier
     * FACL: the four inputs, ACI, leak suspicion, verdict
     * ACGC: the 4-variant comparison (acgc / physics / full / hard) —
       RMSE, marginal coverage, MPIW, Winkler, worst-regime gap
     * RCA : per-regime top drivers, AII, s(AII), regime-aware ACI
2. ``rca_regime_importance.csv`` — full features × regimes share matrix.
3. ``acgc_intervals.csv``        — test-block truth, gated prediction, bounds.

Usage
-----
    python run_dkasc_experiments.py                    # auto-locate CSV
    python run_dkasc_experiments.py --csv path/to/dkasc.csv
    python run_dkasc_experiments.py --channel 2        # pick power channel
    python run_dkasc_experiments.py --list-channels    # see what's available

Get the data (free, no account): https://dkasolarcentre.com.au/download
→ Alice Springs → pick an array (e.g. 1B DKA Centre, mono-Si) → CSV download
→ save it into this project's ``data/`` directory.
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
from src import rca


RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
def _find_csv(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    return dksac.find_site_csv("alice") or dksac.find_site_csv("yulara")


def _fail(msg: str, code: int = 2) -> int:
    print(f"\n[ABORT] {msg}\n")
    return code


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DKASC metered-data experiments "
                                             "(PRAM → FACL → ACGC → RCA)")
    ap.add_argument("--csv", default=None, help="path to a DKASC CSV export")
    ap.add_argument("--channel", type=int, default=0,
                    help="index of the measured power channel (see --list-channels)")
    ap.add_argument("--list-channels", action="store_true",
                    help="list selectable measured PV channels and exit")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="target miscoverage for the conformal intervals")
    ap.add_argument("--n-bins", type=int, default=4,
                    help="Mondrian regime bins (shared by ACGC and RCA)")
    args = ap.parse_args(argv)

    path = _find_csv(args.csv)
    if path is None:
        return _fail(
            "No DKASC CSV found. Download real metered data (free) from\n"
            "  https://dkasolarcentre.com.au/download?location=alice-springs\n"
            "and save it in this project's data/ directory (any filename\n"
            "containing 'dkasc', 'alice' or 'DKA'), or pass --csv <path>.")

    print(f"[1/6] Loading DKASC export: {path}")
    hourly, wmap, power_cols = dksac.load_dksac_frame(path)
    chans = dksac.power_channels(hourly, power_cols)
    if not chans:
        return _fail("The CSV contains no data-bearing measured power channel.")

    if args.list_channels:
        print("\nSelectable measured PV channels:")
        for i, (col, label, peak) in enumerate(chans):
            print(f"  [{i}] {label:<40} peak ≈ {peak:.1f} kW   ({col})")
        return 0

    if not (0 <= args.channel < len(chans)):
        return _fail(f"--channel {args.channel} out of range (0..{len(chans)-1}); "
                     f"run with --list-channels to see options.")
    pcol, plabel, _peak = chans[args.channel]

    d = pd.DataFrame(index=hourly.index)
    d["P_meas"] = hourly[pcol]
    for k in ("ghi", "dhi", "poa", "temp", "rh", "wind", "rain"):
        d[k] = hourly[k] if k in hourly else np.nan
    d = d.dropna(subset=["P_meas", "ghi"])
    if len(d) < 500:
        return _fail(f"Only {len(d)} usable hourly rows — need ≥ 500 for the "
                     f"three-block split. Export a longer period.")

    rated = dksac.estimate_rated_kw(d["P_meas"])
    print(f"      channel = {plabel}   rated ≈ {rated:.1f} kW   "
          f"rows = {len(d)}   period = {d.index.min()} → {d.index.max()}")

    # ---- physics baseline --------------------------------------------------
    print("[2/6] Physics baseline (NOCT thermal model on measured weather)")
    p_model = dksac.model_pv_from_dksac(d, rated)
    base = dksac.validation_metrics(d["P_meas"], p_model, rated)
    print(f"      physics-only: n={base['n']}  RMSE={base['rmse']:.2f} kW  "
          f"nRMSE={base['nrmse_pct']:.1f}%  R²={base['r2']:.3f}")

    # ---- PRAM ---------------------------------------------------------------
    print("[3/6] PRAM (physics-residual attribution)")
    weather = d[["ghi", "dhi", "poa", "temp", "rh", "wind", "rain"]]
    res = pram.run_pram(weather, d["P_meas"], p_model, rated, mode="measured")
    if not res["ok"]:
        return _fail("PRAM failed: " + res["reason"])
    fit, impr = res["fit"], res["impr"]
    print(f"      RMSE physics {impr['rmse_phys']:.3f} → corrected "
          f"{impr['rmse_corr']:.3f} kW ({impr['improvement_pct']:+.1f}%)   "
          f"residual R² = {fit['resid_r2']:.3f}   tier = {res['classify']['tier']}")

    # ---- FACL ---------------------------------------------------------------
    print("[4/6] FACL (fuzzy attribution-confidence)")
    fz = facl.run_facl(res)
    aci = fz["aci"]
    print(f"      ACI = {aci:.1f}/100 ({fz['term']})   "
          f"leak suspicion = {fz.get('leak_suspicion', 0.0):.2f}")

    # ---- ACGC ---------------------------------------------------------------
    print("[5/6] ACGC (confidence-gated conformal correction) — headline table")
    driver = d["poa"].where(d["poa"].notna(), d["ghi"])
    ra = acgc.run_acgc(res, driver, alpha=args.alpha, rated_kw=rated,
                       n_bins=args.n_bins)
    if not ra.get("ok"):
        return _fail("ACGC failed: " + ra.get("reason", "unknown"))
    print("\n" + acgc.summarize(ra) + "\n")

    # ---- RCA ----------------------------------------------------------------
    print("[6/6] RCA (regime-conditional attribution) — shared Mondrian taxonomy")
    drv_test = driver.reindex(fit["X_te"].index).to_numpy(dtype=float)
    edges = ra["variants"]["acgc"]["calibrator"].get("edges")
    rr = rca.run_rca(fit["model"], fit["X_te"], fit["y_te"], drv_test,
                     n_bins=args.n_bins, edges=edges, aci=aci)
    if not rr.get("ok"):
        return _fail("RCA failed: " + rr.get("reason", "unknown"))
    print("\n" + rca.summarize(rr) + "\n")

    # ---- write the results pack ---------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rr["regimes"]["matrix"].to_csv(
        os.path.join(RESULTS_DIR, "rca_regime_importance.csv"))

    v = ra["variants"]["acgc"]
    pd.DataFrame({
        "y_true": ra["y_test"], "driver": ra["driver_test"],
        "yhat_gated": v["yhat"], "lower": v["lower"], "upper": v["upper"],
    }).to_csv(os.path.join(RESULTS_DIR, "acgc_intervals.csv"), index=False)

    md = _results_markdown(path, plabel, rated, d, base, res, fz, ra, rr,
                           args.alpha)
    out_md = os.path.join(RESULTS_DIR, "dkasc_results.md")
    with open(out_md, "w") as f:
        f.write(md)

    print(f"Results pack written:\n"
          f"  {out_md}\n"
          f"  {RESULTS_DIR}/rca_regime_importance.csv\n"
          f"  {RESULTS_DIR}/acgc_intervals.csv")
    return 0


# ---------------------------------------------------------------------------
def _results_markdown(path, plabel, rated, d, base, res, fz, ra, rr,
                      alpha) -> str:
    fit, impr, aero = res["fit"], res["impr"], res["aerosol"]
    L = []
    L.append("# DKASC metered-generation results (PRAM → FACL → ACGC → RCA)\n")
    L.append("Auto-generated by `run_dkasc_experiments.py`. These are the "
             "headline tables for the paper's real-data validation section.\n")

    L.append("## 1. Data provenance\n")
    L.append(f"- File: `{os.path.basename(path)}`")
    L.append(f"- Channel: **{plabel}** (rated ≈ {rated:.1f} kW, "
             f"99.5th-percentile estimate)")
    L.append(f"- Hourly rows: {len(d)} "
             f"({d.index.min():%Y-%m-%d} → {d.index.max():%Y-%m-%d})")
    L.append(f"- Physics-only baseline: RMSE {base['rmse']:.2f} kW, "
             f"nRMSE {base['nrmse_pct']:.1f}%, R² {base['r2']:.3f} "
             f"(operating hours, n={base['n']})\n")

    L.append("## 2. PRAM — physics-residual attribution\n")
    L.append("| Signal | Value |")
    L.append("|---|---|")
    L.append(f"| RMSE physics-only (test) | {impr['rmse_phys']:.3f} kW |")
    L.append(f"| RMSE physics + residual (test) | {impr['rmse_corr']:.3f} kW "
             f"({impr['improvement_pct']:+.1f}%) |")
    L.append(f"| Residual model R² (held-out) | {fit['resid_r2']:.3f} |")
    L.append(f"| Bias MBE physics → corrected | {impr['mbe_phys']:+.3f} → "
             f"{impr['mbe_corr']:+.3f} kW |")
    if aero.get("available"):
        L.append(f"| Aerosol signature | {aero['feature']} @ rank "
                 f"{aero['rank']}"
                 + ("" if aero.get("negative") is None else
                    (", negative" if aero["negative"] else ", positive")) + " |")
    L.append(f"| Crisp tier | {res['classify']['tier']} |\n")

    L.append("## 3. FACL — fuzzy attribution-confidence\n")
    L.append("| Input | Value |")
    L.append("|---|---|")
    for k, val in fz.get("inputs", {}).items():
        L.append(f"| {k} | {val:.3f} |")
    L.append(f"| **ACI** | **{fz['aci']:.1f} / 100 ({fz['term']})** |")
    L.append(f"| Leak suspicion (Gate 1) | "
             f"{fz.get('leak_suspicion', 0.0):.2f} |\n")

    L.append(f"## 4. ACGC — confidence-gated conformal correction "
             f"(alpha = {alpha:.2f})\n")
    g = ra["gate"]
    L.append(f"Gate-block ACI = {g['aci']:.1f} ({g['term']}) → "
             f"**w = {ra['w']:.3f}**; crisp switch w_hard = "
             f"{ra['w_hard']:.0f}. Blocks: gate {ra['n_gate']} / "
             f"calib {ra['n_calib']} / test {ra['n_test']}.\n")
    L.append("| Variant | w | RMSE (kW) | Coverage | MPIW | Winkler | "
             "Worst-regime gap |")
    L.append("|---|---|---|---|---|---|---|")
    for name in ("acgc", "physics", "full", "hard"):
        vv = ra["variants"][name]
        rep = vv["report"]
        L.append(f"| {name} | {vv['w']:.2f} | {vv['rmse']:.3f} | "
                 f"{rep['marginal_coverage']*100:.1f}% | {rep['mpiw']:.3f} | "
                 f"{rep['winkler']:.3f} | "
                 f"{rep.get('worst_slab_gap', float('nan'))*100:.1f} pp |")
    L.append("")

    L.append("## 5. RCA — regime-conditional attribution\n")
    inst = rr["instability"]
    L.append(f"- Mean pairwise weighted Kendall tau: {inst['mean_tau']:+.3f}")
    L.append(f"- **AII = {inst['aii']:.3f}**, retention s(AII) = "
             f"{rr['stability_weight']:.3f}")
    if "aci_regime_aware" in rr:
        L.append(f"- ACI {rr['aci_input']:.1f} → **regime-aware "
                 f"{rr['aci_regime_aware']:.1f}**")
    if inst.get("worst_pair"):
        wp = inst["worst_pair"]
        L.append(f"- Least-agreeing regimes: {wp['pair'][0]} vs "
                 f"{wp['pair'][1]} (tau {wp['tau']:+.3f})")
    L.append("\n| Regime | Driver range (W/m²) | n | Top driver | Share |")
    L.append("|---|---|---|---|---|")
    for r in rr["regimes"]["bins"]:
        top = r["importance"].iloc[0]
        L.append(f"| {r['regime']} | {r['driver_lo']:.0f}–{r['driver_hi']:.0f} "
                 f"| {r['n']} | {top['Feature']} | {top['Share']*100:.1f}% |")
    L.append("\nFull features × regimes matrix: `rca_regime_importance.csv`. "
             "Interval traces: `acgc_intervals.csv`.\n")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
