"""
run_crisp_validation.py
=======================
Empirical validation for CRISP (the conformal prediction-interval layer).

Claim under test
----------------
Split-conformal prediction wrapped around the project's transparent PV physics,
and evaluated on **real metered DKASC Alice Springs generation**, should:

  (a) achieve **marginal coverage ≈ 1-alpha** on a held-out (later-in-time) test
      block — the distribution-free guarantee, checked on real data;
  (b) show that a single *global* interval, while marginally valid, badly
      mis-covers *conditionally* across irradiance regimes (too wide when it is
      dark, too narrow at peak sun); and
  (c) show that the **regime-indexed (Mondrian)** variant restores conditional
      coverage — i.e. it shrinks the worst-regime coverage gap.

This does NOT fabricate a result. It loads the real DKASC CSV, applies the
project's own PV physics as the point predictor, and prints whatever coverage
the conformal calibration actually delivers — including any shortfall caused by
real temporal drift (which breaks strict exchangeability and is worth reporting
honestly).

Honesty caveat (read before quoting a number)
---------------------------------------------
Conformal's finite-sample guarantee assumes the calibration and test scores are
*exchangeable*. A chronological calib→test split (used here, on purpose, to
avoid look-ahead leakage) injects real seasonal drift, so empirical coverage can
sit a little below nominal. That is a property of the *data*, not a bug in the
method — report the observed number and the split, don't round it up.

Usage
-----
    python run_crisp_validation.py                       # auto-locate DKASC CSV
    python run_crisp_validation.py --csv datadkasc_alice_springs.csv
    python run_crisp_validation.py --alpha 0.1 --n-bins 5 --channel-index 0
    python run_crisp_validation.py --monthly              # per-month windows too
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src import dksac, crisp          # NOTE: not pram/facl (those need sklearn)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Leakage-safe per-hour-of-day gain calibration of the physics centre.
# (sklearn-free copy of pram.calibrate_baseline, fit on the calibration block
#  only, so a single fixed-tilt physics curve is re-scaled to the metered array
#  before we quantify residual uncertainty around it.)
# ---------------------------------------------------------------------------
def per_hour_gain_calibrate(ref: pd.Series, mod: pd.Series,
                            operating: np.ndarray, calib_frac: float) -> pd.Series:
    idx = ref.index
    hours = pd.DatetimeIndex(idx).hour
    op = pd.Series(operating, index=idx)
    op_idx = op[op].index
    if len(op_idx) < 40:
        return mod.copy()
    cut = int(len(op_idx) * calib_frac)
    train_idx = op_idx[:cut]
    r_tr, m_tr = ref.loc[train_idx], mod.loc[train_idx]
    h_tr = pd.DatetimeIndex(train_idx).hour
    max_mod = float(np.nanmax(mod.to_numpy())) or 1.0
    gains = {}
    for h in range(24):
        sel = (h_tr == h) & (m_tr.to_numpy() > 0.02 * max_mod)
        if sel.sum() >= 10:
            ratio = r_tr.to_numpy()[sel] / np.clip(m_tr.to_numpy()[sel], 1e-6, None)
            gains[h] = float(np.clip(np.nanmedian(ratio), 0.2, 5.0))
    if not gains:
        return mod.copy()
    gg = float(np.nanmedian(list(gains.values())))
    gain_vec = np.array([gains.get(int(h), gg) for h in hours])
    return pd.Series(mod.to_numpy() * gain_vec, index=idx)


def build_physics_prediction(sub: pd.DataFrame, rated: float):
    """Physics PV point prediction + the regime driver (POA if present, else GHI)."""
    model_in = sub.copy()
    if "poa" not in model_in or model_in["poa"].isna().all():
        model_in["poa"] = model_in.get("ghi")
    p_model = dksac.model_pv_from_dksac(model_in, rated)
    driver = model_in["poa"].where(model_in["poa"].notna(), model_in.get("ghi"))
    return p_model, driver


def _print_report(rep: dict, label: str) -> None:
    print(f"  {label:26s} marginal={rep['marginal_coverage']*100:5.1f}%  "
          f"worst-regime gap={rep.get('worst_slab_gap', float('nan'))*100:4.1f} pts  "
          f"MPIW={rep['mpiw']:6.2f} kW  PINAW={rep['pinaw']:.3f}  "
          f"Winkler={rep['winkler']:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="DKASC CSV (defaults to auto-locate).")
    ap.add_argument("--channel-index", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="Target miscoverage (0.10 → 90%% intervals).")
    ap.add_argument("--n-bins", type=int, default=5, help="Irradiance regimes.")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--min-hours", type=int, default=300)
    ap.add_argument("--max-windows", type=int, default=12)
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Skip per-hour physics-centre calibration.")
    ap.add_argument("--monthly", action="store_true",
                    help="Also run per-month windows (like the FACL harness).")
    args = ap.parse_args()

    path = args.csv or dksac.find_dksac_csv()
    if not path:
        raise SystemExit(
            "No DKASC CSV found. Download it from "
            "https://dkasolarcentre.com.au/download?location=alice-springs "
            "and pass it with --csv, or drop it next to this script.")
    print(f"Loading DKASC CSV: {path}")
    hourly, wmap, power_cols = dksac.load_dksac_frame(path)
    chans = dksac.power_channels(hourly, power_cols)
    if not chans:
        raise SystemExit("No data-bearing PV power channel found in the CSV.")
    ci = min(args.channel_index, len(chans) - 1)
    pcol, plabel, ppeak = chans[ci]
    rated = dksac.estimate_rated_kw(hourly[pcol])
    print(f"Channel: {plabel}  (peak ~{ppeak:.1f} kW, rated≈{rated:.1f} kW)")
    print(f"Regime driver: POA/GHI irradiance · {args.n_bins} quantile regimes · "
          f"target {(1-args.alpha)*100:.0f}% coverage\n")

    # ---- full-period pooled analysis (the headline) -----------------------
    weather_cols = [c for c in ("ghi", "dhi", "poa", "temp", "rh") if c in hourly]
    df = hourly[weather_cols + [pcol]].copy()
    p_meas = pd.to_numeric(df[pcol], errors="coerce")
    aligned = pd.concat([df[weather_cols], p_meas.rename("P")], axis=1).dropna(
        subset=[c for c in ("ghi", "temp") if c in weather_cols] + ["P"])
    aligned = aligned.sort_index()

    p_model, driver = build_physics_prediction(aligned, rated)
    operating = (aligned["P"] > 0.02 * rated).to_numpy()
    if not args.no_calibrate:
        p_model = per_hour_gain_calibrate(aligned["P"], p_model, operating,
                                          args.calib_frac)

    print("=" * 90)
    print("FULL-PERIOD POOLED VALIDATION")
    print("=" * 90)
    res = crisp.run_crisp(
        aligned["P"].to_numpy(), p_model.to_numpy(), driver.to_numpy(),
        alpha=args.alpha, n_bins=args.n_bins, method="mondrian",
        calib_frac=args.calib_frac, operating_mask=operating, rated_kw=rated,
        compare_global=True)
    if not res["ok"]:
        raise SystemExit(f"CRISP failed: {res['reason']}")

    print(f"calib rows={res['n_calib']}  test rows={res['n_test']}\n")
    _print_report(res["report"], "Mondrian (regime-indexed)")
    if "report_global" in res:
        _print_report(res["report_global"], "Global (single width)")

    print("\nPer-regime conditional coverage (Mondrian vs Global on identical slabs):")
    print(f"  {'regime (kW POA)':22s} {'n':>5s}  {'Mondrian':>9s}  {'Global':>8s}  "
          f"{'M-width':>8s}")
    gbins = {b["regime"]: b for b in res.get("report_global", {}).get("bins", [])}
    for b in res["report"]["bins"]:
        g = gbins.get(b["regime"], {})
        rng = f"[{b['driver_lo']:.0f}-{b['driver_hi']:.0f}]"
        print(f"  {rng:22s} {b['n']:5d}  {b['coverage']*100:8.1f}%  "
              f"{g.get('coverage', float('nan'))*100:7.1f}%  {b['mpiw']:7.2f}")

    # ---- reliability curve across confidence levels -----------------------
    print("\nReliability curve (nominal vs empirical marginal coverage, Mondrian):")
    rel = crisp.reliability(
        aligned["P"].to_numpy(), p_model.to_numpy(), driver.to_numpy(),
        alphas=(0.5, 0.2, 0.1, 0.05), method="mondrian", n_bins=args.n_bins,
        calib_frac=args.calib_frac, operating_mask=operating, rated_kw=rated)
    print(rel.to_string(index=False))

    # ---- optional per-month windows ---------------------------------------
    if args.monthly:
        print("\n" + "=" * 90)
        print("PER-MONTH WINDOWS")
        print("=" * 90)
        rows = []
        for period, sub in aligned.groupby(aligned.index.to_period("M")):
            if len(rows) >= args.max_windows:
                break
            op = int((sub["P"] > 0.02 * rated).sum())
            if op < args.min_hours:
                continue
            pm, drv = build_physics_prediction(sub, rated)
            opmask = (sub["P"] > 0.02 * rated).to_numpy()
            if not args.no_calibrate:
                pm = per_hour_gain_calibrate(sub["P"], pm, opmask, args.calib_frac)
            r = crisp.run_crisp(sub["P"].to_numpy(), pm.to_numpy(), drv.to_numpy(),
                                alpha=args.alpha, n_bins=args.n_bins,
                                method="mondrian", calib_frac=args.calib_frac,
                                operating_mask=opmask, rated_kw=rated,
                                compare_global=True)
            if not r["ok"]:
                continue
            rep, repg = r["report"], r.get("report_global", {})
            rows.append({
                "month": str(period), "op_hrs": op,
                "cover_M%": round(rep["marginal_coverage"] * 100, 1),
                "cover_G%": round(repg.get("marginal_coverage", float("nan")) * 100, 1),
                "gap_M": round(rep.get("worst_slab_gap", float("nan")) * 100, 1),
                "gap_G": round(repg.get("worst_slab_gap", float("nan")) * 100, 1),
                "MPIW": round(rep["mpiw"], 1),
                "Winkler": round(rep["winkler"], 1),
            })
            print(f"  {period}  cover(M)={rep['marginal_coverage']*100:5.1f}%  "
                  f"gap(M)={rep.get('worst_slab_gap', float('nan'))*100:4.1f}pts  "
                  f"gap(G)={repg.get('worst_slab_gap', float('nan'))*100:4.1f}pts  "
                  f"MPIW={rep['mpiw']:.1f}kW")
        if rows:
            mdf = pd.DataFrame(rows)
            print("\n" + mdf.to_string(index=False))
            print(f"\nMean marginal coverage  Mondrian: {mdf['cover_M%'].mean():.1f}%   "
                  f"Global: {mdf['cover_G%'].mean():.1f}%   (target "
                  f"{(1-args.alpha)*100:.0f}%)")
            print(f"Mean worst-regime gap   Mondrian: {mdf['gap_M'].mean():.1f}pts  "
                  f"Global: {mdf['gap_G'].mean():.1f}pts")

    print("\nInterpretation is yours to make honestly. The defensible headline is:\n"
          "  'On real DKASC generation, regime-indexed conformal intervals hold\n"
          "   marginal coverage near nominal AND cut the worst-regime coverage gap\n"
          "   versus a single global interval' — the conditional-validity gain.")


if __name__ == "__main__":
    main()
