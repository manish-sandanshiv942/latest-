"""
run_facl_validation.py
======================
Empirical check for FACL (the fuzzy attribution-confidence layer).

Claim under test
----------------
FACL's Attribution-Confidence Index (ACI) should behave sensibly on REAL
measured data: (a) higher ACI windows should have lower held-out corrected-model
error, (b) an *independent* quality signal — the test-period Pearson r between
the corrected model and the measured power, which is NOT a FACL input — should
rank-correlate with ACI, and (c) the leakage guard should trigger on windows
whose residual R² lands in the suspicious region.

This does NOT fabricate a result. It runs PRAM + FACL over successive monthly
windows of the real DKASC Alice Springs CSV and prints whatever it finds,
including the Spearman correlations and any leakage-guard activations.

Honesty caveat (read before quoting a number)
---------------------------------------------
ACI is partly built from the RMSE-improvement, so a monotone ACI-vs-error
relation is *expected by construction* and is a sanity check, not an independent
validation. The genuinely independent check is Spearman(ACI, test_r), because
the test-period correlation r is not one of FACL's four inputs. Report both, and
say so.

Usage
-----
    python run_facl_validation.py --csv datadkasc_alice_springs.csv
    python run_facl_validation.py                 # auto-locates the DKASC CSV
    python run_facl_validation.py --max-windows 6 --min-hours 300
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src import dksac, pram, facl

warnings.filterwarnings("ignore")  # silence sklearn/numpy divide-by-zero notes


def _spearman(a, b) -> float:
    a = pd.Series(a).rank()
    b = pd.Series(b).rank()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(a.corr(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="Path to the DKASC CSV (defaults to auto-locate).")
    ap.add_argument("--max-windows", type=int, default=12)
    ap.add_argument("--min-hours", type=int, default=250,
                    help="Skip a month with fewer aligned operating hours.")
    ap.add_argument("--channel-index", type=int, default=0,
                    help="Which ranked DKASC power channel to validate against.")
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
    print(f"Channel: {plabel}  (peak ~{ppeak:.1f} kW)\n")

    rated = dksac.estimate_rated_kw(hourly[pcol])
    # Keep only well-populated weather columns (this DKASC export has empty wind
    # and near-empty rain). PRAM tolerates scattered NaNs, but a mostly-empty
    # column would gut the residual fit, so drop columns >20% missing.
    cand = [c for c in ("ghi", "dhi", "poa", "temp", "rh", "wind", "rain")
            if c in hourly]
    weather_cols = [c for c in cand if hourly[c].isna().mean() < 0.20]
    essentials = [c for c in ("ghi", "temp") if c in weather_cols]
    print(f"Using weather columns: {weather_cols}")

    rows = []
    # iterate month by month
    for period, sub in hourly.groupby(hourly.index.to_period("M")):
        if len(rows) >= args.max_windows:
            break
        w = sub[weather_cols].copy()
        p_meas = pd.to_numeric(sub[pcol], errors="coerce")
        aligned = pd.concat([w, p_meas.rename("P")], axis=1).dropna(
            subset=essentials + ["P"])
        op = int((aligned["P"] > 0.02 * rated).sum())
        if op < args.min_hours:
            continue

        wv = aligned[weather_cols]
        pm = aligned["P"].to_numpy()
        model_in = aligned.copy()
        if "poa" not in model_in:
            model_in["poa"] = model_in["ghi"]
        p_model = dksac.model_pv_from_dksac(model_in, rated).to_numpy()

        res = pram.run_pram(wv, pm, p_model, rated, mode="measured")
        if not res["ok"]:
            continue
        fz = facl.run_facl(res)

        # Independent quality signal: test-period corr(measured, corrected model).
        fit = res["fit"]
        cut = fit["cut"]
        idx = fit["index"]
        meas_te = pd.Series(pm, index=aligned.index).reindex(idx).to_numpy()[cut:]
        mod_te = res["model"].reindex(idx).to_numpy()[cut:] + fit["pred"]
        mask = np.isfinite(meas_te) & np.isfinite(mod_te)
        test_r = (float(np.corrcoef(meas_te[mask], mod_te[mask])[0, 1])
                  if mask.sum() > 3 else float("nan"))
        mean_meas_te = float(np.nanmean(meas_te[mask])) if mask.any() else 1.0
        test_nrmse = (res["impr"]["rmse_corr"] / mean_meas_te * 100.0
                      if mean_meas_te else float("nan"))

        rows.append({
            "month": str(period),
            "op_hrs": op,
            "impr%": round(res["impr"]["improvement_pct"], 1),
            "resid_R2": round(fit["resid_r2"], 3),
            "test_nRMSE%": round(test_nrmse, 1),
            "test_r": round(test_r, 3),
            "ACI": round(fz["aci"], 1),
            "verdict": fz["term"],
            "crisp": fz["crisp_tier"],
            "leak_guard": round(fz["leak_suspicion"], 2),
        })
        print(f"  {period}  ACI={fz['aci']:5.1f} ({fz['term']:9})  "
              f"crisp={str(fz['crisp_tier']):8}  R2={fit['resid_r2']:.3f}  "
              f"impr={res['impr']['improvement_pct']:+.1f}%  "
              f"test_r={test_r:.3f}  leak={fz['leak_suspicion']:.2f}")

    if not rows:
        raise SystemExit("No window had enough operating hours; lower --min-hours.")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)

    # relationships
    s_err = _spearman(df["ACI"], -df["test_nRMSE%"])   # higher ACI -> lower error?
    s_r = _spearman(df["ACI"], df["test_r"])           # INDEPENDENT check
    n_leak = int((df["leak_guard"] > 0.02).sum())

    print(f"\nWindows evaluated                     : {len(df)}")
    print(f"Spearman(ACI, -test_nRMSE)  [expected +, but partly by construction] : "
          f"{s_err:+.3f}")
    print(f"Spearman(ACI,  test_r)      [INDEPENDENT of FACL inputs]             : "
          f"{s_r:+.3f}")
    print(f"Leakage guard activated on            : {n_leak} window(s)")
    print("\nInterpretation is yours to make honestly: a positive INDEPENDENT "
          "correlation supports 'confidence tracks real quality'; a near-zero one "
          "means FACL is, on this data, mostly re-expressing the crisp signals "
          "more smoothly (still useful, but do not overclaim).")


if __name__ == "__main__":
    main()
