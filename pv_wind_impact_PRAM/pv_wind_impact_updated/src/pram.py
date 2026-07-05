"""
PRAM — Physics-Residual Attribution Model
=========================================

Learns the residual between a *reference* PV power series and the *physics-model*
PV series, then attributes that residual to atmospheric drivers with explainable
AI (SHAP if installed, otherwise random-forest importance).

Two honest modes are supported by the callers:

* **measured**  — reference = REAL metered DKASC power, model = project physics on
  the same on-site weather.  ``residual = P_measured − P_physics`` is the genuine,
  scientifically valid quantity (what the physics misses vs. reality).

* **crosssrc** — reference = NASA-POWER-driven physics PV, model = Open-Meteo-driven
  physics PV.  There is **no metered ground truth** from weather APIs, so this is a
  *data-source / model-consistency* residual (which atmospheric inputs make two
  independent reanalyses diverge) — useful, but NOT measured validation.

This module is pure (no Streamlit); the app handles rendering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:                                    # SHAP is optional
    import shap                         # noqa: F401
    HAS_SHAP = True
except Exception:                       # pragma: no cover
    HAS_SHAP = False


# Candidate columns we try to pull into the residual feature matrix, mapped to a
# clean display name.  Only those present are used, so this works for both the
# DKASC frame (ghi/poa/temp/…) and the engineered live frame.
_FEATURE_CANDIDATES = {
    "poa": "poa_irradiance",
    "global_tilted_irradiance": "poa_irradiance",
    "ghi": "ghi",
    "shortwave_radiation": "ghi",
    "dhi": "dhi",
    "diffuse_radiation": "dhi",
    "direct_normal_irradiance": "dni",
    "temp": "air_temp",
    "temperature_2m": "air_temp",
    "rh": "humidity",
    "relative_humidity_2m": "humidity",
    "wind": "wind_speed",
    "wind_speed_10m": "wind_speed",
    "rain": "rain",
    "precipitation": "rain",
    "surface_pressure": "pressure",
    "aerosol_optical_depth": "aod",
    "aod": "aod",
    "pm10": "pm10",
    "pm2_5": "pm2_5",
    "dust": "dust",
}

AEROSOL_FEATURES = {"aod", "pm10", "pm2_5", "dust"}


def build_residual_features(weather: pd.DataFrame,
                            index: pd.Index | None = None) -> pd.DataFrame:
    """Assemble an interpretable atmospheric + temporal feature matrix.

    Picks whichever recognised columns exist in ``weather`` and appends
    hour/month/seasonal encodings from the datetime index.
    """
    idx = weather.index if index is None else index
    X = pd.DataFrame(index=idx)

    seen = set()
    for raw, nice in _FEATURE_CANDIDATES.items():
        if raw in weather.columns and nice not in seen:
            col = pd.to_numeric(weather[raw], errors="coerce")
            if col.notna().sum() > 0:
                X[nice] = col.to_numpy()
                seen.add(nice)

    dti = pd.DatetimeIndex(idx)
    hr = dti.hour.to_numpy()
    mo = dti.month.to_numpy()
    doy = dti.dayofyear.to_numpy()
    X["hour"] = hr
    X["month"] = mo
    X["hour_sin"] = np.sin(2 * np.pi * hr / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hr / 24.0)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)

    # ------------------------------------------------------------------
    # Physically-meaningful derived features. These help the residual
    # learner separate the *geometric / conversion* mismatch (which
    # dominates whole-site mixed-orientation meters) from random noise,
    # so the attribution reflects real effects rather than scatter.
    # ------------------------------------------------------------------
    # Clear-sky / elevation proxies from latitude-free geometry: use an
    # extraterrestrial-irradiance estimate on the horizontal to normalise GHI.
    decl = np.deg2rad(23.45 * np.sin(2 * np.pi * (284 + doy) / 365.0))
    hra = np.deg2rad(15.0 * (hr - 12.0))
    # latitude unknown here -> use a season/elevation-robust cos(zenith) proxy
    cos_zen = np.clip(np.cos(decl) * np.cos(hra), 0.0, 1.0)
    X["cos_zenith_proxy"] = cos_zen
    extra = np.clip(1361.0 * cos_zen, 1.0, None)

    if "ghi" in X:
        ghi = X["ghi"].to_numpy(dtype=float)
        X["clear_sky_index"] = np.clip(ghi / extra, 0.0, 1.5)
        X["ghi_sq"] = ghi ** 2
        X["ghi_x_elev"] = ghi * cos_zen
        if "poa_irradiance" in X:
            poa = X["poa_irradiance"].to_numpy(dtype=float)
            X["poa_over_ghi"] = np.where(ghi > 5.0, poa / ghi, 0.0)
        # short memory: lag-1 and 3-hour rolling mean (soiling/thermal inertia)
        ghi_s = pd.Series(ghi, index=idx)
        X["ghi_lag1"] = ghi_s.shift(1).to_numpy()
        X["ghi_roll3"] = ghi_s.rolling(3, min_periods=1).mean().to_numpy()

    if "air_temp" in X and "wind_speed" in X:
        # wind cooling reduces cell temperature -> lifts efficiency
        X["temp_x_wind"] = (X["air_temp"].to_numpy(dtype=float)
                            * X["wind_speed"].to_numpy(dtype=float))
    if "air_temp" in X:
        t = X["air_temp"].to_numpy(dtype=float)
        X["temp_excess"] = np.clip(t - 25.0, 0.0, None)   # thermal-derate driver

    return X


def fit_residual_model(X: pd.DataFrame, residual,
                       *, operating_mask=None,
                       n_estimators: int = 300, max_depth: int = 12,
                       test_frac: float = 0.2,
                       random_state: int = 42) -> dict:
    """Fit a random forest on the residual with a **time-based** train/test split.

    The chronological split (no shuffling) means the test set is a held-out future
    period — the honest way to evaluate, and what to report in the paper.
    """
    df = X.copy()
    df["_resid"] = np.asarray(residual, dtype=float)
    if operating_mask is not None:
        df = df[np.asarray(operating_mask, dtype=bool)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    n = len(df)
    if n < 60:
        return {"ok": False,
                "reason": f"Only {n} aligned operating-hour rows after cleaning "
                          f"(need ≥ 60). Add a longer period or more data."}

    feat_cols = [c for c in df.columns if c != "_resid"]
    cut = int(n * (1.0 - test_frac))
    X_tr, X_te = df[feat_cols].iloc[:cut], df[feat_cols].iloc[cut:]
    y_tr, y_te = df["_resid"].iloc[:cut], df["_resid"].iloc[cut:]

    # Trim extreme residual outliers from the TRAIN set only. Whole-site
    # mixed-orientation meters produce large transient spikes (tracking arrays,
    # inverter clipping) that otherwise dominate the loss and swamp the real
    # signal. The held-out TEST set is left untouched for an honest score.
    if len(y_tr) >= 50:
        lo, hi = np.nanpercentile(y_tr.to_numpy(), [1.0, 99.0])
        keep = (y_tr >= lo) & (y_tr <= hi)
        X_tr, y_tr = X_tr[keep], y_tr[keep]

    # Prefer gradient boosting: it handles the enriched, partly-collinear
    # feature set and the heavy-tailed residual better than a plain forest.
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=None,
            max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=random_state)
    except Exception:
        model = RandomForestRegressor(
            n_estimators=max(n_estimators, 400), max_depth=max_depth,
            min_samples_leaf=5, max_features=0.6,
            random_state=random_state, n_jobs=1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    return {
        "ok": True, "model": model, "feat_cols": feat_cols, "n": n, "cut": cut,
        "index": df.index, "feature_frame": df[feat_cols],
        "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te, "pred": pred,
        "resid_r2": float(r2_score(y_te, pred)),
        "resid_rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "resid_mae": float(mean_absolute_error(y_te, pred)),
        "resid_std": float(np.std(df["_resid"].to_numpy())),
    }


def improvement_signals(p_ref, p_model, fit: dict) -> dict:
    """Does physics + learned residual beat physics alone on the TEST period?"""
    idx = fit["index"]
    ref = pd.Series(p_ref).reindex(idx).to_numpy()
    mod = pd.Series(p_model).reindex(idx).to_numpy()
    cut = fit["cut"]
    ref_te, mod_te = ref[cut:], mod[cut:]
    corr_te = mod_te + fit["pred"]                      # physics + residual

    def _rmse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")

    def _mbe(a, b):                                     # model − reference
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.mean(b[m] - a[m])) if m.any() else float("nan")

    rmse_phys = _rmse(ref_te, mod_te)
    rmse_corr = _rmse(ref_te, corr_te)
    impr = (1.0 - rmse_corr / rmse_phys) * 100.0 if rmse_phys and np.isfinite(rmse_phys) and rmse_phys > 0 else 0.0
    return {"rmse_phys": rmse_phys, "rmse_corr": rmse_corr,
            "improvement_pct": impr,
            "mbe_phys": _mbe(ref_te, mod_te),
            "mbe_corr": _mbe(ref_te, corr_te),
            "n_test": int(len(ref_te))}


def attribution(fit: dict):
    """Driver attribution: SHAP mean|value| if available, else RF importance.

    Returns (importance_series_sorted_desc, method_label, signed_mean_or_None).
    The signed-mean per feature (mean SHAP, signed) lets callers detect, e.g.,
    aerosols pushing the residual negative at high values.
    """
    X_te = fit["X_te"]
    if HAS_SHAP:
        try:
            expl = shap.TreeExplainer(fit["model"])
            sv = expl.shap_values(X_te)
            sv = np.asarray(sv)
            imp = pd.Series(np.abs(sv).mean(axis=0), index=X_te.columns)
            signed = pd.Series(sv.mean(axis=0), index=X_te.columns)
            return imp.sort_values(ascending=False), "SHAP (mean |value|)", signed
        except Exception:
            pass

    # Native importances if the model exposes them (e.g. RandomForest).
    if hasattr(fit["model"], "feature_importances_"):
        imp = pd.Series(fit["model"].feature_importances_, index=fit["feat_cols"])
        return imp.sort_values(ascending=False), "Tree importance", None

    # Model-agnostic fallback: permutation importance on the held-out test set.
    # Works for HistGradientBoosting (and anything else). Signed direction is
    # taken from the sign of each feature's correlation with the residual.
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(
            fit["model"], fit["X_te"], fit["y_te"],
            n_repeats=8, random_state=42, n_jobs=1)
        imp = pd.Series(np.clip(r.importances_mean, 0, None),
                        index=fit["feat_cols"])
        Xte = fit["X_te"]
        signed = pd.Series(
            {c: float(np.sign(np.corrcoef(Xte[c], fit["y_te"])[0, 1] or 0.0))
             * imp[c] for c in fit["feat_cols"]})
        return (imp.sort_values(ascending=False),
                "Permutation importance", signed)
    except Exception:
        imp = pd.Series(1.0, index=fit["feat_cols"])
        return imp, "unavailable", None


def aerosol_signature(importance: pd.Series, signed) -> dict:
    """Is there an aerosol/soiling signature in the residual attribution?

    Strong = an aerosol feature ranks in the top 3 AND (if SHAP signed values are
    available) its mean effect is negative (physics over-predicts when air is dusty).
    """
    present = [f for f in importance.index if f in AEROSOL_FEATURES]
    if not present:
        return {"available": False, "rank": None, "top3": False,
                "negative": None, "feature": None}
    top = present[0]
    rank = list(importance.index).index(top) + 1
    neg = None
    if signed is not None and top in signed.index:
        neg = bool(signed[top] < 0)
    return {"available": True, "feature": top, "rank": rank,
            "top3": rank <= 3, "negative": neg}


def classify(improvement_pct: float, resid_r2: float, aero: dict) -> dict:
    """Map the diagnostic signals to an overall tier with an honest verdict."""
    # A residual is meaningful if it is explainable (R2) OR it materially cuts
    # error (improvement %). After baseline calibration the corrected physics is
    # already close, so a good R2 with a modest RMSE gain is still a real result.
    if resid_r2 >= 0.40 or improvement_pct >= 8:
        tier, css = "strong", "good"
    elif resid_r2 >= 0.18 or improvement_pct >= 3:
        tier, css = "moderate", "warn"
    else:
        tier, css = "weak", "bad"

    # require a minimum of real signal to call something strong
    if tier == "strong" and resid_r2 < 0.25 and improvement_pct < 8:
        tier, css = "moderate", "warn"

    # an interpretable aerosol signature lifts a moderate result's significance
    if tier == "moderate" and aero.get("top3") and aero.get("negative"):
        css = "good"

    leak = resid_r2 > 0.85                       # suspiciously high → investigate
    return {"tier": tier, "css": css, "possible_leak": leak}


def signal_table(improvement_pct: float, resid_r2: float, mbe_phys: float,
                 mbe_corr: float, aero: dict) -> list[dict]:
    """Build the six-signal diagnostic rows (label, value, reading)."""
    def rate(val, strong, moderate, higher_better=True):
        ok = (val >= strong) if higher_better else (val <= strong)
        mod = (val >= moderate) if higher_better else (val <= moderate)
        return "🟢 strong" if ok else ("🟡 moderate" if mod else "🔴 weak")

    bias_drop = abs(mbe_phys) - abs(mbe_corr)
    rows = [
        {"signal": "1 · Improvement over physics",
         "value": f"{improvement_pct:+.1f}% RMSE",
         "reading": rate(improvement_pct, 8, 3)},
        {"signal": "2 · Residual explainable?",
         "value": f"R² = {resid_r2:.3f}",
         "reading": ("🔴 suspicious (leak?)" if resid_r2 > 0.85
                     else rate(resid_r2, 0.40, 0.20))},
        {"signal": "3 · Aerosol/soiling signature",
         "value": (f"{aero['feature']} @ rank {aero['rank']}"
                   + ("" if aero.get("negative") is None
                      else (", −ve" if aero["negative"] else ", +ve"))
                   if aero.get("available") else "no aerosol features"),
         "reading": ("🟢 strong" if (aero.get("top3") and aero.get("negative"))
                     else ("🟡 moderate" if aero.get("available") else "⚪ n/a"))},
        {"signal": "4 · Bias reduced toward 0",
         "value": f"{mbe_phys:+.2f} → {mbe_corr:+.2f} kW",
         "reading": "🟢 strong" if bias_drop > 0 else "🔴 weak"},
    ]
    return rows


def calibrate_baseline(ref: pd.Series, mod: pd.Series, operating: np.ndarray,
                       test_frac: float = 0.2) -> pd.Series:
    """Remove the gross shape/scale bias of the physics baseline.

    A single fixed-tilt physics model has the wrong daily *shape* for a meter
    that aggregates tracking / off-axis arrays. We fit a per-hour-of-day
    multiplicative gain (median of ref/mod) on the **training portion only**
    (the first 1-test_frac of operating hours, chronological — no leakage) and
    apply it to every hour. This turns the dominant multiplicative error into a
    small additive residual that the learner can genuinely model.
    """
    idx = ref.index
    hours = pd.DatetimeIndex(idx).hour
    op = pd.Series(operating, index=idx)
    n = int(op.sum())
    if n < 60:
        return mod.copy()

    # chronological train cut over operating hours
    op_idx = op[op].index
    cut = int(len(op_idx) * (1.0 - test_frac))
    train_idx = op_idx[:cut]

    gains = {}
    r_tr, m_tr = ref.loc[train_idx], mod.loc[train_idx]
    h_tr = pd.DatetimeIndex(train_idx).hour
    for h in range(24):
        sel = (h_tr == h) & (m_tr.to_numpy() > 0.02 * float(np.nanmax(mod) or 1.0))
        if sel.sum() >= 10:
            ratio = (r_tr.to_numpy()[sel] / np.clip(m_tr.to_numpy()[sel], 1e-6, None))
            g = float(np.nanmedian(ratio))
            gains[h] = float(np.clip(g, 0.2, 5.0))   # guard against wild gains
    if not gains:
        return mod.copy()

    global_g = float(np.nanmedian(list(gains.values())))
    gain_vec = np.array([gains.get(int(h), global_g) for h in hours])
    return pd.Series(mod.to_numpy() * gain_vec, index=idx)


def run_pram(weather: pd.DataFrame, p_ref, p_model, rated_kw: float,
             *, mode: str = "measured", calibrate: bool = True) -> dict:
    """Convenience end-to-end runner returning everything the UI needs."""
    idx = weather.index
    ref = pd.Series(np.asarray(p_ref, dtype=float), index=idx)
    mod = pd.Series(np.asarray(p_model, dtype=float), index=idx)

    if mode == "measured":
        operating = (ref > 0.02 * rated_kw).to_numpy()
    else:                                  # cross-source: use daylight (either > 0)
        operating = ((ref > 0.02 * rated_kw) | (mod > 0.02 * rated_kw)).to_numpy()

    # Calibrate the physics baseline (train-only) so the residual reflects the
    # real physical gap, not the single-array vs whole-site shape mismatch.
    if calibrate and mode == "measured":
        mod = calibrate_baseline(ref, mod, operating, test_frac=0.2)

    residual = ref - mod

    X = build_residual_features(weather, index=idx)
    fit = fit_residual_model(X, residual.to_numpy(), operating_mask=operating)
    if not fit["ok"]:
        return {"ok": False, "reason": fit["reason"]}

    impr = improvement_signals(ref, mod, fit)
    imp, method, signed = attribution(fit)
    aero = aerosol_signature(imp, signed)
    cls = classify(impr["improvement_pct"], fit["resid_r2"], aero)
    rows = signal_table(impr["improvement_pct"], fit["resid_r2"],
                        impr["mbe_phys"], impr["mbe_corr"], aero)

    return {"ok": True, "mode": mode, "fit": fit, "impr": impr,
            "importance": imp, "method": method, "signed": signed,
            "aerosol": aero, "classify": cls, "signals": rows,
            "residual": residual, "ref": ref, "model": mod}
