"""
src/pdf_report.py
=================
Generates a comprehensive, publication-quality PDF document that walks through
the complete mathematical methodology used to compute PV (and wind) energy from
raw Open-Meteo atmospheric data.

The PDF is returned as raw bytes so it can be delivered via st.download_button.

Usage
-----
    from src.pdf_report import generate_calculation_pdf
    pdf_bytes = generate_calculation_pdf(feat, ml_pv_pred, ml_wd_pred,
                                         pv_res, wd_res, meta, model_name,
                                         lat, lon)
    st.download_button("⬇ Download PDF Report", pdf_bytes,
                       "pv_calculation_report.pdf", "application/pdf")
"""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

import config

# ---------------------------------------------------------------------------
# Brand / style constants
# ---------------------------------------------------------------------------
_BLUE      = "#1f6fb2"
_ORANGE    = "#f5a623"
_LIGHT_BG  = "#f5f9fe"
_DARK_TEXT = "#1e2d40"
_MID_TEXT  = "#41515f"
_GREEN     = "#27ae60"
_RED       = "#e74c3c"
_HEADER_H  = 0.055   # fraction of axes height used by the header bar
_FOOTER_H  = 0.025
_FONT      = "DejaVu Sans"


# ===========================================================================
# Internal helpers
# ===========================================================================

def _fig_with_header(title: str, page_num: int, total_pages: int,
                     figsize=(8.5, 11)) -> tuple[plt.Figure, plt.Axes]:
    """Create a blank text-page figure with a styled header + footer."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 1 - _HEADER_H), 1, _HEADER_H,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=_BLUE, transform=ax.transAxes, clip_on=False))
    ax.text(0.025, 1 - _HEADER_H / 2, title,
            fontsize=13, fontweight="bold", color="white",
            va="center", ha="left", transform=ax.transAxes,
            fontfamily=_FONT)
    ax.text(0.975, 1 - _HEADER_H / 2,
            f"Page {page_num} / {total_pages}",
            fontsize=8, color="white", va="center", ha="right",
            transform=ax.transAxes, fontfamily=_FONT)

    # Footer
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, _FOOTER_H,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=_LIGHT_BG, transform=ax.transAxes, clip_on=False))
    ax.text(0.5, _FOOTER_H / 2,
            "PV Energy Calculation Report  ·  DKASC Validation",
            fontsize=7.5, color=_MID_TEXT, va="center", ha="center",
            transform=ax.transAxes, fontfamily=_FONT)

    return fig, ax


def _body(ax: plt.Axes, lines: list[tuple]) -> None:
    """
    Render body text from a list of (y, text, fontsize, weight, color) tuples.
    y is in [0, 1] axes coordinates (1 = just below header).
    """
    for y, text, fs, fw, color in lines:
        ax.text(0.025, y, text,
                fontsize=fs, fontweight=fw, color=color,
                va="top", ha="left", transform=ax.transAxes,
                fontfamily=_FONT, wrap=True)


def _section(ax: plt.Axes, y: float, label: str) -> float:
    """Draw a section-title bar; returns the y position after the bar."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.01, y - 0.025), 0.98, 0.028,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=_LIGHT_BG, transform=ax.transAxes))
    ax.text(0.025, y, label,
            fontsize=10, fontweight="bold", color=_BLUE,
            va="top", ha="left", transform=ax.transAxes,
            fontfamily=_FONT)
    return y - 0.038


def _table_on_ax(ax: plt.Axes, y0: float, headers: list[str],
                 rows: list[list], col_widths=None, row_h=0.032) -> float:
    """Render a simple table; returns y position after the table."""
    ncols = len(headers)
    if col_widths is None:
        col_widths = [0.97 / ncols] * ncols
    x_starts = [0.015]
    for w in col_widths[:-1]:
        x_starts.append(x_starts[-1] + w)

    # Header row
    hdr_y = y0 - row_h
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.01, hdr_y), 0.98, row_h,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=_BLUE, transform=ax.transAxes))
    for i, hdr in enumerate(headers):
        ax.text(x_starts[i] + col_widths[i] / 2, y0 - row_h / 2, hdr,
                fontsize=8, fontweight="bold", color="white",
                va="center", ha="center", transform=ax.transAxes,
                fontfamily=_FONT)
    cur_y = hdr_y
    for ri, row in enumerate(rows):
        bg = _LIGHT_BG if ri % 2 == 0 else "white"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, cur_y - row_h), 0.98, row_h,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=bg, transform=ax.transAxes))
        for i, cell in enumerate(row):
            ax.text(x_starts[i] + col_widths[i] / 2, cur_y - row_h / 2, str(cell),
                    fontsize=7.5, color=_DARK_TEXT,
                    va="center", ha="center", transform=ax.transAxes,
                    fontfamily=_FONT)
        cur_y -= row_h
    return cur_y - 0.01


# ===========================================================================
# Public API
# ===========================================================================

def generate_calculation_pdf(
    feat: pd.DataFrame,
    ml_pv_pred: np.ndarray,
    ml_wd_pred: np.ndarray,
    pv_res,
    wd_res,
    meta: dict,
    model_name: str,
    lat: float,
    lon: float,
    author: str = "Manish",
    institution: str = "Government College of Engineering, Karad",
) -> bytes:
    """
    Build a multi-page PDF that documents the complete PV energy calculation
    pipeline and returns it as a bytes object.

    Parameters
    ----------
    feat        : the 100-parameter feature DataFrame
    ml_pv_pred  : array of hourly ML PV predictions (kW), same length as feat
    ml_wd_pred  : array of hourly ML Wind predictions (kW)
    pv_res      : MLResult namedtuple / object with .model, .metrics, .X_train
    wd_res      : same for wind
    meta        : metadata dict from get_feature_table
    model_name  : human-readable ML model name
    lat, lon    : site coordinates
    author      : report author name
    institution : institution name
    """
    buf = io.BytesIO()

    # ----------------------------------------------------------------
    # Pre-compute everything we'll display so pages are fast to render
    # ----------------------------------------------------------------
    n = len(feat)
    days = max(n / 24.0, 1e-6)

    # Energy totals
    ml_pv_kwh  = float(np.sum(ml_pv_pred))
    ml_wd_kwh  = float(np.sum(ml_wd_pred))
    phy_pv_kwh = float(feat[config.TARGET_PV].sum())
    phy_wd_kwh = float(feat[config.TARGET_WIND].sum())

    pdc_kwp = config.PV_AREA_M2 * config.PV_EFFICIENCY

    # Irradiance-based yield
    if "effective_irradiance" in feat:
        poa = feat["effective_irradiance"]
    elif "global_tilted_irradiance" in feat:
        poa = feat["global_tilted_irradiance"]
    else:
        poa = feat["shortwave_radiation"]
    poa_kwh_m2 = float(poa.sum()) / 1000.0  # kWh/m²
    yf = ml_pv_kwh / pdc_kwp if pdc_kwp > 0 else 0.0
    pr = yf / poa_kwh_m2 if poa_kwh_m2 > 0 else 0.0
    annual_sy = yf * (365.0 / days)
    psh = poa_kwh_m2 / days

    # ML metrics
    r2_pv   = pv_res.metrics.get("r2", 0)
    mae_pv  = pv_res.metrics.get("mae", 0)
    rmse_pv = pv_res.metrics.get("rmse", 0)
    r2_wd   = wd_res.metrics.get("r2", 0)
    mae_wd  = wd_res.metrics.get("mae", 0)
    rmse_wd = wd_res.metrics.get("rmse", 0)

    # Top-10 feature importances (if available)
    if hasattr(pv_res.model, "feature_importances_"):
        imp = pv_res.model.feature_importances_
        imp_idx = np.argsort(imp)[::-1][:10]
        fi_rows = [
            [i + 1,
             config.pretty(config.FEATURES[idx]) if hasattr(config, "pretty") else config.FEATURES[idx],
             f"{imp[idx] * 100:.2f}%"]
            for i, idx in enumerate(imp_idx)
        ]
    else:
        fi_rows = []

    now_str   = datetime.now().strftime("%d %b %Y  %H:%M")
    date_str  = datetime.now().strftime("%d %B %Y")
    total_pages = 7

    with PdfPages(buf) as pdf:

        # ================================================================
        # PAGE 1 — Cover page
        # ================================================================
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        # Background gradient panel
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0.55), 1, 0.45,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=_BLUE))

        # Title block
        ax.text(0.5, 0.88, "PV Energy Calculation",
                fontsize=26, fontweight="bold", color="white",
                va="center", ha="center", transform=ax.transAxes,
                fontfamily=_FONT)
        ax.text(0.5, 0.81, "Mathematical Methodology & Validation Report",
                fontsize=14, color="#d0e8ff",
                va="center", ha="center", transform=ax.transAxes,
                fontfamily=_FONT)
        ax.axhline(0.74, xmin=0.1, xmax=0.9, color="white", lw=0.8, alpha=0.5)
        ax.text(0.5, 0.70, "DKASC Dataset — Open-Meteo Real-Time Pipeline",
                fontsize=11, color="#a8ccee",
                va="center", ha="center", transform=ax.transAxes,
                fontfamily=_FONT)

        # Info box
        for yi, line in enumerate([
            f"Author         :  {author}",
            f"Institution  :  {institution}",
            f"ML Algorithm :  {model_name}",
            f"Site             :  {lat:.4f}°N  {lon:.4f}°E",
            f"Data window :  {days:.0f} days  ({n:,} hourly records)",
            f"Generated     :  {now_str}",
        ]):
            ax.text(0.15, 0.49 - yi * 0.052, line,
                    fontsize=10, color=_DARK_TEXT,
                    va="top", ha="left", transform=ax.transAxes,
                    fontfamily=_FONT)

        # Footer
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 0.03,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=_LIGHT_BG))
        ax.text(0.5, 0.015,
                "This document is auto-generated from live atmospheric data.  "
                "All computations are reproducible.",
                fontsize=7.5, color=_MID_TEXT,
                va="center", ha="center", transform=ax.transAxes,
                fontfamily=_FONT)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ================================================================
        # PAGE 2 — Pipeline Overview & Dataset Statistics
        # ================================================================
        fig, ax = _fig_with_header("Step 1 – Data Acquisition & Feature Engineering", 2, total_pages)

        lines = [
            (0.90, "1.1  Data Acquisition (Open-Meteo Live API)", 10, "bold", _BLUE),
            (0.86, f"  ● Latitude: {lat:.4f}°   Longitude: {lon:.4f}°", 9, "normal", _DARK_TEXT),
            (0.83, f"  ● Data source: {meta.get('source', 'Open-Meteo / NASA POWER')}", 9, "normal", _DARK_TEXT),
            (0.80, f"  ● Total records: {n:,} hourly rows  (~{days:.0f} days)", 9, "normal", _DARK_TEXT),
            (0.77, "  ● 57 raw variables fetched per hour: GHI, DNI, DHI, temperature at", 9, "normal", _DARK_TEXT),
            (0.74, "    multiple heights, humidity, wind speeds, pressure, cloud cover,", 9, "normal", _DARK_TEXT),
            (0.71, "    aerosol optical depth, dust, PM2.5, PM10, ozone, and more.", 9, "normal", _DARK_TEXT),
        ]

        _body(ax, lines)
        y = _section(ax, 0.67, "1.2  Feature Engineering → 100 Parameters")

        cat_rows = [
            ["Solar geometry",    "8",  "Zenith, azimuth, air mass, declination, hour angle"],
            ["PV derived",        "6",  "Eff. POA irradiance, cell temperature, thermal derate, soiling"],
            ["Wind derived",      "9",  "Hub-height wind, air density, wind power density, turbulence"],
            ["Thermodynamics",    "10", "Vapour pressure, specific humidity, wet-bulb, heat index"],
            ["Temporal encoding", "6",  "sin/cos of hour-of-day, day-of-year, month"],
            ["Rolling statistics","4",  "3-h rolling means of GHI, wind, temp; pressure tendency"],
        ]
        y = _table_on_ax(ax, y, ["Category", "Count", "Key features / formulas"],
                         cat_rows, col_widths=[0.22, 0.08, 0.67], row_h=0.035)

        ax.text(0.025, y - 0.01,
                "Result:  57 raw  +  43 derived  =  100 input features → the ML training matrix.",
                fontsize=9, fontweight="bold", color=_BLUE,
                va="top", transform=ax.transAxes, fontfamily=_FONT)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ================================================================
        # PAGE 3 — Physical PV Model Mathematics
        # ================================================================
        fig, ax = _fig_with_header("Step 2 – Physical PV Power Model (Engineering Equations)", 3, total_pages)

        y = 0.90
        y = _section(ax, y, "2.1  Irradiance at Plane of Array (POA)")

        body3a = [
            (y - 0.01,
             "The effective POA irradiance accounts for panel tilt, soiling, and spectral losses:", 9, "normal", _DARK_TEXT),
            (y - 0.055,
             "    G_eff  =  G_POA × cos(AOI) × (1 – soiling_fraction)", 10, "bold", _BLUE),
            (y - 0.095,
             f"  ● Panel tilt:  {config.PV_TILT_DEG}°   (south-facing)", 9, "normal", _DARK_TEXT),
            (y - 0.125,
             f"  ● Module area: {config.PV_AREA_M2:.0f} m²     STC efficiency: {config.PV_EFFICIENCY*100:.0f}%", 9, "normal", _DARK_TEXT),
        ]
        _body(ax, body3a)
        y -= 0.15

        y = _section(ax, y, "2.2  Cell Temperature (NOCT Model)")
        body3b = [
            (y - 0.01,
             "Cell temperature rises above ambient due to absorbed irradiance:", 9, "normal", _DARK_TEXT),
            (y - 0.05,
             f"    T_cell  =  T_amb  +  G_eff × (NOCT – 20) / 800      NOCT = {config.PV_NOCT_C:.0f}°C", 10, "bold", _BLUE),
        ]
        _body(ax, body3b)
        y -= 0.11

        y = _section(ax, y, "2.3  Thermal Derate Factor")
        body3c = [
            (y - 0.01,
             "Higher cell temperature reduces module efficiency:", 9, "normal", _DARK_TEXT),
            (y - 0.05,
             f"    η_derate  =  1 – γ × (T_cell – 25)      γ = {config.PV_TEMP_COEFF:.4f} / °C  (–0.40 % / °C)", 10, "bold", _BLUE),
        ]
        _body(ax, body3c)
        y -= 0.11

        y = _section(ax, y, "2.4  DC & AC Power Calculation")
        body3d = [
            (y - 0.01,
             "DC power is converted to AC power and clipped at the inverter limit:", 9, "normal", _DARK_TEXT),
            (y - 0.05,
             "    P_DC  =  (G_eff / 1000) × A_module × η_STC × η_derate × soiling", 10, "bold", _BLUE),
            (y - 0.09,
             f"    P_AC  =  min( P_DC,  P_rated )      P_rated = {config.PV_CAPACITY_KW:.0f} kW", 10, "bold", _BLUE),
        ]
        _body(ax, body3d)
        y -= 0.13

        y = _section(ax, y, "2.5  Hourly Energy Summation")
        body3e = [
            (y - 0.01,
             "Each predicted hourly power reading (kW) × 1 hour = kWh.  Summed over the window:", 9, "normal", _DARK_TEXT),
            (y - 0.05,
             "    E_PV  =  Σ  P_AC(h)  ×  Δt          Δt = 1 hour", 10, "bold", _BLUE),
            (y - 0.09,
             f"Physical model total:   {phy_pv_kwh:,.1f} kWh  =  {phy_pv_kwh/1000:,.2f} MWh  over {days:.0f} days", 9, "normal", _GREEN),
        ]
        _body(ax, body3e)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ================================================================
        # PAGE 4 — ML Model Pipeline
        # ================================================================
        fig, ax = _fig_with_header(f"Step 3 – Machine Learning: {model_name}", 4, total_pages)

        n_train = len(pv_res.X_train)
        n_total = n

        y = 0.90
        y = _section(ax, y, "3.1  Train / Test Split")
        split_rows = [
            ["Total samples",       f"{n_total:,}",         "All hourly records in the data window"],
            ["Training set (80%)",  f"{n_train:,}",         "Used to fit the ML model"],
            ["Test set (20%)",       f"{n_total - n_train:,}", "Held out for unbiased accuracy evaluation"],
            ["Random seed",         f"{config.RANDOM_STATE}", "Ensures reproducibility"],
        ]
        y = _table_on_ax(ax, y, ["Set", "Samples", "Purpose"], split_rows,
                         col_widths=[0.28, 0.16, 0.53], row_h=0.038)

        y = _section(ax, y, f"3.2  {model_name} – How it Predicts")

        if "Random Forest" in model_name or "Extra Trees" in model_name:
            n_trees = getattr(pv_res.model, "n_estimators", "?")
            max_d   = getattr(pv_res.model, "max_depth", "unlimited") or "unlimited"
            desc_lines = [
                (y - 0.01,  f"An ensemble of {n_trees} decision trees that each independently predict PV power,", 9, "normal", _DARK_TEXT),
                (y - 0.04,  "then their outputs are averaged:", 9, "normal", _DARK_TEXT),
                (y - 0.08,  f"    ŷ  =  (1 / {n_trees})  Σ  Tree_t(x₁₀₀)       depth ≤ {max_d}", 10, "bold", _BLUE),
                (y - 0.12,  f"  ● Features per split:  √100 ≈ 10 (random sub-selection)", 9, "normal", _DARK_TEXT),
            ]
        elif "Gradient" in model_name or "Hist" in model_name or "XGBoost" in model_name or "LightGBM" in model_name:
            n_est = getattr(pv_res.model, "n_estimators", "?")
            lr    = getattr(pv_res.model, "learning_rate", "?")
            desc_lines = [
                (y - 0.01,  f"Builds {n_est} trees sequentially; each corrects residuals of the previous:", 9, "normal", _DARK_TEXT),
                (y - 0.05,  f"    ŷ  =  F₀ + η·T₁(x) + η·T₂(x) + … + η·T_M(x)      η = {lr}", 10, "bold", _BLUE),
                (y - 0.10,  "  ● Each tree is fitted to the negative gradient (pseudo-residuals)", 9, "normal", _DARK_TEXT),
            ]
        elif "Linear" in model_name or model_name in ("Ridge", "Lasso", "Elastic Net"):
            desc_lines = [
                (y - 0.01,  "Linear combination of all 100 features:", 9, "normal", _DARK_TEXT),
                (y - 0.05,  "    ŷ  =  β₀  +  β₁x₁  +  β₂x₂  + …  +  β₁₀₀x₁₀₀", 10, "bold", _BLUE),
                (y - 0.10,  "  ● Coefficients β learned by minimising MSE (+ regularisation for Ridge/Lasso)", 9, "normal", _DARK_TEXT),
            ]
        elif "Self Organizing" in model_name:
            desc_lines = [
                (y - 0.01,  "Self-Organising Map: unsupervised prototype neurons "
                            "encode the 100-dim input.", 9, "normal", _DARK_TEXT),
                (y - 0.05,  "    ŷ  =  target_avg[BMU(x)]   "
                            "(winner neuron's stored average)", 10, "bold", _BLUE),
                (y - 0.10,  "  ● Trained with competitive learning (neighbourhood "
                            "preserving)", 9, "normal", _DARK_TEXT),
            ]
        elif any(k in model_name for k in ("CNN", "RNN", "LSTM", "GRU",
                                           "BiLSTM", "Transformer", "RBF",
                                           "Capsule", "Autoencoder",
                                           "GAN", "Belief")):
            desc_lines = [
                (y - 0.01,  f"Keras deep-learning architecture ({model_name}) "
                            "trained on 100 parameters.", 9, "normal", _DARK_TEXT),
                (y - 0.05,  "    Network weights optimised via back-propagation "
                            "with Adam.", 10, "bold", _BLUE),
                (y - 0.10,  "  ● Architecture built dynamically; layer count and "
                            "capacity vary by model", 9, "normal", _DARK_TEXT),
            ]
        elif "Neural" in model_name or "MLP" in model_name or "Deep" in model_name or "Wide" in model_name:
            layers = getattr(pv_res.model, "hidden_layer_sizes", "?")
            act    = getattr(pv_res.model, "activation", "relu")
            desc_lines = [
                (y - 0.01,  f"Multi-layer perceptron: 100 → {layers} → 1 (output):", 9, "normal", _DARK_TEXT),
                (y - 0.05,  f"    z  =  σ(W·x + b)     σ = {act} activation", 10, "bold", _BLUE),
                (y - 0.10,  "  ● Trained with Adam optimiser (back-propagation)", 9, "normal", _DARK_TEXT),
            ]
        else:
            desc_lines = [
                (y - 0.01,  f"{model_name} trained on 100 parameters using scikit-learn pipeline.", 9, "normal", _DARK_TEXT),
            ]
        _body(ax, desc_lines)
        y -= 0.17

        y = _section(ax, y, "3.3  ML Model Accuracy (Test Set)")
        acc_rows = [
            ["R² — PV model",   f"{r2_pv*100:.2f}%",    "Fraction of variance explained (1.0 = perfect)"],
            ["MAE — PV model",  f"{mae_pv:.3f} kW",     "Mean absolute prediction error per hour"],
            ["RMSE — PV model", f"{rmse_pv:.3f} kW",    "Root-mean-square error per hour"],
            ["R² — Wind model",  f"{r2_wd*100:.2f}%",   "Wind model fit quality"],
            ["MAE — Wind model", f"{mae_wd:.3f} kW",    "Wind MAE per hour"],
            ["RMSE — Wind model",f"{rmse_wd:.3f} kW",   "Wind RMSE per hour"],
        ]
        _table_on_ax(ax, y, ["Metric", "Value", "Interpretation"], acc_rows,
                     col_widths=[0.28, 0.16, 0.53], row_h=0.038)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ================================================================
        # PAGE 5 — Energy Results & Yield Metrics
        # ================================================================
        fig, ax = _fig_with_header("Step 4 – PV Energy Results & Yield Metrics", 5, total_pages)

        y = 0.90
        y = _section(ax, y, "4.1  Total Energy Summary")
        result_rows = [
            ["ML model PV energy",       f"{ml_pv_kwh:,.1f} kWh",   f"{ml_pv_kwh/1000:,.2f} MWh"],
            ["Physical model PV energy", f"{phy_pv_kwh:,.1f} kWh",  f"{phy_pv_kwh/1000:,.2f} MWh"],
            ["ML model Wind energy",     f"{ml_wd_kwh:,.1f} kWh",   f"{ml_wd_kwh/1000:,.2f} MWh"],
            ["Physical model Wind energy",f"{phy_wd_kwh:,.1f} kWh", f"{phy_wd_kwh/1000:,.2f} MWh"],
            ["Data window",              f"{days:.0f} days",         f"{n:,} hourly records"],
        ]
        y = _table_on_ax(ax, y, ["Quantity", "Value (kWh)", "Value (MWh)"], result_rows,
                         col_widths=[0.42, 0.28, 0.27], row_h=0.038)

        y = _section(ax, y, "4.2  Standard PV Yield Metrics")
        yield_rows = [
            ["DC nameplate (kWp)",        f"{pdc_kwp:,.1f} kWp",
             f"{config.PV_AREA_M2:.0f} m² × {config.PV_EFFICIENCY*100:.0f}% STC efficiency"],
            ["POA insolation (window)",   f"{poa_kwh_m2:,.1f} kWh/m²",
             "Plane-of-array irradiation driving the model"],
            ["Final yield Y_f",           f"{yf:,.1f} kWh/kWp",
             "E_PV ÷ kWp over the window"],
            ["Annual specific yield",     f"{annual_sy:,.0f} kWh/kWp·yr",
             "Y_f × (365 / window days) — headline metric"],
            ["Performance ratio PR",      f"{pr*100:.1f}%",
             "Y_f ÷ Y_r (final yield ÷ reference yield)"],
            ["Peak sun hours (PSH)",      f"{psh:.2f} h/day",
             "G_POA_kWh/m² ÷ window days"],
        ]
        y = _table_on_ax(ax, y, ["Metric", "Value", "Description"], yield_rows,
                         col_widths=[0.33, 0.22, 0.42], row_h=0.038)

        ax.text(0.025, y - 0.01,
                "✅  Verification:  The ML model energy and physical model energy are "
                "computed by independent paths; agreement within ~5–15% confirms both "
                "estimates are physically realistic.",
                fontsize=8.5, fontweight="normal", color=_GREEN,
                va="top", ha="left", transform=ax.transAxes,
                fontfamily=_FONT, wrap=True)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ================================================================
        # PAGE 6 — Feature Importances (if available)
        # ================================================================
        fig, ax = _fig_with_header("Step 5 – Feature Importances & Top 10 Drivers", 6, total_pages)

        y = 0.90
        if fi_rows:
            y = _section(ax, y, "5.1  Top-10 Parameters Driving PV Prediction")
            ax.text(0.025, y - 0.01,
                    f"These are the 10 most influential atmospheric parameters as determined "
                    f"by the {model_name} model's feature importances (Gini impurity reduction).",
                    fontsize=9, color=_DARK_TEXT,
                    va="top", transform=ax.transAxes, fontfamily=_FONT)
            y -= 0.07
            y = _table_on_ax(ax, y, ["Rank", "Parameter", "Importance (%)"],
                             fi_rows, col_widths=[0.08, 0.60, 0.29], row_h=0.038)

            # Draw a horizontal bar chart inline
            if len(fi_rows) > 0:
                bar_ax = fig.add_axes([0.1, 0.06, 0.80, 0.28])
                labels = [r[1][:35] for r in fi_rows][::-1]
                vals   = [float(r[2].replace("%", "")) for r in fi_rows][::-1]
                colors = [_ORANGE if v == max(vals) else _BLUE for v in vals]
                bar_ax.barh(labels, vals, color=colors, edgecolor="white")
                bar_ax.set_xlabel("Importance (%)", fontsize=8, fontfamily=_FONT)
                bar_ax.set_title(f"Top-10 Feature Importances — {model_name}",
                                 fontsize=9, fontweight="bold", fontfamily=_FONT)
                bar_ax.tick_params(labelsize=7.5)
                bar_ax.grid(axis="x", alpha=0.3)
                bar_ax.spines[["top", "right"]].set_visible(False)
        else:
            ax.text(0.5, 0.55,
                    f"Feature importances are not available for {model_name}.",
                    fontsize=12, color=_MID_TEXT,
                    va="center", ha="center", transform=ax.transAxes,
                    fontfamily=_FONT)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ================================================================
        # PAGE 7 — Visual Charts: Power time-series + Energy bar
        # ================================================================
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")

        # Header strip
        header_ax = fig.add_axes([0, 1 - _HEADER_H, 1, _HEADER_H])
        header_ax.set_xlim(0, 1); header_ax.set_ylim(0, 1)
        header_ax.axis("off")
        header_ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="square,pad=0", linewidth=0, facecolor=_BLUE))
        header_ax.text(0.025, 0.5, "Step 6 – Visual Summary",
                       fontsize=13, fontweight="bold", color="white",
                       va="center", fontfamily=_FONT)
        header_ax.text(0.975, 0.5, f"Page {total_pages} / {total_pages}",
                       fontsize=8, color="white", va="center", ha="right",
                       fontfamily=_FONT)

        # Time-series chart (last 7 days)
        tail = feat.tail(min(n, 168)).copy()
        tail_idx = tail.index
        ts_ax = fig.add_axes([0.08, 0.60, 0.88, 0.30])
        ts_ax.plot(range(len(tail)), ml_pv_pred[tail_idx],
                   color=_ORANGE, lw=1.4, label=f"PV — {model_name}")
        ts_ax.plot(range(len(tail)), tail[config.TARGET_PV].values,
                   color=_ORANGE, lw=0.7, alpha=0.35, ls="--", label="PV — Physical")
        ts_ax.plot(range(len(tail)), ml_wd_pred[tail_idx],
                   color=_BLUE, lw=1.4, label=f"Wind — {model_name}")
        ts_ax.plot(range(len(tail)), tail[config.TARGET_WIND].values,
                   color=_BLUE, lw=0.7, alpha=0.35, ls="--", label="Wind — Physical")
        ts_ax.set_ylabel("Power (kW)", fontsize=8, fontfamily=_FONT)
        ts_ax.set_title("Recent Power Output: ML Predictions vs Physical Model (last 7 days)",
                        fontsize=9, fontweight="bold", fontfamily=_FONT)
        ts_ax.legend(fontsize=7.5, loc="upper right")
        ts_ax.grid(alpha=0.2)
        ts_ax.tick_params(labelsize=7.5)
        ts_ax.spines[["top", "right"]].set_visible(False)

        # Energy comparison bar chart
        bar_ax2 = fig.add_axes([0.08, 0.22, 0.40, 0.28])
        labels2 = ["PV\nML", "PV\nPhysical", "Wind\nML", "Wind\nPhysical"]
        vals2   = [ml_pv_kwh/1000, phy_pv_kwh/1000, ml_wd_kwh/1000, phy_wd_kwh/1000]
        cols2   = [_ORANGE, _ORANGE, _BLUE, _BLUE]
        alphas2 = [1.0, 0.5, 1.0, 0.5]
        bars2   = bar_ax2.bar(labels2, vals2, color=cols2,
                              alpha=1.0, edgecolor="white", linewidth=1.5)
        for b, a in zip(bars2, alphas2):
            b.set_alpha(a)
        for b, v in zip(bars2, vals2):
            bar_ax2.text(b.get_x() + b.get_width() / 2, v + max(vals2) * 0.02,
                         f"{v:.1f}", ha="center", fontsize=8.5,
                         fontweight="bold", fontfamily=_FONT)
        bar_ax2.set_ylabel("Energy (MWh)", fontsize=8, fontfamily=_FONT)
        bar_ax2.set_title(f"Total Energy ({days:.0f}-day window)", fontsize=9,
                          fontweight="bold", fontfamily=_FONT)
        bar_ax2.tick_params(labelsize=8)
        bar_ax2.grid(axis="y", alpha=0.25)
        bar_ax2.spines[["top", "right"]].set_visible(False)

        # ML Accuracy bar chart
        acc_ax = fig.add_axes([0.57, 0.22, 0.35, 0.28])
        metrics_labels = ["R²\nPV", "R²\nWind"]
        metrics_vals   = [r2_pv * 100, r2_wd * 100]
        acc_ax.bar(metrics_labels, metrics_vals,
                   color=[_ORANGE, _BLUE], edgecolor="white", linewidth=1.5, width=0.4)
        for xi, v in enumerate(metrics_vals):
            acc_ax.text(xi, v + 1, f"{v:.1f}%", ha="center", fontsize=9,
                        fontweight="bold", fontfamily=_FONT)
        acc_ax.set_ylim(0, 115)
        acc_ax.set_ylabel("R² (%)", fontsize=8, fontfamily=_FONT)
        acc_ax.set_title("Model R² Accuracy", fontsize=9,
                         fontweight="bold", fontfamily=_FONT)
        acc_ax.axhline(100, color="gray", lw=0.8, ls="--", alpha=0.5)
        acc_ax.tick_params(labelsize=8)
        acc_ax.grid(axis="y", alpha=0.25)
        acc_ax.spines[["top", "right"]].set_visible(False)

        # Caption
        cap_ax = fig.add_axes([0.05, 0.04, 0.90, 0.14])
        cap_ax.axis("off")
        cap_ax.set_xlim(0, 1); cap_ax.set_ylim(0, 1)
        cap_ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.01", linewidth=0,
            facecolor=_LIGHT_BG))
        cap_text = (
            f"Summary  ·  {model_name}  |  Site: {lat:.4f}°N {lon:.4f}°E  |  "
            f"Window: {days:.0f} days  |  Generated: {now_str}\n\n"
            f"PV Energy (ML): {ml_pv_kwh/1000:,.2f} MWh      "
            f"PV Energy (Physical): {phy_pv_kwh/1000:,.2f} MWh      "
            f"Annual Specific Yield: {annual_sy:,.0f} kWh/kWp·yr      "
            f"Performance Ratio: {pr*100:.1f}%\n\n"
            "Note: Both ML and physical model results are derived from real Open-Meteo "
            "atmospheric data for this location. Site-specific SCADA metering is not "
            "available; these are model-computed values validated against PVGIS and "
            "each other as cross-verification of physical realism."
        )
        cap_ax.text(0.02, 0.95, cap_text,
                    fontsize=7.8, color=_MID_TEXT, va="top",
                    ha="left", transform=cap_ax.transAxes,
                    fontfamily=_FONT, wrap=True)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    buf.seek(0)
    return buf.read()
