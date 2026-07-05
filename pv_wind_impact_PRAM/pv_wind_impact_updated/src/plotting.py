"""
src/plotting.py
===============
Matplotlib figure helpers shared by the Streamlit app and the CLI report.
Every function returns a Matplotlib Figure (no Streamlit calls here).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")            # safe default; Streamlit/CLI both fine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def scatter(df: pd.DataFrame, xcol: str, ycol: str, color: str):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.scatter(df[xcol], df[ycol], s=6, alpha=0.25, color=color, edgecolors="none")
    ax.set_xlabel(config.pretty(xcol))
    ax.set_ylabel(config.pretty(ycol))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def pred_vs_actual(y_test, y_pred, color: str, label: str):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.scatter(y_test, y_pred, s=6, alpha=0.2, color=color, edgecolors="none")
    hi = float(max(np.max(y_test), np.max(y_pred))) * 1.02 + 1e-6
    ax.plot([0, hi], [0, hi], "k--", lw=1)
    ax.set_xlabel(f"Actual {label} (kW)")
    ax.set_ylabel(f"Predicted {label} (kW)")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def importance_bar(imp: pd.DataFrame, color: str, title: str,
                   top: int = 12, value_col: str = "Share"):
    """Horizontal bar chart of the top-N most important features."""
    d = imp.head(top).iloc[::-1]                       # largest at the top
    labels = d["Label"] if "Label" in d else d["Feature"]
    vals = (d[value_col] * 100).to_numpy() if value_col == "Share" else d[value_col].to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bars = ax.barh(labels, vals, color=color)
    vmax = vals.max() if len(vals) and vals.max() > 0 else 1.0
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + vmax * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%" if value_col == "Share" else f"{val:.3f}",
                va="center", fontsize=8)
    ax.set_xlim(0, vmax * 1.18)
    ax.set_xlabel("Relative impact (%)" if value_col == "Share" else value_col)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def grouped_donut(grp: pd.DataFrame, title: str):
    """Donut chart of importance rolled up by physical category."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    shares = grp["Share"].to_numpy() * 100
    labels = grp["Category"].to_numpy()
    palette = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    wedges, _ = ax.pie(shares, colors=palette, startangle=90,
                       wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.legend(wedges, [f"{l} ({s:.0f}%)" for l, s in zip(labels, shares)],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def power_timeseries(df: pd.DataFrame, hours: int = 168):
    """Recent PV / wind power output over time."""
    d = df.tail(hours)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(d["Datetime"], d[config.TARGET_PV], color=config.PV_COLOR,
            lw=1.2, label="PV (kW)")
    ax.plot(d["Datetime"], d[config.TARGET_WIND], color=config.WIND_COLOR,
            lw=1.2, label="Wind (kW)")
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def accuracy_comparison_bar(pv_lb: pd.DataFrame, wd_lb: pd.DataFrame,
                            metric: str = "R2_pct"):
    """
    Grouped horizontal bar chart comparing every ML algorithm's accuracy
    (default R² %) for PV vs Wind, sorted by the PV score.
    """
    merged = pv_lb[["Model", metric]].merge(
        wd_lb[["Model", metric]], on="Model", suffixes=("_pv", "_wd"))
    merged = merged.sort_values(f"{metric}_pv").reset_index(drop=True)

    models = merged["Model"].to_numpy()
    pv = merged[f"{metric}_pv"].to_numpy()
    wd = merged[f"{metric}_wd"].to_numpy()
    y = np.arange(len(models))
    h = 0.4

    fig, ax = plt.subplots(figsize=(7.6, max(4.5, 0.5 * len(models) + 1)))
    ax.barh(y + h / 2, pv, height=h, color=config.PV_COLOR, label="PV model")
    ax.barh(y - h / 2, wd, height=h, color=config.WIND_COLOR, label="Wind model")
    is_pct = metric.endswith("pct")
    for yi, v in zip(y + h / 2, pv):
        ax.text(v + (0.5 if is_pct else max(pv) * 0.01), yi,
                f"{v:.1f}{'%' if is_pct else ''}", va="center", fontsize=7.5)
    for yi, v in zip(y - h / 2, wd):
        ax.text(v + (0.5 if is_pct else max(wd) * 0.01), yi,
                f"{v:.1f}{'%' if is_pct else ''}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("Accuracy  (R² %)" if is_pct else metric)
    if is_pct:
        ax.set_xlim(min(70, float(min(pv.min(), wd.min())) - 5), 102)
    ax.set_title("ML algorithm accuracy — PV vs Wind")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def critical_param_bar(imp: pd.DataFrame, color: str, title: str):
    """Horizontal bar of the 15 critical parameters by their impact share."""
    d = imp.iloc[::-1]
    labels = d["Label"] if "Label" in d else d["Feature"]
    vals = (d["Share"] * 100).to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.barh(labels, vals, color=color)
    vmax = vals.max() if len(vals) and vals.max() > 0 else 1.0
    for i, v in enumerate(vals):
        ax.text(v + vmax * 0.015, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, vmax * 1.18)
    ax.set_xlabel("Relative impact (%)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def pvgis_monthly_compare(comp: pd.DataFrame):
    """Grouped monthly PV energy: this project vs PVGIS (independent)."""
    months = comp["Month"].tolist()
    x = np.arange(len(months))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.bar(x - w / 2, comp["Energy_kWh"].to_numpy(), width=w,
           color="#2E86AB", label="PVGIS (typical year)")
    proj = comp["Project_kWh"].to_numpy(dtype=float)
    ax.bar(x + w / 2, np.nan_to_num(proj), width=w,
           color=config.PV_COLOR, label="This project (covered months)")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel("PV energy (kWh)")
    ax.set_title("Monthly PV energy — project vs PVGIS")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
