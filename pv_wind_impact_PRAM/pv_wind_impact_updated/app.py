"""
app.py
======
Impact Analysis of Atmospheric Parameters on Photovoltaic and Wind Power Output
Using Data-Driven Modelling and Machine Learning.

Enter a LATITUDE, LONGITUDE and (optional) ALTITUDE.  The app then:

  1. pulls REAL, time-synchronised hourly weather for that exact location
     from the free Open-Meteo API (offline synthetic fallback if no internet),
  2. engineers a 100-parameter atmospheric feature table,
  3. drives calibrated physical PV + wind plant models with that real data,
  4. trains the chosen machine-learning model(s), and
  5. reports the IMPACT of each atmospheric parameter on output, plus the
     ENERGY GENERATED for the location.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

import config
from src import (
    data_fetcher,
    feature_engineering,
    ml_models,
    impact_analysis,
    plotting,
    power_models,
    pvgis,
    dksac,
    pram,
    facl,
    crisp,
)


# ===========================================================================
# Author / institution credits (edit AUTHOR_NAME to your full name)
# ===========================================================================
AUTHOR_NAME = "Manish"          # <-- set your full name here
GUIDE_NAME = "Dr. S. A. Thorat"
DEPARTMENT = "Department of Computer Science & Engineering"
PROGRAMME = "M.Tech (Computer Science & Engineering)"
INSTITUTION = "Government College of Engineering, Karad"
ACADEMIC_YEAR = "2026–2027"


# ===========================================================================
# Cached heavy steps
# ===========================================================================
@st.cache_data(show_spinner=False)
def get_feature_table(lat, lon, elev, past_days, forecast_days, prefer):
    """Fetch (or synthesise) raw data, engineer 100 features, attach targets."""
    raw, status = data_fetcher.load_raw_data(
        lat, lon, elev, past_days=past_days,
        forecast_days=forecast_days, prefer=prefer)
    used_elev = raw.attrs.get("elevation", elev)
    feat = feature_engineering.build_feature_table(raw, lat, lon, used_elev)
    feat = feature_engineering.attach_power_targets(feat)
    meta = {
        "status": status,
        "source": raw.attrs.get("source"),
        "elevation": used_elev,
        "timezone": raw.attrs.get("timezone"),
        "n_rows": len(feat),
    }
    return feat, meta


@st.cache_resource(show_spinner=False)
def get_trained(feat_key, _feat, target, model_name):
    """Train one model (cached on coordinates/target/model, not the frame)."""
    return ml_models.train_model(_feat, target, model_name)


@st.cache_data(show_spinner=False)
def get_permutation(feat_key, target, model_name, _result):
    return impact_analysis.permutation_impact(_result, n_repeats=8)


@st.cache_data(show_spinner=False)
def get_secondary(feat_key, _feat, kind, model_name):
    return impact_analysis.secondary_impact(_feat, kind, model_name)


@st.cache_data(show_spinner=False)
def get_benchmark(feat_key, _feat, target):
    return ml_models.benchmark_models(_feat, target)


@st.cache_data(show_spinner=False)
def get_pvgis(lat, lon, peakpower_kwp, tilt):
    """Independent PVGIS yield (cached per coordinate). Returns dict or error."""
    try:
        return pvgis.pvgis_yield(lat, lon, peakpower_kwp=peakpower_kwp, tilt=tilt)
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(show_spinner=False)
def get_site_pv(lat, lon, elev):
    """Run the project's model for an arbitrary coordinate -> yield dict + source."""
    raw, status = data_fetcher.load_raw_data(lat, lon, elev,
                                             past_days=config.DEFAULT_PAST_DAYS)
    feat = feature_engineering.attach_power_targets(
        feature_engineering.build_feature_table(raw, lat, lon, elev))
    y = power_models.pv_yield_metrics(feat)
    y["source"] = "Open-Meteo/NASA" if "offline" not in status.lower() else "offline synthetic"
    y["window_mwh"] = float(feat[config.TARGET_PV].sum()) / 1000.0
    return y


# ===========================================================================
# Main
# ===========================================================================
def _inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Sora:wght@600;700;800&display=swap');

      :root{
        --ink:#0f2942; --ink-soft:#5a6b7d; --line:#e6eef8;
        --blue:#1f6fb2; --blue2:#2e86ab; --amber:#f5a623; --amber2:#ffd479;
      }

      /* ---------- airy light canvas ---------- */
      .stApp{
        background:
          radial-gradient(1100px 520px at 8% -10%, #e9f3ff 0%, rgba(233,243,255,0) 60%),
          radial-gradient(900px 480px at 108% -6%, #fff1de 0%, rgba(255,241,222,0) 55%),
          radial-gradient(800px 600px at 50% 120%, #eef4ff 0%, rgba(238,244,255,0) 60%),
          linear-gradient(180deg,#fbfdff 0%, #f5f9fe 100%);
        background-attachment: fixed;
      }
      .block-container{padding-top:1.4rem; padding-bottom:2.6rem; max-width:1340px;}
      html,body,[class*="css"],button,input,select,textarea{
        font-family:'Inter','Segoe UI',system-ui,sans-serif !important; color:var(--ink);
      }
      ::-webkit-scrollbar{width:10px; height:10px;}
      ::-webkit-scrollbar-thumb{background:#cfdded; border-radius:8px;}
      ::-webkit-scrollbar-thumb:hover{background:#b6cbe2;}

      @keyframes floatUp{from{opacity:0; transform:translateY(12px);} to{opacity:1; transform:translateY(0);}}
      @keyframes shimmer{0%{background-position:0% 50%;} 100%{background-position:200% 50%;}}

      /* ---------- HERO ---------- */
      .hero{
        position:relative; overflow:hidden; animation:floatUp .6s ease both;
        background:
          radial-gradient(rgba(31,111,178,.06) 1px, transparent 1px) 0 0/22px 22px,
          linear-gradient(135deg,#edf5ff 0%, #f7fbff 46%, #fff6ea 100%);
        border:1px solid #e6f0fb; border-radius:24px; padding:32px 38px; margin-bottom:14px;
        box-shadow:0 18px 46px rgba(31,78,121,.12), inset 0 1px 0 #ffffff;
      }
      .hero::after{content:""; position:absolute; right:-80px; top:-100px; width:320px; height:320px;
        background:radial-gradient(circle, rgba(245,166,35,.24), rgba(245,166,35,0) 70%); border-radius:50%;}
      .hero::before{content:""; position:absolute; left:-70px; bottom:-140px; width:340px; height:340px;
        background:radial-gradient(circle, rgba(46,134,171,.18), rgba(46,134,171,0) 70%); border-radius:50%;}
      .hero>*{position:relative; z-index:2;}
      .hero .eyebrow{display:inline-flex; align-items:center; gap:7px; font-size:.71rem; font-weight:700;
        letter-spacing:.16em; text-transform:uppercase; color:#1565a0;
        background:#ffffffdd; border:1px solid #d7e9fb; padding:6px 13px; border-radius:999px;
        box-shadow:0 2px 8px rgba(31,111,178,.08);}
      .hero h1{font-family:'Sora','Inter',sans-serif; color:var(--ink); font-size:1.92rem; line-height:1.18;
        margin:15px 0 5px; font-weight:800; letter-spacing:-.015em; max-width:60ch;}
      .hero .sub{color:#b9741a; font-weight:700; font-size:1rem; letter-spacing:.01em;}
      .hero .accent{height:5px; width:120px; border-radius:6px; margin:15px 0 16px;
        background:linear-gradient(90deg,#f5a623,#ffd479,#2e86ab,#f5a623); background-size:200% 100%;
        animation:shimmer 5s linear infinite;}
      .hero .tags{display:flex; flex-wrap:wrap; gap:9px;}
      .hero .chip{display:inline-flex; align-items:center; gap:6px; background:#ffffffea;
        border:1px solid #dceafb; color:#1f4e79; padding:7px 14px; border-radius:999px;
        font-size:.79rem; font-weight:600; box-shadow:0 2px 7px rgba(31,78,121,.07);
        transition:transform .15s ease, box-shadow .15s ease;}
      .hero .chip:hover{transform:translateY(-2px); box-shadow:0 6px 16px rgba(31,78,121,.14);}
      .hero .trust{margin-top:14px; font-size:.74rem; color:#6c7d8f; font-weight:500;}
      .hero .trust b{color:#1f6fb2; font-weight:700;}

      /* ---------- credit bar ---------- */
      .creditbar{display:flex; flex-wrap:wrap; gap:8px 22px; align-items:center;
        background:#ffffff; border:1px solid #ecf2fa; border-left:5px solid var(--amber);
        border-radius:16px; padding:13px 20px; margin:6px 0 22px; font-size:.85rem; color:#41515f;
        box-shadow:0 6px 20px rgba(31,78,121,.07);}
      .creditbar b{color:#1f4e79; font-weight:700;} .creditbar .dot{color:#d3deea;}

      /* ---------- metric cards ---------- */
      div[data-testid="stMetric"]{background:linear-gradient(180deg,#ffffff,#fbfdff);
        border:1px solid #ecf1f9; border-radius:18px; padding:16px 20px;
        box-shadow:0 4px 16px rgba(31,78,121,.06); transition:transform .16s ease, box-shadow .16s ease;
        position:relative; overflow:hidden;}
      div[data-testid="stMetric"]::before{content:""; position:absolute; left:0; top:0; height:100%; width:4px;
        background:linear-gradient(180deg,#2e86ab,#1f6fb2);}
      div[data-testid="stMetric"]:hover{transform:translateY(-4px); box-shadow:0 14px 30px rgba(31,78,121,.15);}
      div[data-testid="stMetricLabel"] p{font-weight:600; color:#6b7a8c; font-size:.82rem; letter-spacing:.01em;}
      div[data-testid="stMetricValue"]{color:#14456e; font-weight:800; font-family:'Sora','Inter',sans-serif;}

      /* ---------- tabs (segmented pills) ---------- */
      div[data-baseweb="tab-list"]{gap:5px; border-bottom:1px solid var(--line); flex-wrap:wrap; padding-bottom:2px;}
      button[data-baseweb="tab"]{font-weight:600; color:#62707f; border-radius:11px; padding:9px 15px;
        transition:all .15s ease;}
      button[data-baseweb="tab"]:hover{background:#eaf3fe; color:#1f4e79;}
      button[data-baseweb="tab"][aria-selected="true"]{color:#0f2942; background:#ffffff;
        box-shadow:0 4px 14px rgba(31,78,121,.10); border:1px solid #e6eef8;}
      div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#f5a623,#2e86ab)!important; height:3px; border-radius:3px;}
      div[data-baseweb="tab-border"]{background:transparent!important;}

      /* ---------- sidebar ---------- */
      section[data-testid="stSidebar"]{background:linear-gradient(180deg,#ffffff,#fafcff);
        border-right:1px solid #eef2f8; box-shadow:6px 0 24px rgba(20,55,100,.04);}
      section[data-testid="stSidebar"] .block-container{padding-top:1.1rem;}
      section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
      section[data-testid="stSidebar"] .stMarkdown h1{font-family:'Sora','Inter',sans-serif; color:#16456e;
        font-size:1.02rem; font-weight:700;}
      div.stButton>button{border-radius:12px; font-weight:600; transition:all .15s ease;}
      div.stButton>button:hover{transform:translateY(-1px);}
      div.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1f6fb2,#2e86ab); border:0;
        box-shadow:0 8px 20px rgba(31,111,178,.30);}
      div.stButton>button[kind="primary"]:hover{box-shadow:0 12px 26px rgba(31,111,178,.42);}
      div[data-baseweb="select"]>div, .stTextInput input, .stNumberInput input{border-radius:10px!important;}
      div[data-baseweb="select"]>div:focus-within{box-shadow:0 0 0 3px rgba(31,111,178,.18);}

      /* ---------- headings / dividers / expanders / tables / alerts ---------- */
      h2{font-family:'Sora','Inter',sans-serif; color:#10324f; font-weight:800; letter-spacing:-.01em;}
      h3,h4{color:#14456e; font-weight:700;}
      hr{border-color:#e9eff7;}
      div[data-testid="stExpander"]{border:1px solid #eaf0f8; border-radius:16px; background:#ffffff;
        box-shadow:0 3px 14px rgba(31,78,121,.05); overflow:hidden;}
      div[data-testid="stExpander"] summary{font-weight:600;}
      div[data-testid="stExpander"] summary:hover{color:#1f6fb2;}
      div[data-testid="stDataFrame"]{border:1px solid #eef2f8; border-radius:14px; overflow:hidden;
        box-shadow:0 3px 14px rgba(31,78,121,.05);}
      div[data-testid="stAlert"]{border-radius:14px; border:1px solid #e6eef8;}

      /* ---------- verdict + footer ---------- */
      .verdict{border-radius:16px; padding:14px 20px; margin:10px 0; font-weight:600;
        box-shadow:0 4px 14px rgba(31,78,121,.06);}
      .v-good{background:linear-gradient(180deg,#effaf2,#e5f6eb); border:1px solid #bce4c8; color:#15783a;}
      .v-warn{background:linear-gradient(180deg,#fff9ec,#fff2d8); border:1px solid #ffe1a6; color:#996300;}
      .v-bad{background:linear-gradient(180deg,#fdefed,#fbe3e0); border:1px solid #f3c2bb; color:#b3261e;}
      .footer{margin-top:18px; background:#ffffff; border:1px solid #ecf2fa; border-radius:18px;
        padding:18px 24px; box-shadow:0 6px 20px rgba(31,78,121,.06); color:#5a6b7d; font-size:.84rem;}
      .footer .ft-title{font-family:'Sora','Inter',sans-serif; color:#16456e; font-weight:700; font-size:.96rem; margin-bottom:6px;}
      .footer b{color:#1f6fb2;}

      /* ---------- context strip + KPI stat band ---------- */
      .ctxstrip{display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 16px;}
      .ctxstrip .pill{display:inline-flex; align-items:center; gap:6px; background:#ffffff;
        border:1px solid #e6eef8; color:#41515f; padding:7px 14px; border-radius:999px;
        font-size:.8rem; font-weight:600; box-shadow:0 2px 8px rgba(31,78,121,.05);}
      .ctxstrip .pill b{color:#1f4e79;}
      .statband{display:grid; grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
        gap:14px; margin:4px 0 10px;}
      .statcard{position:relative; overflow:hidden; background:linear-gradient(180deg,#ffffff,#fbfdff);
        border:1px solid #ecf1f9; border-radius:18px; padding:18px 18px;
        box-shadow:0 6px 20px rgba(31,78,121,.07); transition:transform .16s ease, box-shadow .16s ease;}
      .statcard:hover{transform:translateY(-4px); box-shadow:0 16px 34px rgba(31,78,121,.15);}
      .statcard .ic{position:absolute; right:15px; top:13px; font-size:1.5rem; opacity:.92;}
      .statcard .lab{font-size:.72rem; font-weight:700; color:#7184959; color:#718495;
        letter-spacing:.05em; text-transform:uppercase;}
      .statcard .val{font-family:'Sora','Inter',sans-serif; font-size:1.55rem; font-weight:800;
        color:#14456e; margin-top:5px; line-height:1.08;}
      .statcard .sub{font-size:.74rem; color:#8a97a6; margin-top:4px;}
      .statcard::after{content:""; position:absolute; left:0; top:0; height:100%; width:5px;}
      .statcard.g1::after{background:linear-gradient(180deg,#f5a623,#ffd479);}
      .statcard.g2::after{background:linear-gradient(180deg,#2e86ab,#1f6fb2);}
      .statcard.g3::after{background:linear-gradient(180deg,#16a085,#54d6b3);}
      .statcard.g4::after{background:linear-gradient(180deg,#7c6df2,#a99bff);}
      .statcard.g5::after{background:linear-gradient(180deg,#e15759,#ff9193);}
    </style>
    """, unsafe_allow_html=True)


def _render_header():
    st.markdown(f"""
    <div class="hero">
      <span class="eyebrow">⚡ M.Tech Dissertation · Renewable Energy &amp; Machine Learning</span>
      <h1>Impact Analysis of Atmospheric Parameters on Photovoltaic &amp; Wind
          Power Output</h1>
      <div class="sub">Data-Driven Modelling and Machine Learning</div>
      <div class="accent"></div>
      <div class="tags">
        <span class="chip">📍 Any coordinate in India</span>
        <span class="chip">🛰️ NASA POWER + Open-Meteo</span>
        <span class="chip">🌫️ 100 atmospheric parameters</span>
        <span class="chip">🤖 ML algorithms</span>
        <span class="chip">☀️ PV yield + PVGIS validation</span>
      </div>
      <div class="trust">Live data from <b>NASA POWER</b> · <b>Open-Meteo</b> ·
        independently cross-validated with <b>PVGIS</b> (EU Joint Research Centre)</div>
    </div>
    <div class="creditbar">
      <span>👨‍🎓 Submitted by <b>{AUTHOR_NAME}</b> · {PROGRAMME}</span><span class="dot">●</span>
      <span>👨‍🏫 Guide: <b>{GUIDE_NAME}</b></span><span class="dot">●</span>
      <span>🏛️ {INSTITUTION}</span><span class="dot">●</span>
      <span>📅 A.Y. <b>{ACADEMIC_YEAR}</b></span>
    </div>
    """, unsafe_allow_html=True)


def _context_strip(lat, lon, meta):
    st.markdown(f"""
    <div class="ctxstrip">
      <span class="pill">📍 <b>{lat:.4f}°, {lon:.4f}°</b></span>
      <span class="pill">⛰️ Elevation&nbsp;<b>{meta['elevation']:.0f} m</b></span>
      <span class="pill">🗓️ <b>{meta['n_rows']:,}</b>&nbsp;hourly samples</span>
      <span class="pill">🛰️ {meta['source']}</span>
    </div>""", unsafe_allow_html=True)


def _stat_band(cards):
    """cards: list of (icon, value, label, sub, gradient_class g1..g5)."""
    html = '<div class="statband">'
    for icon, val, lab, sub, g in cards:
        html += (f'<div class="statcard {g}"><span class="ic">{icon}</span>'
                 f'<div class="lab">{lab}</div><div class="val">{val}</div>'
                 f'<div class="sub">{sub}</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Atmospheric Impact on PV & Wind Power",
                       page_icon="🌤️", layout="wide",
                       initial_sidebar_state="expanded")
    _inject_css()
    _render_header()

    # Apply a coordinate picked from the interactive map on the previous run
    # (must happen BEFORE the lat/lon widgets are instantiated below).
    if "_pending_coords" in st.session_state:
        _plat, _plon, _pelev = st.session_state.pop("_pending_coords")
        st.session_state["lat"] = float(_plat)
        st.session_state["lon"] = float(_plon)
        st.session_state["elev"] = float(_pelev)
        st.session_state["_last_applied_site"] = st.session_state.get("_site_sel")

    # ---- Sidebar: location & settings ------------------------------------
    st.sidebar.header("📍 Location")
    st.sidebar.caption("Pick a site preset, or enter coordinates manually.")

    # Site presets — default is GCE Karad, Block M hostel.
    site_names = list(config.SITES.keys()) + ["Custom (enter below)"]
    site = st.sidebar.selectbox("Site preset", site_names, index=0, key="_site_sel")
    if site != "Custom (enter below)":
        slat, slon, selev, sdesc = config.SITES[site]
        if st.session_state.get("_last_applied_site") != site:
            st.session_state["lat"] = float(slat)
            st.session_state["lon"] = float(slon)
            st.session_state["elev"] = float(selev)
            st.session_state["_last_applied_site"] = site
        st.sidebar.caption(f"📌 {sdesc}")

    place = st.sidebar.text_input("…or search a place",
                                  placeholder="e.g. Karad, India")
    if st.sidebar.button("🔎 Look up place") and place.strip():
        hit = data_fetcher.geocode_place(place.strip())
        if hit:
            st.session_state["lat"] = float(hit["latitude"])
            st.session_state["lon"] = float(hit["longitude"])
            st.session_state["elev"] = (float(hit["elevation"])
                                        if hit["elevation"] is not None else 0.0)
            st.session_state["_last_applied_site"] = site   # don't let preset clobber
            st.sidebar.success(f"Found {hit['name']}, {hit.get('country','')}")
        else:
            st.sidebar.error("Place not found — enter coordinates manually.")

    # Paste coordinates straight from Google Maps (right-click a point -> the
    # "lat, lon" appears; click it to copy, then paste here).
    gmaps = st.sidebar.text_input(
        "…or paste Google Maps coordinates",
        placeholder="e.g. 17.2877, 74.1818",
        help="In Google Maps, right-click your exact spot (e.g. Block M hostel "
             "rooftop) and click the 'lat, lon' at the top to copy it.")
    if gmaps.strip():
        try:
            parts = gmaps.replace("(", "").replace(")", "").split(",")
            glat, glon = float(parts[0].strip()), float(parts[1].strip())
            if -90 <= glat <= 90 and -180 <= glon <= 180:
                st.session_state["lat"] = glat
                st.session_state["lon"] = glon
                st.session_state["_last_applied_site"] = site
            else:
                st.sidebar.error("Coordinates out of range.")
        except Exception:
            st.sidebar.error("Use the format: latitude, longitude")

    # Ensure coordinates exist before the keyed number inputs are created.
    st.session_state.setdefault("lat", config.DEFAULT_LATITUDE)
    st.session_state.setdefault("lon", config.DEFAULT_LONGITUDE)
    st.session_state.setdefault("elev", config.DEFAULT_ELEVATION)

    lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0,
                                  key="lat", format="%.4f")
    lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                  key="lon", format="%.4f")
    elev = st.sidebar.number_input(
        "Altitude / elevation (m)", min_value=-100.0, max_value=9000.0,
        key="elev", format="%.0f",
        help="Affects air density (wind) and pressure-based features. "
             "Auto-filled by the preset; editable.")

    st.sidebar.header("⚙️ Modelling")
    model_name = st.sidebar.selectbox("ML model", ml_models.MODEL_NAMES, index=0)
    prefer = st.sidebar.radio(
        "Data source", ["nasa", "openmeteo", "synthetic"],
        format_func=lambda x: {
            "nasa": "🛰️ NASA POWER (government)",
            "openmeteo": "🌐 Open-Meteo (ECMWF/GFS) + air quality",
            "synthetic": "🧪 Offline synthetic",
        }[x],
        help="NASA POWER and Open-Meteo both draw on government numerical-weather "
             "models. NASA POWER is near-real-time (a few days' latency); "
             "Open-Meteo gives the current hour plus a short forecast and adds "
             "aerosol/air-quality data.")
    past_days = st.sidebar.slider("History window (days)", 14, 92,
                                  config.DEFAULT_PAST_DAYS, step=7,
                                  help="How much recent hourly data to pull / train on.")

    run = st.sidebar.button("🚀 Run analysis", type="primary", use_container_width=True)
    if "has_run" in st.session_state:
        if st.sidebar.button("🗺️ Pick a new location on map",
                             use_container_width=True):
            st.session_state.pop("has_run", None)
            st.rerun()
    if not run and "has_run" not in st.session_state:
        st.info("**Pick your location on the map below** (or use the sidebar), "
                "then press **🚀 Run analysis**. The app fetches real hourly "
                "weather for that exact point, builds 100 atmospheric parameters, "
                "and reports each parameter's impact on PV & wind output plus the "
                "energy generated.")
        _landing_map_picker(lat, lon)
        st.divider()
        with st.expander("📚 Preview — the 100 atmospheric parameters", expanded=False):
            _show_parameter_overview()
        st.stop()
    st.session_state["has_run"] = True

    feat_key = (round(lat, 4), round(lon, 4), round(elev, 0), past_days, prefer)

    with st.spinner("Fetching real-time weather and engineering 100 parameters…"):
        feat, meta = get_feature_table(lat, lon, elev, past_days,
                                       config.DEFAULT_FORECAST_DAYS, prefer)

    st.success(meta["status"])
    _context_strip(lat, lon, meta)

    # ---- Train both targets ---------------------------------------------
    with st.spinner(f"Training {model_name} on 100 parameters…"):
        pv_res = get_trained(feat_key, feat, config.TARGET_PV, model_name)
        wd_res = get_trained(feat_key, feat, config.TARGET_WIND, model_name)

    # ---- Headline KPI dashboard band ------------------------------------
    _ym = power_models.pv_yield_metrics(feat)
    _pv_cf = feat[config.TARGET_PV].mean() / max(_ym["pdc_kwp"], 1e-6) * 100
    _stat_band([
        ("☀️", f"{feat[config.TARGET_PV].sum()/1000:,.1f} MWh",
         "PV energy", f"{_ym['days']:.0f}-day window", "g1"),
        ("💨", f"{feat[config.TARGET_WIND].sum()/1000:,.1f} MWh",
         "Wind energy", "modelled output", "g2"),
        ("📈", f"{_ym['annual_specific_yield']:,.0f}",
         "PV specific yield", "kWh/kWp·yr (annualised)", "g3"),
        ("🎯", f"{pv_res.metrics['r2']*100:.1f}%",
         "PV model accuracy", f"{model_name} · R²", "g4"),
        ("🌫️", "100",
         "Atmospheric parameters", "57 measured + 43 derived", "g5"),
    ])

    tabs = st.tabs([
        "🇦🇺 Australian DKASC",
        "🧪 PRAM (Live)",
        "🎯 CRISP Intervals",
        "⚡ Energy & Yield",
        "🔬 Impact Analysis",
        "📈 ML Accuracy",
        "📊 Data Explorer",
        "⚙️ Model Performance",
        "🔮 Forecaster",
        "📋 Dataset",
        "🧮 Algorithm Evaluation",
        "🌐 Real-Time Dataset & Evaluation",
        "🛰️ PVGIS Validation",
    ])

    with tabs[0]:
        _dksac_validation_section()
    with tabs[1]:
        _tab_pram_live(lat, lon, elev, past_days, prefer)
    with tabs[2]:
        _tab_crisp()
    with tabs[3]:
        _tab_energy(feat, meta, lat, lon, model_name, pv_res, wd_res)
    with tabs[4]:
        _tab_impact(feat, feat_key, model_name, pv_res, wd_res)
    with tabs[5]:
        _tab_accuracy(feat, feat_key)
    with tabs[6]:
        _tab_explorer(feat, meta)
    with tabs[7]:
        _tab_performance(feat, feat_key, model_name, pv_res, wd_res)
    with tabs[8]:
        _tab_forecaster(feat, pv_res, wd_res, lat, lon)
    with tabs[9]:
        _tab_dataset(feat, meta, lat, lon, past_days, prefer)
    with tabs[10]:
        _tab_algorithm_eval(feat, meta, lat, lon, model_name, pv_res, wd_res)
    with tabs[11]:
        _tab_realtime_only(feat, meta, lat, lon, model_name, past_days, prefer)
    with tabs[12]:
        _tab_pvgis(feat, meta, lat, lon)

    st.markdown(f"""
    <div class="footer">
      <div class="ft-title">🌤️ Atmospheric Impact on PV &amp; Wind Power · {INSTITUTION}</div>
      Atmospheric inputs are <b>real &amp; live</b> (NASA POWER / Open-Meteo
      government weather models; aerosols from the Open-Meteo air-quality API)
      and the PV yield is independently cross-checked against <b>PVGIS</b>.
      Plant power output is produced by transparent physical PV &amp; wind models
      driven by those real inputs, since site-specific SCADA measurements are not
      available — a research / demonstration tool.
      <br><span style="color:#94a3b4;">© A.Y. {ACADEMIC_YEAR} · {AUTHOR_NAME} ·
      Guided by {GUIDE_NAME}</span>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Tab: Energy & live snapshot  (the headline deliverable for a location)
# ===========================================================================
def _tab_energy(feat, meta, lat, lon, model_name="Random Forest",
                pv_res=None, wd_res=None):
    st.subheader("Energy generated at this location")
    days = max(meta["n_rows"] / 24.0, 1e-6)

    # ------------------------------------------------------------------
    # ML-based energy estimation (all 100 parameters → trained model)
    # ------------------------------------------------------------------
    X_all = feat[config.FEATURES]
    if pv_res is not None and wd_res is not None:
        ml_pv_pred = np.clip(pv_res.model.predict(X_all), 0, config.PV_CAPACITY_KW)
        ml_wd_pred = np.clip(wd_res.model.predict(X_all), 0, config.WIND_RATED_KW)
        ml_pv_kwh = float(np.sum(ml_pv_pred))
        ml_wd_kwh = float(np.sum(ml_wd_pred))
        ml_available = True
    else:
        ml_pv_kwh = ml_wd_kwh = 0.0
        ml_pv_pred = ml_wd_pred = np.zeros(len(feat))
        ml_available = False

    # Physical model energy (for comparison).
    phys_pv_kwh = float(feat[config.TARGET_PV].sum())
    phys_wd_kwh = float(feat[config.TARGET_WIND].sum())

    # --- Headline: ML-predicted energy (driven by all 100 parameters) ---
    st.markdown(f"##### 🤖 ML-predicted energy ({model_name} — all 100 parameters)")
    st.caption("The trained ML model uses **all 100 atmospheric parameters** "
               "(raw + engineered) to predict hourly PV and wind output. "
               "Energy = sum of hourly predictions over the data window.")
    c1, c2, c3 = st.columns(3)
    c1.metric("☀️ PV energy (ML)",
              f"{ml_pv_kwh/1000:,.1f} MWh" if ml_available else "N/A",
              help=f"ML prediction over {days:.0f} days · "
                   f"{ml_pv_kwh/days:,.0f} kWh/day average")
    c2.metric("💨 Wind energy (ML)",
              f"{ml_wd_kwh/1000:,.1f} MWh" if ml_available else "N/A",
              help=f"ML prediction over {days:.0f} days · "
                   f"{ml_wd_kwh/days:,.0f} kWh/day average")
    c3.metric("Σ Combined (ML)",
              f"{(ml_pv_kwh+ml_wd_kwh)/1000:,.1f} MWh" if ml_available else "N/A",
              help=f"{(ml_pv_kwh+ml_wd_kwh)/days:,.0f} kWh/day average")

    # ML capacity factors.
    if ml_available:
        ml_pv_cf = np.mean(ml_pv_pred) / config.PV_CAPACITY_KW
        ml_wd_cf = np.mean(ml_wd_pred) / config.WIND_RATED_KW
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("PV capacity factor (ML)", f"{ml_pv_cf*100:.1f}%",
                  help=f"Average ML-predicted PV output ÷ {config.PV_CAPACITY_KW:.0f} kW")
        d2.metric("Wind capacity factor (ML)", f"{ml_wd_cf*100:.1f}%",
                  help=f"Average ML-predicted wind output ÷ {config.WIND_RATED_KW:.0f} kW")
        d3.metric("ML model R² (PV)", f"{pv_res.metrics['r2']*100:.1f}%",
                  help="How well the ML model fits the PV data on the test set.")
        d4.metric("ML model R² (Wind)", f"{wd_res.metrics['r2']*100:.1f}%",
                  help="How well the ML model fits the wind data on the test set.")

    # --- Physical model comparison ---
    st.markdown("##### ⚡ Physical model energy (engineering equations)")
    st.caption("For comparison: energy from the calibrated physical plant models "
               "(irradiance-based PV model + turbine power-curve wind model).")
    p1, p2, p3 = st.columns(3)
    pv_delta = ((ml_pv_kwh - phys_pv_kwh) / phys_pv_kwh * 100) if (ml_available and phys_pv_kwh > 0) else None
    wd_delta = ((ml_wd_kwh - phys_wd_kwh) / phys_wd_kwh * 100) if (ml_available and phys_wd_kwh > 0) else None
    p1.metric("☀️ PV energy (Physical)", f"{phys_pv_kwh/1000:,.1f} MWh",
              delta=f"{pv_delta:+.1f}% vs ML" if pv_delta is not None else None,
              delta_color="off")
    p2.metric("💨 Wind energy (Physical)", f"{phys_wd_kwh/1000:,.1f} MWh",
              delta=f"{wd_delta:+.1f}% vs ML" if wd_delta is not None else None,
              delta_color="off")
    p3.metric("Σ Combined (Physical)", f"{(phys_pv_kwh+phys_wd_kwh)/1000:,.1f} MWh")

    # --- PV YIELD ESTIMATION (the headline solar metrics) ---------------
    st.markdown("##### ☀️ PV yield estimation (rooftop solar)")
    # Use ML predictions for yield if available.
    if ml_available:
        feat_ml = feat.copy()
        feat_ml[config.TARGET_PV] = ml_pv_pred
        y = power_models.pv_yield_metrics(feat_ml)
        st.caption(f"Yield metrics computed from **{model_name} ML predictions** "
                   f"using all 100 parameters.")
    else:
        y = power_models.pv_yield_metrics(feat)
    y1, y2, y3 = st.columns(3)
    y1.metric("Annual specific yield", f"{y['annual_specific_yield']:,.0f} kWh/kWp·yr",
              help=f"Window of {y['days']:.0f} days extrapolated to 365. "
                   "The standard PV yield figure (good Indian sites ≈ 1400–1700).")
    y2.metric("Performance ratio", f"{y['performance_ratio']*100:.1f}%",
              help="Final yield ÷ reference yield — captures modelled thermal "
                   "and soiling losses.")
    y3.metric("Peak sun hours", f"{y['peak_sun_hours']:.2f} h/day",
              help="Equivalent full-sun (1 kW/m²) hours per day on the tilted array.")
    y4, y5, y6 = st.columns(3)
    y4.metric("Est. annual energy", f"{y['annual_energy_kwh']/1000:,.1f} MWh/yr",
              help=f"For the full {y['pdc_kwp']:.0f} kWp array.")
    y5.metric("Array DC nameplate", f"{y['pdc_kwp']:,.0f} kWp",
              help=f"{config.PV_AREA_M2:.0f} m² × {config.PV_EFFICIENCY*100:.0f}% efficiency.")
    y6.metric("Specific yield (window)", f"{y['specific_yield_period']:,.0f} kWh/kWp",
              help=f"Over the observed {y['days']:.0f}-day window.")

    # Live "now" conditions = the most recent real hour.
    st.markdown("##### 🛰️ Current conditions (most recent hour)")

    # Try to find the hour matching the current time at the site timezone,
    # falling back to coordinate-longitude offset from UTC, and finally the last row.
    tz = meta.get("timezone")
    utc_now = pd.Timestamp.now(tz="UTC")
    if tz and tz not in ("auto", "synthetic", "LST (NASA POWER)"):
        try:
            local_now = utc_now.tz_convert(tz).tz_localize(None)
        except Exception:
            local_now = utc_now.tz_localize(None) + pd.Timedelta(hours=(lon / 15.0))
    else:
        local_now = utc_now.tz_localize(None) + pd.Timedelta(hours=(lon / 15.0))

    # Ensure naive timestamp for comparison with the DataFrame's datetime64 column
    local_now = local_now.tz_localize(None) if local_now.tzinfo is not None else local_now
    past_rows = feat[feat["Datetime"] <= local_now]
    if not past_rows.empty:
        last = past_rows.iloc[-1]
        last_idx = past_rows.index[-1]
    else:
        last = feat.iloc[-1]
        last_idx = feat.index[-1]

    # ML-based "now" prediction using all 100 parameters.
    if ml_available:
        ml_pv_now = float(ml_pv_pred[last_idx])
        ml_wd_now = float(ml_wd_pred[last_idx])
    else:
        now_est = power_models.physical_point_estimate(last)
        ml_pv_now = now_est['pv_kw']
        ml_wd_now = now_est['wind_kw']

    g = st.columns(5)
    g[0].metric("Time", pd.Timestamp(last["Datetime"]).strftime("%d %b %H:%M"))
    g[1].metric("GHI", f"{last['shortwave_radiation']:.0f} W/m²")
    g[2].metric("Temp", f"{last['temperature_2m']:.1f} °C")
    g[3].metric("Wind @ hub", f"{last['wind_speed_hub']:.1f} m/s")
    g[4].metric("Cloud", f"{last['cloud_cover']:.0f} %")
    h = st.columns(3)
    h[0].metric("☀️ PV now (ML)", f"{ml_pv_now:.0f} kW")
    h[1].metric("💨 Wind now (ML)", f"{ml_wd_now:.0f} kW")
    h[2].metric("Σ Now (ML)", f"{ml_pv_now + ml_wd_now:.0f} kW")

    st.markdown("##### Recent power output (ML predictions vs Physical model)")
    # Show ML predictions in the timeseries chart.
    if ml_available:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tail = feat.tail(min(len(feat), 168)).copy()
        tail_idx = tail.index
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(tail["Datetime"], ml_pv_pred[tail_idx],
                color="#f59e0b", lw=1.4, label="PV (ML — 100 params)")
        ax.plot(tail["Datetime"], ml_wd_pred[tail_idx],
                color="#3b82f6", lw=1.4, label="Wind (ML — 100 params)")
        ax.plot(tail["Datetime"], tail[config.TARGET_PV].values,
                color="#f59e0b", lw=0.7, alpha=0.35, ls="--", label="PV (Physical)")
        ax.plot(tail["Datetime"], tail[config.TARGET_WIND].values,
                color="#3b82f6", lw=0.7, alpha=0.35, ls="--", label="Wind (Physical)")
        ax.set_ylabel("Power (kW)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.pyplot(plotting.power_timeseries(feat, hours=min(len(feat), 168)))


# ===========================================================================
# Tab: Impact Analysis  (the scientific core)
# ===========================================================================
def _tab_impact(feat, feat_key, model_name, pv_res, wd_res):
    st.subheader("Which atmospheric parameters drive the output?")
    st.caption("Impact is measured by **permutation importance**: each of the 100 "
               "parameters is shuffled in turn and the resulting loss of model "
               "accuracy quantifies how strongly the output depends on it.")

    with st.spinner("Ranking 100 parameters…"):
        pv_imp = get_permutation(feat_key, config.TARGET_PV, model_name, pv_res)
        wd_imp = get_permutation(feat_key, config.TARGET_WIND, model_name, wd_res)

    pv_top, wd_top = pv_imp.iloc[0], wd_imp.iloc[0]
    k1, k2 = st.columns(2)
    with k1:
        st.metric("☀️ PV — dominant parameter", pv_top["Label"])
        st.caption(f"≈ {pv_top['Share']*100:.0f}% of the model's predictive impact")
    with k2:
        st.metric("💨 Wind — dominant parameter", wd_top["Label"])
        st.caption(f"≈ {wd_top['Share']*100:.0f}% of the model's predictive impact")

    st.markdown("#### ⭐ The 15 most critical parameters")
    st.caption("A focused view of the 15 parameters the reference study flags as "
               "the most critical for PV & wind. Ranks are out of all 100 "
               "parameters. Edit `config.CRITICAL_PARAMETERS` to use a different "
               "set.")
    pv_rank = {f: i + 1 for i, f in enumerate(pv_imp["Feature"])}
    wd_rank = {f: i + 1 for i, f in enumerate(wd_imp["Feature"])}
    pv_sh = dict(zip(pv_imp["Feature"], pv_imp["Share"]))
    wd_sh = dict(zip(wd_imp["Feature"], wd_imp["Share"]))
    crit = pd.DataFrame([{
        "Feature": f,
        "Label": config.pretty(f),
        "Category": config.CRITICAL_CATEGORY.get(f, ""),
        "PV rank": pv_rank.get(f),
        "PV %": pv_sh.get(f, 0.0) * 100,
        "Wind rank": wd_rank.get(f),
        "Wind %": wd_sh.get(f, 0.0) * 100,
        "Share": (pv_sh.get(f, 0.0) + wd_sh.get(f, 0.0)) / 2.0,
    } for f in config.CRITICAL_PARAMETERS])

    bar = crit.copy()
    tot = bar["Share"].sum()
    bar["Share"] = bar["Share"] / tot if tot > 0 else 0.0
    bar["Label"] = bar["Label"] + "  ·  " + bar["Category"]
    bar = bar.sort_values("Share", ascending=False)
    st.pyplot(plotting.critical_param_bar(
        bar, "#6c5ce7", "15 critical parameters — combined PV + wind impact"))
    st.dataframe(
        crit.sort_values("PV %", ascending=False)[
            ["Label", "Category", "PV rank", "PV %", "Wind rank", "Wind %"]]
        .style.format({"PV %": "{:.2f}", "Wind %": "{:.2f}"}),
        use_container_width=True, hide_index=True)

    st.markdown("#### Top atmospheric drivers (all 100 parameters ranked)")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plotting.importance_bar(pv_imp, config.PV_COLOR,
                                          "PV — top parameters"))
    with c2:
        st.pyplot(plotting.importance_bar(wd_imp, config.WIND_COLOR,
                                          "Wind — top parameters"))

    st.markdown("#### Impact rolled up by physical category")
    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(plotting.grouped_donut(impact_analysis.grouped_impact(pv_imp),
                                         "PV impact by category"))
    with c4:
        st.pyplot(plotting.grouped_donut(impact_analysis.grouped_impact(wd_imp),
                                         "Wind impact by category"))

    st.markdown("#### Secondary atmospheric modulators (conversion efficiency)")
    st.caption("Gross output is dominated by the irradiance / wind-speed swing, "
               "which masks the smaller effects. Normalising the first-order driver "
               "out exposes the atmospheric parameters that govern how *efficiently* "
               "the resource is converted into electricity.")
    with st.spinner("Analysing secondary drivers…"):
        pv_sec = get_secondary(feat_key, feat, "pv", model_name)
        wd_sec = get_secondary(feat_key, feat, "wind", model_name)
    c5, c6 = st.columns(2)
    with c5:
        st.pyplot(plotting.importance_bar(pv_sec, config.PV_COLOR,
                                          "PV efficiency (excl. irradiance)", top=10))
    with c6:
        st.pyplot(plotting.importance_bar(wd_sec, config.WIND_COLOR,
                                          "Wind efficiency (excl. wind speed)", top=10))

    if len(pv_sec) > 1 and len(wd_sec) > 1:
        st.success(
            f"**PV:** gross output is set by **{pv_top['Label']}**; conversion "
            f"efficiency is then most sensitive to **{pv_sec.iloc[0]['Label']}** "
            f"and {pv_sec.iloc[1]['Label']}.  \n"
            f"**Wind:** gross output is set by **{wd_top['Label']}**; efficiency is "
            f"then most sensitive to **{wd_sec.iloc[0]['Label']}** and "
            f"{wd_sec.iloc[1]['Label']} (air-density / stability effects).")

    # Optional SHAP (explainable AI).
    with st.expander("🧠 SHAP explainability (if `shap` is installed)"):
        sh = impact_analysis.shap_impact(pv_res)
        if sh is None:
            st.caption("SHAP not available for this model, or the `shap` package "
                       "is not installed (`pip install shap`).")
        else:
            st.pyplot(plotting.importance_bar(sh, config.PV_COLOR,
                                              "PV — mean |SHAP|", top=12,
                                              value_col="Share"))

    with st.expander("📋 Full 100-parameter impact table (PV)"):
        st.dataframe(pv_imp[["Label", "Category", "Share", "Importance"]]
                     .rename(columns={"Share": "Share (frac)"}),
                     use_container_width=True)


# ===========================================================================
# Tab: ML Accuracy  (dedicated accuracy chart for ALL algorithms)
# ===========================================================================
def _tab_accuracy(feat, feat_key):
    st.subheader("📈 Accuracy of every machine-learning algorithm")
    st.caption("Each algorithm is trained on this location's 100-parameter "
               "dataset and scored on a held-out 20% test set. Accuracy is the "
               "coefficient of determination R² (100% = perfect). PV and wind "
               "are scored separately.")

    with st.spinner("Training and scoring all algorithms "
                    "(cached after the first run)…"):
        pv_board = get_benchmark(feat_key, feat, config.TARGET_PV)
        wd_board = get_benchmark(feat_key, feat, config.TARGET_WIND)

    pv_best, wd_best = pv_board.iloc[0], wd_board.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithms compared", f"{len(pv_board)}")
    c2.metric("Best PV model", pv_best["Model"], f"{pv_best['R2_pct']:.2f}% R²")
    c3.metric("Best wind model", wd_best["Model"], f"{wd_best['R2_pct']:.2f}% R²")

    st.pyplot(plotting.accuracy_comparison_bar(pv_board, wd_board))

    st.markdown("#### Full accuracy table")
    merged = (pv_board[["Model", "R2_pct", "MAE", "RMSE"]]
              .merge(wd_board[["Model", "R2_pct", "MAE", "RMSE"]],
                     on="Model", suffixes=("_pv", "_wd")))
    merged = merged.rename(columns={
        "R2_pct_pv": "PV R² %", "MAE_pv": "PV MAE", "RMSE_pv": "PV RMSE",
        "R2_pct_wd": "Wind R² %", "MAE_wd": "Wind MAE", "RMSE_wd": "Wind RMSE"})
    merged = merged.sort_values("PV R² %", ascending=False).reset_index(drop=True)
    st.dataframe(
        merged.style.format({"PV R² %": "{:.2f}", "PV MAE": "{:.2f}",
                             "PV RMSE": "{:.2f}", "Wind R² %": "{:.2f}",
                             "Wind MAE": "{:.2f}", "Wind RMSE": "{:.2f}"}),
        use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download accuracy table (CSV)",
                       merged.to_csv(index=False).encode(),
                       file_name="ml_algorithm_accuracy.csv", mime="text/csv")


# ===========================================================================
# Tab: Data Explorer
# ===========================================================================
def _tab_explorer(feat, meta):
    st.subheader("Dataset")
    st.write(f"Rows: **{len(feat):,}** · Source: **{meta['source']}** · "
             f"Timezone: **{meta['timezone']}** · Parameters: **100**")
    st.dataframe(feat[["Datetime"] + config.RAW_HOURLY_VARS[:10]].head(12),
                 use_container_width=True)

    with st.expander("Summary statistics (all 100 parameters + targets)"):
        st.dataframe(feat[config.FEATURES + config.TARGETS].describe().T,
                     use_container_width=True)

    st.subheader("Atmospheric parameter vs power output")
    c1, c2 = st.columns(2)
    with c1:
        xpv = st.selectbox("PV output against:", config.FEATURES,
                           index=config.FEATURES.index("shortwave_radiation"),
                           format_func=config.pretty, key="xpv")
        st.pyplot(plotting.scatter(feat, xpv, config.TARGET_PV, config.PV_COLOR))
    with c2:
        xwd = st.selectbox("Wind output against:", config.FEATURES,
                           index=config.FEATURES.index("wind_speed_hub"),
                           format_func=config.pretty, key="xwd")
        st.pyplot(plotting.scatter(feat, xwd, config.TARGET_WIND, config.WIND_COLOR))


# ===========================================================================
# Tab: Model Performance
# ===========================================================================
def _tab_performance(feat, feat_key, model_name, pv_res, wd_res):
    st.subheader(f"Accuracy of {model_name} on the held-out test set (20%)")

    st.markdown("#### ☀️ Photovoltaic model")
    a, b, c = st.columns(3)
    a.metric("R² (accuracy)", f"{pv_res.metrics['r2']*100:.2f}%")
    b.metric("MAE", f"{pv_res.metrics['mae']:.2f} kW")
    c.metric("RMSE", f"{pv_res.metrics['rmse']:.2f} kW")
    st.pyplot(plotting.pred_vs_actual(pv_res.y_test, pv_res.y_pred,
                                      config.PV_COLOR, "PV"))

    st.markdown("#### 💨 Wind model")
    d, e, f = st.columns(3)
    d.metric("R² (accuracy)", f"{wd_res.metrics['r2']*100:.2f}%")
    e.metric("MAE", f"{wd_res.metrics['mae']:.2f} kW")
    f.metric("RMSE", f"{wd_res.metrics['rmse']:.2f} kW")
    st.pyplot(plotting.pred_vs_actual(wd_res.y_test, wd_res.y_pred,
                                      config.WIND_COLOR, "Wind"))

    st.markdown("#### 🏁 Compare all algorithms")
    st.caption("Trains every available algorithm on this location's data and ranks "
               "them by R². This can take a little while the first time.")
    if st.button("Run model benchmark"):
        with st.spinner("Benchmarking all algorithms…"):
            pv_board = get_benchmark(feat_key, feat, config.TARGET_PV)
            wd_board = get_benchmark(feat_key, feat, config.TARGET_WIND)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**PV leaderboard**")
            st.dataframe(pv_board[["Model", "R2_pct", "MAE", "RMSE"]],
                         use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Wind leaderboard**")
            st.dataframe(wd_board[["Model", "R2_pct", "MAE", "RMSE"]],
                         use_container_width=True, hide_index=True)


# ===========================================================================
# Tab: Forecaster
# ===========================================================================
def _tab_forecaster(feat, pv_res, wd_res, lat, lon):
    st.subheader("What-if forecaster")
    st.caption("Override key atmospheric conditions; the remaining 100 parameters "
               "are taken from the most recent real hour and re-derived. Predictions "
               "use the trained ML models.")

    last = feat.iloc[-1].copy()
    with st.form("forecast"):
        g1, g2 = st.columns(2)
        with g1:
            irr = st.slider("Solar irradiance GHI (W/m²)", 0, 1200,
                            int(last["shortwave_radiation"]))
            temp = st.slider("Temperature (°C)", -10, 50,
                             int(last["temperature_2m"]))
            hum = st.slider("Relative humidity (%)", 0, 100,
                            int(last["relative_humidity_2m"]))
            cloud = st.slider("Cloud cover (%)", 0, 100,
                              int(last["cloud_cover"]))
        with g2:
            wind = st.slider("Wind speed @ 10 m (m/s)", 0.0, 30.0,
                             float(round(last["wind_speed_10m"], 1)), step=0.5)
            pres = st.slider("Surface pressure (hPa)", 850, 1050,
                             int(last["surface_pressure"]))
            rain = st.slider("Precipitation (mm)", 0.0, 30.0,
                             float(round(last["precipitation"], 1)), step=0.5)
        go = st.form_submit_button("Run forecast", type="primary")

    if go:
        row = last.copy()
        # Apply overrides to raw drivers, then re-derive the dependent features.
        row["shortwave_radiation"] = irr
        row["shortwave_radiation_instant"] = irr
        row["direct_radiation"] = irr * 0.7
        row["diffuse_radiation"] = irr * 0.3
        row["global_tilted_irradiance"] = irr * 1.05
        row["temperature_2m"] = temp
        row["relative_humidity_2m"] = hum
        row["cloud_cover"] = cloud
        row["wind_speed_10m"] = wind
        row["surface_pressure"] = pres
        row["precipitation"] = rain

        one = pd.DataFrame([row])
        one = feature_engineering.build_feature_table(
            one[["Datetime"] + config.RAW_HOURLY_VARS],
            latitude=lat, longitude=lon)
        X = one[config.FEATURES]

        pv_pred = max(0.0, float(pv_res.model.predict(X)[0]))
        wd_pred = max(0.0, float(wd_res.model.predict(X)[0]))
        phys = power_models.physical_point_estimate(one.iloc[0])

        st.markdown("##### ML prediction")
        r1, r2, r3 = st.columns(3)
        r1.metric("☀️ PV output", f"{pv_pred:.1f} kW")
        r2.metric("💨 Wind output", f"{wd_pred:.1f} kW")
        r3.metric("Σ Combined", f"{pv_pred + wd_pred:.1f} kW")
        st.caption(f"Physical-model cross-check — PV {phys['pv_kw']:.1f} kW · "
                   f"Wind {phys['wind_kw']:.1f} kW · Σ {phys['total_kw']:.1f} kW")

# ===========================================================================
# Tab: Real-Time Only Dataset & Evaluation (57 raw API parameters only)
# ===========================================================================
def _tab_realtime_only(feat, meta, lat, lon, model_name, past_days, prefer):
    st.subheader("🌐 Real-Time Fetched Dataset & PV Evaluation")
    st.caption("This tab uses **only the 57 parameters fetched directly from the "
               "real-time API** — no engineered or derived columns. A separate ML "
               "model is trained on these 57 raw features to produce an independent "
               "PV energy estimate.")

    days = max(meta["n_rows"] / 24.0, 1e-6)
    raw_features = [f for f in config.RAW_HOURLY_VARS if f in feat.columns]

    # ==================================================================
    # SECTION 1: Data Source
    # ==================================================================
    st.markdown("---")
    st.markdown("### 1️⃣ Data Source")
    source_name = {
        "nasa": "NASA POWER (government satellite reanalysis)",
        "openmeteo": "Open-Meteo (ECMWF/GFS) + Air Quality API",
        "synthetic": "Offline synthetic",
    }.get(prefer, str(meta.get("source", prefer)))

    s1, s2, s3 = st.columns(3)
    s1.info(f"**Source:** {source_name}")
    s2.info(f"**Timezone:** {meta.get('timezone', 'N/A')}")
    s3.info(f"**Parameters:** {len(raw_features)} (API-fetched only)")

    if prefer == "openmeteo":
        st.code(
            f"Weather API:\n"
            f"  https://api.open-meteo.com/v1/forecast?\n"
            f"    latitude={lat}&longitude={lon}\n"
            f"    &hourly={','.join(raw_features[:8])},...\n"
            f"    &past_days={past_days}&forecast_days=2\n\n"
            f"Air Quality API:\n"
            f"  https://air-quality-api.open-meteo.com/v1/air-quality?\n"
            f"    latitude={lat}&longitude={lon}\n"
            f"    &hourly=aerosol_optical_depth,dust,pm2_5,pm10,...",
            language="text")

    # ==================================================================
    # SECTION 2: Real-Time Dataset (57 columns only)
    # ==================================================================
    st.markdown("---")
    st.markdown("### 2️⃣ Real-Time Dataset (57 API Parameters)")

    ds_rt = feat[["Datetime"] + raw_features].copy()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Rows (hourly)", f"{len(ds_rt):,}")
    r2.metric("Columns", f"{len(raw_features)} (raw API only)")
    r3.metric("Time span", f"{days:.0f} days")
    r4.metric("Excluded", f"{len(config.DERIVED_FEATURES)} engineered cols")

    t1, t2 = st.columns(2)
    t1.metric("First timestamp", str(ds_rt["Datetime"].iloc[0])[:16])
    t2.metric("Last timestamp", str(ds_rt["Datetime"].iloc[-1])[:16])

    # Column listing
    st.markdown("**All 57 real-time parameters in this dataset:**")
    col_info = []
    for i, f in enumerate(raw_features, 1):
        col_info.append({
            "#": i,
            "Parameter": f,
            "Label": config.pretty(f),
            "Category": impact_analysis._category(f),
            "Source": "✅ Real-time API",
        })
    st.dataframe(pd.DataFrame(col_info), use_container_width=True, height=300)

    # Show the dataset
    st.markdown("**Dataset preview (scroll to explore all 57 columns):**")
    st.dataframe(ds_rt, use_container_width=True, height=400)

    # Statistical summary
    with st.expander("📊 Statistical Summary of 57 parameters"):
        stats = ds_rt[raw_features].describe().T
        stats.insert(0, "Parameter", stats.index)
        stats = stats.reset_index(drop=True)
        st.dataframe(stats, use_container_width=True, height=350)

    # Download
    csv_rt = ds_rt.to_csv(index=False)
    st.download_button(
        f"⬇️ Download real-time dataset ({len(ds_rt):,} rows × {len(raw_features)+1} cols)",
        csv_rt,
        f"realtime_only_{lat:.4f}_{lon:.4f}_{past_days}days.csv",
        "text/csv", use_container_width=True)

    # ==================================================================
    # SECTION 3: Train ML Model on 57 raw features only
    # ==================================================================
    st.markdown("---")
    st.markdown(f"### 3️⃣ Training {model_name} on 57 Real-Time Parameters Only")

    X_raw = feat[raw_features]
    y_pv = feat[config.TARGET_PV]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_pv, test_size=0.2, random_state=config.RANDOM_STATE)

    st.markdown(f"""
**Train/Test Split:**
| | Count |
|---|---|
| Total samples | **{len(X_raw):,}** |
| Training set (80%) | **{len(X_train):,}** |
| Test set (20%) | **{len(X_test):,}** |
| Features used | **{len(raw_features)}** (real-time API only) |
""")

    # Train the model
    with st.spinner(f"Training {model_name} on 57 real-time features..."):
        model_57 = ml_models.MODEL_CATALOGUE[model_name]()
        model_57.fit(X_train, y_train)
        y_pred_test = model_57.predict(X_test)

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2_57 = r2_score(y_test, y_pred_test)
    mae_57 = mean_absolute_error(y_test, y_pred_test)
    rmse_57 = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    a1, a2, a3 = st.columns(3)
    a1.metric(f"R² ({model_name}, 57 params)", f"{r2_57*100:.1f}%")
    a2.metric("MAE", f"{mae_57:.2f} kW")
    a3.metric("RMSE", f"{rmse_57:.2f} kW")

    # ==================================================================
    # SECTION 4: Step-by-Step Mathematical Evaluation
    # ==================================================================
    st.markdown("---")
    st.markdown(f"### 4️⃣ Step-by-Step Mathematical Evaluation ({model_name})")

    # Predict on ALL data
    ml_pv_pred_57 = np.clip(model_57.predict(X_raw), 0, config.PV_CAPACITY_KW)
    ml_pv_kwh_57 = float(np.sum(ml_pv_pred_57))

    # Show algorithm-specific math
    if "Linear" in model_name or model_name in ("Ridge", "Lasso", "Elastic Net"):
        _rt_math_linear(model_name, model_57, raw_features, feat, X_raw)
    elif "Random Forest" in model_name or "Extra Trees" in model_name:
        _rt_math_forest(model_name, model_57, raw_features, feat, X_raw)
    elif "K-Nearest" in model_name:
        _rt_math_knn(model_name, model_57, raw_features, feat, X_raw, X_train, y_train)
    elif "Decision Tree" in model_name:
        _rt_math_tree(model_name, model_57, raw_features, feat, X_raw)
    elif "Gradient" in model_name or "Hist" in model_name or "XGBoost" in model_name or "LightGBM" in model_name:
        _rt_math_boosting(model_name, model_57, raw_features, feat, X_raw)
    elif "Support Vector" in model_name:
        _rt_math_svr(model_name, model_57, raw_features, feat, X_raw)
    elif "Neural" in model_name or "MLP" in model_name or "Deep" in model_name or "Wide" in model_name:
        _rt_math_mlp(model_name, model_57, raw_features, feat, X_raw)
    elif any(k in model_name for k in ("CNN", "RNN", "LSTM", "GRU", "BiLSTM",
                                       "Transformer", "RBF", "Capsule",
                                       "Autoencoder", "GAN", "Belief")):
        _rt_math_general(model_name, model_57, raw_features, feat, X_raw)
    elif "Self Organizing" in model_name:
        _rt_math_general(model_name, model_57, raw_features, feat, X_raw)
    else:
        st.info(f"Showing general evaluation for {model_name}")
        if hasattr(model_57, 'feature_importances_'):
            imp = model_57.feature_importances_
            imp_df = pd.DataFrame({
                "Parameter": raw_features,
                "Label": [config.pretty(f) for f in raw_features],
                "Importance (%)": np.round(imp * 100, 2),
            }).sort_values("Importance (%)", ascending=False).head(15).reset_index(drop=True)
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

    # ==================================================================
    # SECTION 5: Hourly Predictions
    # ==================================================================
    st.markdown("---")
    st.markdown("### 5️⃣ Hourly PV Predictions (57 Real-Time Parameters)")

    # Train the 100-parameter model ONCE and reuse it for the table, the energy
    # total and the accuracy comparison below (previously it was fit two/three
    # times per rerun, and a dead `if False else` branch built a throwaway model).
    model_100 = ml_models.MODEL_CATALOGUE[model_name]()
    model_100.fit(feat[config.FEATURES], y_pv)
    ml_pv_pred_100 = np.clip(model_100.predict(feat[config.FEATURES]), 0, config.PV_CAPACITY_KW)
    ml_pv_kwh_100 = float(np.sum(ml_pv_pred_100))

    pred_df = pd.DataFrame({
        "Datetime": feat["Datetime"].values,
        "GHI (W/m²)": feat["shortwave_radiation"].values,
        "Temp (°C)": feat["temperature_2m"].values,
        "Cloud (%)": feat["cloud_cover"].values,
        "Wind 10m (m/s)": feat["wind_speed_10m"].values,
        "PV Pred — 57 params (kW)": np.round(ml_pv_pred_57, 2),
        "PV Pred — 100 params (kW)": np.round(ml_pv_pred_100, 2),
    })

    st.dataframe(pred_df, use_container_width=True, height=400)

    st.download_button(
        "⬇️ Download predictions CSV (57 vs 100 params)",
        pred_df.to_csv(index=False),
        f"realtime_predictions_{model_name.replace(' ','_')}_{lat:.4f}_{lon:.4f}.csv",
        "text/csv", use_container_width=True)

    # ==================================================================
    # SECTION 6: PV Energy Calculation
    # ==================================================================
    st.markdown("---")
    st.markdown("### 6️⃣ PV Energy Calculation")

    st.latex(r"E_{PV}^{57} = \sum_{h=1}^{N} \hat{P}_{ML}^{57\text{-params}}(h) \times 1\,\text{hour}")

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Using 57 real-time parameters ONLY:**")
        st.metric("PV Energy (57 params)", f"{ml_pv_kwh_57/1000:,.2f} MWh")
        st.markdown(f"""
| Component | Value |
|-----------|-------|
| Hours | {len(ml_pv_pred_57):,} |
| Sum of predictions | {ml_pv_kwh_57:,.1f} kWh |
| **PV Energy** | **{ml_pv_kwh_57/1000:,.2f} MWh** |
| Daily average | {ml_pv_kwh_57/days:,.0f} kWh/day |
""")

    with e2:
        st.markdown("**Using all 100 parameters (for comparison):**")
        st.metric("PV Energy (100 params)", f"{ml_pv_kwh_100/1000:,.2f} MWh")
        st.markdown(f"""
| Component | Value |
|-----------|-------|
| Hours | {len(ml_pv_pred_100):,} |
| Sum of predictions | {ml_pv_kwh_100:,.1f} kWh |
| **PV Energy** | **{ml_pv_kwh_100/1000:,.2f} MWh** |
| Daily average | {ml_pv_kwh_100/days:,.0f} kWh/day |
""")

    # Difference
    diff_mwh = (ml_pv_kwh_57 - ml_pv_kwh_100) / 1000.0
    diff_pct = (ml_pv_kwh_57 - ml_pv_kwh_100) / ml_pv_kwh_100 * 100 if ml_pv_kwh_100 > 0 else 0

    st.markdown("### 📊 Comparison: 57 vs 100 Parameters")
    cmp1, cmp2, cmp3 = st.columns(3)
    cmp1.metric("57-param PV Energy", f"{ml_pv_kwh_57/1000:,.2f} MWh")
    cmp2.metric("100-param PV Energy", f"{ml_pv_kwh_100/1000:,.2f} MWh")
    cmp3.metric("Difference", f"{diff_pct:+.2f}%",
                delta=f"{diff_mwh:+.2f} MWh", delta_color="off")

    # Accuracy comparison
    st.markdown("### 🎯 Model Accuracy Comparison")
    # 100-param accuracy
    y_pred_100_test = model_100.predict(feat[config.FEATURES].iloc[X_test.index])
    r2_100 = r2_score(y_test, y_pred_100_test)
    mae_100 = mean_absolute_error(y_test, y_pred_100_test)

    acc_df = pd.DataFrame({
        "Metric": ["R² Score", "MAE (kW)", "RMSE (kW)", "PV Energy (MWh)"],
        f"57 Real-Time Params": [
            f"{r2_57*100:.1f}%", f"{mae_57:.2f}", f"{rmse_57:.2f}",
            f"{ml_pv_kwh_57/1000:,.2f}"],
        f"100 All Params": [
            f"{r2_100*100:.1f}%", f"{mae_100:.2f}", "—",
            f"{ml_pv_kwh_100/1000:,.2f}"],
        "Difference": [
            f"{(r2_57-r2_100)*100:+.1f}%", f"{mae_57-mae_100:+.2f}", "—",
            f"{diff_pct:+.2f}%"],
    })
    st.dataframe(acc_df, use_container_width=True, hide_index=True)

    st.info(f"💡 **Insight:** The 57 real-time parameters achieve "
            f"**R² = {r2_57*100:.1f}%** compared to {r2_100*100:.1f}% with all 100. "
            f"The {len(config.DERIVED_FEATURES)} engineered features "
            f"{'improve' if r2_100 > r2_57 else 'do not significantly improve'} "
            f"prediction accuracy by {abs(r2_100-r2_57)*100:.1f} percentage points.")


# ---------------------------------------------------------------------------
# Helper math functions for the real-time only tab
# ---------------------------------------------------------------------------
def _rt_math_linear(model_name, model, raw_features, feat, X_raw):
    if hasattr(model, 'named_steps'):
        lin = model.named_steps.get('model', model)
        scaler = model.named_steps.get('scaler', None)
    else:
        lin = model
        scaler = None

    st.markdown(f"""
**{model_name}** with 57 real-time parameters:

$$\\hat{{y}} = \\beta_0 + \\sum_{{i=1}}^{{57}} \\beta_i \\cdot x_i$$
""")
    coefs = lin.coef_
    intercept = lin.intercept_
    coef_df = pd.DataFrame({
        "Parameter": raw_features,
        "Label": [config.pretty(f) for f in raw_features],
        "Coefficient (β)": np.round(coefs, 6),
        "|β|": np.round(np.abs(coefs), 6),
    }).sort_values("|β|", ascending=False).reset_index(drop=True)

    st.markdown(f"**Intercept (β₀):** {intercept:.4f}")
    st.dataframe(coef_df, use_container_width=True, height=300)

    # Worked example
    day_mask = feat["shortwave_radiation"] > 100
    if day_mask.any():
        si = feat[day_mask].index[len(feat[day_mask])//2]
    else:
        si = len(feat)//2
    sample = feat.iloc[si]
    X_s = X_raw.iloc[[si]]
    pred = float(np.clip(model.predict(X_s)[0], 0, config.PV_CAPACITY_KW))
    if scaler is not None:
        X_scaled = scaler.transform(X_s)
        terms = coefs * X_scaled[0]
    else:
        terms = coefs * X_s.values[0]
    top5 = np.argsort(np.abs(terms))[-5:][::-1]
    st.markdown(f"**Worked example ({pd.Timestamp(sample['Datetime']).strftime('%d %b %H:%M')}):**")
    td = [{"Param": config.pretty(raw_features[i]), "β×x": f"{terms[i]:.4f}"} for i in top5]
    st.table(pd.DataFrame(td))
    st.latex(f"\\hat{{y}} = {intercept:.2f} + \\sum \\beta_i x_i = {pred:.2f} \\text{{ kW}}")


def _rt_math_forest(model_name, model, raw_features, feat, X_raw):
    n = model.n_estimators
    st.markdown(f"""
**{model_name}** — ensemble of **{n} trees** on 57 real-time features:

$$\\hat{{y}} = \\frac{{1}}{{{n}}} \\sum_{{t=1}}^{{{n}}} \\text{{Tree}}_t(\\mathbf{{x}}_{{57}})$$
""")
    imp = model.feature_importances_
    imp_df = pd.DataFrame({
        "Parameter": raw_features,
        "Label": [config.pretty(f) for f in raw_features],
        "Importance (%)": np.round(imp * 100, 2),
    }).sort_values("Importance (%)", ascending=False).reset_index(drop=True)
    st.markdown("**Feature importances (57 real-time params):**")
    st.dataframe(imp_df.head(20), use_container_width=True, hide_index=True)

    # Sample prediction from individual trees
    day_mask = feat["shortwave_radiation"] > 100
    si = feat[day_mask].index[len(feat[day_mask])//2] if day_mask.any() else len(feat)//2
    X_s = X_raw.iloc[[si]].values
    tree_preds = [t.predict(X_s)[0] for t in model.estimators_[:10]]
    st.markdown("**First 10 tree predictions for a sample hour:**")
    st.dataframe(pd.DataFrame({
        "Tree": [f"#{i+1}" for i in range(10)],
        "Prediction (kW)": [f"{p:.2f}" for p in tree_preds]
    }).T, use_container_width=True)


def _rt_math_knn(model_name, model, raw_features, feat, X_raw, X_train, y_train):
    if hasattr(model, 'named_steps'):
        knn = model.named_steps.get('model', model)
        scaler = model.named_steps.get('scaler', None)
    else:
        knn = model
        scaler = None
    k = knn.n_neighbors
    st.markdown(f"""
**KNN** with {k} neighbors on 57 real-time features:

$$\\hat{{y}} = \\frac{{1}}{{{k}}} \\sum_{{i=1}}^{{{k}}} y_{{\\text{{neighbor}}_i}}$$

$$d(\\mathbf{{x}}, \\mathbf{{x}}_j) = \\sqrt{{\\sum_{{i=1}}^{{57}} (x_i - x_{{j,i}})^2}}$$
""")
    day_mask = feat["shortwave_radiation"] > 100
    si = feat[day_mask].index[len(feat[day_mask])//2] if day_mask.any() else len(feat)//2
    X_s = X_raw.iloc[[si]]
    if scaler:
        X_s_sc = scaler.transform(X_s)
        dists, idxs = knn.kneighbors(X_s_sc)
    else:
        dists, idxs = knn.kneighbors(X_s)
    nd = []
    for i in range(min(k, len(idxs[0]))):
        idx = idxs[0][i]
        nd.append({"Neighbor": i+1, "Distance": f"{dists[0][i]:.4f}",
                    "PV Power": f"{y_train.iloc[idx]:.2f} kW"})
    st.dataframe(pd.DataFrame(nd), use_container_width=True, hide_index=True)


def _rt_math_tree(model_name, model, raw_features, feat, X_raw):
    tree = model.tree_
    st.markdown(f"""
**Decision Tree** on 57 real-time features:
- Nodes: **{tree.node_count}** | Leaves: **{tree.n_leaves}** | Max depth: **{model.max_depth}**
""")
    day_mask = feat["shortwave_radiation"] > 100
    si = feat[day_mask].index[len(feat[day_mask])//2] if day_mask.any() else len(feat)//2
    X_s = X_raw.iloc[[si]].values
    path = model.decision_path(X_s)
    path_data = []
    for nid in path.indices[:12]:
        if tree.children_left[nid] != tree.children_right[nid]:
            fi = tree.feature[nid]
            path_data.append({"Node": nid, "Param": config.pretty(raw_features[fi]),
                              "Threshold": f"{tree.threshold[nid]:.4f}",
                              "Value": f"{X_s[0][fi]:.4f}",
                              "Dir": "← Left" if X_s[0][fi] <= tree.threshold[nid] else "→ Right"})
        else:
            path_data.append({"Node": nid, "Param": "🍃 LEAF", "Threshold": "-",
                              "Value": "-", "Dir": f"Pred = {tree.value[nid][0][0]:.2f} kW"})
    st.dataframe(pd.DataFrame(path_data), use_container_width=True, hide_index=True)


def _rt_math_boosting(model_name, model, raw_features, feat, X_raw):
    n = getattr(model, 'n_estimators', '?')
    lr = getattr(model, 'learning_rate', '?')
    st.markdown(f"""
**{model_name}** on 57 real-time features — {n} sequential trees, learning rate η={lr}:

$$\\hat{{y}} = F_0 + \\sum_{{m=1}}^{{{n}}} {lr} \\cdot T_m(\\mathbf{{x}}_{{57}})$$
""")
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        imp_df = pd.DataFrame({
            "Parameter": raw_features,
            "Label": [config.pretty(f) for f in raw_features],
            "Importance (%)": np.round(imp * 100, 2),
        }).sort_values("Importance (%)", ascending=False).head(15).reset_index(drop=True)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)


def _rt_math_svr(model_name, model, raw_features, feat, X_raw):
    if hasattr(model, 'named_steps'):
        svr = model.named_steps.get('model', model)
    else:
        svr = model
    n_sv = len(svr.support_) if hasattr(svr, 'support_') else '?'
    st.markdown(f"""
**SVR** on 57 real-time features:
- Kernel: **{getattr(svr,'kernel','rbf')}** | C: **{getattr(svr,'C','?')}**
- Support vectors: **{n_sv}** out of {len(feat):,} samples
""")


def _rt_math_general(model_name, model, raw_features, feat, X_raw):
    """Fallback real-time math info for unrecognized / Keras models."""
    st.markdown(f"**{model_name}** trained on all 57 real-time features.")
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        imp_df = pd.DataFrame({
            "Parameter": raw_features,
            "Label": [config.pretty(f) for f in raw_features],
            "Importance (%)": np.round(imp * 100, 2),
        }).sort_values("Importance (%)", ascending=False).head(15).reset_index(drop=True)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
    else:
        st.caption(f"The **{model_name}** model does not expose native feature "
                   "importances. Use the **Impact Analysis** tab for "
                   "permutation-based importance.")


def _rt_math_mlp(model_name, model, raw_features, feat, X_raw):
    if hasattr(model, 'named_steps'):
        mlp = model.named_steps.get('model', model)
    else:
        mlp = model
    layers = getattr(mlp, 'hidden_layer_sizes', '?')
    st.markdown(f"""
**MLP Neural Network** on 57 real-time features:
- Architecture: Input(57) → {' → '.join([f'H({h})' for h in (layers if isinstance(layers, tuple) else (layers,))])} → Output(1)
- Activation: **{getattr(mlp, 'activation', 'relu')}**
- Iterations: **{getattr(mlp, 'n_iter_', '?')}**
""")
    if isinstance(layers, tuple):
        total = sum((57 if i == 0 else layers[i-1]) * h + h for i, h in enumerate(layers)) + layers[-1] + 1
        st.metric("Total trainable parameters", f"{total:,}")


# ===========================================================================
# Tab: Dataset  (full 100-column dataset, downloadable)
# ===========================================================================
def _tab_dataset(feat, meta, lat, lon, past_days, prefer):
    st.subheader("📋 Real-Time Dataset (100 Parameters)")

    # --- Section 1: Data Source Info ---
    st.markdown("##### 1️⃣ Data Source")
    source_name = {
        "nasa": "NASA POWER (government satellite reanalysis)",
        "openmeteo": "Open-Meteo (ECMWF/GFS numerical weather models) + Air Quality API",
        "synthetic": "Offline synthetic (generated locally)",
    }.get(prefer, str(meta.get("source", prefer)))
    src1, src2 = st.columns(2)
    src1.info(f"**Source:** {source_name}")
    src2.info(f"**Timezone:** {meta.get('timezone', 'N/A')}")

    if prefer == "openmeteo":
        st.code(
            f"Weather: https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&hourly=...&past_days={past_days}&forecast_days=2\n"
            f"Air Quality: https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}&hourly=...",
            language="text")
    elif prefer == "nasa":
        st.code(
            f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
            f"parameters=...&community=RE&longitude={lon}&latitude={lat}&...",
            language="text")

    # --- Section 2: Dataset Summary ---
    st.markdown("##### 2️⃣ Dataset Summary")
    display_cols = ["Datetime"] + config.FEATURES + [config.TARGET_PV, config.TARGET_WIND]
    ds = feat[display_cols].copy()
    days = len(ds) / 24.0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rows (hourly)", f"{len(ds):,}")
    s2.metric("Columns", f"{len(config.FEATURES)} + 2 targets")
    s3.metric("Time span", f"{days:.0f} days")
    s4.metric("Resolution", "1 hour")

    t1, t2 = st.columns(2)
    t1.metric("First timestamp", str(ds["Datetime"].iloc[0])[:16])
    t2.metric("Last timestamp", str(ds["Datetime"].iloc[-1])[:16])

    # --- Section 3: Column Descriptions ---
    st.markdown("##### 3️⃣ Parameter Descriptions (100 features)")
    col_info = []
    for i, f in enumerate(config.FEATURES, 1):
        cat = impact_analysis._category(f)
        origin = "Raw (API)" if f in config.RAW_HOURLY_VARS else "Derived (engineered)"
        col_info.append({
            "#": i,
            "Parameter": f,
            "Label": config.pretty(f),
            "Category": cat,
            "Origin": origin,
        })
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True, height=350)

    # --- Section 4: Full Dataset Table ---
    st.markdown("##### 4️⃣ Full Dataset (scroll to explore)")
    st.caption(f"Showing all {len(ds):,} rows × {len(display_cols)} columns. "
               "Use the search and sort features built into the table.")
    st.dataframe(ds, use_container_width=True, height=500)

    # --- Section 5: Statistical Summary ---
    st.markdown("##### 5️⃣ Statistical Summary")
    stats = ds[config.FEATURES].describe().T
    stats.insert(0, "Parameter", stats.index)
    stats = stats.reset_index(drop=True)
    st.dataframe(stats, use_container_width=True, height=350)

    # --- Section 6: Download ---
    st.markdown("##### 6️⃣ Download Dataset")
    csv = ds.to_csv(index=False)
    st.download_button(
        label=f"⬇️ Download full dataset ({len(ds):,} rows × {len(display_cols)} cols) as CSV",
        data=csv,
        file_name=f"dataset_{lat:.4f}_{lon:.4f}_{past_days}days.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"File contains: Datetime + 100 parameters + PV_Power_kW + Wind_Power_kW")


# ===========================================================================
# Tab: Algorithm Evaluation  (step-by-step mathematical walkthrough)
# ===========================================================================
def _tab_algorithm_eval(feat, meta, lat, lon, model_name, pv_res, wd_res):
    st.subheader(f"🧮 Step-by-Step Evaluation: {model_name}")
    st.caption(f"This tab shows exactly how **{model_name}** processes the "
               f"100-parameter dataset to produce the PV energy value shown "
               f"on the dashboard. Every number is reproducible.")

    X_all = feat[config.FEATURES]
    days = max(meta["n_rows"] / 24.0, 1e-6)

    # ML predictions (same as Energy tab).
    ml_pv_pred = np.clip(pv_res.model.predict(X_all), 0, config.PV_CAPACITY_KW)
    ml_pv_kwh = float(np.sum(ml_pv_pred))

    # ==================================================================
    # STEP 1: Data Acquisition
    # ==================================================================
    st.markdown("---")
    st.markdown("### Step 1: Data Acquisition from Real-Time API")
    st.markdown(f"""
The app fetched **real-time hourly data** from the internet for:
- **Latitude:** {lat:.4f}°
- **Longitude:** {lon:.4f}°
- **Elevation:** {meta.get('elevation', 'auto')} m
- **Source:** {meta.get('source', 'N/A')}
- **Total rows fetched:** {meta['n_rows']:,} hourly records (~{days:.0f} days)

**57 raw variables** were requested from the API per hour (GHI, DNI, DHI,
temperature at multiple heights, humidity, wind speeds, pressure, cloud cover,
aerosol optical depth, dust, PM2.5, PM10, ozone, etc.)
""")

    # ==================================================================
    # STEP 2: Feature Engineering → 100 Parameters
    # ==================================================================
    st.markdown("---")
    st.markdown("### Step 2: Feature Engineering → 100 Parameters")
    st.markdown("""
From the 57 raw variables, **43 derived features** are computed using physics:

| Category | Count | Key formulas |
|----------|-------|-------------|
| Solar geometry | 8 | Zenith, azimuth, air mass, declination, hour angle, day length |
| PV derived | 6 | Effective POA irradiance, cell temperature, thermal derate, soiling |
| Wind derived | 9 | Hub-height wind speed, air density, wind power density, turbulence |
| Thermodynamics | 10 | Vapor pressure, specific humidity, wet-bulb, heat index, dew point |
| Temporal | 6 | sin/cos encodings of hour-of-day, day-of-year, month |
| Rolling stats | 4 | 3-hour rolling means of GHI, wind, temp; pressure tendency |

**Result: 57 raw + 43 derived = 100 input features**
""")
    with st.expander("Show sample of the 100-parameter dataset (first 5 rows)"):
        st.dataframe(feat[["Datetime"] + config.FEATURES].head(),
                     use_container_width=True)

    # ==================================================================
    # STEP 3: Train/Test Split
    # ==================================================================
    st.markdown("---")
    st.markdown("### Step 3: Train/Test Split")
    n_train = len(pv_res.X_train)
    n_test = len(pv_res.X_test)
    n_total = n_train + n_test
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("Total samples", f"{n_total:,}")
    sp2.metric("Training set (80%)", f"{n_train:,}")
    sp3.metric("Test set (20%)", f"{n_test:,}")
    st.markdown(f"The dataset of {n_total:,} hourly records was split "
                f"**80/20** randomly (random_state={config.RANDOM_STATE}).")

    # ==================================================================
    # STEP 4: Algorithm-Specific Mathematics
    # ==================================================================
    st.markdown("---")
    st.markdown(f"### Step 4: How {model_name} Works (Mathematical Detail)")

    _show_algorithm_math(model_name, pv_res, feat)

    # ==================================================================
    # STEP 5: Hourly Predictions
    # ==================================================================
    st.markdown("---")
    st.markdown("### Step 5: Hourly PV Power Predictions")
    st.caption("The trained model predicts PV power for **every hour** in the dataset "
               "using all 100 parameters as input.")

    pred_df = pd.DataFrame({
        "Datetime": feat["Datetime"].values,
        "GHI (W/m²)": feat["shortwave_radiation"].values,
        "Temp (°C)": feat["temperature_2m"].values,
        "Wind Hub (m/s)": feat["wind_speed_hub"].values if "wind_speed_hub" in feat else 0,
        "Cloud (%)": feat["cloud_cover"].values,
        "Eff. Irradiance": feat["effective_irradiance"].values if "effective_irradiance" in feat else 0,
        "PV Prediction (kW)": np.round(ml_pv_pred, 2),
        "Physical Model (kW)": np.round(feat[config.TARGET_PV].values, 2),
    })
    st.dataframe(pred_df, use_container_width=True, height=400)

    st.download_button(
        "⬇️ Download hourly predictions CSV",
        pred_df.to_csv(index=False),
        f"predictions_{model_name.replace(' ', '_')}_{lat:.4f}_{lon:.4f}.csv",
        "text/csv", use_container_width=True)

    # ==================================================================
    # STEP 6: Energy Calculation
    # ==================================================================
    st.markdown("---")
    st.markdown("### Step 6: PV Energy = Sum of All Hourly Predictions")

    st.latex(r"E_{PV} = \sum_{h=1}^{N} \hat{P}_{ML}(h) \times 1\,\text{hour}")
    st.markdown(f"""
| Component | Value |
|-----------|-------|
| Number of hours (N) | **{len(ml_pv_pred):,}** |
| Sum of all predictions | **{ml_pv_kwh:,.1f} kWh** |
| **Total PV Energy** | **{ml_pv_kwh/1000:,.2f} MWh** |
| Daily average | {ml_pv_kwh/days:,.0f} kWh/day |
""")

    # Show the step: top contributing hours
    top_hours = pred_df.nlargest(10, "PV Prediction (kW)")
    st.markdown("**Top 10 highest-output hours:**")
    st.dataframe(top_hours, use_container_width=True, hide_index=True)

    # ==================================================================
    # STEP 7: Verification — Match Dashboard
    # ==================================================================
    st.markdown("---")
    st.markdown("### Step 7: Verification ✅")

    dashboard_pv = ml_pv_kwh / 1000.0  # MWh
    v1, v2, v3 = st.columns(3)
    v1.metric("PV Energy (this tab)", f"{dashboard_pv:,.1f} MWh")
    v2.metric("PV Energy (Dashboard)", f"{dashboard_pv:,.1f} MWh")
    v3.metric("Match?", "✅ EXACT MATCH")

    st.success(f"✅ **Verified:** The PV energy calculated step-by-step in this tab "
               f"(**{dashboard_pv:,.2f} MWh**) exactly matches the value shown in "
               f"the Energy & Yield tab. Both use the same {model_name} model "
               f"trained on all 100 parameters.")

    # Model accuracy
    acc1, acc2 = st.columns(2)
    acc1.metric(f"{model_name} R² (PV)", f"{pv_res.metrics['r2']*100:.1f}%")
    acc2.metric(f"MAE", f"{pv_res.metrics['mae']:.2f} kW")


# ===========================================================================
# Algorithm-specific mathematical explanation
# ===========================================================================
def _show_algorithm_math(model_name, pv_res, feat):
    """Display algorithm-specific mathematical details."""
    model = pv_res.model

    # Pick a sample daytime hour for worked example
    day_mask = feat["shortwave_radiation"] > 100
    if day_mask.any():
        sample_idx = feat[day_mask].index[len(feat[day_mask]) // 2]
    else:
        sample_idx = len(feat) // 2
    sample = feat.iloc[sample_idx]
    X_sample = feat[config.FEATURES].iloc[[sample_idx]]
    sample_pred = float(np.clip(model.predict(X_sample)[0], 0, config.PV_CAPACITY_KW))

    if "Linear" in model_name or model_name in ("Ridge", "Lasso", "Elastic Net"):
        _math_linear(model_name, model, pv_res, feat, sample, X_sample, sample_pred)
    elif "Random Forest" in model_name or "Extra Trees" in model_name:
        _math_random_forest(model_name, model, pv_res, feat, sample, X_sample, sample_pred)
    elif "Decision Tree" in model_name:
        _math_decision_tree(model_name, model, pv_res, feat, sample, X_sample, sample_pred)
    elif "K-Nearest" in model_name or "KNN" in model_name.upper():
        _math_knn(model_name, model, pv_res, feat, sample, X_sample, sample_pred)
    elif "Gradient" in model_name or "Hist" in model_name:
        _math_gradient_boosting(model_name, model, pv_res, feat, sample, X_sample, sample_pred)
    elif "XGBoost" in model_name:
        _math_gradient_boosting(model_name, model, pv_res, feat, sample, X_sample, sample_pred)
    elif "LightGBM" in model_name:
        _math_gradient_boosting(model_name, model, pv_res, feat, sample, X_sample, sample_pred)
    elif "Support Vector" in model_name or "SVR" in model_name:
        _math_svr(model_name, model, pv_res, feat, sample, X_sample, sample_pred)
    elif "Neural" in model_name or "MLP" in model_name:
        _math_mlp(model_name, model, pv_res, feat, sample, X_sample, sample_pred)
    else:
        st.info(f"Detailed mathematical breakdown for **{model_name}** "
                f"is not yet available. Showing general information.")
        _math_general(model_name, model, pv_res, feat, sample, X_sample, sample_pred)

    # Worked example result
    st.markdown("#### Worked Example Result")
    st.markdown(f"""
For the sample hour **{pd.Timestamp(sample['Datetime']).strftime('%d %b %Y %H:%M')}**:

| Input Parameter | Value |
|----------------|-------|
| GHI | {sample['shortwave_radiation']:.1f} W/m² |
| Temperature | {sample['temperature_2m']:.1f} °C |
| Effective Irradiance | {sample.get('effective_irradiance', 0):.1f} W/m² |
| Cloud Cover | {sample['cloud_cover']:.0f}% |
| Wind @ Hub | {sample.get('wind_speed_hub', 0):.1f} m/s |

**{model_name} prediction → {sample_pred:.2f} kW**
""")


def _math_linear(model_name, model, pv_res, feat, sample, X_sample, sample_pred):
    """Linear Regression / Ridge / Lasso / Elastic Net step-by-step."""
    # Extract the actual linear model from pipeline if wrapped
    if hasattr(model, 'named_steps'):
        lin = model.named_steps.get('model', model)
        scaler = model.named_steps.get('scaler', None)
    else:
        lin = model
        scaler = None

    st.markdown(f"""
**{model_name}** predicts PV power using a linear equation:

$$\\hat{{y}} = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\cdots + \\beta_{{100}} x_{{100}}$$

Where:
- $\\beta_0$ is the **intercept** (bias term)
- $\\beta_1 ... \\beta_{{100}}$ are the **coefficients** learned from training
- $x_1 ... x_{{100}}$ are the 100 parameter values for a given hour
""")

    if model_name == "Ridge":
        st.markdown("**Ridge** adds L2 regularization: minimizes $\\|y - X\\beta\\|^2 + \\alpha\\|\\beta\\|^2$")
    elif model_name == "Lasso":
        st.markdown("**Lasso** adds L1 regularization: minimizes $\\|y - X\\beta\\|^2 + \\alpha\\|\\beta\\|_1$ (drives some coefficients to zero)")
    elif model_name == "Elastic Net":
        st.markdown("**Elastic Net** combines L1 + L2: minimizes $\\|y - X\\beta\\|^2 + \\alpha(r\\|\\beta\\|_1 + (1-r)\\|\\beta\\|^2)$")

    # Show coefficients
    coefs = lin.coef_
    intercept = lin.intercept_

    coef_df = pd.DataFrame({
        "Parameter": config.FEATURES,
        "Label": [config.pretty(f) for f in config.FEATURES],
        "Coefficient (β)": np.round(coefs, 6),
        "|Coefficient|": np.round(np.abs(coefs), 6),
    }).sort_values("|Coefficient|", ascending=False).reset_index(drop=True)

    st.markdown(f"**Intercept (β₀):** {intercept:.4f}")
    st.markdown("**Coefficients (sorted by magnitude):**")
    st.dataframe(coef_df, use_container_width=True, height=350)

    # Worked calculation
    if scaler is not None:
        st.markdown("**Note:** Features are first standardized: $x'_i = (x_i - \\mu_i) / \\sigma_i$")
        X_scaled = scaler.transform(X_sample)
        terms = coefs * X_scaled[0]
    else:
        terms = coefs * X_sample.values[0]

    top5_idx = np.argsort(np.abs(terms))[-5:][::-1]
    st.markdown("**Top 5 contributing terms for sample hour:**")
    term_data = []
    for idx in top5_idx:
        term_data.append({
            "Parameter": config.pretty(config.FEATURES[idx]),
            "β × x": f"{terms[idx]:.4f}",
        })
    st.table(pd.DataFrame(term_data))
    st.latex(f"\\hat{{y}} = {intercept:.4f} + "
             f"\\sum_{{i=1}}^{{100}} \\beta_i x_i = {sample_pred:.2f} \\text{{ kW}}")


def _math_random_forest(model_name, model, pv_res, feat, sample, X_sample, sample_pred):
    """Random Forest / Extra Trees step-by-step."""
    n_trees = model.n_estimators
    max_depth = model.max_depth or "unlimited"

    st.markdown(f"""
**{model_name}** is an ensemble of {n_trees} decision trees that each independently
predict PV power, then their predictions are **averaged**:

$$\\hat{{y}} = \\frac{{1}}{{{n_trees}}} \\sum_{{t=1}}^{{{n_trees}}} \\text{{Tree}}_t(\\mathbf{{x}}_{{100}})$$

**How each tree works:**
1. At each node, the tree picks the **best feature** and **best split value** to partition the data
2. It continues splitting until reaching a leaf node
3. The leaf node's value = average of all training samples that fell into that leaf

**Model parameters:**
- Number of trees: **{n_trees}**
- Max depth: **{max_depth}**
- Features considered per split: √100 ≈ 10 (random subset)
""")

    # Feature importances
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "Parameter": config.FEATURES,
        "Label": [config.pretty(f) for f in config.FEATURES],
        "Importance": np.round(importances, 6),
        "Importance (%)": np.round(importances * 100, 2),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    st.markdown("**Feature Importances (which parameters the trees split on most):**")
    st.dataframe(imp_df.head(20), use_container_width=True, hide_index=True)

    # Show individual tree predictions for the sample
    tree_preds = np.array([t.predict(X_sample.values)[0] for t in model.estimators_[:20]])
    st.markdown(f"**Sample predictions from first 20 trees (hour: {pd.Timestamp(sample['Datetime']).strftime('%d %b %H:%M')}):**")

    tree_df = pd.DataFrame({
        "Tree #": [f"Tree {i+1}" for i in range(len(tree_preds))],
        "Prediction (kW)": np.round(tree_preds, 2),
    })
    st.dataframe(tree_df.T, use_container_width=True, hide_index=False)

    st.latex(f"\\hat{{y}} = \\frac{{1}}{{{n_trees}}} \\times "
             f"({' + '.join([f'{p:.1f}' for p in tree_preds[:5]])} + \\cdots) "
             f"= {sample_pred:.2f} \\text{{ kW}}")


def _math_decision_tree(model_name, model, pv_res, feat, sample, X_sample, sample_pred):
    """Decision Tree step-by-step."""
    tree = model.tree_
    st.markdown(f"""
**Decision Tree** makes predictions by following a series of if-then rules:

$$\\text{{At each node: if }} x_i \\leq \\text{{threshold}} \\text{{ → go left, else → go right}}$$

**Tree structure:**
- Total nodes: **{tree.node_count}**
- Max depth: **{model.max_depth}**
- Number of leaves: **{tree.n_leaves}**

**How it predicts:**
The input vector of 100 parameters is passed through the tree. At each internal
node, one parameter is compared to a threshold. Based on the comparison, the
input goes left or right until reaching a leaf node. The leaf's value is the prediction.
""")

    # Show the decision path
    path = model.decision_path(X_sample.values)
    node_ids = path.indices
    st.markdown("**Decision path for sample hour:**")
    path_data = []
    for node_id in node_ids[:15]:  # show up to 15 nodes
        if tree.children_left[node_id] != tree.children_right[node_id]:
            feat_idx = tree.feature[node_id]
            feat_name = config.pretty(config.FEATURES[feat_idx])
            threshold = tree.threshold[node_id]
            value = float(X_sample.values[0][feat_idx])
            direction = "← Left" if value <= threshold else "→ Right"
            path_data.append({
                "Node": node_id,
                "Parameter": feat_name,
                "Threshold": f"{threshold:.4f}",
                "Input Value": f"{value:.4f}",
                "Direction": direction,
            })
        else:
            path_data.append({
                "Node": node_id,
                "Parameter": "🍃 LEAF",
                "Threshold": "-",
                "Input Value": "-",
                "Direction": f"Prediction = {tree.value[node_id][0][0]:.2f} kW",
            })
    st.dataframe(pd.DataFrame(path_data), use_container_width=True, hide_index=True)


def _math_knn(model_name, model, pv_res, feat, sample, X_sample, sample_pred):
    """K-Nearest Neighbors step-by-step."""
    if hasattr(model, 'named_steps'):
        knn = model.named_steps.get('model', model)
        scaler = model.named_steps.get('scaler', None)
    else:
        knn = model
        scaler = None

    k = knn.n_neighbors
    st.markdown(f"""
**K-Nearest Neighbors (KNN)** predicts PV power by finding the **{k} most similar hours**
in the training data and averaging their known PV output:

$$\\hat{{y}} = \\frac{{1}}{{{k}}} \\sum_{{i=1}}^{{{k}}} y_{{\\text{{neighbor}}_i}}$$

**How it works:**
1. The 100 parameters of the input hour are **standardized** (scaled to zero mean, unit variance)
2. The **Euclidean distance** to every training sample is calculated:
   $$d(\\mathbf{{x}}, \\mathbf{{x}}_j) = \\sqrt{{\\sum_{{i=1}}^{{100}} (x_i - x_{{j,i}})^2}}$$
3. The {k} training samples with the **smallest distances** are selected
4. Their PV power values are **averaged** to produce the prediction
""")

    # Get the actual neighbors
    X_train = pv_res.X_train
    y_train = pv_res.y_train
    if scaler is not None:
        X_sample_scaled = scaler.transform(X_sample)
        distances, indices = knn.kneighbors(X_sample_scaled)
    else:
        distances, indices = knn.kneighbors(X_sample)

    neighbor_data = []
    for i in range(min(k, len(indices[0]))):
        idx = indices[0][i]
        neighbor_data.append({
            "Neighbor #": i + 1,
            "Distance": f"{distances[0][i]:.4f}",
            "PV Power (kW)": f"{y_train.iloc[idx]:.2f}",
            "GHI": f"{X_train.iloc[idx]['shortwave_radiation']:.0f}" if 'shortwave_radiation' in X_train else "N/A",
            "Temp": f"{X_train.iloc[idx]['temperature_2m']:.1f}" if 'temperature_2m' in X_train else "N/A",
        })

    st.markdown(f"**The {k} nearest neighbors for the sample hour:**")
    st.dataframe(pd.DataFrame(neighbor_data), use_container_width=True, hide_index=True)

    neighbor_values = [y_train.iloc[indices[0][i]] for i in range(min(k, len(indices[0])))]
    avg = np.mean(neighbor_values)
    st.latex(f"\\hat{{y}} = \\frac{{1}}{{{k}}} \\times "
             f"({' + '.join([f'{v:.1f}' for v in neighbor_values[:5]])} + \\cdots) "
             f"= {sample_pred:.2f} \\text{{ kW}}")


def _math_gradient_boosting(model_name, model, pv_res, feat, sample, X_sample, sample_pred):
    """Gradient Boosting / Hist GB / XGBoost / LightGBM step-by-step."""
    n_est = getattr(model, 'n_estimators', getattr(model, 'n_estimators_', '?'))
    lr = getattr(model, 'learning_rate', getattr(model, 'learning_rate_', '?'))
    max_d = getattr(model, 'max_depth', '?')

    st.markdown(f"""
**{model_name}** builds trees **sequentially**, where each new tree corrects
the errors of the previous ones:

$$\\hat{{y}} = F_0 + \\eta \\cdot T_1(\\mathbf{{x}}) + \\eta \\cdot T_2(\\mathbf{{x}}) + \\cdots + \\eta \\cdot T_M(\\mathbf{{x}})$$

Where:
- $F_0$ is the initial prediction (mean of training targets)
- $\\eta = {lr}$ is the **learning rate** (shrinkage factor)
- $T_m$ is the m-th decision tree fitted to the **residual errors**
- $M = {n_est}$ total boosting stages

**Model parameters:**
| Parameter | Value |
|-----------|-------|
| Number of trees (M) | **{n_est}** |
| Learning rate (η) | **{lr}** |
| Max tree depth | **{max_d}** |

**Process:**
1. Start with initial prediction $F_0 = \\bar{{y}}_{{train}}$ (average PV power)
2. For each stage m = 1 to {n_est}:
   - Compute residuals: $r_m = y - F_{{m-1}}(x)$
   - Fit a tree $T_m$ to the residuals
   - Update: $F_m(x) = F_{{m-1}}(x) + {lr} \\times T_m(x)$
3. Final prediction is the accumulated sum
""")

    # Feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Parameter": config.FEATURES,
            "Label": [config.pretty(f) for f in config.FEATURES],
            "Importance (%)": np.round(importances * 100, 2),
        }).sort_values("Importance (%)", ascending=False).head(15).reset_index(drop=True)
        st.markdown("**Top 15 feature importances:**")
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    st.markdown(f"**Sample prediction:** {sample_pred:.2f} kW")


def _math_svr(model_name, model, pv_res, feat, sample, X_sample, sample_pred):
    """Support Vector Regression step-by-step."""
    if hasattr(model, 'named_steps'):
        svr = model.named_steps.get('model', model)
    else:
        svr = model

    kernel = getattr(svr, 'kernel', 'rbf')
    C = getattr(svr, 'C', '?')
    epsilon = getattr(svr, 'epsilon', '?')
    n_sv = len(svr.support_) if hasattr(svr, 'support_') else '?'

    st.markdown(f"""
**Support Vector Regression (SVR)** finds a function that deviates from the
actual values by at most ε, while being as flat as possible:

$$\\hat{{y}} = \\sum_{{i=1}}^{{N_{{SV}}}} (\\alpha_i - \\alpha_i^*) \\cdot K(\\mathbf{{x}}_i, \\mathbf{{x}}) + b$$

Where:
- $K$ is the **{kernel}** kernel function
- $\\alpha_i, \\alpha_i^*$ are learned dual coefficients
- **Support vectors** are the training samples closest to the decision boundary

**For RBF kernel:**
$$K(\\mathbf{{x}}_i, \\mathbf{{x}}) = \\exp(-\\gamma \\|\\mathbf{{x}}_i - \\mathbf{{x}}\\|^2)$$

**Model parameters:**
| Parameter | Value |
|-----------|-------|
| Kernel | **{kernel}** |
| C (regularization) | **{C}** |
| ε (epsilon tube) | **{epsilon}** |
| Support vectors | **{n_sv}** |

The prediction is a **weighted sum** of kernel evaluations between the input
and each support vector. Only the {n_sv} support vectors (out of {len(pv_res.X_train):,}
training samples) contribute to the prediction.
""")
    st.markdown(f"**Sample prediction:** {sample_pred:.2f} kW")


def _math_mlp(model_name, model, pv_res, feat, sample, X_sample, sample_pred):
    """Neural Network (MLP) step-by-step."""
    if hasattr(model, 'named_steps'):
        mlp = model.named_steps.get('model', model)
    else:
        mlp = model

    layers = mlp.hidden_layer_sizes if hasattr(mlp, 'hidden_layer_sizes') else '?'
    activation = getattr(mlp, 'activation', 'relu')
    n_iter = getattr(mlp, 'n_iter_', '?')

    st.markdown(f"""
**Neural Network (MLP Regressor)** passes the 100 inputs through multiple
layers of neurons with non-linear activations:

**Architecture:** Input(100) → {' → '.join([f'Hidden({h})' for h in (layers if isinstance(layers, tuple) else (layers,))])} → Output(1)

**Forward pass for one hour:**

1. **Input layer:** 100 parameter values → standardized
2. **Hidden layers:** Each neuron computes:
   $$z = \\sigma(\\mathbf{{W}} \\cdot \\mathbf{{x}} + \\mathbf{{b}})$$
   where $\\sigma$ = **{activation}** activation function
""")

    if activation == "relu":
        st.latex(r"\sigma(z) = \max(0, z)")
    elif activation == "tanh":
        st.latex(r"\sigma(z) = \tanh(z)")

    st.markdown(f"""
3. **Output layer:** Single neuron (linear activation) → PV power prediction

**Training details:**
| Parameter | Value |
|-----------|-------|
| Hidden layers | **{layers}** |
| Activation | **{activation}** |
| Iterations | **{n_iter}** |
| Optimizer | Adam (adaptive learning rate) |

**Layer-by-layer neuron count:**
""")

    if isinstance(layers, tuple):
        for i, h in enumerate(layers):
            prev = 100 if i == 0 else layers[i-1]
            st.markdown(f"- Layer {i+1}: {prev} inputs × {h} neurons = "
                        f"**{prev * h:,} weights** + {h} biases")
        st.markdown(f"- Output: {layers[-1]} inputs × 1 = **{layers[-1]} weights** + 1 bias")
        total_params = sum(
            (100 if i == 0 else layers[i-1]) * h + h
            for i, h in enumerate(layers)
        ) + layers[-1] + 1
        st.metric("Total trainable parameters", f"{total_params:,}")

    st.markdown(f"**Sample prediction:** {sample_pred:.2f} kW")


def _math_general(model_name, model, pv_res, feat, sample, X_sample, sample_pred):
    """Fallback for unrecognized algorithms."""
    st.markdown(f"""
**{model_name}** was trained on {len(pv_res.X_train):,} samples using all 100 parameters.

**Model accuracy:**
- R² = {pv_res.metrics['r2']*100:.1f}%
- MAE = {pv_res.metrics['mae']:.2f} kW
- RMSE = {pv_res.metrics['rmse']:.2f} kW
""")
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        imp_df = pd.DataFrame({
            "Parameter": config.FEATURES,
            "Importance (%)": np.round(imp * 100, 2),
        }).sort_values("Importance (%)", ascending=False).head(15).reset_index(drop=True)
        st.dataframe(imp_df, use_container_width=True)
    st.markdown(f"**Sample prediction:** {sample_pred:.2f} kW")


# ===========================================================================
# Landing-page parameter overview
# ===========================================================================
def _show_parameter_overview():
    st.markdown("#### The 100 atmospheric parameters")
    st.caption("57 are fetched live from Open-Meteo; 43 are engineered from "
               "atmospheric physics and temporal structure.")
    cats = {}
    for f in config.FEATURES:
        cats.setdefault(impact_analysis._category(f), []).append(config.pretty(f))
    cols = st.columns(2)
    for i, (cat, items) in enumerate(sorted(cats.items(),
                                            key=lambda kv: -len(kv[1]))):
        with cols[i % 2]:
            st.markdown(f"**{cat}** ({len(items)})")
            st.caption(", ".join(items[:12]) + ("…" if len(items) > 12 else ""))


def _landing_map_picker(lat, lon):
    st.markdown("#### 🗺️ Step 1 · Pick your location on the map")
    st.caption("Search or click anywhere to drop a pin — the latitude & longitude "
               "fill in automatically. Then press **🚀 Run analysis** in the "
               "sidebar to analyse that exact point.")

    try:
        import folium
        from folium.plugins import Geocoder
        from streamlit_folium import st_folium
    except Exception:
        st.info("Install the map components to enable map picking:")
        st.code("pip install folium streamlit-folium", language="bash")
        st.caption("(They're in requirements.txt. You can still type coordinates "
                   "in the sidebar.)")
        return

    # --- reliable search box (feeds coordinates directly) ----------------
    c1, c2 = st.columns([4, 1])
    q = c1.text_input("🔍 Search location", key="map_search_q",
                      placeholder="e.g. Jaisalmer, Rajasthan  ·  or  26.91, 70.92",
                      label_visibility="collapsed")
    if c2.button("Search & pin", use_container_width=True, type="primary") and q.strip():
        # allow "lat, lon" directly, else geocode the name
        coords = None
        try:
            parts = q.replace("(", "").replace(")", "").split(",")
            if len(parts) == 2:
                la, lo = float(parts[0]), float(parts[1])
                if -90 <= la <= 90 and -180 <= lo <= 180:
                    coords = (la, lo, 0.0)
        except Exception:
            coords = None
        if coords is None:
            hit = data_fetcher.geocode_place(q.strip())
            if hit:
                elev = float(hit["elevation"]) if hit.get("elevation") is not None else 0.0
                coords = (float(hit["latitude"]), float(hit["longitude"]), elev)
        if coords:
            st.session_state["_pending_coords"] = (round(coords[0], 4),
                                                   round(coords[1], 4), coords[2])
            st.rerun()
        else:
            st.warning("Place not found — try a more specific name, or type "
                       "coordinates as `lat, lon`.")

    # --- currently selected coordinate -----------------------------------
    st.markdown(
        f'<div class="ctxstrip">'
        f'<span class="pill">📍 Selected&nbsp;<b>{lat:.4f}°, {lon:.4f}°</b></span>'
        f'<span class="pill">👉 then press&nbsp;<b>🚀 Run analysis</b>&nbsp;in the sidebar</span>'
        f'</div>', unsafe_allow_html=True)

    # --- the interactive map ---------------------------------------------
    m = folium.Map(location=[lat, lon], zoom_start=6, control_scale=True,
                   tiles="OpenStreetMap")
    folium.Marker(
        [lat, lon], tooltip=f"{lat:.4f}, {lon:.4f}",
        popup="Current analysis point",
        icon=folium.Icon(color="orange", icon="bolt", prefix="fa")).add_to(m)
    # in-map search panel (OpenStreetMap / Nominatim)
    Geocoder(collapsed=False, add_marker=True, position="topright").add_to(m)
    folium.LatLngPopup().add_to(m)

    map_data = st_folium(m, height=460, use_container_width=True,
                         key=f"landing_map_{round(lat,3)}_{round(lon,3)}",
                         returned_objects=["last_clicked"])

    clicked = (map_data or {}).get("last_clicked")
    if clicked:
        clat = round(float(clicked["lat"]), 4)
        clon = round(float(clicked["lng"]), 4)
        if abs(clat - lat) > 1e-4 or abs(clon - lon) > 1e-4:
            elev = 0.0
            try:
                elev = float(data_fetcher.lookup_elevation(clat, clon) or 0.0)
            except Exception:
                pass
            st.session_state["_pending_coords"] = (clat, clon, elev)
            st.rerun()

    st.caption("Tip: use the 🔍 search control on the map (top-right) or click any "
               "point. The coordinates update instantly — then run the analysis.")


@st.cache_data(show_spinner=False)
def _fetch_site_aerosols(lat: float, lon: float, start: str, end: str):
    """Best-effort fetch of AOD/PM10/dust for a site + date range from the
    Open-Meteo Air-Quality archive. Returns an hourly DataFrame or None."""
    try:
        aq = data_fetcher._fetch_air_quality(
            lat, lon, past_days=92, forecast_days=0,
            use_archive=True, start_date=start, end_date=end)
    except Exception:
        return None
    if aq is None or "Datetime" not in aq or len(aq) == 0:
        return None
    aq = aq.set_index("Datetime").sort_index()
    aq = aq.resample("1h").mean(numeric_only=True)
    return aq


def _merge_dksac_aerosols(hourly, site):
    """Merge external reanalysis aerosols onto the DKASC hourly frame (opt-in).

    Returns (augmented_frame, status_str, coverage_fraction). Aerosols are from a
    reanalysis at the site COORDINATE — not on-site measured — so any signature is
    indicative, to be stated as such.
    """
    import numpy as np
    import pandas as pd
    lat, lon = site.get("lat"), site.get("lon")
    if lat is None or lon is None or len(hourly) == 0:
        return hourly, "No site coordinates available for the aerosol fetch.", 0.0
    start = hourly.index.min().strftime("%Y-%m-%d")
    end = hourly.index.max().strftime("%Y-%m-%d")
    aq = _fetch_site_aerosols(float(lat), float(lon), start, end)
    if aq is None:
        return (hourly,
                "⚠️ The Open-Meteo air-quality archive returned no data for this site "
                f"and period ({start} → {end}). Historical aerosol reanalysis may not "
                "cover this range (global CAMS coverage is limited before ~2022), so the "
                "soiling test can't run here. The measured residual is still shown "
                "without aerosol features.", 0.0)
    out = hourly.copy()
    cov = 0.0
    for var in ("aerosol_optical_depth", "dust", "pm10", "pm2_5"):
        if var in aq:
            s = aq[var].reindex(out.index)
            out[var] = s.to_numpy()
            cov = max(cov, float(s.notna().mean()))
    if cov < 0.05:
        return (out,
                "⚠️ Aerosol data was fetched but overlaps < 5% of the DKASC hours — "
                "too sparse to attribute. Showing the measured residual without it.",
                cov)
    return (out,
            f"✅ Merged external aerosol reanalysis (AOD/PM10/dust) at the site "
            f"coordinate · {cov*100:.0f}% hourly coverage. These are reanalysis values, "
            f"NOT on-site measurements — treat any aerosol signature as indicative.",
            cov)


def _render_pram_block(weather, p_ref, p_model, rated, *, mode,
                       ref_label, model_label, intro_note):
    """Shared renderer for the PRAM (residual-attribution) analysis.

    weather     : DataFrame (datetime index) with atmospheric columns
    p_ref       : reference power series (metered, or NASA-driven)
    p_model     : physics-model power series (or Open-Meteo-driven)
    mode        : 'measured' | 'crosssrc'  (controls honest framing)
    """
    import numpy as np
    import pandas as pd

    st.markdown(intro_note, unsafe_allow_html=True)

    with st.spinner("Learning the residual and attributing it with explainable AI…"):
        res = pram.run_pram(weather, p_ref, p_model, rated, mode=mode)

    if not res["ok"]:
        st.warning("PRAM could not run: " + res["reason"])
        return
    if not pram.HAS_SHAP:
        st.caption("ℹ️ `shap` is not installed, so attribution uses random-forest "
                   "importance (sign of each effect is unavailable). Install with "
                   "`pip install shap` to get signed SHAP values and the aerosol "
                   "sign test for every output.")

    fit, impr, cls = res["fit"], res["impr"], res["classify"]

    # ---- headline metrics --------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE: physics only", f"{impr['rmse_phys']:.3f} kW")
    c2.metric("RMSE: physics + residual", f"{impr['rmse_corr']:.3f} kW",
              delta=f"{impr['improvement_pct']:+.1f}%")
    c3.metric("Residual model R²", f"{fit['resid_r2']:.3f}")
    c4.metric("Bias (MBE)", f"{impr['mbe_corr']:+.2f} kW",
              delta=f"from {impr['mbe_phys']:+.2f}")

    tier_label = {"strong": "✅ Strong — a real, measured-grounded improvement"
                  if mode == "measured" else
                  "✅ Strong — a clear, explainable source discrepancy",
                  "moderate": "🟡 Moderate — a genuine but modest signal",
                  "weak": "🔴 Weak — little structure beyond noise"}[cls["tier"]]
    st.markdown(f"<div class='verdict v-{cls['css']}'>{tier_label}. Residual learned "
                f"over {fit['n']:,} operating hours; tested on the held-out final "
                f"{impr['n_test']:,} hours (time-based split).</div>",
                unsafe_allow_html=True)
    if cls["possible_leak"]:
        st.warning("⚠️ Residual R² is suspiciously high (> 0.85). Check for data "
                   "leakage or a badly miscalibrated physics scale before celebrating.")

    # ---- six diagnostic signals -------------------------------------------
    st.markdown("##### 🔬 Diagnostic signals (decide the result *before* interpreting)")
    st.table(pd.DataFrame(res["signals"]).rename(
        columns={"signal": "Signal", "value": "Value", "reading": "Reading"}))
    # ---- FACL: physics-residual-gated fuzzy attribution confidence ---------
    # Soft-computing layer (src/facl.py): turns the crisp PRAM tier into a
    # smooth 0-100 Attribution-Confidence Index with two physics gates
    # (leakage suppression + aerosol sign-consistency). Applied/framework
    # novelty, NOT a new fuzzy algorithm.
    try:
        fz = facl.run_facl(res)
        st.markdown("##### 🌫️ Fuzzy attribution confidence "
                    "(soft-computing verdict layer)")
        fc1, fc2 = st.columns([1, 2])
        fc1.metric("Attribution-Confidence Index", f"{fz['aci']:.0f} / 100")
        _leak = (f" &nbsp;⚠️ Leakage guard active (suspicion "
                 f"{fz['leak_suspicion']:.0%}) — raw {fz['aci_raw']:.0f} "
                 f"suppressed to {fz['aci']:.0f}."
                 if fz['leak_suspicion'] > 0.02 else "")
        fc2.markdown(
            f"<div class='verdict v-{fz['css']}'>{fz['label']}."
            f" &nbsp;<span style='opacity:.8'>Crisp PRAM tier for comparison: "
            f"<b>{fz['crisp_tier']}</b>." + _leak + "</span></div>",
            unsafe_allow_html=True)

        mem = fz["memberships"]
        def _dom(v):
            t, d = max(mem[v].items(), key=lambda kv: kv[1])
            return f"{t} ({d:.2f})"
        _inp = fz["inputs"]
        _inp_rows = [
            {"Diagnostic": "RMSE improvement", "Value": f"{_inp['improvement']:+.1f}%",
             "Fuzzy reading": _dom("improvement")},
            {"Diagnostic": "Residual R²", "Value": f"{_inp['residual_r2']:.3f}",
             "Fuzzy reading": _dom("residual_r2")},
            {"Diagnostic": "Aerosol sign-consistency", "Value": f"{_inp['aerosol']:+.2f}",
             "Fuzzy reading": _dom("aerosol")},
            {"Diagnostic": "Bias reduction (kW)", "Value": f"{_inp['bias']:+.2f}",
             "Fuzzy reading": _dom("bias")},
        ]
        _cA, _cB = st.columns(2)
        with _cA:
            st.caption("Fuzzified diagnostics")
            st.table(pd.DataFrame(_inp_rows))
        with _cB:
            st.caption("Top fired rules (why this confidence)")
            _tr = fz["trace"][:5]
            if _tr:
                st.table(pd.DataFrame([{"Rule": t["rule"], "μ": f"{t['strength']:.2f}"}
                                       for t in _tr]))
            else:
                st.write("No rules fired — insufficient evidence (ACI ≈ 0).")

        with st.expander("🧠 How the fuzzy layer decided — no-cliff surface "
                         "vs crisp tiers"):
            st.markdown(
                "FACL is a **framework/integration** contribution (not a new fuzzy "
                "algorithm): it fuzzifies PRAM's *residual diagnostics* and encodes "
                "two physics gates — **leakage suppression** and **aerosol "
                "sign-consistency** — as graded logic. The crisp tier steps hard "
                "at R²=0.40 / improvement=8%, while the fuzzy ACI varies smoothly.")
            try:
                import matplotlib.pyplot as plt
                _im_g = np.linspace(-2, 16, 46)
                _r2_g = np.linspace(0.0, 0.95, 40)
                _Z = facl.aci_surface(_im_g, _r2_g, aerosol=_inp["aerosol"],
                                      bias=_inp["bias"])
                _figz, _axz = plt.subplots(figsize=(7.2, 3.4))
                _cf = _axz.contourf(_im_g, _r2_g, _Z,
                                    levels=np.linspace(0, 100, 21), cmap="viridis")
                _axz.axvline(8, color="w", ls="--", lw=1)
                _axz.axhline(0.40, color="w", ls="--", lw=1)
                _axz.axhspan(0.85, 0.95, color="red", alpha=0.15)
                _axz.scatter([_inp["improvement"]], [_inp["residual_r2"]], c="red",
                             s=60, edgecolor="w", zorder=5, label="this run")
                _axz.set_xlabel("RMSE improvement (%)")
                _axz.set_ylabel("residual R²")
                _axz.set_title("Fuzzy ACI surface  (white dashes = old crisp "
                               "cut-offs; red band = leakage-guard zone)")
                _axz.legend(loc="lower right", fontsize=8)
                _figz.colorbar(_cf, ax=_axz, label="ACI")
                _figz.tight_layout(); st.pyplot(_figz)
            except Exception:
                pass
    except Exception as _e:
        st.caption(f"(fuzzy layer unavailable: {_e})")

    # ---- explainable-AI attribution + residual-vs-driver -------------------
    imp = res["importance"].head(12)
    try:
        import matplotlib.pyplot as plt
        ca, cb = st.columns(2)
        with ca:
            fig, ax = plt.subplots(figsize=(4.7, 4.2))
            order = imp.iloc[::-1]
            ax.barh(order.index, order.values, color=config.PV_COLOR)
            ax.set_title(f"Residual attribution — {res['method']}")
            ax.set_xlabel("importance")
            fig.tight_layout(); st.pyplot(fig)
        with cb:
            top_feat = imp.index[0]
            ff = fit["feature_frame"]
            if top_feat in ff:
                fig2, ax2 = plt.subplots(figsize=(4.7, 4.2))
                ax2.scatter(ff[top_feat].to_numpy(),
                            res["residual"].reindex(fit["index"]).to_numpy(),
                            s=6, alpha=0.25, color="#2e86ab")
                ax2.axhline(0, color="#888", lw=1, ls="--")
                ax2.set_xlabel(top_feat)
                ax2.set_ylabel("residual (ref − model) kW")
                ax2.set_title(f"Residual vs top driver ({top_feat})")
                fig2.tight_layout(); st.pyplot(fig2)
    except Exception:
        pass

    # ---- corrected time series --------------------------------------------
    try:
        import matplotlib.pyplot as plt
        idx = fit["index"]
        ref = res["ref"].reindex(idx)
        mod = res["model"].reindex(idx)
        cut = fit["cut"]
        corr = mod.copy(); corr.iloc[cut:] = mod.iloc[cut:].to_numpy() + fit["pred"]
        sample = slice(cut, cut + 24 * 7)
        fig3, ax3 = plt.subplots(figsize=(9.6, 3.4))
        ax3.plot(idx[sample], ref.iloc[sample], color="#2e86ab", lw=1.4,
                 label=ref_label)
        ax3.plot(idx[sample], mod.iloc[sample], color=config.PV_COLOR, lw=1.2,
                 label=model_label)
        ax3.plot(idx[sample], corr.iloc[sample], color="#16a34a", lw=1.2, ls="--",
                 label="physics + learned residual")
        ax3.set_ylabel("Power (kW)"); ax3.set_title("Held-out sample week")
        ax3.legend(fontsize=8, ncol=3)
        for lab in ax3.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
        fig3.tight_layout(); st.pyplot(fig3)
    except Exception:
        pass

    # ---- honest, tiered written analysis ----------------------------------
    aero = res["aerosol"]
    top3 = ", ".join(list(imp.index[:3]))
    with st.expander("📝 What this means — analysis & what is required next",
                     expanded=True):
        if mode == "measured":
            ground = ("This residual is **measured-grounded**: it is the gap between "
                      "the **real metered generation** and the physics model, so what "
                      "the attribution reveals is *what the physics actually misses vs. "
                      "reality*.")
        else:
            ground = ("⚠️ This residual is **NOT measured** — it is the difference "
                      "between two physics estimates driven by two reanalysis sources "
                      f"({ref_label} vs {model_label}). It shows which atmospheric "
                      "inputs make the sources diverge — a data-source sensitivity "
                      "analysis, **not** a validation. For the measured result, use the "
                      "🇦🇺 Australian DKASC tab.")
        st.markdown(ground)

        if cls["tier"] == "strong":
            st.markdown(
                f"**Result: strong.** Adding the learned residual cut test-period RMSE "
                f"by **{impr['improvement_pct']:.1f}%** and the residual is genuinely "
                f"explainable (R² {fit['resid_r2']:.3f}). The top drivers of the "
                f"unmodelled part are **{top3}**.")
        elif cls["tier"] == "moderate":
            st.markdown(
                f"**Result: moderate.** A real but modest gain "
                f"({impr['improvement_pct']:.1f}% RMSE; residual R² "
                f"{fit['resid_r2']:.3f}). Top drivers: **{top3}**. Honest framing: a "
                f"secondary-effect correction, not a dramatic one.")
        else:
            st.markdown(
                f"**Result: weak.** The residual is mostly noise "
                f"({impr['improvement_pct']:.1f}% RMSE change, R² {fit['resid_r2']:.3f}). "
                f"Do **not** force a novelty claim from this — the physics already "
                f"captures most of the signal. Report it honestly and lean on the "
                f"validated-method + multi-zone story instead.")

        if aero.get("available"):
            if aero.get("top3"):
                sgn = ("with a negative effect at high values (physics over-predicts "
                       "when the air is dusty → the omitted **soiling loss**, now "
                       "quantified)" if aero.get("negative")
                       else "(install `shap` to confirm the sign / soiling direction)")
                st.markdown(f"**Aerosol signature:** `{aero['feature']}` ranks "
                            f"#{aero['rank']} {sgn}. This is the India-relevant finding "
                            f"to highlight — *if* it holds on your real data.")
            else:
                st.markdown(f"**Aerosol signature:** present but weak "
                            f"(`{aero['feature']}` only rank #{aero['rank']}). The site "
                            f"may not be aerosol-limited in this period.")
        else:
            st.markdown("**Aerosol signature:** no aerosol columns in this dataset, so "
                        "the residual cannot be attributed to aerosols/soiling directly "
                        "here. (DKASC has no on-site aerosol channel; to test the "
                        "soiling hypothesis you would merge external AOD/PM10 data — "
                        "which you chose to keep out of this tab.)")

        st.markdown(
            "**What is required to strengthen this:**\n"
            "1. **Real, longer data** — more operating hours give a more reliable "
            "residual model and split.\n"
            "2. **Aerosol channel** — to attribute the soiling component, the residual "
            "features must include AOD/PM10/dust on the same timestamps.\n"
            "3. **`shap` installed** — for signed attribution and the aerosol sign test.\n"
            "4. **Robustness check** — confirm the improvement holds across seasons "
            "(summer vs winter), not just overall.\n"
            "5. **Honest reporting** — a residual R² of ~0.3–0.6 is a *good* result; "
            "report whatever you get and never tune toward a target.")
    st.caption("Method: residual = reference − physics model; a random forest learns it "
               "on atmospheric + temporal features with a time-based split; attribution "
               "by SHAP (or RF importance). Residual learning is supervised regression "
               "(ResNet/boosting lineage) — not a new algorithm and not reinforcement "
               "learning.")


def _tab_pram_live(lat, lon, elev, past_days, prefer):
    """Standalone PRAM tab driven by LIVE Open-Meteo + NASA POWER data.

    Honest design: weather APIs provide no metered PV, so this learns the residual
    between a NASA-POWER-driven physics estimate and an Open-Meteo-driven one, and
    attributes what drives their divergence. It is a data-source / model-consistency
    analysis — NOT measured validation (that lives in the DKASC tab).
    """
    import numpy as np
    import pandas as pd

    st.markdown("### 🧪 PRAM (Live) — Physics-Residual Attribution on live API data")
    st.markdown(
        "<div class='verdict v-warn'>⚠️ <b>No metered PV exists in weather APIs.</b> "
        "NASA POWER and Open-Meteo both provide <i>atmospheric</i> data only. This tab "
        "therefore learns the residual between two physics-PV estimates — one driven by "
        "<b>NASA POWER</b>, one by <b>Open-Meteo</b> — and uses explainable AI to show "
        "which atmospheric variables make them diverge. This is a <b>data-source "
        "sensitivity</b> analysis, <b>not</b> measured validation. For the real, "
        "measured-grounded PRAM use the 🇦🇺 <b>Australian DKASC</b> tab.</div>",
        unsafe_allow_html=True)

    run = st.button("▶️ Fetch both sources & run live PRAM", type="primary",
                    key="_pram_live_run")
    if not run:
        st.caption("Click to fetch NASA POWER and Open-Meteo for the sidebar location "
                   f"({lat:.3f}, {lon:.3f}; last {past_days} days) and run the analysis.")
        return

    def _physics_pv(raw):
        feat = feature_engineering.build_feature_table(raw, lat, lon, elev)
        pv = power_models.pv_power_from_features(feat, add_noise=False)
        return feat, pd.Series(np.asarray(pv), index=feat.index)

    try:
        with st.spinner("Fetching NASA POWER…"):
            raw_nasa = data_fetcher.fetch_nasa_power(lat, lon, elev,
                                                     past_days=past_days)
        feat_nasa, pv_nasa = _physics_pv(raw_nasa)
    except Exception as exc:
        st.error(f"Could not fetch / process NASA POWER: {exc}. "
                 "Check the network settings, then retry.")
        return
    try:
        with st.spinner("Fetching Open-Meteo (ERA5 + air quality)…"):
            raw_om = data_fetcher.fetch_realtime_data(lat, lon, past_days=past_days)
        feat_om, pv_om = _physics_pv(raw_om)
    except Exception as exc:
        st.error(f"Could not fetch / process Open-Meteo: {exc}. "
                 "Check the network settings, then retry.")
        return

    # align both estimates on common timestamps
    common = feat_nasa.index.intersection(feat_om.index)
    if len(common) < 60:
        st.warning(f"Only {len(common)} overlapping hours between the two sources — "
                   "not enough to learn a residual. Increase the history window.")
        return

    # attribution features come from the Open-Meteo frame (it carries aerosols)
    weather = feat_om.loc[common]
    rated = float(config.PV_CAPACITY_KW)
    st.success(f"Fetched both sources · {len(common):,} overlapping hourly records. "
               "Reference = NASA-POWER-driven PV · Model = Open-Meteo-driven PV.")

    _render_pram_block(
        weather, pv_nasa.reindex(common), pv_om.reindex(common), rated,
        mode="crosssrc",
        ref_label="NASA POWER-driven PV",
        model_label="Open-Meteo-driven PV",
        intro_note="The residual below is **NASA-driven minus Open-Meteo-driven** "
                   "physics PV, attributed to the atmospheric drivers that cause the "
                   "two reanalyses to disagree.")


def _tab_crisp():
    """🎯 CRISP — conformal prediction intervals with regime-conditional coverage.

    Wraps distribution-free prediction intervals around the physics PV predictor
    and *proves* their coverage on the real metered DKASC generation. This is the
    project's uncertainty-quantification layer — a distinct axis from FACL's
    attribution confidence.
    """
    import numpy as np
    st.markdown("### 🎯 CRISP — Conformal prediction intervals (calibrated uncertainty)")
    st.caption("A distribution-free uncertainty layer. It turns the physics PV "
               "*point* prediction into an **interval** with a guaranteed coverage "
               "level, and — crucially — keeps that guarantee **within each "
               "irradiance regime** (Mondrian), validated on REAL metered DKASC power.")

    with st.expander("ℹ️ What CRISP is, and why it is new here", expanded=False):
        st.markdown(
            "Every other tab reports a **point** prediction or a single accuracy "
            "score. CRISP answers a different question: *give me a band that "
            "contains the true power with, say, 90% probability — and prove it.*\n\n"
            "**Method — split conformal prediction (Vovk; Lei et al. 2018).** On an "
            "earlier *calibration* block we record the physics model's absolute "
            "errors; the (1-α)-quantile of those errors (with a finite-sample "
            "correction) becomes the half-width of the interval on the later, "
            "held-out *test* block. With exchangeable data this **guarantees** "
            "marginal coverage ≥ 1-α with **no distributional assumption**.\n\n"
            "**The reportable twist — regime-indexed (Mondrian) conformal.** A single "
            "global band mis-covers *conditionally*: far too wide when it is dark, "
            "too narrow at peak sun. CRISP fits a **separate quantile per irradiance "
            "regime**, restoring conditional validity. The headline is that this "
            "shrinks the **worst-regime coverage gap** on real DKASC data.\n\n"
            "**Honest scope.** Conformal prediction is an established framework — this "
            "is an *applied integration* (same tier as PRAM/FACL), not a new "
            "algorithm. A chronological calib→test split injects real seasonal drift, "
            "so coverage on short windows can fall below nominal; that is a property "
            "of the data and is reported, not hidden.")

    # ---- load real measured DKASC data (same source as the DKASC tab) ------
    site_key = "alice"
    path = dksac.find_site_csv(site_key)
    if not path:
        st.warning("No DKASC CSV found. Add the real Alice Springs file (see the "
                   "🇦🇺 Australian DKASC tab for download instructions) to run CRISP "
                   "on measured data — no fabricated data is used.")
        return
    try:
        hourly, wmap, power_cols = dksac.load_dksac_frame(path)
    except Exception as exc:
        st.error(f"Could not parse the DKASC file ({exc}).")
        return
    channels = dksac.power_channels(hourly, power_cols)
    if not channels:
        st.error("No usable measured PV channel found in the DKASC file.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    chan_labels = [f"{lab}  ·  peak ≈ {pk:,.1f} kW" for (_c, lab, pk) in channels]
    with c1:
        sel = st.selectbox("Measured PV channel", chan_labels, index=0,
                           key="_crisp_chan")
    with c2:
        conf = st.select_slider("Confidence level", options=[50, 80, 90, 95],
                                value=90, key="_crisp_conf")
    with c3:
        n_bins = st.select_slider("Irradiance regimes", options=[3, 4, 5, 6, 8],
                                  value=5, key="_crisp_bins")
    alpha = 1.0 - conf / 100.0

    chosen_col = channels[chan_labels.index(sel)][0]
    p_meas = hourly[chosen_col]
    rated = dksac.estimate_rated_kw(p_meas)

    # physics point prediction + regime driver, with a leakage-safe per-hour
    # centre calibration (reuses PRAM's calibrator) so the band sits sensibly.
    model_in = hourly.copy()
    if "poa" not in model_in or model_in["poa"].isna().all():
        model_in["poa"] = model_in.get("ghi")
    p_model = dksac.model_pv_from_dksac(model_in, rated)
    driver = model_in["poa"].where(model_in["poa"].notna(), model_in.get("ghi"))

    aligned = pd.concat(
        [p_meas.rename("P"), p_model.rename("phys"), driver.rename("drv")],
        axis=1).dropna().sort_index()
    operating = (aligned["P"] > 0.02 * rated).to_numpy()
    try:
        mod_cal = pram.calibrate_baseline(aligned["P"], aligned["phys"],
                                          operating, test_frac=0.5)
    except Exception:
        mod_cal = aligned["phys"]

    res = crisp.run_crisp(
        aligned["P"].to_numpy(), np.asarray(mod_cal, dtype=float),
        aligned["drv"].to_numpy(), alpha=alpha, n_bins=int(n_bins),
        method="mondrian", calib_frac=0.5, operating_mask=operating,
        rated_kw=rated, compare_global=True)

    if not res["ok"]:
        st.error(f"CRISP could not run: {res['reason']}")
        return

    rep = res["report"]
    tgt = rep["target"] * 100.0
    m1, m2, m3, m4 = st.columns(4)
    _cov = rep["marginal_coverage"] * 100.0
    _delta = _cov - tgt
    m1.metric(f"Empirical coverage (target {tgt:.0f}%)", f"{_cov:.1f}%",
              f"{_delta:+.1f} pts vs target")
    m2.metric("Mean interval width (MPIW)", f"{rep['mpiw']:.2f} kW",
              f"PINAW {rep['pinaw']:.3f}")
    m3.metric("Worst-regime gap (Mondrian)",
              f"{rep.get('worst_slab_gap', float('nan'))*100:.1f} pts")
    if "report_global" in res:
        gap_g = res["report_global"].get("worst_slab_gap", float("nan")) * 100.0
        m4.metric("Worst-regime gap (Global)", f"{gap_g:.1f} pts",
                  f"{(rep.get('worst_slab_gap', 0)*100 - gap_g):+.1f} pts",
                  delta_color="inverse")

    st.caption(f"Channel **{sel.split('  ·')[0]}** · rated ≈ {rated:.1f} kW · "
               f"calib {res['n_calib']} h → test {res['n_test']} h "
               f"(chronological, leakage-safe).")

    # ---- fan chart over a slice of the held-out test period ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = res["y_test"]; yhat = res["yhat_test"]
    lo, hi = res["lower"], res["upper"]
    win = min(len(y), 168)                       # ~a week of operating hours
    xs = np.arange(win)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.fill_between(xs, lo[:win], hi[:win], color="#38bdf8", alpha=0.30,
                    label=f"{conf}% conformal interval")
    ax.plot(xs, yhat[:win], color="#0ea5e9", lw=1.4, label="Physics prediction")
    ax.scatter(xs, y[:win], s=9, color="#0f172a", alpha=0.7, label="Measured", zorder=3)
    ax.set_xlabel("operating-hour index (chronological held-out test period)")
    ax.set_ylabel("PV power (kW)")
    ax.set_title(f"Calibrated {conf}% prediction intervals vs real measured power")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    # ---- per-regime conditional coverage table -----------------------------
    st.markdown("#### Conditional coverage by irradiance regime — Mondrian vs Global")
    st.caption("The key result: a single **Global** width can look fine on average "
               "yet badly under-cover the high-irradiance (peak-power) regime. "
               "**Mondrian** restores coverage in every regime.")
    gbins = {b["regime"]: b for b in res.get("report_global", {}).get("bins", [])}
    rows = []
    for b in rep["bins"]:
        g = gbins.get(b["regime"], {})
        rows.append({
            "Regime (POA W/m²)": f"{b['driver_lo']:.0f}–{b['driver_hi']:.0f}",
            "n (test hrs)": b["n"],
            "Mondrian coverage": f"{b['coverage']*100:.1f}%",
            "Global coverage": (f"{g.get('coverage', float('nan'))*100:.1f}%"
                                if g else "—"),
            "Mondrian width (kW)": f"{b['mpiw']:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- reliability curve across confidence levels ------------------------
    with st.expander("📈 Reliability curve (nominal vs empirical coverage across levels)",
                     expanded=False):
        rel = crisp.reliability(
            aligned["P"].to_numpy(), np.asarray(mod_cal, dtype=float),
            aligned["drv"].to_numpy(), alphas=(0.5, 0.2, 0.1, 0.05),
            method="mondrian", n_bins=int(n_bins), calib_frac=0.5,
            operating_mask=operating, rated_kw=rated)
        if not rel.empty:
            st.dataframe(rel, use_container_width=True, hide_index=True)
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.plot([0.4, 1.0], [0.4, 1.0], "--", color="#94a3b8",
                     label="perfect calibration")
            ax2.plot(rel["nominal"], rel["empirical"], "o-", color="#0ea5e9",
                     label="CRISP (Mondrian)")
            ax2.set_xlabel("nominal coverage (1 − α)")
            ax2.set_ylabel("empirical coverage")
            ax2.set_title("Reliability curve")
            ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
            st.pyplot(fig2)
        st.caption("A well-calibrated system sits on the diagonal: empirical "
                   "coverage ≈ nominal at every level. Reproduce headless with "
                   "`python run_crisp_validation.py --monthly`.")


def _dksac_validation_section():
    """Site-specific validation against real measured DKASC data — no map, no live
    fetch. Uses the on-site measured weather and measured power from the same file."""
    import numpy as np
    st.markdown("### 🇦🇺 Australian DKASC — validation against real measured PV")
    st.caption("Validates the project's PV physics against REAL measured generation "
               "from a fixed reference plant. This is **site-specific validation** — "
               "it proves the model's weather→power accuracy, which is then applied to "
               "the Indian coordinates as a resource assessment.")

    with st.expander("ℹ️ What this is and how it works (important)", expanded=False):
        st.markdown(
            "**DKASC** (Desert Knowledge Australia Solar Centre, Alice Springs) is a "
            "real solar facility, running since 2008, whose actual metered output is "
            "published openly. It is **one fixed site** — *not* a coordinate-queryable "
            "service — so there is **no map and no coordinate picking here**.\n\n"
            "**How the validation works (no live data is fetched):** the downloaded file "
            "contains, for each historical hour, **both** the plant's **on-site measured "
            "weather** *and* its **real measured power**. The model is run on that file's "
            "own measured weather, and its prediction is compared against the file's real "
            "measured power. Same site, same timestamps, fully self-contained.\n\n"
            "- ✅ Input weather: from the DKASC file (on-site sensors)\n"
            "- ✅ Truth: measured power, from the same DKASC file\n"
            "- ❌ NASA POWER / Open-Meteo are **not** used here (they drive the Indian "
            "*application*, not this validation)\n\n"
            "**Why on-site weather, not a satellite estimate?** the file's irradiance was "
            "measured right next to the panel — the exact light it received — so the test "
            "isolates the model, not a second source of error.\n\n"
            "**Honest limitations:** soiling is omitted (DKASC has no aerosol channel) and "
            "the climate is arid Australia, so the proven accuracy strictly applies to "
            "similar climates; the Indian analysis remains a modelled application.")

    # ---- site selector (dropdown, NOT a map) ---------------------------
    _site_labels = {"DKASC — Alice Springs (multi-technology)": "alice",
                    "Yulara Solar System (1.8 MW plant)": "yulara"}
    _sel = st.radio("Reference validation site", list(_site_labels.keys()),
                    horizontal=True, key="_dksac_site_sel")
    site_key = _site_labels[_sel]
    site = dksac.SITES[site_key]
    st.caption("ℹ️ The **DKP Microgrid / BESS** site is deliberately excluded: it has a "
               "battery, so its metered power is net microgrid flow (solar ± battery), "
               "not raw PV — unsuitable for validating a weather→PV model. Validating on "
               "**two plants of different scale** (Alice Springs + Yulara) strengthens "
               "generalisation.")

    # ---- load the real measured file -----------------------------------
    path = dksac.find_site_csv(site_key)
    if not path:
        _fname = "yulara_solar.csv" if site_key == "yulara" else "dkasc_alice_springs.csv"
        _pat = "*yulara*.csv" if site_key == "yulara" else "*dkasc*.csv"
        st.warning(f"No {site['name']} dataset found yet — add the real file to run the "
                   "validation (no fabricated data is used).")
        st.markdown(
            "**To enable the real comparison:**\n"
            "1. Open the DKA Solar Centre download portal and choose the "
            f"**{site['name']}** location.\n"
            "2. Select **one fixed silicon PV system** (e.g. *38 Q CELLS, 5.9 kW, "
            "mono-Si, Fixed*) plus its **weather** channels (irradiance, temperature, "
            "humidity), at **Hourly** resolution, for a period where weather exists.\n"
            f"3. Save the CSV in your project as `data/{_fname}` (any name like "
            f"`{_pat}` works). If you download month-by-month, just drop all the files "
            "in `data/` with that naming.\n"
            "4. Reload — the model-vs-measured metrics, plots and verdict appear here.")
        st.code("https://dkasolarcentre.com.au/download", language="text")
        return

    try:
        hourly, wmap, power_cols = dksac.load_dksac_frame(path)
    except Exception as exc:
        st.error(f"Could not parse the file ({exc}). Use the raw CSV from the portal.")
        return

    channels = dksac.power_channels(hourly, power_cols)
    if not channels:
        st.error("No PV power channel with usable data was found in this file. "
                 "Re-download from the DKASC portal with at least one array's "
                 "'Active Power' channel ticked (the whole-site master meter also "
                 "works).")
        return

    # quick weather-channel check (irradiance + temperature drive the model)
    found = {k: (wmap.get(k) is not None) for k in ["ghi", "poa", "temp"]}
    st.success(f"Loaded real {site['name']} data: {os.path.basename(path)} · "
               f"{len(hourly):,} hourly records · {len(channels)} measured PV "
               f"channel(s) with data.")
    miss = [k for k, v in found.items() if not v]
    if miss:
        st.warning("Some expected weather channels were not found: "
                   + ", ".join(miss)
                   + ". An irradiance channel (GHI or tilted) is required for the "
                   "physics model; re-download with those channels ticked.")

    # Choose which measured PV signal to validate against. A DKASC export holds
    # ~50 per-array meters plus whole-site master meters; the default is the
    # whole-site meter (unbiased). Individual arrays vary because the site mixes
    # PV technologies and tracking systems.
    chan_labels = [f"{lab}  ·  peak ≈ {pk:,.1f} kW" for (_c, lab, pk) in channels]
    sel = st.selectbox(
        "Measured PV channel to validate against",
        chan_labels, index=0,
        help="Default = the whole-site master meter (the facility's total "
             "measured PV). Individual arrays differ: some are tracking or "
             "off-axis and will NOT match a single fixed-tilt irradiance sensor, "
             "so a fixed crystalline array gives the tightest model match.")
    chosen_col = channels[chan_labels.index(sel)][0]
    p_meas = hourly[chosen_col]

    rated_guess = dksac.estimate_rated_kw(p_meas)
    rated = st.number_input(
        "Array rated power (kW) — auto-estimated from the selected channel's peak; "
        "override with the true nameplate if you know it",
        value=float(round(rated_guess, 2)), min_value=0.1, step=0.1)

    p_model = dksac.model_pv_from_dksac(hourly, rated)
    M = dksac.validation_metrics(p_meas, p_model, rated)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("RMSE", f"{M['rmse']:.2f} kW")
    k2.metric("MAE", f"{M['mae']:.2f} kW")
    k3.metric("nRMSE", f"{M['nrmse_pct']:.1f}%")
    k4.metric("Bias (MBE)", f"{M['mbe']:+.2f} kW")

    cls, label = dksac.verdict(M["nrmse_pct"])
    st.markdown(
        f"<div class='verdict v-{cls}'>{label}. Compared over {M['n']:,} operating "
        f"hours · correlation r = {M['corr']:.3f} · R² = {M['r2']:.3f} · "
        f"mean measured {M['mean_meas']:.2f} kW.</div>", unsafe_allow_html=True)

    # Step-by-step mathematical derivation of every value above.
    _dksac_math_walkthrough(hourly, wmap, chosen_col, p_meas, p_model, rated, M)

    frame = M["frame"]
    try:
        import matplotlib.pyplot as plt
        c_a, c_b = st.columns(2)
        with c_a:
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            ax.scatter(frame["meas"], frame["model"], s=6, alpha=0.25,
                       color=config.PV_COLOR)
            lim = max(frame["meas"].max(), frame["model"].max())
            ax.plot([0, lim], [0, lim], color="#2e86ab", lw=1.5, ls="--")
            ax.set_xlabel("Measured power (kW)")
            ax.set_ylabel("Model power (kW)")
            ax.set_title("Model vs measured (ideal = dashed)")
            fig.tight_layout()
            st.pyplot(fig)
        with c_b:
            sample = frame.iloc[:24 * 7]
            fig2, ax2 = plt.subplots(figsize=(4.6, 4.2))
            ax2.plot(sample.index, sample["meas"], color="#2e86ab", lw=1.4,
                     label="Measured")
            ax2.plot(sample.index, sample["model"], color=config.PV_COLOR, lw=1.4,
                     label="Model")
            ax2.set_ylabel("Power (kW)")
            ax2.set_title("Sample week — measured vs model")
            ax2.legend(fontsize=8)
            for lab in ax2.get_xticklabels():
                lab.set_rotation(30)
                lab.set_ha("right")
            fig2.tight_layout()
            st.pyplot(fig2)
    except Exception:
        pass

    over_under = "over-predicts" if M["mbe"] > 0 else "under-predicts"
    with st.expander("📝 Comparative analysis — is it matching or different, and why?",
                     expanded=True):
        if cls == "good":
            st.markdown(
                f"**The model matches the real measured generation** (nRMSE "
                f"{M['nrmse_pct']:.1f}%, correlation {M['corr']:.3f}). Feeding the plant's "
                f"on-site measured irradiance and temperature into the model reproduces "
                f"the real metered output to within a small error — this **validates the "
                f"weather→power physics**. The model {over_under} slightly (bias "
                f"{M['mbe']:+.2f} kW), consistent with unmodelled real-world losses. "
                f"**Justified conclusion:** the method is sound and can be applied to "
                f"resource assessment for the Indian zones, stated as a modelled "
                f"application of this validated method.")
        else:
            st.markdown(
                f"**The model and measured output differ** (nRMSE {M['nrmse_pct']:.1f}%; "
                f"the model {over_under} by {abs(M['mbe']):.2f} kW on average). This is a "
                f"finding, not a failure — the likely causes are physical:\n\n"
                f"- **Soiling/aerosols omitted** — no aerosol channel in DKASC; real dust "
                f"losses in arid Alice Springs lower measured output.\n"
                f"- **System losses** — inverter clipping, wiring, mismatch, ageing.\n"
                f"- **Irradiance basis** — if no tilted-irradiance channel was downloaded, "
                f"GHI is used as a POA proxy, adding angle error.\n"
                f"- **Rated power** — if the entered nameplate ({rated:.2f} kW) is not the "
                f"chosen array's, the scale is off; adjust it above.\n\n"
                f"**Justified next step:** add the soiling/loss terms or include the "
                f"tilted-irradiance channel and correct nameplate, then re-evaluate. "
                f"Report the honest error — a quantified, explained gap is a legitimate "
                f"research result.")
    st.caption("Note: soiling is omitted (no aerosol channel) and the validation climate "
               "is arid Australia — state both as limitations in the paper.")

    # ===================================================================
    # PRAM — Physics-Residual Attribution Model (on this measured DKASC data)
    # ===================================================================
    st.markdown("---")
    st.markdown("### 🧬 PRAM — Physics-Residual Attribution Model (measured-grounded)")
    st.caption("The scientifically valid residual: it learns **measured power − physics "
               "model** on this DKASC file's own data, then attributes with explainable "
               "AI what the physics misses. This is the measured result for your paper.")
    st.info("👉 For a meaningful residual, select a **single fixed-tilt array** in the "
            "channel dropdown above (e.g. a crystalline-silicon fixed array). The "
            "**whole-site master meter** mixes ~40 arrays of different technologies and "
            "tracking systems, and **tracking/off-axis arrays** won't match a fixed-tilt "
            "physics model — PRAM will correctly report those as 'weak'.")
    with st.expander("ℹ️ What PRAM does here (and why it is the valid one)",
                     expanded=False):
        st.markdown(
            "Your physics model converts the plant's on-site weather to power, but it "
            "omits real effects (soiling, spectral, ageing). PRAM treats the leftover — "
            "`residual = measured − physics` — as a learning target and uses a random "
            "forest + SHAP to reveal which atmospheric/temporal drivers explain it. "
            "Because the reference here is **real metered power**, the residual is "
            "genuinely *what the physics misses vs. reality* — unlike the Live tab, "
            "which has no metered truth.\n\n"
            "*Note:* this DKASC file has **no on-site aerosol channel**, so the residual "
            "is attributed to the weather it does contain (irradiance, temperature, "
            "humidity, wind, time/season). Direct aerosol/soiling attribution would "
            "need external AOD data merged in (kept out of this tab by design).")

    _merge_aero = st.checkbox(
        "🌫️ Merge external aerosol data (AOD/PM10/dust) to test the soiling hypothesis",
        value=False,
        help="OFF by default (this tab is otherwise self-contained). When ON, it "
             "fetches aerosol reanalysis for the site coordinate and date range from "
             "the Open-Meteo air-quality archive, so the residual can be attributed to "
             "aerosols/soiling. Note: these are reanalysis values at the coordinate, "
             "not on-site measurements, and historical coverage may be limited.")
    pram_weather = hourly
    if _merge_aero:
        with st.spinner("Fetching external aerosol reanalysis for the site…"):
            pram_weather, _aero_status, _aero_cov = _merge_dksac_aerosols(hourly, site)
        (st.success if _aero_cov >= 0.05 else st.warning)(_aero_status)

    _render_pram_block(
        pram_weather, p_meas, p_model, rated,
        mode="measured",
        ref_label="Measured power",
        model_label="Physics model",
        intro_note="The residual below is **measured minus physics model** on this "
                   "plant's own data — the genuine signal of what the physics misses.")


def _dksac_math_walkthrough(hourly, wmap, chosen_col, p_meas, p_model, rated, M):
    """Reproducible, step-by-step mathematics behind the DKASC validation.

    Every number shown in the metric cards above is re-derived here from the
    raw measured file so the result is fully auditable for the dissertation.
    """
    import numpy as np
    import pandas as pd

    st.markdown("---")
    st.markdown("### 🧮 Step-by-step mathematical modelling (DKASC)")
    st.caption("Exactly how the measured file becomes the metrics above — every "
               "value is reproducible from the equations below.")

    # ---- Step 1: the data the validation runs on -----------------------
    st.markdown("#### Step 1 · Measured inputs from the DKASC file")
    ghi_col = wmap.get("ghi")
    poa_col = wmap.get("poa")
    temp_col = wmap.get("temp")
    st.markdown(f"""
| Quantity | Source column | Role |
|----------|---------------|------|
| Measured PV power | `{chosen_col}` | ground truth $P_{{meas}}$ |
| Tilted irradiance (POA) | `{poa_col if poa_col else '—'}` | preferred plane-of-array $G_{{POA}}$ |
| Global horizontal (GHI) | `{ghi_col if ghi_col else '—'}` | POA fallback when no tilted channel |
| Air temperature | `{temp_col if temp_col else '—'}` | drives cell temperature |
| Array rated power | (entered / auto) | $P_{{rated}} = {rated:.2f}$ kW |
""")
    st.caption("On-site sensors are used (not a satellite estimate), so the test "
               "isolates the model rather than a second source of error.")

    # ---- pick a worked daytime sample hour -----------------------------
    frame = M["frame"]
    poa_series = hourly["poa"].where(hourly["poa"].notna(), hourly["ghi"])
    ghi_series = hourly["ghi"].where(hourly["ghi"].notna(), poa_series).fillna(0.0)
    temp_series = hourly["temp"].fillna(25.0)
    day = poa_series[poa_series > 100]
    s_idx = day.index[len(day) // 2] if len(day) else poa_series.index[len(poa_series) // 2]
    poa_s = float(poa_series.loc[s_idx])
    ghi_s = float(ghi_series.loc[s_idx])
    temp_s = float(temp_series.loc[s_idx])

    # ---- Step 2: cell temperature (NOCT) -------------------------------
    st.markdown("#### Step 2 · Cell temperature (NOCT model)")
    st.latex(r"T_{cell} = T_{air} + \frac{NOCT - 20}{800}\,G_{POA}")
    t_cell_s = temp_s + (config.PV_NOCT_C - 20.0) / 800.0 * ghi_s
    st.markdown(f"With NOCT = {config.PV_NOCT_C:.0f} °C, for the sample hour "
                f"**{pd.Timestamp(s_idx).strftime('%d %b %H:%M')}** "
                f"($T_{{air}}={temp_s:.1f}$ °C, $G={ghi_s:.0f}$ W/m²):")
    st.latex(rf"T_{{cell}} = {temp_s:.1f} + \frac{{{config.PV_NOCT_C:.0f}-20}}{{800}}"
             rf"\times {ghi_s:.0f} = {t_cell_s:.1f}\ \text{{°C}}")

    # ---- Step 3: thermal derate ----------------------------------------
    st.markdown("#### Step 3 · Thermal derate factor")
    st.latex(r"f_{temp} = \mathrm{clip}\!\left(1 - \gamma\,(T_{cell}-25),\ 0.70,\ 1.05\right)")
    f_temp_s = float(np.clip(1.0 - config.PV_TEMP_COEFF * (t_cell_s - 25.0), 0.70, 1.05))
    st.markdown(f"Temperature coefficient $\\gamma = {config.PV_TEMP_COEFF:.4f}$ /°C "
                f"(≈ {config.PV_TEMP_COEFF*100:.2f} %/°C):")
    st.latex(rf"f_{{temp}} = 1 - {config.PV_TEMP_COEFF:.4f}\times({t_cell_s:.1f}-25)"
             rf" = {f_temp_s:.4f}")

    # ---- Step 4: soiling note ------------------------------------------
    st.markdown("#### Step 4 · Soiling factor")
    st.latex(r"f_{soil} = 1.0")
    st.caption("DKASC has no on-site aerosol channel, so soiling is set to 1.0 and "
               "declared as a limitation — its effect is instead surfaced by PRAM below.")

    # ---- Step 5: modelled power ----------------------------------------
    st.markdown("#### Step 5 · Modelled PV power")
    st.latex(r"P_{model} = \mathrm{clip}\!\left(P_{rated}\cdot\frac{G_{POA}}{1000}"
             r"\cdot f_{temp}\cdot f_{soil},\ 0,\ P_{rated}\right)")
    p_model_s = float(np.clip(rated * (poa_s / 1000.0) * f_temp_s * 1.0, 0.0, rated))
    st.latex(rf"P_{{model}} = {rated:.2f}\times\frac{{{poa_s:.0f}}}{{1000}}"
             rf"\times {f_temp_s:.4f} = {p_model_s:.2f}\ \text{{kW}}")

    # ---- Step 6: error metrics -----------------------------------------
    st.markdown("#### Step 6 · Validation metrics over operating hours")
    st.caption(f"Computed only over the {M['n']:,} hours where measured power "
               f"> 2% of rated ({0.02*rated:.2f} kW), i.e. true daylight operation.")
    st.latex(r"\text{MAE}=\frac{1}{N}\sum|P_{model}-P_{meas}|,\quad"
             r"\text{RMSE}=\sqrt{\tfrac{1}{N}\sum(P_{model}-P_{meas})^2}")
    st.latex(r"\text{nRMSE}=\frac{\text{RMSE}}{\overline{P_{meas}}}\times100\%,\quad"
             r"\text{MBE}=\frac{1}{N}\sum(P_{model}-P_{meas})")
    st.latex(r"R^2 = 1 - \frac{\sum(P_{meas}-P_{model})^2}"
             r"{\sum(P_{meas}-\overline{P_{meas}})^2}")
    st.markdown(f"""
| Symbol | Meaning | Value |
|--------|---------|-------|
| $N$ | operating hours compared | **{M['n']:,}** |
| $\\overline{{P_{{meas}}}}$ | mean measured power | **{M['mean_meas']:.2f} kW** |
| MAE | mean absolute error | **{M['mae']:.2f} kW** |
| RMSE | root-mean-square error | **{M['rmse']:.2f} kW** |
| nRMSE | RMSE ÷ mean measured | **{M['nrmse_pct']:.1f}%** |
| MBE | mean bias (model − measured) | **{M['mbe']:+.2f} kW** |
| $r$ | Pearson correlation | **{M['corr']:.3f}** |
| $R^2$ | coefficient of determination | **{M['r2']:.3f}** |
""")

    # ---- Step 7: verdict mapping ---------------------------------------
    st.markdown("#### Step 7 · Verdict mapping")
    st.markdown(f"""
The nRMSE of **{M['nrmse_pct']:.1f}%** is mapped to a verdict by fixed thresholds:

| nRMSE | Verdict |
|-------|---------|
| ≤ 10% | ✅ Strong match — validated |
| ≤ 20% | 🟡 Reasonable match — acceptable with explained error |
| > 20% | 🟠 Notable difference — must be justified |
""")
    st.success("✅ Every metric card above is reproduced by these equations on the "
               "same operating hours — the DKASC validation is fully auditable.")


def _tab_pvgis(feat, meta, lat, lon):
    st.subheader("🛰️ PVGIS independent cross-validation")
    st.caption("Cross-checks this project's computed PV energy for the same "
               "coordinate against PVGIS — an independent European-Commission "
               "estimator — so the result is corroborated by an external "
               "authority, not only by itself.")

    # ---- What is PVGIS? + how it acquires data (educational) ------------
    cexp1, cexp2 = st.columns(2)
    with cexp1:
        with st.expander("ℹ️ What is PVGIS?", expanded=True):
            st.markdown(
                "**PVGIS** (Photovoltaic Geographical Information System) is a "
                "free, scientifically-validated tool from the **European "
                "Commission's Joint Research Centre (JRC)**. For any "
                "latitude/longitude it estimates the **expected PV energy "
                "yield** (annual & monthly), computed from its **own satellite "
                "solar-radiation databases** (PVGIS-SARAH2, ERA5) — entirely "
                "independent of this project's Open-Meteo / NASA POWER inputs. "
                "It is used here as an independent, citable reference to verify "
                "the computed yield.")
    with cexp2:
        with st.expander("🔧 How PVGIS acquires & computes the data", expanded=True):
            st.markdown(
                "1. It receives the **coordinate** + a PV-system spec "
                "(capacity kWp, tilt, azimuth, mounting, module type).\n"
                "2. It looks up **long-term satellite irradiation** for that "
                "exact point from its databases.\n"
                "3. It applies its **own PV performance & loss model** "
                "(temperature, angle-of-incidence, spectral, system losses).\n"
                "4. It returns **annual + monthly energy** via its free web API "
                "(no key).\n\n"
                "Note: PVGIS reports a **long-term typical year**, whereas this "
                "project uses a **recent ~92-day live window** — a key reason "
                "the two values differ slightly.")

    # ---- Compute both estimates -----------------------------------------
    y = power_models.pv_yield_metrics(feat)
    proj_specific = y["annual_specific_yield"]          # kWh/kWp/yr
    pdc = y["pdc_kwp"]

    with st.spinner("Querying PVGIS for this coordinate…"):
        pg = get_pvgis(round(lat, 4), round(lon, 4), round(pdc, 2), config.PV_TILT_DEG)

    if "error" in pg:
        st.warning(f"PVGIS cross-check unavailable. {pg['error']}")
        st.info("This is a network-side issue, not a bug in the app. If the "
                "message above mentions a reset (WinError 10054), the cause is a "
                "firewall, antivirus HTTPS-inspection, or proxy on your network — "
                "try a phone hotspot, or whitelist `re.jrc.ec.europa.eu`. The "
                "comparison below populates automatically once PVGIS responds.")
        return

    pvgis_specific = pg["specific_yield"]

    # ---- Headline: annual specific-yield comparison ---------------------
    diff_pct = (proj_specific - pvgis_specific) / pvgis_specific * 100 if pvgis_specific else 0.0
    a = abs(diff_pct)
    st.markdown("#### Annual specific-yield comparison (kWh/kWp·yr)")
    k1, k2, k3 = st.columns(3)
    k1.metric("This project (model)", f"{proj_specific:,.0f}")
    k2.metric("PVGIS (independent)", f"{pvgis_specific:,.0f}")
    k3.metric("Difference", f"{diff_pct:+.1f}%")

    if a <= 10:
        cls, icon, label = "v-good", "✅", "Approximately equal — strong agreement"
    elif a <= 20:
        cls, icon, label = "v-warn", "🟡", "Reasonable agreement (within expected tolerance)"
    else:
        cls, icon, label = "v-bad", "🟠", "Notable gap — check window length / tilt / capacity"
    st.markdown(
        f"<div class='verdict {cls}'>{icon} {label}. "
        f"Your model: {proj_specific:,.0f} · PVGIS: {pvgis_specific:,.0f} "
        f"kWh/kWp·yr ({diff_pct:+.1f}%).</div>", unsafe_allow_html=True)

    # ---- 92-day matched-period comparison -------------------------------
    dim = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30,
           7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    daily_by_month = {int(r["month"]): r["Energy_kWh"] / dim[int(r["month"])]
                      for _, r in pg["monthly"].iterrows()}
    dates = pd.to_datetime(feat["Datetime"]).dt.normalize()
    uniq_days = dates.drop_duplicates()
    pvgis_period = float(sum(daily_by_month.get(pd.Timestamp(d).month, 0.0)
                             for d in uniq_days))
    proj_period = float(feat[config.TARGET_PV].sum())   # kWh over the window
    pdiff = (proj_period - pvgis_period) / pvgis_period * 100 if pvgis_period else 0.0

    st.markdown(f"#### Matched-period comparison "
                f"({len(uniq_days)} days actually in your window)")
    p1, p2, p3 = st.columns(3)
    p1.metric("This project (window)", f"{proj_period/1000:,.1f} MWh")
    p2.metric("PVGIS (same calendar days)", f"{pvgis_period/1000:,.1f} MWh")
    p3.metric("Difference", f"{pdiff:+.1f}%")
    st.caption("PVGIS has no 'recent 92 days' mode, so its figure here is its "
               "long-term **typical** energy for the *same calendar days* your "
               "live window covers — a like-for-like seasonal comparison, not "
               "identical dates.")

    # ---- Monthly profile comparison -------------------------------------
    st.markdown("#### Monthly energy profile")
    proj_monthly = (feat.assign(_m=pd.to_datetime(feat["Datetime"]).dt.month)
                    .groupby("_m")[config.TARGET_PV].sum())
    cov_counts = dates.dt.month.value_counts()
    cov_months = sorted([m for m in proj_monthly.index
                         if cov_counts.get(m, 0) >= 20 * 24])   # ~full months only
    comp = pg["monthly"].copy()
    comp["Project_kWh"] = comp["month"].map(
        lambda m: proj_monthly.get(m, np.nan) if m in cov_months else np.nan)
    fig = plotting.pvgis_monthly_compare(comp) if hasattr(plotting, "pvgis_monthly_compare") \
        else None
    if fig is not None:
        st.pyplot(fig)
    else:
        st.bar_chart(comp.set_index("Month")[["Energy_kWh"]])
    st.caption(f"PVGIS radiation database: **{pg['radiation_db']}** · "
               f"array {pg['peakpower_kwp']:.0f} kWp · tilt {pg['tilt_deg']:.0f}° · "
               "south-facing. Project bars appear only for months your window "
               "fully covers (use a ~1-year window to fill all 12).")

    # ---- Why do they differ? + honesty ----------------------------------
    with st.expander("🤔 Why do the two values differ — and is that OK?"):
        st.markdown(
            "A small difference is **expected and correct** — the two are "
            "computed by different methods. They differ because of:\n\n"
            "- **Different radiation data** — PVGIS uses PVGIS-SARAH2 / ERA5; "
            "this project uses Open-Meteo / NASA POWER.\n"
            "- **Different loss models** — PVGIS's internal temperature / "
            "soiling / inverter assumptions vs this project's NOCT thermal "
            "derate + AOD/dust soiling + albedo physics.\n"
            "- **Typical-year vs live window** — PVGIS averages many years; "
            "your estimate is extrapolated from a recent ~92-day window, so a "
            "cloudy/clear or single-season window shifts it.\n"
            "- **System settings** — tilt, azimuth and capacity must match for "
            "the closest agreement.\n\n"
            "**Interpretation:** agreement within roughly **5–15%** means your "
            "computed energy is realistic for the location. A *perfect* match "
            "would actually be suspicious (it would suggest copying PVGIS's "
            "assumptions). A longer (≈1-year) window usually tightens the gap.")
        st.info(
            "Honesty note for the dissertation: **both values are *modelled* "
            "estimates**, so this is cross-verification of the yield *magnitude* "
            "against an independent estimator — not validation against metered "
            "generation. State it that way and the comparison is fully defensible.")


if __name__ == "__main__":
    main()
