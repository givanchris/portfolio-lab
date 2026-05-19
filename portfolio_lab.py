"""
Portfolio Lab — interactive Monte Carlo sandbox.

Run with:
    streamlit run portfolio_lab.py

Features:
  - Preset portfolios (My Portfolio, Conservative, Moderate Growth, Aggressive AI, 100% Index)
  - Easy ticker management via multiselect + simple shares editor
  - Scenario buttons (Base / AI Boom / Risk-Off / Mild Bear / Custom)
  - Advisor plain-English summary card
  - Advanced risk-model expander for power users
  - Live price refresh from yfinance on every load (cached 60s)
  - Multi-portfolio comparison (up to 3 books side-by-side)
  - Adjustable sim params: paths, horizon, lump sum, monthly contrib
  - Outputs: percentile table, fan chart, terminal histogram, drawdown stats
  - Save/load scenarios to JSON for later comparison
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from build_pdf_report import build_report as _build_pdf_report

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE               = Path(__file__).parent
HOLDINGS_FP        = HERE / os.environ.get("HOLDINGS_FILE", "holdings.json")
SCENARIOS_FP       = HERE / "lab_scenarios.json"
PRICE_CACHE_FP     = HERE / "price_cache.json"
CACHE_MAX_AGE_DAYS = 1

NAVY = "#1e3a5f"
GOLD = "#b8860b"
RED  = "#c0392b"

st.set_page_config(page_title="Portfolio Lab", layout="wide", page_icon="📊")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ══ Design tokens ════════════════════════════════════════════════ */
  :root {
    --bg:     #e8edf5;
    --card:   #ffffff;
    --navy:   #1e3a5f;
    --navy2:  #2d5282;
    --accent: #3b82f6;
    --green:  #059669;
    --red:    #dc2626;
    --amber:  #d97706;
    --text:   #0f172a;
    --text2:  #374151;
    --muted:  #64748b;
    --border: #e2e8f0;
    --sh1: 0 1px 3px rgba(15,23,42,.07), 0 1px 2px rgba(15,23,42,.04);
    --sh2: 0 4px 12px rgba(15,23,42,.09), 0 2px 4px rgba(15,23,42,.05);
    --sh3: 0 12px 32px rgba(15,23,42,.14), 0 4px 8px rgba(15,23,42,.07);
    --r1: 6px; --r2: 10px; --r3: 16px;
  }

  /* ══ Base ═════════════════════════════════════════════════════════ */
  html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg) !important;
  }
  .block-container { padding-top: 1.5rem !important; max-width: 1440px; }
  h2, h3 { color: var(--text) !important; font-weight: 600; }

  /* ══ Sidebar ══════════════════════════════════════════════════════ */
  [data-testid="stSidebar"] {
    background: #060d1a !important;
    border-right: 1px solid #0d1829 !important;
  }
  [data-testid="stSidebar"] * { color: #94a3b8 !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] strong { color: #f1f5f9 !important; }
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] select {
    background: #1e293b !important;
    border-color: #334155 !important;
    color: #e2e8f0 !important;
    border-radius: var(--r1) !important;
  }
  [data-testid="stSidebar"] [data-baseweb="slider"] > div:first-child {
    background: #1e293b !important;
  }
  [data-testid="stSidebar"] [data-baseweb="slider"] > div:first-child > div {
    background: var(--accent) !important;
  }
  [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #60a5fa !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.28) !important;
  }

  /* ══ Sidebar brand & sections ═════════════════════════════════════ */
  .sidebar-brand {
    padding: 18px 0 22px 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 22px;
  }
  .sb-name {
    font-size: 1.05rem; font-weight: 700;
    color: #f1f5f9 !important; letter-spacing: -0.02em;
  }
  .sb-sub {
    font-size: 0.63rem; color: #475569 !important;
    margin-top: 3px; text-transform: uppercase; letter-spacing: 0.14em;
  }
  .sidebar-section {
    margin: 22px 0 10px 0;
    padding-bottom: 7px;
    border-bottom: 1px solid #1e293b;
  }
  .sidebar-section span {
    font-size: 0.64rem; font-weight: 700;
    color: #475569 !important; text-transform: uppercase; letter-spacing: 0.14em;
  }

  /* ══ Hero ══════════════════════════════════════════════════════════ */
  .hero {
    background: linear-gradient(135deg, #060d1a 0%, #1e3a5f 48%, #2d5282 100%);
    border-radius: var(--r3);
    padding: 36px 44px;
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--sh3);
  }
  .hero::before {
    content: ''; position: absolute;
    top: -55%; right: -5%; width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(59,130,246,.18) 0%, transparent 65%);
    pointer-events: none;
  }
  .hero-left { position: relative; z-index: 1; flex: 1; }
  .hero-eyebrow {
    font-size: 0.62rem; font-weight: 700;
    color: #60a5fa; letter-spacing: 0.20em;
    text-transform: uppercase; margin-bottom: 10px;
  }
  .hero h1 {
    font-size: 2.25rem; font-weight: 800;
    color: #f8fafc !important; margin: 0 0 10px 0;
    letter-spacing: -0.045em; line-height: 1.1;
  }
  .hero-sub {
    color: #94a3b8; font-size: 0.87rem;
    margin: 0 0 20px 0; line-height: 1.65; max-width: 520px;
  }
  .hero-pills { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .hero-pill {
    background: rgba(255,255,255,0.07);
    color: #cbd5e1; border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px; padding: 4px 13px;
    font-size: 0.70rem; font-weight: 500;
  }
  .hero-pill.live {
    background: rgba(5,150,105,.18);
    color: #6ee7b7; border-color: rgba(5,150,105,.28);
  }
  .hero-right { display: flex; gap: 10px; position: relative; z-index: 1; flex-shrink: 0; }
  .hero-stat {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: var(--r2); padding: 18px 22px; text-align: center; min-width: 115px;
  }
  .hs-val {
    font-size: 1.30rem; font-weight: 700;
    color: #f8fafc; letter-spacing: -0.025em;
  }
  .hs-lbl {
    font-size: 0.63rem; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.12em; margin-top: 5px;
  }

  /* ══ Section labels ════════════════════════════════════════════════ */
  .section-label {
    margin: 30px 0 14px 0; padding-left: 12px;
    border-left: 3px solid var(--navy); display: flex; align-items: center;
  }
  .section-label span {
    font-size: 0.70rem; font-weight: 700;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.13em;
  }

  /* ══ KPI cards ═════════════════════════════════════════════════════ */
  .kpi-row { display: flex; gap: 10px; margin: 10px 0; }
  .kpi {
    flex: 1; min-width: 0;
    background: var(--card); border-radius: var(--r2);
    padding: 18px 20px; border: 1px solid var(--border);
    border-top: 3px solid #e2e8f0; box-shadow: var(--sh2);
    transition: transform .12s, box-shadow .12s;
  }
  .kpi:hover { transform: translateY(-1px); box-shadow: var(--sh3); }
  .kpi.green { border-top-color: var(--green); }
  .kpi.red   { border-top-color: var(--red); }
  .kpi.blue  { border-top-color: var(--accent); }
  .kpi.gold  { border-top-color: var(--amber); }
  .kpi .kval {
    font-size: 1.45rem; font-weight: 700;
    color: var(--text); line-height: 1.15; letter-spacing: -0.025em;
  }
  .kpi .klbl {
    font-size: 0.66rem; color: var(--muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.10em; margin-top: 6px;
  }

  /* ══ Preset cards ══════════════════════════════════════════════════ */
  .preset-card {
    background: var(--card); border: 1.5px solid var(--border);
    border-radius: var(--r2); padding: 16px; margin-bottom: 8px;
    box-shadow: var(--sh1); transition: border-color .15s, box-shadow .15s;
    min-height: 120px;
  }
  .preset-card.active {
    border-color: var(--navy);
    background: linear-gradient(135deg, #f0f5ff 0%, #f8faff 100%);
    box-shadow: 0 0 0 3px rgba(30,58,95,.10), var(--sh2);
  }
  .preset-card:hover:not(.active) { border-color: #cbd5e1; box-shadow: var(--sh2); }
  .pc-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
  .pc-icon { font-size: 1.5rem; line-height: 1; }
  .pc-badge {
    font-size: 0.58rem; font-weight: 700; color: var(--navy);
    background: rgba(30,58,95,.10); border-radius: 10px;
    padding: 3px 8px; text-transform: uppercase; letter-spacing: 0.08em; display: none;
  }
  .preset-card.active .pc-badge { display: inline-block; }
  .pc-name { font-size: 0.86rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
  .pc-desc { font-size: 0.71rem; color: var(--muted); line-height: 1.55; }

  /* ══ Scenario cards ════════════════════════════════════════════════ */
  .scenario-card {
    background: var(--card); border: 1.5px solid var(--border);
    border-radius: var(--r2); padding: 14px 16px; margin-bottom: 8px;
    box-shadow: var(--sh1); transition: border-color .15s, box-shadow .15s; min-height: 90px;
  }
  .scenario-card.active {
    border-color: var(--accent);
    background: linear-gradient(135deg, #eff6ff 0%, #f8faff 100%);
    box-shadow: 0 0 0 3px rgba(59,130,246,.12), var(--sh2);
  }
  .scenario-card:hover:not(.active) { border-color: #cbd5e1; box-shadow: var(--sh2); }
  .sc-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .sc-icon { font-size: 1.1rem; }
  .sc-name { font-size: 0.84rem; font-weight: 700; color: var(--text); }
  .sc-badge {
    margin-left: auto; font-size: 0.58rem; font-weight: 700; color: var(--accent);
    background: rgba(59,130,246,.12); border-radius: 10px;
    padding: 3px 8px; text-transform: uppercase; letter-spacing: 0.08em;
  }
  .sc-desc { font-size: 0.70rem; color: var(--muted); line-height: 1.55; }

  /* ══ Advisor card ══════════════════════════════════════════════════ */
  .advisor-card {
    background: var(--card); border: 1px solid var(--border);
    border-left: 4px solid var(--navy); border-radius: var(--r2);
    padding: 22px 26px; margin: 16px 0 22px 0;
    line-height: 1.85; color: var(--text2); box-shadow: var(--sh2);
  }
  .advisor-card strong { color: var(--text); font-weight: 600; }
  .advisor-card h4 {
    color: var(--text); margin: 0 0 12px 0; font-size: 0.92rem; font-weight: 700;
    border-bottom: 1px solid var(--border); padding-bottom: 10px;
  }

  /* ══ Tabs ══════════════════════════════════════════════════════════ */
  .stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #d8dfe9;
    border-radius: var(--r1); padding: 4px; border: 1px solid #c8d0de;
  }
  .stTabs [data-baseweb="tab"] {
    color: var(--muted) !important; font-weight: 500 !important;
    border-radius: 4px !important; padding: 6px 18px !important; font-size: 0.84rem !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--card) !important; color: var(--text) !important;
    font-weight: 600 !important; box-shadow: var(--sh1) !important;
  }

  /* ══ Buttons ═══════════════════════════════════════════════════════ */
  .stButton > button {
    border-radius: var(--r1) !important; font-weight: 500 !important;
    font-size: 0.82rem !important; background: var(--card) !important;
    color: var(--text2) !important; border: 1px solid var(--border) !important;
    box-shadow: var(--sh1) !important; transition: all .12s ease !important;
  }
  .stButton > button[kind="primary"] {
    background: var(--navy) !important; border-color: var(--navy) !important;
    color: #ffffff !important; box-shadow: var(--sh2) !important;
  }
  .stButton > button:hover { opacity: 0.84 !important; }

  /* ══ Expanders ═════════════════════════════════════════════════════ */
  [data-testid="stExpander"] {
    border: 1px solid var(--border) !important; border-radius: var(--r2) !important;
    background: var(--card) !important; box-shadow: var(--sh1) !important;
  }

  /* ══ Metric tiles ══════════════════════════════════════════════════ */
  [data-testid="metric-container"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--r2); padding: 16px 20px; box-shadow: var(--sh2);
  }

  /* ══ Dataframes ════════════════════════════════════════════════════ */
  [data-testid="stDataFrame"] {
    border-radius: var(--r2) !important; overflow: hidden;
    border: 1px solid var(--border) !important; box-shadow: var(--sh1);
  }

  /* ══ Misc ══════════════════════════════════════════════════════════ */
  .custom-divider { border: none; border-top: 1px solid var(--border); margin: 28px 0; }
  .stAlert { border-radius: var(--r2) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chart style defaults
# ---------------------------------------------------------------------------

_CHART_RC: dict = {
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    "#fafbfc",
    "figure.facecolor":  "white",
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.color":        "#dde3ee",
    "grid.linewidth":    0.7,
    "axes.labelcolor":   "#4a5568",
    "xtick.color":       "#718096",
    "ytick.color":       "#718096",
    "axes.labelsize":    9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
}

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _divider():
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


def _section_label(text: str):
    st.markdown(f'<div class="section-label"><span>{text}</span></div>', unsafe_allow_html=True)


def _kpi(label: str, value: str, tone: str = "blue") -> str:
    return (f'<div class="kpi {tone}">'
            f'<div class="kval">{value}</div>'
            f'<div class="klbl">{label}</div>'
            f'</div>')


def _kpi_row(*cards: str):
    st.markdown('<div class="kpi-row">' + "".join(cards) + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Default per-ticker risk model
#   mu_annual, sigma_annual, (f_market, f_ai, f_power)
# ---------------------------------------------------------------------------

DEFAULT_RISK_MODEL = {
    "COHR": (0.12, 0.55, 0.10, 0.55, 0.00),
    "AVGO": (0.10, 0.30, 0.20, 0.50, 0.00),
    "MRVL": (0.11, 0.50, 0.15, 0.55, 0.00),
    "GOOG": (0.09, 0.28, 0.35, 0.30, 0.00),
    "ALAB": (0.10, 0.60, 0.10, 0.60, 0.00),
    "AMAT": (0.09, 0.38, 0.20, 0.50, 0.00),
    "CRWV": (0.08, 0.70, 0.05, 0.55, 0.00),
    "ASML": (0.10, 0.32, 0.20, 0.50, 0.00),
    "CEG":  (0.08, 0.38, 0.15, 0.10, 0.40),
    "LEU":  (0.09, 0.55, 0.10, 0.05, 0.45),
    "HOOD": (0.09, 0.55, 0.30, 0.15, 0.00),
    "UUUU": (0.08, 0.65, 0.10, 0.00, 0.40),
    "SOLS": (0.08, 0.45, 0.20, 0.10, 0.00),
    "APLD": (0.07, 0.75, 0.05, 0.50, 0.00),
    "PATH": (0.05, 0.50, 0.20, 0.20, 0.00),
    "VELO": (0.03, 0.80, 0.05, 0.05, 0.00),
    "NVDA": (0.12, 0.40, 0.15, 0.55, 0.00),
    "TSM":  (0.10, 0.35, 0.25, 0.45, 0.00),
    "MSFT": (0.10, 0.26, 0.30, 0.40, 0.00),
    "META": (0.10, 0.34, 0.30, 0.35, 0.00),
    "AAPL": (0.08, 0.26, 0.40, 0.25, 0.00),
    "AMZN": (0.10, 0.30, 0.30, 0.30, 0.00),
    "VT":   (0.08, 0.16, 0.95, 0.00, 0.00),
    "VOO":  (0.08, 0.16, 0.95, 0.00, 0.00),
    "SPY":  (0.08, 0.16, 0.95, 0.00, 0.00),
    "QQQ":  (0.10, 0.22, 0.65, 0.30, 0.00),
    "SGOV": (0.045, 0.01, 0.01, 0.00, 0.00),
    "BIL":  (0.045, 0.01, 0.01, 0.00, 0.00),
}

GENERIC = (0.08, 0.40, 0.20, 0.20, 0.00)

SPX_MU    = 0.08
SPX_SIGMA = 0.16
SMH_MU    = 0.18   # AI/semis ETF (SMH) — hand-set per house rules
SMH_SIGMA = 0.28
URA_MU    = 0.10   # uranium ETF (URA) — hand-set
URA_SIGMA = 0.35

REGIMES = {
    "Base case": {"mu_shift": 0.00, "ai_shift": 0.00, "vol_mult": 1.00},
    "AI boom":   {"mu_shift": 0.00, "ai_shift": 0.20, "vol_mult": 0.85},
    "Risk-off":  {"mu_shift": -0.20, "ai_shift": -0.25, "vol_mult": 1.40},
    "Mild bear": {"mu_shift": -0.10, "ai_shift": -0.10, "vol_mult": 1.20},
    "Custom":    {"mu_shift": 0.00, "ai_shift": 0.00, "vol_mult": 1.00},
}

# Two-state regime-switching parameters for the MC engine.
# Calm→stress 5.5%/mo (~18-mo avg calm spell); stress persists 70%/mo (~3.3-mo avg).
# Stationary stress fraction ≈ 15.5% — consistent with NBER recession coverage.
STRESS_P_ENTER  = 0.055   # P(calm → stress) per month
STRESS_P_STAY   = 0.700   # P(stress → stress) per month
STRESS_VOL_MULT = 1.50    # total vol scale in stress
# Multiplier on the systematic variance *fraction* in stress.  1.5 → a ticker
# that is 50% systematic in calm becomes 75% systematic in stress (capped at 92%).
# This directly spikes pairwise correlations toward 1 during drawdown periods.
STRESS_SYS_BOOST = 1.50

# Preset definitions: list of (ticker, target_dollars)
# Shares are computed from live prices at load time; total notional = $10,000
PRESETS: dict[str, list[tuple[str, float]] | None] = {
    "My Portfolio":    None,   # loaded from holdings.json
    "Conservative":    [("VOO", 5000), ("SGOV", 3000), ("QQQ", 2000)],
    "Moderate Growth": [("VOO", 6000), ("QQQ", 2000), ("META", 1000), ("NVDA", 1000)],
    "Aggressive AI":   [("NVDA", 1250), ("AVGO", 1250), ("MRVL", 1250), ("ALAB", 1250),
                        ("COHR", 1250), ("ASML", 1250), ("GOOG", 1250), ("AMAT", 1250)],
    "100% Index":      [("VOO", 10000)],
}

_PRESET_META: dict[str, dict] = {
    "My Portfolio":    {"icon": "💼", "desc": "Live holdings from holdings.json with cash position"},
    "Conservative":    {"icon": "🛡️", "desc": "60% VOO · 30% SGOV · 10% QQQ — capital preservation"},
    "Moderate Growth": {"icon": "📈", "desc": "60% VOO · 20% QQQ · 20% growth — balanced risk/return"},
    "Aggressive AI":   {"icon": "🚀", "desc": "8-stock AI/semis basket — NVDA AVGO MRVL ALAB COHR ASML GOOG AMAT"},
    "100% Index":      {"icon": "📊", "desc": "100% VOO — pure market-cap exposure, SPX beta ≈ 1"},
}

_SCENARIO_META: dict[str, dict] = {
    "Base case": {"icon": "📊", "label": "Base Case",  "desc": "Historical average returns, normal vol regime"},
    "AI boom":   {"icon": "🚀", "label": "AI Boom",    "desc": "AI factor μ +20%, systematic vol compressed −15%"},
    "Risk-off":  {"icon": "🛡️", "label": "Risk-Off",   "desc": "Broad μ −20%, AI factor −25%, vol +40%"},
    "Mild bear": {"icon": "🐻", "label": "Mild Bear",  "desc": "μ −10%, vol ×1.2 — moderate multi-quarter correction"},
    "Custom":    {"icon": "⚙️", "label": "Custom",     "desc": "Set μ shift, AI boost, and vol multiplier manually"},
}

# ---------------------------------------------------------------------------
# Price cache helpers
# ---------------------------------------------------------------------------

def _load_price_cache() -> dict:
    if not PRICE_CACHE_FP.exists():
        return {}
    try:
        return json.loads(PRICE_CACHE_FP.read_text())
    except Exception:
        return {}


def _save_price_cache(cache: dict) -> None:
    try:
        PRICE_CACHE_FP.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Live price fetcher — 3-tier resolver, cached 60s
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...]) -> dict[str, dict]:
    if not tickers:
        return {}

    cache  = _load_price_cache()
    now    = datetime.now()
    result: dict[str, dict] = {}

    needs_retry: list[str] = []
    try:
        data = yf.download(
            list(tickers), period="5d", interval="1d",
            auto_adjust=True, progress=False, threads=True, group_by="ticker",
        )
        for t in tickers:
            try:
                close = (data["Close"].dropna().iloc[-1] if len(tickers) == 1
                         else data[t]["Close"].dropna().iloc[-1])
                result[t] = {"price": float(close), "source": "live",
                             "ts": now.isoformat(timespec="seconds"), "retried": False}
            except Exception:
                needs_retry.append(t)
    except Exception:
        needs_retry = list(tickers)

    still_missing: list[str] = []
    for t in needs_retry:
        time.sleep(0.3)
        try:
            hist  = yf.Ticker(t).history(period="5d", auto_adjust=True)
            close = hist["Close"].dropna().iloc[-1]
            result[t] = {"price": float(close), "source": "live",
                         "ts": now.isoformat(timespec="seconds"), "retried": True}
        except Exception:
            still_missing.append(t)

    for t, info in result.items():
        cache[t] = {"price": info["price"], "ts": info["ts"]}
    if result:
        _save_price_cache(cache)

    for t in still_missing:
        entry = cache.get(t)
        if entry:
            age_days = (now - datetime.fromisoformat(entry["ts"])).days
            if age_days <= CACHE_MAX_AGE_DAYS:
                result[t] = {"price": entry["price"],
                             "source": f"cached ({age_days}d old)",
                             "ts": entry["ts"], "retried": True}
            else:
                result[t] = {"price": float("nan"), "source": "missing",
                             "ts": entry["ts"], "retried": True}
        else:
            result[t] = {"price": float("nan"), "source": "missing",
                         "ts": None, "retried": True}

    return result


# ---------------------------------------------------------------------------
# Holdings I/O
# ---------------------------------------------------------------------------

def _row_from_ticker(ticker: str, shares: float = 1.0, avg_cost: float = float("nan"),
                     thesis: str = "") -> dict:
    t  = ticker.upper()
    rm = DEFAULT_RISK_MODEL.get(t, GENERIC)
    return {"ticker": t, "shares": shares, "avg_cost": avg_cost,
            "mu": rm[0], "sigma": rm[1],
            "f_market": rm[2], "f_ai": rm[3], "f_power": rm[4],
            "thesis": thesis}


def load_default_holdings() -> pd.DataFrame:
    if not HOLDINGS_FP.exists():
        return _empty_holdings()
    with open(HOLDINGS_FP) as f:
        data = json.load(f)
    rows = [_row_from_ticker(h["ticker"], h["shares"],
                             h.get("avg_cost", float("nan")), h.get("thesis", ""))
            for h in data.get("holdings", [])]
    account = data.get("account", {})
    cash = float(account.get("cash", 0))
    if cash > 0:
        rows.append({
            "ticker": "CASH", "shares": cash, "avg_cost": 1.0, "thesis": "",
            "mu": 0.04, "sigma": 0.002, "f_market": 0.0, "f_ai": 0.0, "f_power": 0.0,
        })
    return pd.DataFrame(rows)


def _empty_holdings() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "shares", "avg_cost", "mu", "sigma",
                                  "f_market", "f_ai", "f_power", "thesis"])


def _portfolio_anchor_mv() -> float:
    """Total MV of the user's actual holdings (holdings.json), cached in session state.
    Presets are scaled to this anchor so every book starts at the same total value."""
    if "anchor_mv" in st.session_state:
        return st.session_state["anchor_mv"]
    fallback = 10000.0
    mv = fallback
    try:
        if HOLDINGS_FP.exists():
            with open(HOLDINGS_FP) as f:
                data = json.load(f)
            rows = data.get("holdings", [])
            if rows:
                tickers = tuple(h["ticker"].upper() for h in rows)
                prices = fetch_prices(tickers)
                total = 0.0
                for h in rows:
                    p = prices.get(h["ticker"].upper(), {}).get("price", float("nan"))
                    if not np.isnan(p):
                        total += float(h["shares"]) * float(p)
                total += float(data.get("account", {}).get("cash", 0))
                if total > 0:
                    mv = total
    except Exception:
        pass
    st.session_state["anchor_mv"] = float(mv)
    return float(mv)


def build_preset(preset_name: str) -> pd.DataFrame:
    spec = PRESETS[preset_name]
    if spec is None:
        return load_default_holdings()
    tickers = tuple(t for t, _ in spec)
    prices  = fetch_prices(tickers)
    spec_total = sum(d for _, d in spec)
    target_mv  = _portfolio_anchor_mv()
    scale      = (target_mv / spec_total) if spec_total > 0 else 1.0
    rows = []
    for ticker, dollars in spec:
        price = prices.get(ticker, {}).get("price", float("nan"))
        shares = (dollars * scale / price) if price and not np.isnan(price) else 1.0
        rows.append(_row_from_ticker(ticker, round(shares, 4)))
    return pd.DataFrame(rows)


def enrich_with_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(price=[], mv=[], weight=[], source=[])
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    fetch_tickers = tuple(t for t in df["ticker"].dropna() if t and t != "CASH")
    price_data    = fetch_prices(fetch_tickers)
    st.session_state["_price_data"] = price_data
    df["price"]  = df["ticker"].map(lambda t: price_data.get(t, {}).get("price", float("nan"))).astype(float)
    df["source"] = df["ticker"].map(lambda t: price_data.get(t, {}).get("source", "missing"))
    cash_mask = df["ticker"] == "CASH"
    if cash_mask.any():
        df.loc[cash_mask, "price"]  = 1.0
        df.loc[cash_mask, "source"] = "synthetic"
    df["mv"]     = df["shares"].astype(float) * df["price"]
    total_mv     = df["mv"].sum()
    df["weight"] = (df["mv"] / total_mv) if total_mv > 0 else 0.0
    return df


def detect_price_failures(enriched: pd.DataFrame) -> list[str]:
    if enriched.empty or "source" not in enriched.columns:
        return []
    return enriched.loc[enriched["source"] == "missing", "ticker"].tolist()


def sync_risk_model(simple_df: pd.DataFrame, existing_full: pd.DataFrame) -> pd.DataFrame:
    """Merge a simple (ticker, shares) df with existing risk model rows, adding defaults for new tickers."""
    existing_map = {row["ticker"]: row for row in existing_full.to_dict("records")}
    rows = []
    for _, r in simple_df.iterrows():
        t = str(r["ticker"]).upper() if pd.notna(r["ticker"]) else ""
        if not t:
            continue
        if t in existing_map:
            rec = dict(existing_map[t])
            rec["shares"] = r["shares"]
            if "avg_cost" in r and pd.notna(r.get("avg_cost")):
                rec["avg_cost"] = r["avg_cost"]
        else:
            rec = _row_from_ticker(t, r["shares"], r.get("avg_cost", float("nan")) if "avg_cost" in r else float("nan"))
        rows.append(rec)
    return pd.DataFrame(rows) if rows else _empty_holdings()


# ---------------------------------------------------------------------------
# What-if rebalance helpers
# ---------------------------------------------------------------------------

def rebalance_pro_rata(weights: dict[str, float], anchor: str,
                       new_anchor_w: float) -> dict[str, float]:
    """Anchor takes new_anchor_w; remaining weights scale to fill (1 − new_anchor_w)."""
    new_anchor_w = max(0.0, min(1.0, float(new_anchor_w)))
    out = dict(weights)
    out[anchor] = new_anchor_w
    others = [k for k in out if k != anchor]
    other_sum = sum(out[k] for k in others)
    remainder = 1.0 - new_anchor_w
    if other_sum > 1e-12 and others:
        scale = remainder / other_sum
        for k in others:
            out[k] = out[k] * scale
    elif others and remainder > 0:
        each = remainder / len(others)
        for k in others:
            out[k] = each
    return out


def weights_to_shares(weights: dict[str, float], prices: dict[str, float],
                      total_mv: float) -> dict[str, float]:
    """Convert target weights → fractional shares given live prices and dollar budget."""
    out = {}
    for t, w in weights.items():
        p = prices.get(t, float("nan"))
        if p and not np.isnan(p) and p > 0:
            out[t] = (float(w) * float(total_mv)) / float(p)
        else:
            out[t] = 0.0
    return out


def concentration_metrics(weights) -> dict:
    """HHI, effective N (1/HHI), top-1 and top-3 weight share."""
    if isinstance(weights, dict):
        arr = np.asarray(list(weights.values()), dtype=float)
    else:
        arr = np.asarray(list(weights), dtype=float)
    arr = arr[~np.isnan(arr)]
    s = arr.sum()
    if s <= 0:
        return {"hhi": float("nan"), "eff_n": float("nan"),
                "top1": float("nan"), "top3": float("nan")}
    arr = arr / s
    sorted_w = np.sort(arr)[::-1]
    hhi = float(np.sum(arr ** 2))
    return {
        "hhi":   hhi,
        "eff_n": float(1.0 / hhi) if hhi > 0 else float("nan"),
        "top1":  float(sorted_w[0]),
        "top3":  float(sorted_w[:3].sum()),
    }


def factor_exposure(df: pd.DataFrame,
                    weights_override: dict[str, float] | None = None) -> dict:
    """Portfolio-weighted average of factor loadings."""
    if df.empty:
        return {"f_market": 0.0, "f_ai": 0.0, "f_power": 0.0, "f_idio": 0.0}
    if weights_override:
        w = df["ticker"].map(weights_override).fillna(0.0).astype(float).to_numpy()
    else:
        w = df["weight"].fillna(0.0).astype(float).to_numpy()
    s = w.sum()
    if s > 1e-12:
        w = w / s
    fm = float(np.sum(w * df["f_market"].fillna(0.0).astype(float).to_numpy()))
    fa = float(np.sum(w * df["f_ai"].fillna(0.0).astype(float).to_numpy()))
    fp = float(np.sum(w * df["f_power"].fillna(0.0).astype(float).to_numpy()))
    fi = max(0.0, 1.0 - fm - fa - fp)
    return {"f_market": fm, "f_ai": fa, "f_power": fp, "f_idio": fi}


# ---------------------------------------------------------------------------
# Monte Carlo engine
# ---------------------------------------------------------------------------

def _stress_loadings(
    b_M: np.ndarray, b_AI: np.ndarray, b_PW: np.ndarray, s_idio: np.ndarray,
    vol_mult: float, sys_boost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stress-state factor loadings.

    Boosts the systematic variance *fraction* by sys_boost (capped at 92%) so
    that more variance is shared across tickers — pairwise correlations spike
    toward 1.  Total monthly vol is scaled by vol_mult.  Each systematic factor
    grows proportionally so their relative weights are preserved.
    """
    var_calm = b_M**2 + b_AI**2 + b_PW**2 + s_idio**2
    s_m_s    = np.sqrt(var_calm) * vol_mult          # stress monthly vol
    f_sys    = (b_M**2 + b_AI**2 + b_PW**2) / np.maximum(var_calm, 1e-30)
    f_sys_s  = np.minimum(f_sys * sys_boost, 0.92)   # stress systematic fraction
    ratio    = np.where(f_sys > 1e-12, f_sys_s / f_sys, 0.0)
    bM_s     = s_m_s * np.sqrt(b_M**2  / np.maximum(var_calm, 1e-30) * ratio)
    bAI_s    = s_m_s * np.sqrt(b_AI**2 / np.maximum(var_calm, 1e-30) * ratio)
    bPW_s    = s_m_s * np.sqrt(b_PW**2 / np.maximum(var_calm, 1e-30) * ratio)
    si_s     = s_m_s * np.sqrt(np.maximum(1.0 - f_sys_s, 0.0))
    return bM_s, bAI_s, bPW_s, si_s


def _generate_states(
    rng: np.random.Generator, n_paths: int, n_months: int,
    p_enter: float, p_stay: float,
) -> np.ndarray:
    """Generate Markov calm/stress state sequences.

    Returns bool array (n_paths, n_months); True = stress.
    Initial states drawn from the stationary distribution.
    """
    pi_stress = p_enter / (p_enter + (1.0 - p_stay))
    states = np.empty((n_paths, n_months), dtype=bool)
    states[:, 0] = rng.random(n_paths) < pi_stress
    for m in range(1, n_months):
        u = rng.random(n_paths)
        states[:, m] = np.where(states[:, m - 1], u < p_stay, u < p_enter)
    return states


def simulate(
    df: pd.DataFrame,
    n_paths: int,
    n_months: int,
    starting_mv: float,
    lump_sum: float,
    lump_month: int,
    monthly_contrib: float,
    regime: dict,
    seed: int = 42,
    two_state: bool = True,
):
    rng = np.random.default_rng(seed)
    n   = len(df)
    if n == 0 or starting_mv <= 0:
        return None, None

    w        = df["weight"].to_numpy()
    mu_ann   = df["mu"].to_numpy() + regime["mu_shift"]
    sig_ann  = df["sigma"].to_numpy() * regime["vol_mult"]
    f_M      = df["f_market"].to_numpy()
    f_AI     = df["f_ai"].to_numpy()
    f_PW     = df["f_power"].to_numpy()
    ai_shift = regime["ai_shift"]

    s_m    = sig_ann * np.sqrt(1/12)
    b_M    = s_m * np.sqrt(np.clip(f_M, 0, 1))
    b_AI   = s_m * np.sqrt(np.clip(f_AI, 0, 1))
    b_PW   = s_m * np.sqrt(np.clip(f_PW, 0, 1))
    s_idio = s_m * np.sqrt(np.clip(1 - f_M - f_AI - f_PW, 0, 1))
    drift  = (mu_ann - sig_ann**2 / 2) / 12
    ai_drift_per_month = ai_shift / 12

    if two_state:
        bM_s, bAI_s, bPW_s, si_s = _stress_loadings(
            b_M, b_AI, b_PW, s_idio, STRESS_VOL_MULT, STRESS_SYS_BOOST
        )
        stress_seq = _generate_states(rng, n_paths, n_months, STRESS_P_ENTER, STRESS_P_STAY)
        spx_sm_s = SPX_SIGMA * regime["vol_mult"] * STRESS_VOL_MULT * np.sqrt(1/12)

    V = np.broadcast_to(starting_mv * w, (n_paths, n)).copy()
    port = np.zeros((n_paths, n_months + 1))
    port[:, 0] = starting_mv

    spx       = np.zeros((n_paths, n_months + 1))
    spx[:, 0] = starting_mv
    spx_drift = (SPX_MU + regime["mu_shift"] - (SPX_SIGMA * regime["vol_mult"])**2 / 2) / 12
    spx_sm    = SPX_SIGMA * regime["vol_mult"] * np.sqrt(1/12)

    smh       = np.zeros((n_paths, n_months + 1))
    smh[:, 0] = starting_mv
    smh_drift = (SMH_MU + regime["mu_shift"] - (SMH_SIGMA * regime["vol_mult"])**2 / 2) / 12
    smh_sm    = SMH_SIGMA * regime["vol_mult"] * np.sqrt(1/12)

    ura       = np.zeros((n_paths, n_months + 1))
    ura[:, 0] = starting_mv
    ura_drift = (URA_MU + regime["mu_shift"] - (URA_SIGMA * regime["vol_mult"])**2 / 2) / 12
    ura_sm    = URA_SIGMA * regime["vol_mult"] * np.sqrt(1/12)

    for m in range(1, n_months + 1):
        Z_M  = rng.standard_normal(n_paths)
        Z_AI = rng.standard_normal(n_paths)
        Z_PW = rng.standard_normal(n_paths)
        Z_i  = rng.standard_normal((n_paths, n))

        if two_state:
            is_s = stress_seq[:, m - 1, None]          # (n_paths, 1) bool
            bM   = np.where(is_s, bM_s,  b_M)          # (n_paths, n)
            bAI  = np.where(is_s, bAI_s, b_AI)
            bPW  = np.where(is_s, bPW_s, b_PW)
            si   = np.where(is_s, si_s,  s_idio)
            sm   = np.where(stress_seq[:, m - 1], spx_sm_s, spx_sm)  # (n_paths,)
            log_r = (drift[None, :]
                     + bM  * Z_M[:, None]
                     + bAI * Z_AI[:, None]
                     + bPW * Z_PW[:, None]
                     + si  * Z_i
                     + (ai_drift_per_month * f_AI)[None, :])
        else:
            sm = spx_sm
            log_r = (drift[None, :]
                     + b_M[None, :]    * Z_M[:, None]
                     + b_AI[None, :]   * Z_AI[:, None]
                     + b_PW[None, :]   * Z_PW[:, None]
                     + s_idio[None, :] * Z_i
                     + (ai_drift_per_month * f_AI)[None, :])

        V *= np.exp(log_r)

        if lump_sum > 0 and m == lump_month:
            V += lump_sum * w[None, :]
        if monthly_contrib > 0:
            V += monthly_contrib * w[None, :]

        port[:, m] = V.sum(axis=1)
        spx[:, m]  = spx[:, m-1] * np.exp(spx_drift + sm * Z_M)
        smh[:, m]  = smh[:, m-1] * np.exp(smh_drift + smh_sm * Z_AI)
        ura[:, m]  = ura[:, m-1] * np.exp(ura_drift + ura_sm * Z_PW)

        if lump_sum > 0 and m == lump_month:
            spx[:, m] += lump_sum
            smh[:, m] += lump_sum
            ura[:, m] += lump_sum
        if monthly_contrib > 0:
            spx[:, m] += monthly_contrib
            smh[:, m] += monthly_contrib
            ura[:, m] += monthly_contrib

    return port, spx, smh, ura


def summarize(port, spx, smh, ura, deposits) -> dict:
    end     = port[:, -1]
    end_spx = spx[:, -1]
    peak    = np.maximum.accumulate(port, axis=1)
    dd      = (port / peak - 1.0).min(axis=1)
    return {
        "median":        float(np.percentile(end, 50)),
        "p05":           float(np.percentile(end, 5)),
        "p10":           float(np.percentile(end, 10)),
        "p25":           float(np.percentile(end, 25)),
        "p75":           float(np.percentile(end, 75)),
        "p90":           float(np.percentile(end, 90)),
        "p95":           float(np.percentile(end, 95)),
        "mean":          float(end.mean()),
        "prob_beat_spx": float(np.mean(end > end_spx)),
        "prob_beat_smh": float(np.mean(end > smh[:, -1])),
        "prob_beat_ura": float(np.mean(end > ura[:, -1])),
        "prob_loss":     float(np.mean(end < deposits)),
        "prob_double":   float(np.mean(end > 2 * deposits)),
        "prob_halve":    float(np.mean(end < 0.5 * deposits)),
        "dd_p10":        float(np.percentile(dd, 10)),
        "dd_p50":        float(np.percentile(dd, 50)),
        "dd_p90":        float(np.percentile(dd, 90)),
        "deposits":      float(deposits),
    }


def _probability_ladder(port: np.ndarray, horizon_y: int, starting_mv: float) -> dict:
    n_months = port.shape[1]
    mpy = max(1, (n_months - 1) // horizon_y) if horizon_y > 0 else 12
    years = [y for y in [5, 10, 15, 20, 25, 30] if y <= horizon_y]
    if horizon_y not in years:
        years = sorted(set(years + [horizon_y]))
    multiples = [2, 3, 5, 10]
    grid = {}
    for y in years:
        idx = min(int(y * mpy), n_months - 1)
        slc = port[:, idx]
        grid[y] = {m: float((slc >= m * starting_mv).mean()) for m in multiples}
    return {"years": years, "multiples": multiples, "grid": grid}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def fan_chart(port, title=""):
    pcts  = [10, 25, 50, 75, 90]
    bands = np.percentile(port, pcts, axis=0)
    months = np.arange(port.shape[1])
    with plt.rc_context(_CHART_RC):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.fill_between(months, bands[0], bands[4], alpha=0.12, color=NAVY, label="P10–P90")
        ax.fill_between(months, bands[1], bands[3], alpha=0.25, color=NAVY, label="P25–P75")
        ax.plot(months, bands[2], color=GOLD, lw=2.5, label="Median", zorder=5)
        ax.plot(months, bands[0], color=NAVY, lw=0.9, alpha=0.45, linestyle="--")
        ax.plot(months, bands[4], color=NAVY, lw=0.9, alpha=0.45, linestyle="--")
        ax.set_xlabel("Month")
        ax.set_ylabel("Portfolio value")
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold", color=NAVY, pad=10)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.spines["left"].set_color("#dde3ee")
        ax.spines["bottom"].set_color("#dde3ee")
        plt.tight_layout()
    return fig


def hist_chart(port, title="", capital: float | None = None):
    end = port[:, -1]
    p10, p50, p90 = np.percentile(end, [10, 50, 90])
    with plt.rc_context(_CHART_RC):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.hist(end, bins=80, color=NAVY, alpha=0.70, edgecolor="white", linewidth=0.3)
        if capital is not None:
            ax.axvline(capital, color="#718096", lw=1.2, linestyle=":",
                       label=f"Capital  ${capital:,.0f}", zorder=4)
        ax.axvline(p10, color=RED,       lw=2,   label=f"P10  ${p10:,.0f}", zorder=5)
        ax.axvline(p50, color=GOLD,      lw=2.5, label=f"P50  ${p50:,.0f}", zorder=5)
        ax.axvline(p90, color="#27ae60", lw=2,   label=f"P90  ${p90:,.0f}", zorder=5)
        ax.set_xlabel("Terminal value")
        ax.set_ylabel("Paths")
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold", color=NAVY, pad=10)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.spines["left"].set_color("#dde3ee")
        ax.spines["bottom"].set_color("#dde3ee")
        plt.tight_layout()
    return fig


def allocation_donut(enriched: pd.DataFrame) -> plt.Figure | None:
    d = enriched.dropna(subset=["mv"]).copy()
    d = d[d["mv"] > 0].sort_values("mv", ascending=False)
    if d.empty:
        return None

    MAX_SLICES = 9
    if len(d) > MAX_SLICES:
        top = d.iloc[:MAX_SLICES - 1].copy()
        other_mv = d.iloc[MAX_SLICES - 1:]["mv"].sum()
        other_w  = d.iloc[MAX_SLICES - 1:]["weight"].sum()
        other_row = pd.DataFrame([{"ticker": "Other", "mv": other_mv, "weight": other_w}])
        d = pd.concat([top[["ticker", "mv", "weight"]], other_row], ignore_index=True)

    weights = d["weight"].values
    labels  = d["ticker"].values

    palette = [
        "#1e3a5f", "#2d5282", "#4a7fb5", "#6499cc", "#8ab4d9",
        "#b8860b", "#d4a017", "#f0c040", "#f5d060", "#fae090",
    ]
    colors = palette[:len(weights)]

    with plt.rc_context(_CHART_RC):
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        fig.patch.set_facecolor("white")
        wedges, _ = ax.pie(
            weights, labels=None, colors=colors, startangle=90,
            wedgeprops=dict(width=0.54, edgecolor="white", linewidth=2.5),
        )
        legend_labels = [f"{t}  {w*100:.1f}%" for t, w in zip(labels, weights)]
        ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(0.92, 0.5),
                  fontsize=8, frameon=False)
        ax.set_title("Allocation", fontsize=10, fontweight="bold", color=NAVY, pad=8)
        plt.tight_layout()
    return fig


def factor_exposure_bar(enriched: pd.DataFrame) -> plt.Figure | None:
    d = enriched.dropna(subset=["weight"]).copy()
    if d.empty or "f_market" not in d.columns:
        return None

    d["f_idio"] = (1 - d["f_market"] - d["f_ai"] - d["f_power"]).clip(0, 1)
    factor_map = {
        "Market":        ("f_market", "#2d5282"),
        "AI / Tech":     ("f_ai",     "#b8860b"),
        "Power":         ("f_power",  "#27ae60"),
        "Idiosyncratic": ("f_idio",   "#94a3b8"),
    }

    labels, vals, colors = [], [], []
    for fname, (col, color) in factor_map.items():
        if col in d.columns:
            v = float((d[col] * d["weight"]).sum())
            labels.append(fname)
            vals.append(v)
            colors.append(color)

    if not vals:
        return None

    with plt.rc_context(_CHART_RC):
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        bars = ax.barh(labels, vals, color=colors, height=0.48, edgecolor="none")
        for bar, v in zip(bars, vals):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v*100:.0f}%", va="center", fontsize=9, color="#4a5568", fontweight="600")
        ax.set_xlim(0, max(vals) * 1.28 if vals else 1)
        ax.set_xlabel("Weighted exposure")
        ax.set_title("Factor exposure", fontsize=10, fontweight="bold", color=NAVY, pad=8)
        ax.spines["left"].set_color("#dde3ee")
        ax.spines["bottom"].set_color("#dde3ee")
        plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Historical analytics helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_prices(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    """Daily adjusted closes; separate cache from fetch_prices (different period/interval)."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            list(tickers), period=period, interval="1d",
            auto_adjust=True, progress=False, threads=True, group_by="ticker",
        )
        out = {}
        for t in tickers:
            try:
                s = raw["Close"].dropna() if len(tickers) == 1 else raw[t]["Close"].dropna()
                out[t] = s
            except Exception:
                pass
        return pd.DataFrame(out).dropna(how="all") if out else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


FACTOR_PROXIES: dict[str, str] = {"market": "SPY", "ai": "SMH", "power": "URA"}


@st.cache_data(ttl=3600, show_spinner=False)
def estimate_risk_model(tickers: tuple[str, ...], period: str = "2y",
                        min_obs: int = 60) -> dict:
    """Per-ticker σ + factor loadings (variance fractions) from historical returns.

    Method: Gram-Schmidt orthogonalize factor proxies (SPY → SMH⊥SPY → URA⊥{SPY,SMH}),
    then OLS each ticker's daily log returns against orthogonalized factors. Variance
    fractions sum to ≤ 1 (the remainder is idiosyncratic / model R²).

    Returns: {ticker: {"ok": bool, "sigma": float, "f_market": float, "f_ai": float,
                       "f_power": float, "n_obs": int, "r2": float}}
    """
    tickers = tuple(t for t in tickers if t != "CASH")
    if not tickers:
        return {}
    proxies = list(FACTOR_PROXIES.values())
    all_tk = tuple(sorted(set(list(tickers) + proxies)))
    prices = fetch_historical_prices(all_tk, period=period)
    if prices.empty or any(p not in prices.columns for p in proxies):
        return {t: {"ok": False, "n_obs": 0, "reason": "no proxy data"} for t in tickers}

    rets = np.log(prices / prices.shift(1)).dropna(how="all")

    # Gram-Schmidt orthogonalization
    df_m = rets[["SPY"]].dropna()
    df_ai = pd.concat([rets["SMH"], rets["SPY"]], axis=1, join="inner").dropna()
    df_ai.columns = ["smh", "m"]
    if len(df_ai) < min_obs:
        return {t: {"ok": False, "n_obs": len(df_ai), "reason": "proxy history too short"}
                for t in tickers}
    var_m = float(np.var(df_ai["m"], ddof=1))
    beta_ai_m = (np.cov(df_ai["smh"], df_ai["m"], ddof=1)[0, 1] / var_m) if var_m > 0 else 0
    F_AI = df_ai["smh"] - beta_ai_m * df_ai["m"]

    df_pw = pd.concat([rets["URA"], df_ai["m"], F_AI], axis=1, join="inner").dropna()
    df_pw.columns = ["ura", "m", "ai"]
    if len(df_pw) < min_obs:
        return {t: {"ok": False, "n_obs": len(df_pw), "reason": "proxy history too short"}
                for t in tickers}
    X_pw = df_pw[["m", "ai"]].to_numpy()
    y_pw = df_pw["ura"].to_numpy()
    coefs_pw, *_ = np.linalg.lstsq(X_pw, y_pw, rcond=None)
    F_PW = pd.Series(y_pw - X_pw @ coefs_pw, index=df_pw.index)

    factors = pd.concat([df_pw["m"], df_pw["ai"], F_PW], axis=1, join="inner").dropna()
    factors.columns = ["m", "ai", "pw"]
    var_f = {k: float(np.var(factors[k], ddof=1)) for k in ("m", "ai", "pw")}

    out: dict = {}
    for t in tickers:
        if t not in rets.columns:
            out[t] = {"ok": False, "n_obs": 0, "reason": "no price history"}
            continue
        joined = pd.concat([rets[t], factors], axis=1, join="inner").dropna()
        if len(joined) < min_obs:
            out[t] = {"ok": False, "n_obs": len(joined),
                      "reason": f"only {len(joined)} obs (need {min_obs})"}
            continue
        joined.columns = ["r", "m", "ai", "pw"]
        var_r = float(np.var(joined["r"], ddof=1))
        if var_r <= 0:
            out[t] = {"ok": False, "n_obs": len(joined), "reason": "zero variance"}
            continue
        X = joined[["m", "ai", "pw"]].to_numpy()
        y = joined["r"].to_numpy()
        betas, *_ = np.linalg.lstsq(X, y, rcond=None)
        f_m  = float(np.clip(betas[0] ** 2 * var_f["m"]  / var_r, 0, 1))
        f_ai = float(np.clip(betas[1] ** 2 * var_f["ai"] / var_r, 0, 1))
        f_pw = float(np.clip(betas[2] ** 2 * var_f["pw"] / var_r, 0, 1))
        s = f_m + f_ai + f_pw
        if s > 1.0:  # rare numerical edge case
            f_m, f_ai, f_pw = f_m / s, f_ai / s, f_pw / s
            s = 1.0
        out[t] = {
            "ok": True,
            "sigma":    float(joined["r"].std() * np.sqrt(252)),
            "f_market": f_m,
            "f_ai":     f_ai,
            "f_power":  f_pw,
            "n_obs":    int(len(joined)),
            "r2":       float(s),
        }
    return out


def apply_historical_overrides(df: pd.DataFrame, estimates: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Override σ + factor loadings on `df` with historical estimates where available.

    Returns (df_overridden, overridden_tickers, fallback_tickers).
    """
    if df.empty or not estimates:
        return df, [], list(df["ticker"].astype(str)) if not df.empty else []
    out = df.copy()
    overridden, fallbacks = [], []
    for i, row in out.iterrows():
        t = str(row["ticker"]).upper()
        est = estimates.get(t, {})
        if est.get("ok"):
            out.at[i, "sigma"]    = est["sigma"]
            out.at[i, "f_market"] = est["f_market"]
            out.at[i, "f_ai"]     = est["f_ai"]
            out.at[i, "f_power"]  = est["f_power"]
            overridden.append(t)
        else:
            fallbacks.append(t)
    return out, overridden, fallbacks


def build_portfolio_history(price_df: pd.DataFrame, shares_map: dict[str, float]) -> pd.Series:
    """Reconstructs daily portfolio value using current shares (held-throughout assumption)."""
    cols = [t for t in shares_map if t in price_df.columns and shares_map[t] > 0]
    if not cols:
        return pd.Series(dtype=float)
    sub = price_df[cols].dropna(how="all")
    series = sum(sub[t].ffill() * shares_map[t] for t in cols)
    return series.dropna()


def compute_performance_metrics(port_series: pd.Series, spx_series: pd.Series) -> dict:
    port = port_series.dropna()
    if len(port) < 5:
        return {}
    daily_ret = port.pct_change().dropna()
    total_ret = port.iloc[-1] / port.iloc[0] - 1
    n_days    = len(port)
    ann_ret   = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol   = daily_ret.std() * np.sqrt(252)
    rf_daily  = 0.045 / 252
    sharpe    = ((daily_ret.mean() - rf_daily) / daily_ret.std() * np.sqrt(252)
                 if daily_ret.std() > 0 else 0.0)
    max_dd    = (port / port.cummax() - 1).min()

    spx     = spx_series.dropna()
    aligned = pd.concat([daily_ret, spx.pct_change().dropna()], axis=1, join="inner").dropna()
    aligned.columns = ["port", "spx"]
    spx_var = np.var(aligned["spx"], ddof=1)
    beta = (np.cov(aligned["port"], aligned["spx"], ddof=1)[0, 1] / spx_var
            if len(aligned) > 5 and spx_var > 0 else float("nan"))

    return {"total_ret": total_ret, "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "max_dd": max_dd, "beta": beta}


def historical_chart(port_series: pd.Series, spx_series: pd.Series, title: str = "") -> plt.Figure | None:
    """Portfolio vs SPY indexed to 100 at start of period."""
    p = port_series.dropna()
    s = spx_series.reindex(p.index, method="ffill").dropna()
    common = p.index.intersection(s.index)
    if common.empty:
        return None
    p_norm = p.loc[common] / p.loc[common].iloc[0] * 100
    s_norm = s.loc[common] / s.loc[common].iloc[0] * 100
    with plt.rc_context(_CHART_RC):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(p_norm.index, p_norm.values, color=NAVY, lw=2.2, label="Portfolio")
        ax.plot(s_norm.index, s_norm.values, color="#94a3b8", lw=1.5,
                linestyle="--", label="SPY")
        ax.set_ylabel("Indexed (start = 100)")
        if title:
            ax.set_title(title, fontsize=10, fontweight="bold", color=NAVY, pad=10)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.spines["left"].set_color("#dde3ee")
        ax.spines["bottom"].set_color("#dde3ee")
        plt.tight_layout()
    return fig


def rolling_vol_chart(port_series: pd.Series, window: int = 90) -> plt.Figure:
    """Rolling annualized volatility chart."""
    rolling_vol = port_series.pct_change().dropna().rolling(window).std().dropna() * np.sqrt(252) * 100
    with plt.rc_context(_CHART_RC):
        fig, ax = plt.subplots(figsize=(9, 3.0))
        ax.plot(rolling_vol.index, rolling_vol.values, color=GOLD, lw=1.8)
        ax.fill_between(rolling_vol.index, 0, rolling_vol.values, alpha=0.15, color=GOLD)
        ax.set_ylabel("Annualized vol (%)")
        ax.set_title(f"Rolling {window}-day volatility", fontsize=10,
                     fontweight="bold", color=NAVY, pad=8)
        ax.spines["left"].set_color("#dde3ee")
        ax.spines["bottom"].set_color("#dde3ee")
        plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Factor correlation matrix
# ---------------------------------------------------------------------------

def compute_factor_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Implied pairwise correlation from the systematic factor loadings.
    Off-diagonal: ρᵢⱼ = Σₖ √(fᵢₖ · fⱼₖ). Diagonal forced to 1.0 (a security is
    perfectly correlated with itself; the off-diagonals reflect only the shared
    systematic part of variance, idiosyncratic risk is uncorrelated by construction)."""
    d = df[df["source"] != "missing"].copy() if "source" in df.columns else df.copy()
    if d.empty:
        return pd.DataFrame()
    tickers = d["ticker"].tolist()
    n = len(tickers)
    mat = np.zeros((n, n))
    rows = d[["f_market", "f_ai", "f_power"]].to_numpy()
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
            else:
                mat[i, j] = sum(
                    (rows[i, k] * rows[j, k]) ** 0.5
                    for k in range(3)
                )
    return pd.DataFrame(mat, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Advisor summary card
# ---------------------------------------------------------------------------

def advisor_summary_html(name: str, stats: dict, horizon_y: int,
                         regime_name: str, starting_mv: float) -> str:
    prob_beat  = stats["prob_beat_spx"] * 100
    prob_loss  = stats["prob_loss"] * 100
    prob_dbl   = stats["prob_double"] * 100
    dd_bad     = abs(stats["dd_p10"]) * 100
    years_from_now = datetime.now().year + horizon_y
    return f"""
<div class="advisor-card">
  <h4>📊 {horizon_y}-Year Outlook — {name} &nbsp;·&nbsp; {regime_name}</h4>
  Starting from <strong>${starting_mv:,.0f}</strong>, your portfolio's most likely outcome is
  <strong>${stats['median']:,.0f}</strong> (median) by {years_from_now}.<br>
  In a great scenario (top 10%), you could reach <strong>${stats['p90']:,.0f}</strong>.<br>
  Chance of beating the S&amp;P 500: <strong>{prob_beat:.0f}%</strong> &nbsp;·&nbsp;
  Chance of losing money: <strong>{prob_loss:.1f}%</strong> &nbsp;·&nbsp;
  Chance of doubling: <strong>{prob_dbl:.0f}%</strong><br>
  In a bad drawdown year, expect up to <strong>−{dd_bad:.0f}%</strong> peak-to-trough.
</div>
"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="sb-name">Portfolio Lab</div>
  <div class="sb-sub">Simulation Parameters</div>
</div>
""", unsafe_allow_html=True)

n_paths   = st.sidebar.number_input("Simulation paths", 500, 50_000, 10_000, 500)
horizon_y = st.sidebar.slider("Horizon (years)", 1, 20, 5)
n_months  = horizon_y * 12

st.sidebar.markdown('<div class="sidebar-section"><span>Cash Flows</span></div>', unsafe_allow_html=True)
lump_sum = st.sidebar.number_input("Lump sum ($)", 0.0, 1_000_000.0, 0.0, 100.0)
lump_mo  = st.sidebar.slider("Lump sum month", 0, n_months, min(1, n_months))
monthly  = st.sidebar.number_input("Monthly contribution ($)", 0.0, 100_000.0, 0.0, 50.0)

seed    = st.sidebar.number_input("Random seed", 0, 99999, 42)
n_books = st.sidebar.radio("Portfolios to compare", [1, 2, 3], horizontal=True)

st.sidebar.markdown('<div class="sidebar-section"><span>Risk Model</span></div>', unsafe_allow_html=True)
use_hist_risk = st.sidebar.checkbox(
    "Use historical estimates (2y)",
    value=False,
    help="Override σ and factor loadings with values estimated from 2 years of daily "
         "returns. Factors are SPY (market), SMH (AI/semis), URA (power) — "
         "Gram-Schmidt orthogonalized. Tickers with <60 trading days fall back to defaults."
)
hist_period = st.sidebar.selectbox(
    "History window", ["1y", "2y", "3y", "5y"], index=1,
    disabled=not use_hist_risk,
    help="Lookback window for vol & factor regression."
)

if st.sidebar.button("Force price refresh"):
    fetch_prices.clear()
    fetch_historical_prices.clear()
    estimate_risk_model.clear()
    st.session_state.pop("anchor_mv", None)
    st.cache_data.clear()  # also wipes news cache on the companion page
    st.rerun()

with st.sidebar.expander("Price diagnostics", expanded=False):
    _pd = st.session_state.get("_price_data", {})
    if _pd:
        st.dataframe(
            pd.DataFrame([
                {"Ticker": t, "Source": v["source"],
                 "Last fetch": v["ts"] or "—", "Retried": "yes" if v["retried"] else "no"}
                for t, v in _pd.items()
            ]),
            hide_index=True, use_container_width=True,
        )
    else:
        st.caption("Prices not yet loaded.")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "books" not in st.session_state:
    a = load_default_holdings()
    st.session_state.books = {
        "A — Current":  a.copy(),
        "B — Variant":  a.copy(),
        "C — Variant":  a.copy(),
    }

if "regime_name" not in st.session_state:
    st.session_state.regime_name = "Base case"

if "active_preset" not in st.session_state:
    st.session_state.active_preset = "My Portfolio"

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

def _latest_price_ts() -> str:
    """Most recent fetch timestamp from the on-disk price cache.
    Falls back to '—' if the cache is empty/missing."""
    try:
        cache = _load_price_cache()
        if not cache:
            return "—"
        latest = max(
            (datetime.fromisoformat(v["ts"]) for v in cache.values() if v.get("ts")),
            default=None,
        )
        if latest is None:
            return "—"
        delta = datetime.now() - latest
        if delta.total_seconds() < 90:
            return f"just now ({latest.strftime('%H:%M:%S')})"
        if delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() // 60)}m ago ({latest.strftime('%H:%M:%S')})"
        if delta.days < 1:
            return f"{int(delta.total_seconds() // 3600)}h ago"
        return f"{delta.days}d ago"
    except Exception:
        return "—"

_mv_cached  = st.session_state.get("anchor_mv")
_mv_str     = f"${_mv_cached:,.0f}" if _mv_cached else "—"
_n_hold     = len(st.session_state.books.get("A — Current", pd.DataFrame()))
_n_hold_str = str(_n_hold) if _n_hold else "—"
st.markdown(f"""
<div class="hero">
  <div class="hero-left">
    <div class="hero-eyebrow">Portfolio Analytics Platform</div>
    <h1>Portfolio Lab</h1>
    <p class="hero-sub">Factor-model Monte Carlo simulator with 2-state Markov regime-switching
    correlations, live prices, and multi-benchmark analytics against SPY, SMH, and URA.</p>
    <div class="hero-pills">
      <span class="hero-pill">3-Factor GBM</span>
      <span class="hero-pill">Regime Switching</span>
      <span class="hero-pill">SPY · SMH · URA</span>
      <span class="hero-pill live">Prices {_latest_price_ts()}</span>
    </div>
  </div>
  <div class="hero-right">
    <div class="hero-stat">
      <div class="hs-val">{_mv_str}</div>
      <div class="hs-lbl">Portfolio MV</div>
    </div>
    <div class="hero-stat">
      <div class="hs-val">{_n_hold_str}</div>
      <div class="hs-lbl">Positions</div>
    </div>
    <div class="hero-stat">
      <div class="hs-val">{horizon_y}y</div>
      <div class="hs-lbl">Horizon</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

price_warn = st.empty()

KNOWN_TICKERS = sorted(DEFAULT_RISK_MODEL.keys())

# ---------------------------------------------------------------------------
# Preset quick-load bar
# ---------------------------------------------------------------------------

_section_label("Portfolio preset")
preset_cols = st.columns(len(PRESETS))
for col, preset_name in zip(preset_cols, PRESETS):
    meta = _PRESET_META[preset_name]
    is_active = st.session_state.active_preset == preset_name
    with col:
        st.markdown(
            f'<div class="preset-card{"  active" if is_active else ""}">'
            f'<div class="pc-top"><span class="pc-icon">{meta["icon"]}</span>'
            f'<span class="pc-badge">Active</span></div>'
            f'<div class="pc-name">{preset_name}</div>'
            f'<div class="pc-desc">{meta["desc"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "✓ Loaded" if is_active else "Load →",
            key=f"preset_{preset_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            preset_df = build_preset(preset_name)
            st.session_state.books["A — Current"] = preset_df
            st.session_state.active_preset = preset_name
            known = [t for t in preset_df["ticker"].tolist() if t in KNOWN_TICKERS]
            st.session_state["multisel_A — Current"] = known
            st.rerun()

_divider()

# ---------------------------------------------------------------------------
# Scenario buttons
# ---------------------------------------------------------------------------

_section_label("Market scenario")
scen_cols = st.columns(len(_SCENARIO_META))
for col, (key, meta) in zip(scen_cols, _SCENARIO_META.items()):
    is_active = st.session_state.regime_name == key
    with col:
        badge = '<span class="sc-badge">Active</span>' if is_active else ''
        st.markdown(
            f'<div class="scenario-card{"  active" if is_active else ""}">'
            f'<div class="sc-header"><span class="sc-icon">{meta["icon"]}</span>'
            f'<span class="sc-name">{meta["label"]}</span>{badge}</div>'
            f'<div class="sc-desc">{meta["desc"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "✓ Active" if is_active else "Apply",
            key=f"scen_{key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            disabled=is_active,
        ):
            st.session_state.regime_name = key
            st.rerun()

regime = dict(REGIMES[st.session_state.regime_name])
if st.session_state.regime_name == "Custom":
    c1, c2, c3 = st.columns(3)
    with c1:
        regime["mu_shift"] = st.slider("μ shift (annual)", -0.30, 0.30, 0.00, 0.01)
    with c2:
        regime["ai_shift"] = st.slider("AI factor μ shift", -0.40, 0.40, 0.00, 0.01)
    with c3:
        regime["vol_mult"] = st.slider("Vol multiplier", 0.5, 3.0, 1.0, 0.05)

_divider()

# ---------------------------------------------------------------------------
# Known tickers for multiselect suggestions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main — portfolio tabs
# ---------------------------------------------------------------------------

book_names = list(st.session_state.books.keys())[:n_books]
tabs = st.tabs(book_names + ["What-if", "Analytics", "Factor diagnostics"])
whatif_tab    = tabs[-3]
analytics_tab = tabs[-2]
factor_tab    = tabs[-1]

results = {}
holdings_dfs = {}

# Pre-compute historical risk model once across all books (shared cache key)
risk_estimates: dict = {}
if use_hist_risk:
    all_book_tickers = sorted({
        str(t).upper()
        for nm in book_names
        for t in st.session_state.books[nm]["ticker"].dropna().tolist()
    })
    if all_book_tickers:
        with st.spinner(f"Estimating risk model from {hist_period} of history…"):
            risk_estimates = estimate_risk_model(tuple(all_book_tickers), period=hist_period)

for tab, name in zip(tabs, book_names):
    with tab:

        # --- Preset quick-load (variant books only) -----------------------
        if name != "A — Current":
            _section_label(f"Load preset into {name}")
            active_key = f"active_preset_{name}"
            if active_key not in st.session_state:
                st.session_state[active_key] = None
            p_cols = st.columns(len(PRESETS))
            for p_col, preset_name in zip(p_cols, PRESETS):
                meta = _PRESET_META[preset_name]
                is_active = st.session_state[active_key] == preset_name
                with p_col:
                    st.markdown(
                        f'<div class="preset-card{"  active" if is_active else ""}">'
                        f'<div class="pc-top"><span class="pc-icon">{meta["icon"]}</span>'
                        f'<span class="pc-badge">Active</span></div>'
                        f'<div class="pc-name">{preset_name}</div>'
                        f'<div class="pc-desc">{meta["desc"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "✓ Loaded" if is_active else "Load →",
                        key=f"preset_{name}_{preset_name}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        preset_df = build_preset(preset_name)
                        st.session_state.books[name] = preset_df
                        st.session_state[active_key] = preset_name
                        known = [t for t in preset_df["ticker"].tolist() if t in KNOWN_TICKERS]
                        st.session_state[f"multisel_{name}"] = known
                        st.rerun()

        # --- Ticker management -------------------------------------------
        _section_label(f"Holdings — {name}")

        current_tickers = st.session_state.books[name]["ticker"].tolist()
        known_in_current = [t for t in current_tickers if t in KNOWN_TICKERS]
        custom_in_current = [t for t in current_tickers if t not in KNOWN_TICKERS]

        col_sel, col_custom = st.columns([3, 1])
        with col_sel:
            selected_known = st.multiselect(
                "Add / remove tickers",
                options=KNOWN_TICKERS,
                default=known_in_current,
                key=f"multisel_{name}",
                help="Select from known tickers. Use the field on the right for anything else.",
            )
        with col_custom:
            custom_input = st.text_input(
                "Custom ticker",
                value="",
                key=f"custom_ticker_{name}",
                placeholder="e.g. TSLA",
                help="Type any ticker symbol and press Enter to add.",
            )

        # Reconcile multiselect + custom input → full ticker list
        new_tickers = [t.upper() for t in selected_known]
        if custom_input.strip():
            for ct in custom_input.strip().upper().split():
                if ct and ct not in new_tickers:
                    new_tickers.append(ct)
        # Keep any custom tickers already in the book that aren't in the multiselect
        for ct in custom_in_current:
            if ct not in new_tickers:
                new_tickers.append(ct)

        # Rebuild full book from new ticker list, preserving existing rows
        existing_full = st.session_state.books[name]
        existing_map  = {r["ticker"]: r for r in existing_full.to_dict("records")}
        full_rows = []
        for t in new_tickers:
            if t in existing_map:
                full_rows.append(existing_map[t])
            else:
                full_rows.append(_row_from_ticker(t, 1.0))
        new_full_df = pd.DataFrame(full_rows) if full_rows else _empty_holdings()

        # --- Simple shares editor ----------------------------------------
        if not new_full_df.empty:
            simple_df = new_full_df[["ticker", "shares"]].copy()
            edited_simple = st.data_editor(
                simple_df,
                num_rows="dynamic",
                key=f"simple_editor_{name}",
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "shares": st.column_config.NumberColumn("Shares", format="%.4f"),
                },
                use_container_width=True,
                hide_index=True,
            )
            merged = sync_risk_model(edited_simple, new_full_df)
        else:
            merged = _empty_holdings()

        st.session_state.books[name] = merged

        # --- Advanced risk model expander --------------------------------
        with st.expander("Advanced: risk model overrides (μ, σ, factor loadings)", expanded=False):
            st.caption("Edit expected return (μ), volatility (σ), and factor loadings. "
                       "f_market + f_ai + f_power ≤ 1; remainder is idiosyncratic.")
            adv_edited = st.data_editor(
                merged,
                num_rows="dynamic",
                key=f"adv_editor_{name}",
                column_config={
                    "ticker":   st.column_config.TextColumn("Ticker", width="small"),
                    "shares":   st.column_config.NumberColumn("Shares", format="%.4f"),
                    "mu":       st.column_config.NumberColumn("μ (annual)", format="%.3f", min_value=-0.5, max_value=0.5),
                    "sigma":    st.column_config.NumberColumn("σ (annual)", format="%.3f", min_value=0.0, max_value=2.0),
                    "f_market": st.column_config.NumberColumn("f_market", format="%.2f", min_value=0.0, max_value=1.0),
                    "f_ai":     st.column_config.NumberColumn("f_ai", format="%.2f", min_value=0.0, max_value=1.0),
                    "f_power":  st.column_config.NumberColumn("f_power", format="%.2f", min_value=0.0, max_value=1.0),
                },
                use_container_width=True,
            )
            if not adv_edited.equals(merged):
                st.session_state.books[name] = adv_edited
                merged = adv_edited

        # --- Apply historical overrides BEFORE enrichment so downstream uses them
        hist_overridden, hist_fallbacks = [], []
        merged_hand = merged.copy()  # preserve hand-set values for the diff display
        if use_hist_risk and risk_estimates:
            merged, hist_overridden, hist_fallbacks = apply_historical_overrides(
                merged, risk_estimates
            )

        # --- Live snapshot -----------------------------------------------
        enriched = enrich_with_prices(merged)

        if use_hist_risk and risk_estimates:
            if hist_overridden:
                detail = " · ".join(
                    f"{t} (n={risk_estimates[t]['n_obs']}, R²={risk_estimates[t]['r2']*100:.0f}%)"
                    for t in hist_overridden[:6]
                )
                more = f" +{len(hist_overridden)-6} more" if len(hist_overridden) > 6 else ""
                st.info(
                    f"📊 **Historical risk model active** ({hist_period}): "
                    f"{len(hist_overridden)}/{len(hist_overridden)+len(hist_fallbacks)} "
                    f"tickers overridden with regression-derived σ and factor loadings.\n\n"
                    f"_{detail}{more}_"
                )

                # Side-by-side: hand-set defaults → historical estimates
                hand_map = {r["ticker"]: r for r in merged_hand.to_dict("records")}
                rows = []
                for t in hist_overridden:
                    h = hand_map.get(t, {})
                    e = risk_estimates[t]
                    rows.append({
                        "Ticker":    t,
                        "σ default": f"{h.get('sigma', 0)*100:5.1f}%",
                        "σ hist":    f"{e['sigma']*100:5.1f}%",
                        "Δσ":        f"{(e['sigma'] - h.get('sigma', 0))*100:+.1f}pp",
                        "f_mkt def": f"{h.get('f_market', 0):.2f}",
                        "f_mkt hist":f"{e['f_market']:.2f}",
                        "f_ai def":  f"{h.get('f_ai', 0):.2f}",
                        "f_ai hist": f"{e['f_ai']:.2f}",
                        "f_pwr def": f"{h.get('f_power', 0):.2f}",
                        "f_pwr hist":f"{e['f_power']:.2f}",
                        "R²":        f"{e['r2']*100:.0f}%",
                        "n_obs":     e["n_obs"],
                    })
                with st.expander(
                    f"📈 Effective risk model (defaults vs historical) — {len(hist_overridden)} tickers",
                    expanded=False
                ):
                    st.caption(
                        "Side-by-side: your hand-set defaults vs the values the simulation "
                        "is actually using (regression-derived from the selected window). "
                        "These are the effective inputs flowing into the MC engine and the PDF."
                    )
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            if hist_fallbacks:
                fb_reasons = ", ".join(
                    f"**{t}** ({risk_estimates.get(t, {}).get('reason', 'no data')})"
                    for t in hist_fallbacks
                )
                st.warning(
                    f"⚠️ Insufficient history — using hand-set defaults for: {fb_reasons}"
                )

        holdings_dfs[name] = enriched
        st.session_state[f"enriched_{name}"] = enriched

        missing_tickers = detect_price_failures(enriched)
        if missing_tickers:
            price_warn.warning(
                f"⚠️ No price found for: **{', '.join(missing_tickers)}** — "
                "check for delisting or typo. These positions are excluded from MV and sim."
            )

        if not enriched.empty:
            display = enriched.copy()
            display["price"]  = display["price"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
            display["mv"]     = display["mv"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
            display["weight"] = display["weight"].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
            total_mv_display = enriched["mv"].sum()
            col_snap, col_mv = st.columns([3, 1])
            with col_snap:
                st.dataframe(
                    display[["ticker", "price", "mv", "weight", "source"]],
                    use_container_width=True, hide_index=True,
                )
            with col_mv:
                st.metric("Total market value", f"${total_mv_display:,.2f}")
            total_mv = total_mv_display
        else:
            st.info("Add at least one ticker to begin.")
            total_mv = 0.0

        # --- Portfolio composition charts --------------------------------
        if total_mv > 0:
            _section_label("Portfolio composition")
            ch1, ch2 = st.columns(2)
            with ch1:
                fig_donut = allocation_donut(enriched)
                if fig_donut:
                    st.pyplot(fig_donut)
                    plt.close(fig_donut)
            with ch2:
                fig_factor = factor_exposure_bar(enriched)
                if fig_factor:
                    st.pyplot(fig_factor)
                    plt.close(fig_factor)

        # --- Simulation --------------------------------------------------
        if total_mv > 0:
            enriched_sim = enriched.dropna(subset=["price"]).copy()
            port, spx, smh, ura = simulate(
                enriched_sim, int(n_paths), n_months, total_mv,
                lump_sum, lump_mo, monthly, regime, seed=int(seed),
            )
            deposits = total_mv + lump_sum + monthly * n_months
            stats    = summarize(port, spx, smh, ura, deposits)
            results[name] = (port, spx, smh, ura, stats)

            # Advisor summary card
            st.markdown(
                advisor_summary_html(name, stats, horizon_y,
                                     st.session_state.regime_name, total_mv),
                unsafe_allow_html=True,
            )

            # KPI cards — row 1: outcomes
            _section_label("Simulation results")
            _kpi_row(
                _kpi("Median End",  f"${stats['median']:,.0f}", "blue"),
                _kpi("P10 · Bad",   f"${stats['p10']:,.0f}",   "red"),
                _kpi("P90 · Great", f"${stats['p90']:,.0f}",   "green"),
                _kpi("Median DD",   f"{stats['dd_p50']*100:.1f}%", "red"),
            )

            # KPI cards — row 2: benchmark beat probabilities
            _kpi_row(
                _kpi("Beat S&P 500", f"{stats['prob_beat_spx']*100:.1f}%",
                     "green" if stats["prob_beat_spx"] > 0.5 else "red"),
                _kpi("Beat SMH",     f"{stats['prob_beat_smh']*100:.1f}%",
                     "green" if stats["prob_beat_smh"] > 0.5 else "red"),
                _kpi("Beat URA",     f"{stats['prob_beat_ura']*100:.1f}%",
                     "green" if stats["prob_beat_ura"] > 0.5 else "red"),
            )

            # KPI cards — row 3: outcome probabilities
            _kpi_row(
                _kpi("Prob Loss",   f"{stats['prob_loss']*100:.1f}%",   "red"),
                _kpi("Prob Double", f"{stats['prob_double']*100:.1f}%", "green"),
                _kpi("Prob Halve",  f"{stats['prob_halve']*100:.1f}%",  "red"),
            )

            # Probability ladder
            with st.expander("Probability ladder"):
                ladder = _probability_ladder(port, horizon_y, total_mv)
                _yrs = ladder["years"]
                _mul = ladder["multiples"]
                _grd = ladder["grid"]
                ladder_rows = {}
                for m in _mul:
                    lbl = f"{m}×  (≥ ${total_mv * m:,.0f})"
                    ladder_rows[lbl] = {f"Yr {y}": f"{_grd[y][m]*100:.0f}%" for y in _yrs}
                ladder_df = pd.DataFrame(ladder_rows).T

                def _ladder_color(val):
                    try:
                        p = float(str(val).strip("%")) / 100
                    except Exception:
                        return ""
                    if p >= 0.50:
                        return "background-color: #16a34a22; color: #15803d; font-weight:600"
                    if p >= 0.25:
                        return "background-color: #ca8a0422; color: #b45309"
                    return "background-color: #dc262622; color: #dc2626"

                st.dataframe(
                    ladder_df.style.applymap(_ladder_color),
                    use_container_width=True,
                )

            _section_label("Simulated paths")
            cc1, cc2 = st.columns(2)
            with cc1:
                fig_fan = fan_chart(port, f"{name}  ·  {horizon_y}-year fan")
                st.pyplot(fig_fan)
                plt.close(fig_fan)
            with cc2:
                fig_hist = hist_chart(port, f"{name}  ·  terminal distribution", capital=deposits)
                st.pyplot(fig_hist)
                plt.close(fig_hist)

            # PDF download
            _divider()
            risk_meta = {
                "use_hist": bool(use_hist_risk),
                "hist_period": hist_period if use_hist_risk else None,
                "n_overridden": len(hist_overridden),
                "n_fallback": len(hist_fallbacks),
                "fallback_tickers": hist_fallbacks,
            }
            pdf_bytes = _build_pdf_report(
                [{"name": name, "enriched_df": enriched,
                  "port": port, "spx": spx, "smh": smh, "ura": ura, "stats": stats}],
                horizon_y, st.session_state.regime_name,
                int(n_paths), lump_sum, monthly,
                risk_model_meta=risk_meta,
            )
            st.download_button(
                "📄 Download PDF report",
                data=pdf_bytes,
                file_name=f"portfolio_report_{name.replace(' — ', '_').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{name}",
            )

# ---------------------------------------------------------------------------
# What-if rebalance tab
# ---------------------------------------------------------------------------

with whatif_tab:
    _section_label("What-if rebalance")
    st.caption(
        "Drag weight sliders to test alternate allocations without touching your real shares. "
        "Composition, concentration, and factor exposure update instantly. "
        "Click **Run Monte Carlo** to re-simulate on the new weights, or **Apply** to push the "
        "what-if portfolio into Book B or Book C as actual share counts."
    )

    avail_books = [
        b for b in book_names
        if isinstance(st.session_state.get(f"enriched_{b}"), pd.DataFrame)
        and not st.session_state[f"enriched_{b}"].empty
    ]

    if not avail_books:
        st.info("Load and price at least one book first — open a book tab, then return here.")
    else:
        col_src, col_mode, col_reset = st.columns([2, 2, 1])
        with col_src:
            src = st.radio("Start from book", avail_books, horizontal=True, key="whatif_src")
        with col_mode:
            st.radio(
                "Slider mode", ["Pro-rata", "Free"],
                horizontal=True, key="whatif_mode",
                help=("Pro-rata: dragging one slider auto-scales the others to keep total at 100%. "
                      "Free: each slider independent — use Normalize button to rescale to 100%."),
            )
        with col_reset:
            st.write(" ")
            reset_clicked = st.button("↺ Reset", use_container_width=True, key="whatif_reset")

        src_df = st.session_state[f"enriched_{src}"].dropna(subset=["price"]).copy()
        src_df = src_df[src_df["price"] > 0].copy()
        src_tickers = src_df["ticker"].astype(str).tolist()
        src_weights = {t: float(w) for t, w in zip(src_df["ticker"], src_df["weight"])}
        src_prices  = {t: float(p) for t, p in zip(src_df["ticker"], src_df["price"])}
        src_mv      = float(src_df["mv"].sum())

        # Init or re-init what-if weights when source book or its tickers change.
        # Also handles deferred Normalize requested on the previous run (flag pattern
        # — slider session_state keys can only be written before the slider widgets
        # are instantiated below).
        init_signature = (src, tuple(src_tickers))
        normalize_pending = st.session_state.pop("whatif_normalize_pending", False)
        if reset_clicked or st.session_state.get("whatif_init_sig") != init_signature:
            st.session_state.whatif_weights = dict(src_weights)
            st.session_state["whatif_init_sig"] = init_signature
            for t, w in src_weights.items():
                st.session_state[f"wf_slider_{t}"] = w * 100.0
            st.session_state.pop("whatif_mc_result", None)
            if reset_clicked:
                st.rerun()
        elif normalize_pending:
            cur_pct = {t: float(st.session_state.get(f"wf_slider_{t}", 0.0))
                       for t in src_tickers}
            s = sum(cur_pct.values())
            if s > 0:
                for t, v in cur_pct.items():
                    norm_pct = (v / s) * 100.0
                    st.session_state[f"wf_slider_{t}"] = norm_pct
            st.session_state.whatif_weights = {
                t: float(st.session_state[f"wf_slider_{t}"]) / 100.0
                for t in src_tickers
            }
            st.session_state.pop("whatif_mc_result", None)

        # Slider on_change callback (handles pro-rata rebalance)
        def _on_wf_slider_change(ticker: str):
            new_pct = float(st.session_state.get(f"wf_slider_{ticker}", 0.0))
            new_w = max(0.0, min(1.0, new_pct / 100.0))
            current = dict(st.session_state.get("whatif_weights", {}))
            if st.session_state.get("whatif_mode") == "Pro-rata" and len(current) > 1:
                rebalanced = rebalance_pro_rata(current, ticker, new_w)
                st.session_state.whatif_weights = rebalanced
                for t, w in rebalanced.items():
                    if t != ticker:
                        st.session_state[f"wf_slider_{t}"] = w * 100.0
            else:
                current[ticker] = new_w
                st.session_state.whatif_weights = current
            # MC result becomes stale — clear so user re-runs
            st.session_state.pop("whatif_mc_result", None)

        # Sliders in 3-column grid, sorted by current weight (heaviest first)
        wf_w = st.session_state.whatif_weights
        sorted_tickers = sorted(src_tickers, key=lambda t: -src_weights.get(t, 0.0))

        st.markdown("**Target weights** — drag to test alternate allocations")
        n_cols = 3
        for i in range(0, len(sorted_tickers), n_cols):
            chunk = sorted_tickers[i:i + n_cols]
            cols = st.columns(n_cols)
            for col, t in zip(cols, chunk):
                with col:
                    cur_pct = src_weights.get(t, 0.0) * 100.0
                    st.slider(
                        f"{t}  ·  was {cur_pct:.1f}%",
                        min_value=0.0, max_value=100.0,
                        step=0.1, format="%.1f%%",
                        key=f"wf_slider_{t}",
                        on_change=_on_wf_slider_change, args=(t,),
                    )

        # Refresh whatif_weights from slider session state (pro-rata callback already
        # syncs them, but in Free mode the dict isn't updated until next interaction)
        wf_w = {
            t: float(st.session_state.get(f"wf_slider_{t}", src_weights.get(t, 0.0) * 100.0)) / 100.0
            for t in src_tickers
        }
        st.session_state.whatif_weights = wf_w
        total = sum(wf_w.values())

        # Total weight strip + Normalize
        col_total, col_norm = st.columns([3, 1])
        with col_total:
            ok = abs(total - 1.0) < 0.005
            color = "#27ae60" if ok else "#c0392b"
            tail = " ✓" if ok else " — click Normalize to rescale to 100%"
            st.markdown(
                f"<div style='font-size:1rem; padding:6px 0;'>"
                f"Total weight: <b style='color:{color}'>{total*100:.1f}%</b>{tail}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_norm:
            if st.button("⚖ Normalize to 100%", use_container_width=True,
                         key="whatif_normalize", disabled=(total <= 0)):
                st.session_state["whatif_normalize_pending"] = True
                st.rerun()

        # Normalized weights for downstream computations (so previews stay sensible
        # even if sliders don't sum to 100%)
        wf_norm = {t: (w / total) for t, w in wf_w.items()} if total > 0 else dict(wf_w)

        _divider()

        # ---- Concentration & factor exposure (instant, no MC) -----------
        cur_conc = concentration_metrics(src_weights)
        wf_conc  = concentration_metrics(wf_norm)
        cur_fac  = factor_exposure(src_df)
        wf_fac   = factor_exposure(src_df, weights_override=wf_norm)

        _section_label("Concentration profile  ·  current → what-if")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top-1 weight",
                  f"{wf_conc['top1']*100:.1f}%",
                  f"{(wf_conc['top1']-cur_conc['top1'])*100:+.1f}pp",
                  delta_color="inverse")
        c2.metric("Top-3 weight",
                  f"{wf_conc['top3']*100:.1f}%",
                  f"{(wf_conc['top3']-cur_conc['top3'])*100:+.1f}pp",
                  delta_color="inverse")
        c3.metric("HHI",
                  f"{wf_conc['hhi']:.3f}",
                  f"{wf_conc['hhi']-cur_conc['hhi']:+.3f}",
                  delta_color="inverse")
        c4.metric("Effective N",
                  f"{wf_conc['eff_n']:.1f}",
                  f"{wf_conc['eff_n']-cur_conc['eff_n']:+.1f}")

        _section_label("Factor exposure  ·  current → what-if")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Market β proxy",
                  f"{wf_fac['f_market']:.2f}",
                  f"{wf_fac['f_market']-cur_fac['f_market']:+.2f}")
        f2.metric("AI / semis",
                  f"{wf_fac['f_ai']:.2f}",
                  f"{wf_fac['f_ai']-cur_fac['f_ai']:+.2f}")
        f3.metric("Power / utilities",
                  f"{wf_fac['f_power']:.2f}",
                  f"{wf_fac['f_power']-cur_fac['f_power']:+.2f}")
        f4.metric("Idiosyncratic",
                  f"{wf_fac['f_idio']:.2f}",
                  f"{wf_fac['f_idio']-cur_fac['f_idio']:+.2f}",
                  delta_color="inverse")

        # ---- Composition donuts side-by-side ----------------------------
        wf_df = src_df.copy()
        wf_df["weight"] = wf_df["ticker"].map(wf_norm).fillna(0.0).astype(float)
        wf_df["mv"]     = wf_df["weight"] * src_mv

        _section_label("Composition  ·  current → what-if")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("<div style='text-align:center; font-weight:600; color:#374151;'>Current</div>",
                        unsafe_allow_html=True)
            fig = allocation_donut(src_df)
            if fig:
                st.pyplot(fig); plt.close(fig)
        with cc2:
            st.markdown("<div style='text-align:center; font-weight:600; color:#374151;'>What-if</div>",
                        unsafe_allow_html=True)
            fig = allocation_donut(wf_df)
            if fig:
                st.pyplot(fig); plt.close(fig)

        _divider()

        # ---- Run MC + Apply actions -------------------------------------
        col_run, col_apply_b, col_apply_c = st.columns([2, 1, 1])
        with col_run:
            run_mc = st.button(
                "⚡ Run Monte Carlo on what-if",
                use_container_width=True, type="primary", key="whatif_run_mc",
            )
        def _commit_whatif_to_book(dest: str):
            shares_map = weights_to_shares(wf_norm, src_prices, src_mv)
            wf_simple = pd.DataFrame({
                "ticker": list(shares_map.keys()),
                "shares": list(shares_map.values()),
            })
            merged = sync_risk_model(wf_simple, src_df)
            st.session_state.books[dest] = merged
            st.session_state[f"multisel_{dest}"] = [
                t for t in wf_simple["ticker"] if t in KNOWN_TICKERS
            ]
            st.success(f"✓ Applied to {dest}. Switch to that tab to view.")

        with col_apply_b:
            disabled_b = src == "B — Variant"
            help_b = ("Source book is Book B — switch source to apply." if disabled_b
                      else "Convert what-if weights to shares and write to Book B.")
            if st.button("Apply → Book B", use_container_width=True, key="whatif_apply_b",
                         disabled=disabled_b, help=help_b):
                _commit_whatif_to_book("B — Variant")
        with col_apply_c:
            disabled_c = src == "C — Variant"
            help_c = ("Source book is Book C — switch source to apply." if disabled_c
                      else "Convert what-if weights to shares and write to Book C.")
            if st.button("Apply → Book C", use_container_width=True, key="whatif_apply_c",
                         disabled=disabled_c, help=help_c):
                _commit_whatif_to_book("C — Variant")

        if run_mc:
            wf_sim_df = wf_df.dropna(subset=["price"]).copy()
            with st.spinner("Running MC on what-if weights…"):
                wf_port, wf_spx, wf_smh, wf_ura = simulate(
                    wf_sim_df, int(n_paths), n_months, src_mv,
                    lump_sum, lump_mo, monthly, regime, seed=int(seed),
                )
            if wf_port is None:
                st.warning("No paths simulated — check ticker prices.")
            else:
                wf_deposits = src_mv + lump_sum + monthly * n_months
                wf_stats    = summarize(wf_port, wf_spx, wf_smh, wf_ura, wf_deposits)
                st.session_state.whatif_mc_result = {
                    "src":       src,
                    "port":      wf_port,
                    "spx":       wf_spx,
                    "smh":       wf_smh,
                    "ura":       wf_ura,
                    "stats":     wf_stats,
                    "deposits":  wf_deposits,
                    "weights":   dict(wf_norm),
                    "n_paths":   int(n_paths),
                    "n_months":  n_months,
                    "regime":    st.session_state.regime_name,
                    "seed":      int(seed),
                }

        wf_mc = st.session_state.get("whatif_mc_result")
        if wf_mc and wf_mc["src"] == src:
            wf_stats = wf_mc["stats"]
            cur_results = results.get(src)
            _section_label("Monte Carlo results  ·  current vs what-if")
            st.caption(
                f"Both runs: {wf_mc['n_paths']:,} paths · "
                f"{wf_mc['n_months']//12}y horizon · "
                f"regime '{wf_mc['regime']}' · seed {wf_mc['seed']}."
            )

            if cur_results is not None:
                _, _, _, _, cur_stats = cur_results

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Median End",
                          f"${wf_stats['median']:,.0f}",
                          f"${wf_stats['median']-cur_stats['median']:+,.0f}")
                k2.metric("P10 · Bad",
                          f"${wf_stats['p10']:,.0f}",
                          f"${wf_stats['p10']-cur_stats['p10']:+,.0f}")
                k3.metric("P90 · Great",
                          f"${wf_stats['p90']:,.0f}",
                          f"${wf_stats['p90']-cur_stats['p90']:+,.0f}")
                k4.metric("Median Drawdown",
                          f"{wf_stats['dd_p50']*100:.1f}%",
                          f"{(wf_stats['dd_p50']-cur_stats['dd_p50'])*100:+.1f}pp",
                          delta_color="inverse")

                k5, k6, k7 = st.columns(3)
                k5.metric("Beat S&P 500",
                          f"{wf_stats['prob_beat_spx']*100:.1f}%",
                          f"{(wf_stats['prob_beat_spx']-cur_stats['prob_beat_spx'])*100:+.1f}pp")
                k6.metric("Beat SMH",
                          f"{wf_stats['prob_beat_smh']*100:.1f}%",
                          f"{(wf_stats['prob_beat_smh']-cur_stats['prob_beat_smh'])*100:+.1f}pp")
                k7.metric("Beat URA",
                          f"{wf_stats['prob_beat_ura']*100:.1f}%",
                          f"{(wf_stats['prob_beat_ura']-cur_stats['prob_beat_ura'])*100:+.1f}pp")

                k8, k9, k10 = st.columns(3)
                k8.metric("Prob Loss",
                          f"{wf_stats['prob_loss']*100:.1f}%",
                          f"{(wf_stats['prob_loss']-cur_stats['prob_loss'])*100:+.1f}pp",
                          delta_color="inverse")
                k9.metric("Prob Double",
                          f"{wf_stats['prob_double']*100:.1f}%",
                          f"{(wf_stats['prob_double']-cur_stats['prob_double'])*100:+.1f}pp")
                k10.metric("Prob Halve",
                           f"{wf_stats['prob_halve']*100:.1f}%",
                           f"{(wf_stats['prob_halve']-cur_stats['prob_halve'])*100:+.1f}pp",
                           delta_color="inverse")

                # Fan chart side by side
                _section_label("Fan charts  ·  current vs what-if")
                fc1, fc2 = st.columns(2)
                with fc1:
                    cur_port, _, _, _, _ = cur_results
                    fig = fan_chart(cur_port, f"Current  ·  {src}")
                    st.pyplot(fig); plt.close(fig)
                with fc2:
                    fig = fan_chart(wf_mc["port"], "What-if")
                    st.pyplot(fig); plt.close(fig)
            else:
                # No baseline run — show what-if alone
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Median End",      f"${wf_stats['median']:,.0f}")
                k2.metric("P10 · Bad",       f"${wf_stats['p10']:,.0f}")
                k3.metric("P90 · Great",     f"${wf_stats['p90']:,.0f}")
                k4.metric("Median Drawdown", f"{wf_stats['dd_p50']*100:.1f}%")
                fig = fan_chart(wf_mc["port"], "What-if")
                st.pyplot(fig); plt.close(fig)
                st.info(f"Add shares to '{src}' on its tab first to enable side-by-side comparison.")
        elif wf_mc and wf_mc["src"] != src:
            st.info("Source book changed — re-run Monte Carlo to refresh.")

# ---------------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------------

ANALYTICS_PERIODS = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y"}

with analytics_tab:
    _section_label("Historical analytics — Book A")
    st.caption(
        "Performance is reconstructed using *current* share counts held throughout the selected "
        "period. This does not reflect actual trade history."
    )

    enriched_a = st.session_state.get("enriched_A — Current")
    if enriched_a is None or enriched_a.empty:
        st.info("Load and price Book A first — open the 'A — Current' tab, then return here.")
    else:
        period_key = st.radio(
            "Time range", list(ANALYTICS_PERIODS.keys()),
            index=4, horizontal=True, key="ana_period",
        )
        period_str = ANALYTICS_PERIODS[period_key]

        valid      = enriched_a.dropna(subset=["price", "shares"])
        valid      = valid[(valid["shares"].astype(float) > 0) & (valid["ticker"] != "CASH")]
        tickers_h  = tuple(valid["ticker"].tolist())
        shares_map = dict(zip(valid["ticker"], valid["shares"].astype(float)))

        if not tickers_h:
            st.warning("No valid holdings with prices found in Book A.")
        else:
            fetch_tickers = tickers_h if "SPY" in tickers_h else tickers_h + ("SPY",)
            with st.spinner("Fetching historical prices…"):
                price_df = fetch_historical_prices(fetch_tickers, period_str)

            if price_df.empty:
                st.error("Could not fetch historical data. Try the Force price refresh button or check your connection.")
            else:
                port_series = build_portfolio_history(price_df, shares_map)
                spx_series  = price_df["SPY"].dropna() if "SPY" in price_df.columns else pd.Series(dtype=float)

                if port_series.empty:
                    st.warning("Insufficient price history to reconstruct portfolio — some tickers may be too new.")
                else:
                    metrics = compute_performance_metrics(port_series, spx_series)
                    if metrics:
                        ret_tone = "green" if metrics["total_ret"] >= 0 else "red"
                        ann_tone = "green" if metrics["ann_ret"]   >= 0 else "red"
                        beta_str = f"{metrics['beta']:.2f}" if not np.isnan(metrics["beta"]) else "—"
                        _kpi_row(
                            _kpi("Total Return",      f"{metrics['total_ret']*100:+.1f}%", ret_tone),
                            _kpi("Annualized Return", f"{metrics['ann_ret']*100:+.1f}%",   ann_tone),
                            _kpi("Sharpe Ratio",      f"{metrics['sharpe']:.2f}",          "blue"),
                            _kpi("Max Drawdown",      f"{metrics['max_dd']*100:.1f}%",     "red"),
                            _kpi("Beta to SPX",       beta_str,                            "gold"),
                            _kpi("Ann. Volatility",   f"{metrics['ann_vol']*100:.1f}%",    "blue"),
                        )

                    _section_label("Historical performance vs SPY")
                    fig_line = historical_chart(
                        port_series, spx_series,
                        title=f"Portfolio vs SPY — {period_key}",
                    )
                    if fig_line:
                        st.pyplot(fig_line)
                        plt.close(fig_line)

                    if len(port_series) >= 91:
                        _section_label("Rolling 90-day volatility")
                        fig_rvol = rolling_vol_chart(port_series)
                        st.pyplot(fig_rvol)
                        plt.close(fig_rvol)
                    else:
                        st.caption("Rolling vol requires 90+ trading days — select a longer time range.")

# ---------------------------------------------------------------------------
# Factor diagnostics tab
# ---------------------------------------------------------------------------

with factor_tab:
    _section_label("Implied factor correlations")
    sel = st.radio("Compute from book:", book_names, horizontal=True, key="hmap_book")
    enriched_sel = st.session_state.get(f"enriched_{sel}")
    if enriched_sel is None:
        st.info("Load and run the selected book first — prices need to resolve before correlations can be computed.")
    else:
        corr = compute_factor_correlations(enriched_sel)
        if corr.empty:
            st.warning("No resolved tickers to compute correlations — all prices are missing.")
        else:
            with plt.rc_context(_CHART_RC):
                fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.6), max(5, len(corr) * 0.55)))
                sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", center=0,
                            cmap="RdBu_r", linewidths=0.4, square=True, vmin=0, vmax=1)
                ax.set_title(f"Implied pairwise correlations — {sel}", fontsize=11,
                             fontweight="bold", color=NAVY, pad=10)
                plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            vals = corr.to_numpy()
            off_mask = ~np.eye(len(corr), dtype=bool)
            off_vals = vals[off_mask]
            max_idx  = np.unravel_index(np.argmax(vals * off_mask), vals.shape)
            max_pair = f"{corr.index[max_idx[0]]} / {corr.columns[max_idx[1]]}"
            max_val  = vals[max_idx]
            _kpi_row(
                _kpi("Mean off-diagonal", f"{off_vals.mean():.2f}", "blue"),
                _kpi("Highest pair",      f"{max_pair}",            "gold"),
                _kpi("Pairs > 0.5",       f"{int((off_vals > 0.5).sum())} of {len(off_vals)}", "red"),
            )

            st.caption(
                "Implied correlations from static factor model. "
                "Real correlations spike toward 1 in drawdowns — this understates tail risk."
            )

# ---------------------------------------------------------------------------
# Comparison panel
# ---------------------------------------------------------------------------

if len(results) > 1:
    _divider()
    _section_label("Side-by-side comparison")
    rows = []
    for name, (_p, _s, _sm, _ur, stats) in results.items():
        rows.append({
            "Portfolio":       name,
            "Median end":      f"${stats['median']:,.0f}",
            "P10":             f"${stats['p10']:,.0f}",
            "P90":             f"${stats['p90']:,.0f}",
            "Mean":            f"${stats['mean']:,.0f}",
            "Beat SPX":        f"{stats['prob_beat_spx']*100:.1f}%",
            "Beat SMH":        f"{stats['prob_beat_smh']*100:.1f}%",
            "Beat URA":        f"{stats['prob_beat_ura']*100:.1f}%",
            "Prob loss":       f"{stats['prob_loss']*100:.1f}%",
            "Prob double":     f"{stats['prob_double']*100:.1f}%",
            "Median DD":       f"{stats['dd_p50']*100:.1f}%",
            "Worst-decile DD": f"{stats['dd_p10']*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    books_payload = [
        {"name": n, "enriched_df": holdings_dfs[n],
         "port": p, "spx": s, "smh": sm, "ura": ur, "stats": st_}
        for n, (p, s, sm, ur, st_) in results.items()
    ]
    cmp_risk_meta = {
        "use_hist": bool(use_hist_risk),
        "hist_period": hist_period if use_hist_risk else None,
        "n_overridden": None,  # varies per book; approximate aggregate
        "n_fallback": None,
        "fallback_tickers": [],
    }
    cmp_pdf = _build_pdf_report(books_payload, horizon_y,
                                st.session_state.regime_name, int(n_paths),
                                lump_sum, monthly,
                                risk_model_meta=cmp_risk_meta)
    st.download_button(
        "📄 Download comparison PDF",
        data=cmp_pdf,
        file_name=f"portfolio_comparison_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        key="dl_comparison_pdf",
    )

# ---------------------------------------------------------------------------
# Save / load scenarios
# ---------------------------------------------------------------------------

_divider()
with st.expander("Save / load scenario snapshots"):
    cs1, cs2 = st.columns(2)
    with cs1:
        snap_name = st.text_input("Snapshot name",
                                   value=f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        if st.button("Save current snapshot"):
            existing = {}
            if SCENARIOS_FP.exists():
                existing = json.loads(SCENARIOS_FP.read_text())
            existing[snap_name] = {
                "ts":            datetime.now().isoformat(timespec="seconds"),
                "regime":        st.session_state.regime_name,
                "horizon_years": horizon_y,
                "lump_sum":      lump_sum,
                "monthly":       monthly,
                "books":         {n: df.to_dict(orient="records")
                                  for n, df in st.session_state.books.items()
                                  if n in book_names},
                "results":       {n: r[4] for n, r in results.items()},
            }
            SCENARIOS_FP.write_text(json.dumps(existing, indent=2, default=str))
            st.success(f"Saved → {SCENARIOS_FP.name}")
    with cs2:
        if SCENARIOS_FP.exists():
            snaps  = json.loads(SCENARIOS_FP.read_text())
            choice = st.selectbox("Load snapshot", [""] + sorted(snaps.keys(), reverse=True))
            if choice and st.button("Restore selected"):
                snap = snaps[choice]
                st.session_state.books = {n: pd.DataFrame(rows)
                                          for n, rows in snap["books"].items()}
                st.session_state.regime_name = snap.get("regime", "Base case")
                st.success(f"Restored {choice}. Rerunning…")
                st.rerun()
        else:
            st.caption("No snapshots saved yet.")

st.caption(
    "Engine: factor-model GBM (market + AI + power factors + idiosyncratic) with two-state "
    "Markov regime switching. Stress state (~15% of months) scales vol ×1.5 and shifts variance "
    "from idiosyncratic to systematic factors, spiking cross-asset correlations toward 1 as "
    "observed in historical drawdowns. Lognormal returns, no fat tails. "
    "Defaults loaded from holdings.json on first launch."
)
