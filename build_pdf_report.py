"""
build_pdf_report.py — Professional PDF generator for Portfolio Lab.

Public API:
    build_report(books, horizon_y, regime_name, n_paths,
                 lump_sum=0, monthly=0) -> bytes
"""

from __future__ import annotations

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image,
    NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

NAVY  = colors.HexColor("#1e3a5f")
GOLD  = colors.HexColor("#b8860b")
RED   = colors.HexColor("#c0392b")
AMBER = colors.HexColor("#e67e22")
LIGHT = colors.HexColor("#F7F7F7")
GHOST = colors.HexColor("#F0F4F8")
MID   = colors.HexColor("#D0D8E4")
MUTED = colors.HexColor("#888888")
WHITE = colors.white
BLACK = colors.HexColor("#1a1a1a")

SANS   = "Helvetica"
SANS_B = "Helvetica-Bold"
SERIF  = "Times-Roman"
SERIF_B = "Times-Bold"

# matplotlib equivalents
MPLNAVY = "#1e3a5f"
MPLGOLD = "#b8860b"
MPLRED  = "#c0392b"

PAGE_W, PAGE_H = LETTER
MARGIN    = 0.65 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

_base = getSampleStyleSheet()

def _style(name, **kw):
    return ParagraphStyle(name, parent=_base["Normal"], **kw)

SECTION_H   = _style("SECTION_H",   fontSize=12, textColor=WHITE,  fontName=SANS_B,  leading=15)
H3          = _style("H3",          fontSize=10, textColor=NAVY,   fontName=SANS_B,  leading=13,
                      spaceBefore=12, spaceAfter=4)
BODY        = _style("BODY",        fontSize=10, textColor=BLACK,  fontName=SERIF,   leading=15, spaceAfter=7)
BODY_B      = _style("BODY_B",      fontSize=10, textColor=BLACK,  fontName=SERIF_B, leading=15, spaceAfter=4)
SMALL       = _style("SMALL",       fontSize=8,  textColor=MUTED,  fontName=SANS,    leading=11, spaceAfter=4)
CAVEAT      = _style("CAVEAT",      fontSize=7.5,textColor=MUTED,  fontName=SANS,    leading=11, spaceAfter=4)
SUMMARY_LINE= _style("SUMMARY_LINE",fontSize=10, textColor=NAVY,   fontName=SANS_B,  leading=14,
                      spaceAfter=10, spaceBefore=4, alignment=TA_LEFT)
CARD_LABEL  = _style("CARD_LABEL",  fontSize=7,  textColor=MUTED,  fontName=SANS,    leading=9,  alignment=TA_CENTER)
CARD_VAL_N  = _style("CARD_VAL_N",  fontSize=16, textColor=NAVY,   fontName=SANS_B,  leading=19, alignment=TA_CENTER)
CARD_VAL_R  = _style("CARD_VAL_R",  fontSize=16, textColor=RED,    fontName=SANS_B,  leading=19, alignment=TA_CENTER)
RISK_TITLE  = _style("RISK_TITLE",  fontSize=9.5,textColor=BLACK,  fontName=SANS_B,  leading=13)
RISK_BODY   = _style("RISK_BODY",   fontSize=9,  textColor=BLACK,  fontName=SERIF,   leading=13, spaceAfter=2)
BADGE_H     = _style("BADGE_H",     fontSize=7,  textColor=WHITE,  fontName=SANS_B,  leading=9,  alignment=TA_CENTER)
BADGE_M     = _style("BADGE_M",     fontSize=7,  textColor=WHITE,  fontName=SANS_B,  leading=9,  alignment=TA_CENTER)

# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------

def _make_doc(buf: io.BytesIO, title: str, run_date: str) -> BaseDocTemplate:
    doc = BaseDocTemplate(
        buf, pagesize=LETTER,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.35 * inch, bottomMargin=0.80 * inch,
        title=title, author="Portfolio Lab",
    )

    # Cover data is populated by build_report before doc.build() — one entry per book
    doc._covers     = []
    doc._cover_idx  = [0]

    def _cover_canvas(canvas, doc):
        idx = doc._cover_idx[0]
        doc._cover_idx[0] += 1
        covers = doc._covers
        if idx >= len(covers):
            return
        data = covers[idx]

        canvas.saveState()
        stripe_h = PAGE_H * 0.40

        # NAVY stripe — top 40%
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - stripe_h, PAGE_W, stripe_h, fill=1, stroke=0)

        # Report title in stripe
        canvas.setFillColor(WHITE)
        canvas.setFont(SANS_B, 28)
        canvas.drawString(MARGIN, PAGE_H - stripe_h + stripe_h * 0.52, "PORTFOLIO LAB")
        canvas.setFont(SANS, 14)
        canvas.drawString(MARGIN, PAGE_H - stripe_h + stripe_h * 0.30, "Monte Carlo Report")

        # Gold accent line at bottom of stripe
        canvas.setStrokeColor("#b8860b")
        canvas.setLineWidth(2.5)
        canvas.line(0, PAGE_H - stripe_h, PAGE_W, PAGE_H - stripe_h)

        # Portfolio name — below stripe, with left margin
        y = PAGE_H - stripe_h - 0.60 * inch
        canvas.setFillColor("#1e3a5f")
        canvas.setFont(SANS_B, 22)
        canvas.drawString(MARGIN, y, data["name"])

        # Subtitle
        y -= 0.38 * inch
        canvas.setFillColor("#888888")
        canvas.setFont(SANS, 11)
        subtitle = (f"{data['horizon_y']}-Year Monte Carlo Simulation  ·  "
                    f"{data['n_paths']:,} Paths  ·  Factor-Model GBM  ·  "
                    f"{data['regime_name']}")
        canvas.drawString(MARGIN, y, subtitle)

        # Gold rule — directly under subtitle block
        y -= 0.38 * inch
        canvas.setStrokeColor("#b8860b")
        canvas.setLineWidth(1.5)
        canvas.line(MARGIN, y, PAGE_W - MARGIN, y)

        # Prepared by + date — more breathing room below the rule
        y -= 0.42 * inch
        canvas.setFillColor("#888888")
        canvas.setFont(SANS, 9)
        canvas.drawString(MARGIN, y, "Prepared by Portfolio Lab")
        y -= 0.22 * inch
        canvas.drawString(MARGIN, y, data["run_date"])

        canvas.restoreState()

    def _main_header_footer(canvas, doc):
        canvas.saveState()
        # Narrow NAVY header bar
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 0.75 * inch, PAGE_W, 0.75 * inch, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont(SANS_B, 11)
        canvas.drawString(MARGIN, PAGE_H - 0.42 * inch, "Portfolio Lab  —  Monte Carlo Report")
        canvas.setFont(SANS, 8)
        canvas.setFillColor(colors.HexColor("#a0b8d0"))
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.42 * inch, run_date)
        # Footer rule
        canvas.setStrokeColor(MID)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 0.72 * inch, PAGE_W - MARGIN, 0.72 * inch)
        # Footer text: left / center / right
        canvas.setFont(SANS, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 0.52 * inch, "Portfolio Lab — Monte Carlo Report")
        canvas.drawCentredString(PAGE_W / 2, 0.52 * inch, f"Page {doc.page}")
        canvas.drawRightString(PAGE_W - MARGIN, 0.52 * inch, run_date)
        # Disclaimer
        canvas.setFont(SANS, 7)
        canvas.drawCentredString(
            PAGE_W / 2, 0.34 * inch,
            "Informational only · Not investment advice · "
            "Lognormal model, no fat tails · Static factor correlations"
        )
        canvas.restoreState()

    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, id="cover",
                        leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0)
    main_frame  = Frame(MARGIN, 0.80 * inch, CONTENT_W,
                        PAGE_H - 0.80 * inch - 1.35 * inch, id="body")

    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=_cover_canvas),
        PageTemplate(id="main",  frames=[main_frame],  onPage=_main_header_footer),
    ])
    return doc

# ---------------------------------------------------------------------------
# Reusable components
# ---------------------------------------------------------------------------

def _section_header(text: str) -> Table:
    t = Table([[Paragraph(text, SECTION_H)]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    return t

def _table_style(header_bg=NAVY, header_fg=WHITE,
                 alt=LIGHT, grid=MID, numeric_from_col=1) -> TableStyle:
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  header_fg),
        ("FONTNAME",      (0, 0), (-1, 0),  SANS_B),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("FONTNAME",      (0, 1), (-1, -1), SANS),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, alt]),
        ("GRID",          (0, 0), (-1, -1), 0.25, grid),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
        ("ALIGN",         (numeric_from_col, 0), (-1, -1), "RIGHT"),
    ])

def _fig_to_image(fig, width: float, height: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width, height=height)

# ---------------------------------------------------------------------------
# Metric card grid
# ---------------------------------------------------------------------------

# Index → (label, is_risk)
_METRIC_DEFS = [
    ("Median End Value",        False),
    ("P10 Downside",            True),
    ("P90 Upside",              False),
    ("Mean Outcome",            False),
    ("Prob. Beat S&P 500",      False),
    ("Prob. Loss on Deposits",  True),
    ("Prob. Double",            False),
    ("Median Max Drawdown",     True),
]

def _metrics_grid(stats: dict) -> Table:
    values = [
        f"${stats['median']:,.0f}",
        f"${stats['p10']:,.0f}",
        f"${stats['p90']:,.0f}",
        f"${stats['mean']:,.0f}",
        f"{stats['prob_beat_spx']*100:.1f}%",
        f"{stats['prob_loss']*100:.1f}%",
        f"{stats['prob_double']*100:.1f}%",
        f"{stats['dd_p50']*100:.1f}%",
    ]
    cell_w = CONTENT_W / 4
    rows = []
    for row_start in (0, 4):
        label_row, val_row = [], []
        for i in range(row_start, row_start + 4):
            label, is_risk = _METRIC_DEFS[i]
            val_style = CARD_VAL_R if is_risk else CARD_VAL_N
            label_row.append(Paragraph(label.upper(), CARD_LABEL))
            val_row.append(Paragraph(values[i], val_style))
        rows += [label_row, val_row]

    t = Table(rows, colWidths=[cell_w] * 4,
              rowHeights=[0.20 * inch, 0.52 * inch] * 2)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), WHITE),
        ("BOX",           (0, 0), (-1, -1), 0.5, MID),
        ("INNERGRID",     (0, 0), (-1, -1), 0.25, MID),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _holdings_table(enriched_df: pd.DataFrame) -> Table:
    has_cost = "avg_cost" in enriched_df.columns and enriched_df["avg_cost"].notna().any()
    if has_cost:
        header = ["Ticker", "Shares", "Avg Cost", "Price", "Market Value",
                  "Cost Basis", "Unrealized $", "Unrealized %", "Weight"]
    else:
        header = ["Ticker", "Shares", "Price", "Market Value", "Weight", "μ", "σ"]
    rows = [header]
    total_mv, total_cb, total_unr = 0.0, 0.0, 0.0
    pnl_color_rows = []
    for i, (_, r) in enumerate(enriched_df.iterrows(), start=1):
        mv = r.get("mv", float("nan"))
        if has_cost:
            ac = r.get("avg_cost", float("nan"))
            shares = r.get("shares", float("nan"))
            cb = ac * shares if pd.notna(ac) and pd.notna(shares) else float("nan")
            unr = mv - cb if pd.notna(mv) and pd.notna(cb) else float("nan")
            unr_pct = (unr / cb * 100) if pd.notna(unr) and pd.notna(cb) and cb > 0 else float("nan")
            if pd.notna(unr):
                pnl_color_rows.append((i, unr >= 0))
            rows.append([
                r["ticker"],
                f"{r['shares']:.4f}",
                f"${ac:,.2f}" if pd.notna(ac) else "—",
                f"${r['price']:,.2f}" if pd.notna(r.get("price")) else "—",
                f"${mv:,.2f}" if pd.notna(mv) else "—",
                f"${cb:,.2f}" if pd.notna(cb) else "—",
                f"${unr:+,.2f}" if pd.notna(unr) else "—",
                f"{unr_pct:+.1f}%" if pd.notna(unr_pct) else "—",
                f"{r['weight']*100:.1f}%" if pd.notna(r.get("weight")) else "—",
            ])
            if pd.notna(mv):  total_mv  += mv
            if pd.notna(cb):  total_cb  += cb
            if pd.notna(unr): total_unr += unr
        else:
            rows.append([
                r["ticker"],
                f"{r['shares']:.4f}",
                f"${r['price']:,.2f}" if pd.notna(r.get("price")) else "—",
                f"${mv:,.2f}" if pd.notna(mv) else "—",
                f"{r['weight']*100:.1f}%" if pd.notna(r.get("weight")) else "—",
                f"{r['mu']*100:.1f}%",
                f"{r['sigma']*100:.1f}%",
            ])
            if pd.notna(mv): total_mv += mv

    if has_cost:
        total_unr_pct = (total_unr / total_cb * 100) if total_cb > 0 else 0.0
        rows.append(["Total", "", "", "",
                     f"${total_mv:,.2f}", f"${total_cb:,.2f}",
                     f"${total_unr:+,.2f}", f"{total_unr_pct:+.1f}%", "100.0%"])
        ratios = [0.55, 0.6, 0.7, 0.7, 0.95, 0.9, 0.85, 0.75, 0.6]
    else:
        rows.append(["Total", "", "", f"${total_mv:,.2f}", "100.0%", "", ""])
        ratios = [0.7, 0.7, 0.85, 1.1, 0.75, 0.6, 0.6]

    col_w = [c * CONTENT_W / sum(ratios) for c in ratios]
    t = Table(rows, colWidths=col_w)
    style = _table_style()
    style.add("FONTNAME",   (0, len(rows)-1), (-1, len(rows)-1), SANS_B)
    style.add("BACKGROUND", (0, len(rows)-1), (-1, len(rows)-1), GHOST)
    if has_cost:
        # Color unrealized $ and % columns by sign
        green = colors.HexColor("#1f7a3a")
        red   = RED
        for row_idx, is_pos in pnl_color_rows:
            c = green if is_pos else red
            style.add("TEXTCOLOR", (6, row_idx), (7, row_idx), c)
        # Total row P&L color
        c = colors.HexColor("#1f7a3a") if total_unr >= 0 else RED
        style.add("TEXTCOLOR", (6, len(rows)-1), (7, len(rows)-1), c)
        # Compact font for wider table
        style.add("FONTSIZE",      (0, 0), (-1,  0), 8)
        style.add("FONTSIZE",      (0, 1), (-1, -1), 7.5)
        style.add("LEFTPADDING",   (0, 0), (-1, -1), 4)
        style.add("RIGHTPADDING",  (0, 0), (-1, -1), 4)
        style.add("TOPPADDING",    (0, 0), (-1, -1), 4)
        style.add("BOTTOMPADDING", (0, 0), (-1, -1), 4)
    t.setStyle(style)
    return t

def _quant_tables(stats: dict) -> list:
    pct_data = [
        ["", "P5", "P10", "P25", "P50 (Median)", "P75", "P90", "P95"],
        ["End value",
         f"${stats['p05']:,.0f}", f"${stats['p10']:,.0f}", f"${stats['p25']:,.0f}",
         f"${stats['median']:,.0f}", f"${stats['p75']:,.0f}",
         f"${stats['p90']:,.0f}", f"${stats['p95']:,.0f}"],
    ]
    col_w = [c * CONTENT_W / sum([1.1] + [0.84]*7) for c in [1.1] + [0.84]*7]
    pct_t = Table(pct_data, colWidths=col_w)
    pct_t.setStyle(_table_style())

    dd_data = [
        ["Drawdown",       "Median (P50)",         "Bad scenario (P10)",    "Mild (P90)"],
        ["Peak-to-trough", f"{stats['dd_p50']*100:.1f}%",
         f"{stats['dd_p10']*100:.1f}%", f"{stats['dd_p90']*100:.1f}%"],
    ]
    col_w2 = [c * CONTENT_W / sum([1.8, 1.5, 1.6, 1.4]) for c in [1.8, 1.5, 1.6, 1.4]]
    dd_t = Table(dd_data, colWidths=col_w2)
    dd_t.setStyle(_table_style())
    return [pct_t, Spacer(1, 0.15 * inch), dd_t]

# Theme classification for portfolio composition breakdown
THEMES: dict[str, str] = {
    "MRVL": "AI / Semis",       "AVGO": "AI / Semis",       "GOOG": "AI / Semis",
    "ALAB": "AI / Semis",       "AMAT": "AI / Semis",       "ASML": "AI / Semis",
    "COHR": "AI / Semis",       "CRWV": "AI / Semis",       "APLD": "AI / Semis",
    "CEG":  "Nuclear / Power",  "LEU":  "Nuclear / Power",  "UUUU": "Nuclear / Power",
    "SOLS": "Nuclear / Power",  "MIR":  "Nuclear / Power",
    "MOG-A":"Defense / Industrial", "HUBB": "Defense / Industrial",
    "PATH": "Software / Automation",
    "TCEHY":"China / Mega-cap Tech",
    "HOOD": "Fintech",
    "VELO": "Speculative / Other", "XRP": "Speculative / Other",
}

def _theme_breakdown_table(enriched_df: pd.DataFrame) -> Table:
    df = enriched_df.copy()
    df["theme"] = df["ticker"].map(lambda t: THEMES.get(t, "Other"))
    grouped = df.groupby("theme").agg(
        mv=("mv", "sum"),
        weight=("weight", "sum"),
        n=("ticker", "count"),
        names=("ticker", lambda x: ", ".join(sorted(x))),
    ).reset_index().sort_values("mv", ascending=False)

    header = ["Theme", "Positions", "Tickers", "Market Value", "Weight"]
    rows = [header]
    name_style = ParagraphStyle("ThemeName", parent=_base["Normal"],
                                fontSize=8, fontName=SANS, leading=11)
    for _, r in grouped.iterrows():
        rows.append([
            r["theme"],
            f"{int(r['n'])}",
            Paragraph(r["names"], name_style),
            f"${r['mv']:,.2f}",
            f"{r['weight']*100:.1f}%",
        ])
    total_mv = grouped["mv"].sum()
    rows.append(["Total", f"{int(grouped['n'].sum())}", "", f"${total_mv:,.2f}", "100.0%"])

    ratios = [1.4, 0.6, 2.6, 1.0, 0.7]
    col_w = [c * CONTENT_W / sum(ratios) for c in ratios]
    t = Table(rows, colWidths=col_w)
    style = _table_style()
    style.add("FONTNAME",   (0, len(rows)-1), (-1, len(rows)-1), SANS_B)
    style.add("BACKGROUND", (0, len(rows)-1), (-1, len(rows)-1), GHOST)
    style.add("VALIGN",     (0, 0), (-1, -1), "MIDDLE")
    style.add("FONTSIZE",   (0, 1), (-1, -1), 8)
    t.setStyle(style)
    return t


def _milestones_table(port: np.ndarray, horizon_y: int, starting_mv: float) -> Table:
    """Year-by-year P10/P50/P90 portfolio value at standard checkpoints."""
    n_months = port.shape[1]
    months_per_year = max(1, (n_months - 1) // horizon_y) if horizon_y > 0 else 12
    candidate_years = [1, 3, 5, 10, 15, 20, 25, 30]
    checkpoints = [y for y in candidate_years if y <= horizon_y]
    if horizon_y not in checkpoints:
        checkpoints.append(horizon_y)
    checkpoints = sorted(set(checkpoints))

    header = ["Year", "P10 (Bad)", "P50 (Median)", "P90 (Good)", "Median Multiple"]
    rows = [header]
    for y in checkpoints:
        idx = min(int(y * months_per_year), n_months - 1)
        slice_ = port[:, idx]
        p10, p50, p90 = np.percentile(slice_, [10, 50, 90])
        mult = p50 / starting_mv if starting_mv > 0 else 0
        rows.append([
            f"Year {y}",
            f"${p10:,.0f}", f"${p50:,.0f}", f"${p90:,.0f}",
            f"{mult:.1f}×",
        ])

    ratios = [0.9, 1.1, 1.1, 1.1, 0.9]
    col_w = [c * CONTENT_W / sum(ratios) for c in ratios]
    t = Table(rows, colWidths=col_w)
    style = _table_style()
    style.add("ALIGN", (1, 0), (-1, -1), "RIGHT")
    t.setStyle(style)
    return t


def _factor_exposure_table(enriched_df: pd.DataFrame) -> tuple[Table, dict]:
    """Portfolio-weighted factor exposures + summary dict for narrative."""
    df = enriched_df.copy()
    w = df["weight"].fillna(0)
    f_market = float((df["f_market"] * w).sum())
    f_ai     = float((df["f_ai"]     * w).sum())
    f_power  = float((df["f_power"]  * w).sum())
    f_idio   = max(0.0, 1.0 - f_market - f_ai - f_power)

    rows = [
        ["Factor", "Weighted Variance Share", "Plain English"],
        ["Market (broad equity beta)",  f"{f_market*100:.1f}%",
         "Moves with the overall stock market — captured by SPY."],
        ["AI / Semi capex cycle",       f"{f_ai*100:.1f}%",
         "Moves with hyperscaler AI infrastructure spending — SMH proxy."],
        ["Power / Nuclear / Energy",    f"{f_power*100:.1f}%",
         "Moves with electricity demand & uranium cycle — URA / utilities."],
        ["Idiosyncratic (stock-specific)", f"{f_idio*100:.1f}%",
         "Company-specific events: earnings, guidance, single-name news."],
    ]
    ratios = [1.4, 1.1, 3.4]
    col_w = [c * CONTENT_W / sum(ratios) for c in ratios]
    t = Table(rows, colWidths=col_w)
    style = _table_style()
    style.add("ALIGN",    (0, 0), (-1, -1), "LEFT")
    style.add("ALIGN",    (1, 0), (1, -1),  "CENTER")
    style.add("FONTSIZE", (0, 1), (-1, -1), 8.5)
    t.setStyle(style)
    return t, {"market": f_market, "ai": f_ai, "power": f_power, "idio": f_idio}


def _factor_table(enriched_df: pd.DataFrame) -> Table:
    header = ["Ticker", "μ (ann.)", "σ (ann.)", "f_market", "f_ai", "f_power"]
    rows = [header]
    for _, r in enriched_df.iterrows():
        rows.append([
            r["ticker"],
            f"{r['mu']*100:.1f}%", f"{r['sigma']*100:.1f}%",
            f"{r['f_market']:.2f}", f"{r['f_ai']:.2f}", f"{r['f_power']:.2f}",
        ])
    col_w = [c * CONTENT_W / sum([0.9, 0.85, 0.85, 0.85, 0.75, 0.85])
             for c in [0.9, 0.85, 0.85, 0.85, 0.75, 0.85]]
    t = Table(rows, colWidths=col_w)
    style = _table_style()
    style.add("ALIGN",         (1, 0),  (-1, -1), "CENTER")
    style.add("FONTSIZE",      (0, 0),  (-1,  0), 8)
    style.add("FONTSIZE",      (0, 1),  (-1, -1), 7.5)
    style.add("TOPPADDING",    (0, 0),  (-1, -1), 3)
    style.add("BOTTOMPADDING", (0, 0),  (-1, -1), 3)
    t.setStyle(style)
    return t

def _comparison_table(books: list[dict]) -> Table:
    header = ["Metric"] + [b["name"] for b in books]
    def _row(label, fn):
        return [label] + [fn(b["stats"]) for b in books]
    rows = [
        header,
        _row("Median end value",   lambda s: f"${s['median']:,.0f}"),
        _row("P10 downside",       lambda s: f"${s['p10']:,.0f}"),
        _row("P90 upside",         lambda s: f"${s['p90']:,.0f}"),
        _row("Mean",               lambda s: f"${s['mean']:,.0f}"),
        _row("Beat S&P 500",       lambda s: f"{s['prob_beat_spx']*100:.1f}%"),
        _row("Prob. loss",         lambda s: f"{s['prob_loss']*100:.1f}%"),
        _row("Prob. double",       lambda s: f"{s['prob_double']*100:.1f}%"),
        _row("Median drawdown",    lambda s: f"{s['dd_p50']*100:.1f}%"),
        _row("Worst-decile DD",    lambda s: f"{s['dd_p10']*100:.1f}%"),
    ]
    n = len(books)
    col_w = [2.0 * inch] + [(CONTENT_W - 2.0 * inch) / n] * n
    t = Table(rows, colWidths=col_w)
    style = _table_style()
    style.add("ALIGN", (1, 1), (-1, -1), "CENTER")
    t.setStyle(style)
    return t

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _fan_chart_fig(port, title="", starting_mv=None):
    pcts  = [10, 25, 50, 75, 90]
    bands = np.percentile(port, pcts, axis=0)
    months = np.arange(port.shape[1])
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)

    # Loss zone
    if starting_mv and starting_mv > 0:
        ax.fill_between(months, 0, starting_mv,
                        alpha=0.06, color=MPLRED, zorder=0, label="_nolegend_")
        ax.axhline(starting_mv, color=MPLRED, lw=0.9, ls=":", alpha=0.7,
                   label=f"Start  ${starting_mv:,.0f}", zorder=1)

    # Percentile fill bands
    ax.fill_between(months, bands[0], bands[4], alpha=0.10, color=MPLNAVY)
    ax.fill_between(months, bands[1], bands[3], alpha=0.20, color=MPLNAVY)

    # Percentile curves
    cfg = [
        (0, "#4a90c4", 0.9, "--", "P10"),
        (1, "#7aaec8", 0.8, "--", "P25"),
        (2, MPLGOLD,   2.2, "-",  "Median"),
        (3, "#7aaec8", 0.8, "--", "P75"),
        (4, "#4a90c4", 0.9, "--", "P90"),
    ]
    for idx, color, lw, ls, label in cfg:
        ax.plot(months, bands[idx], color=color, lw=lw, ls=ls, label=label, zorder=2)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Month", fontsize=8, color="#444444")
    ax.set_ylabel("Portfolio value", fontsize=8, color="#444444")
    ax.set_title(title, fontsize=9, color=MPLNAVY, fontweight="bold", pad=6)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85,
              ncol=3, columnspacing=1.0)
    ax.grid(True, alpha=0.2, linestyle=":", color="#aaaaaa")
    ax.tick_params(labelsize=7, colors="#444444")
    ax.spines[["top", "right"]].set_visible(False)
    return fig

def _hist_chart_fig(port, title="", starting_mv=None):
    end = port[:, -1]
    p10, p50, p90 = np.percentile(end, [10, 50, 90])
    p99 = np.percentile(end, 99)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)

    ax.hist(end[end <= p99], bins=80, color=MPLNAVY, alpha=0.68, edgecolor="none", zorder=2)
    ax.set_xlim(left=0, right=p99)

    for x, c, lab in [(p10, MPLRED,     f"P10  ${p10:,.0f}"),
                      (p50, MPLGOLD,    f"P50  ${p50:,.0f}"),
                      (p90, "#27ae60",  f"P90  ${p90:,.0f}")]:
        ax.axvline(x, color=c, lw=2.0, label=lab, zorder=3)

    if starting_mv and starting_mv > 0:
        ax.axvline(starting_mv, color=MPLRED, lw=1.4, ls="--",
                   label=f"Start  ${starting_mv:,.0f}", zorder=4, alpha=0.7)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Terminal value", fontsize=8, color="#444444")
    ax.set_ylabel("Frequency", fontsize=8, color="#444444")
    ax.set_title(title, fontsize=9, color=MPLNAVY, fontweight="bold", pad=6)
    ax.legend(fontsize=7, framealpha=0.85, ncol=2, columnspacing=1.0)
    ax.grid(True, alpha=0.2, linestyle=":", color="#aaaaaa", axis="y")
    ax.tick_params(labelsize=7, colors="#444444")
    ax.spines[["top", "right"]].set_visible(False)
    return fig

# ---------------------------------------------------------------------------
# Risk cards
# ---------------------------------------------------------------------------

def _risk_card(title: str, body: str, impact: str) -> Table:
    badge_bg    = RED if impact == "HIGH" else AMBER
    badge_style = BADGE_H if impact == "HIGH" else BADGE_M

    badge_cell = Table(
        [[Paragraph(impact, badge_style)]],
        colWidths=[0.62 * inch],
    )
    badge_cell.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), badge_bg),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ("ROUNDEDCORNERS",(0,0),(-1,-1), [2,2,2,2]),
    ]))

    header_row = [Paragraph(title, RISK_TITLE), badge_cell]
    body_row   = [Paragraph(body, RISK_BODY),   ""]

    t = Table(
        [header_row, body_row],
        colWidths=[CONTENT_W - 0.72 * inch, 0.72 * inch],
    )
    t.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 0.5, MID),
        ("BACKGROUND",    (0, 0), (-1,  0), GHOST),
        ("BACKGROUND",    (0, 1), (-1,  1), WHITE),
        ("SPAN",          (0, 1), ( 1,  1)),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return t

# ---------------------------------------------------------------------------
# Narrative functions
# ---------------------------------------------------------------------------

def _interpret_results(stats: dict, horizon_y: int, starting_mv: float) -> list[str]:
    deposits  = stats["deposits"]
    median    = stats["median"]
    p10, p90  = stats["p10"], stats["p90"]
    beat      = stats["prob_beat_spx"] * 100
    prob_loss = stats["prob_loss"] * 100
    dd_med    = abs(stats["dd_p50"]) * 100
    loss_pct  = (1 - p10 / deposits) * 100 if deposits > 0 else 0
    gain_pct  = (median / starting_mv - 1) * 100
    dep_note  = (f" (${deposits:,.0f} including all contributions)"
                 if abs(deposits - starting_mv) > 1 else "")
    p10_context = (f"a {loss_pct:.1f}% loss on total capital deployed"
                   if loss_pct >= 0
                   else f"a {abs(loss_pct):.1f}% gain on total capital deployed")

    return [
        f"The median simulated portfolio value after {horizon_y} years is "
        f"<b>${median:,.0f}</b>, a {gain_pct:.1f}% gain on the "
        f"<b>${starting_mv:,.0f}</b> starting value{dep_note}. "
        f"The mean outcome is ${stats['mean']:,.0f} — above the median, which is "
        f"characteristic of equity distributions where a small number of outstanding "
        f"paths pull the average upward.",

        f"The more informative number is the downside. At the 10th percentile you end "
        f"with <b>${p10:,.0f}</b> — {p10_context} — "
        f"while the 90th percentile reaches ${p90:,.0f}. The wide gap between P10 and "
        f"P90 reflects concentration: a handful of correlated positions dominate the "
        f"outcome distribution. Budget emotionally for a median peak-to-trough drawdown "
        f"of {dd_med:.0f}% at some point during this window. That is not a tail "
        f"scenario — it is the typical experience.",

        f"Probability of beating a passive S&amp;P 500 index is <b>{beat:.1f}%</b>, and "
        f"probability of ending below total capital deployed is {prob_loss:.1f}%. A beat "
        f"rate below 50% is not unusual for concentrated, high-volatility books: even "
        f"when arithmetic expected return exceeds the index, high variance creates a "
        f"geometric return drag that the index — at roughly 16% annual vol — does not "
        f"face. You are paying with variance for the option on the right tail.",
    ]

def _identify_risks(enriched_df: pd.DataFrame, stats: dict) -> list[tuple[str, str, str]]:
    risks = []
    ai_exp = float((enriched_df["f_ai"] * enriched_df["weight"]).sum())
    if ai_exp > 0.45:
        top_ai = enriched_df.sort_values("weight", ascending=False).head(4)["ticker"].tolist()
        risks.append((
            f"Correlated AI-infrastructure sell-off ({ai_exp*100:.0f}% factor exposure)",
            f"Roughly {ai_exp*100:.0f}% of portfolio variance is driven by the AI capex "
            f"factor. {', '.join(top_ai)} move together on AI spending cycle news. A "
            f"sustained negative factor shock — hyperscaler pullback, capex slowdown, "
            f"regulatory overhang — craters every core position simultaneously. This is "
            f"the single largest driver of the left tail.",
            "HIGH"
        ))

    heavy = enriched_df[enriched_df["weight"] > 0.08].sort_values("weight", ascending=False)
    if not heavy.empty:
        names = ", ".join(f"{r['ticker']} ({r['weight']*100:.1f}%)"
                          for _, r in heavy.iterrows())
        impact = "HIGH" if any(r["weight"] > 0.10 for _, r in heavy.iterrows()) else "MEDIUM"
        risks.append((
            f"Single-stock concentration ({names})",
            f"Position(s) above 8% are large enough that a single company-specific "
            f"event — earnings miss, customer cancellation, guidance cut — moves the "
            f"total book by 4–10% in a single session. High weight combined with high "
            f"σ is the primary driver of idiosyncratic tail events.",
            impact
        ))

    if abs(stats["dd_p50"]) > 0.35:
        risks.append((
            f"Geometric return drag (median drawdown {abs(stats['dd_p50'])*100:.0f}%)",
            f"High-volatility books sacrifice geometric return even without a catastrophe. "
            f"Volatility itself reduces compounded outcomes below what the weighted-μ "
            f"suggests. A {abs(stats['dd_p50'])*100:.0f}% median peak-to-trough drawdown "
            f"is the price of the concentrated position — not a tail scenario.",
            "MEDIUM"
        ))

    small = enriched_df[enriched_df["weight"] < 0.02]
    if len(small) >= 2:
        risks.append((
            f"Sub-scale positions ({', '.join(small['ticker'].tolist())})",
            f"Positions below 2% are too small to move the needle but still require "
            f"monitoring and conviction maintenance. They dilute focus without improving "
            f"the distribution. Not a tail-risk driver, but a portfolio-hygiene drag.",
            "MEDIUM"
        ))

    return risks[:4]

def _recommend(enriched_df: pd.DataFrame, stats: dict) -> list[tuple[str, str, str, str]]:
    recs = []
    ai_exp = float((enriched_df["f_ai"] * enriched_df["weight"]).sum())
    if ai_exp > 0.60:
        recs.append((
            "Add diversifier sleeve",
            f"{ai_exp*100:.0f}% AI exposure",
            "15% VT + 5% SGOV",
            "Largest single drawdown reducer. Moves median DD down ~10pp, improves P10 ~30–40%."
        ))

    heavy = enriched_df[enriched_df["weight"] > 0.08].sort_values("weight", ascending=False)
    for _, r in heavy.iterrows():
        recs.append((
            f"Cap {r['ticker']} weight",
            f"{r['weight']*100:.1f}%",
            "8.0%",
            "Reduces idiosyncratic blowup risk. Proceeds recycled into highest-conviction names."
        ))
        if len(recs) >= 3:
            break

    small = enriched_df[enriched_df["weight"] < 0.02]
    if len(small) >= 2 and len(recs) < 3:
        recs.append((
            f"Consolidate {', '.join(small['ticker'].tolist())}",
            f"{len(small)} positions < 2%",
            "Fold into top names",
            "Eliminates clutter. Focus conviction capital on highest-confidence ideas."
        ))

    if not recs:
        recs.append((
            "Maintain & rebalance",
            "Current",
            "Monitor drift",
            "No structural changes flagged. Rebalance if any position drifts above 10%."
        ))

    return recs[:4]

def _action_table(recs: list[tuple]) -> Table:
    header = ["Action", "Current", "Target", "Rationale"]
    col_ratios = [1.5, 1.1, 1.1, 2.6]
    col_w = [c * CONTENT_W / sum(col_ratios) for c in col_ratios]
    cell_style = ParagraphStyle("ActionCell", parent=_base["Normal"],
                                fontSize=8.5, fontName=SANS, leading=12)
    def _wrap(v):
        return Paragraph(str(v), cell_style)
    rows = [header] + [[_wrap(v) for v in r] for r in recs]
    t = Table(rows, colWidths=col_w)
    style = _table_style(numeric_from_col=0)
    style.add("ALIGN", (0, 0), (-1, -1), "LEFT")
    t.setStyle(style)
    return t

# ---------------------------------------------------------------------------
# Deep analytics — computation helpers
# ---------------------------------------------------------------------------

def _concentration_metrics(enriched_df: pd.DataFrame) -> dict:
    w = enriched_df["weight"].fillna(0).to_numpy()
    w = w[w > 0]
    if len(w) == 0:
        return {"hhi": 0, "eff_n": 0, "top3": 0, "top5": 0, "gini": 0,
                "n": 0, "max_w": 0, "max_ticker": ""}
    hhi   = float(np.sum(w ** 2))
    eff_n = 1.0 / hhi if hhi > 0 else 0
    sw    = np.sort(w)[::-1]
    top3  = float(sw[:3].sum())
    top5  = float(sw[:5].sum())
    # Gini coefficient on weights
    sorted_w = np.sort(w)
    n_pos = len(sorted_w)
    cum   = np.cumsum(sorted_w)
    gini  = float((2.0 * np.sum((np.arange(1, n_pos + 1)) * sorted_w)) /
                  (n_pos * cum[-1]) - (n_pos + 1) / n_pos) if cum[-1] > 0 else 0
    max_idx = enriched_df["weight"].fillna(0).idxmax()
    return {
        "hhi": hhi, "eff_n": eff_n, "top3": top3, "top5": top5, "gini": gini,
        "n": int(len(enriched_df)),
        "max_w": float(enriched_df.loc[max_idx, "weight"]),
        "max_ticker": str(enriched_df.loc[max_idx, "ticker"]),
    }


def _probability_ladder(port: np.ndarray, horizon_y: int, starting_mv: float) -> dict:
    n_months = port.shape[1]
    months_per_year = max(1, (n_months - 1) // horizon_y) if horizon_y > 0 else 12
    years = [y for y in [5, 10, 15, 20, 25, 30] if y <= horizon_y]
    if horizon_y not in years:
        years = sorted(set(years + [horizon_y]))
    multiples = [2, 3, 5, 10]
    grid = {}
    for y in years:
        idx = min(int(y * months_per_year), n_months - 1)
        slc = port[:, idx]
        grid[y] = {m: float((slc >= m * starting_mv).mean()) for m in multiples}
    return {"years": years, "multiples": multiples, "grid": grid}


def _drawdown_stats(port: np.ndarray) -> dict:
    peak = np.maximum.accumulate(port, axis=1)
    dd_series = port / peak - 1.0
    max_dd = dd_series.min(axis=1)
    # Time to recover from -25% drawdown
    recoveries = []
    for path_idx in range(min(2000, port.shape[0])):
        path = port[path_idx]
        peak_path = np.maximum.accumulate(path)
        dd_path = path / peak_path - 1.0
        breach_idxs = np.where(dd_path <= -0.25)[0]
        if len(breach_idxs) == 0:
            continue
        breach = breach_idxs[0]
        peak_val = peak_path[breach]
        recover_after = np.where(path[breach:] >= peak_val)[0]
        if len(recover_after) > 0:
            recoveries.append(recover_after[0])
    months_to_recover_25 = float(np.median(recoveries)) if recoveries else float("nan")
    return {
        "p10_dd": float(np.percentile(max_dd, 10)),
        "p25_dd": float(np.percentile(max_dd, 25)),
        "p50_dd": float(np.percentile(max_dd, 50)),
        "p75_dd": float(np.percentile(max_dd, 75)),
        "p90_dd": float(np.percentile(max_dd, 90)),
        "worst": float(np.min(max_dd)),
        "frac_dd_30": float((max_dd <= -0.30).mean()),
        "frac_dd_50": float((max_dd <= -0.50).mean()),
        "frac_dd_70": float((max_dd <= -0.70).mean()),
        "median_months_to_recover_25": months_to_recover_25,
        "max_dd_array": max_dd,
    }


# Stress scenarios: instantaneous shocks expressed as factor sensitivities.
# Each scenario shocks a subset of factors; portfolio impact = sum(weight * sensitivity).
STRESS_SCENARIOS = [
    {
        "name": "AI capex slowdown",
        "desc": "Hyperscaler infra spend cuts; AI factor −40%, market −10%.",
        "shocks": {"f_market": -0.10, "f_ai": -0.40, "f_power": 0.0},
    },
    {
        "name": "Broad equity crash",
        "desc": "Recession-style market drawdown; market −30%, AI −20%, power −15%.",
        "shocks": {"f_market": -0.30, "f_ai": -0.20, "f_power": -0.15},
    },
    {
        "name": "Power demand collapse",
        "desc": "Datacenter buildout pause; power factor −35%, AI −15%.",
        "shocks": {"f_market": -0.05, "f_ai": -0.15, "f_power": -0.35},
    },
    {
        "name": "Dot-com style tech bear",
        "desc": "Sustained 18-mo bear; AI/Semis −50%, market −25%, power −10%.",
        "shocks": {"f_market": -0.25, "f_ai": -0.50, "f_power": -0.10},
    },
    {
        "name": "Stagflation",
        "desc": "Persistent inflation + slow growth; market −15%, AI −10%, power +10% (energy bid).",
        "shocks": {"f_market": -0.15, "f_ai": -0.10, "f_power": +0.10},
    },
]


def _apply_stress(enriched_df: pd.DataFrame, shocks: dict) -> dict:
    """Return per-position and portfolio-level $ and % impact under shock."""
    df = enriched_df.copy()
    # Per-position pct shock = sum_factor( factor_loading * factor_shock )
    pct = (df["f_market"].fillna(0) * shocks.get("f_market", 0)
           + df["f_ai"].fillna(0)    * shocks.get("f_ai", 0)
           + df["f_power"].fillna(0) * shocks.get("f_power", 0))
    # Idiosyncratic exposure absorbs no scripted shock (assumed avg 0)
    impact_dollar = (pct * df["mv"].fillna(0)).sum()
    total_mv = df["mv"].fillna(0).sum()
    return {
        "pct": float(impact_dollar / total_mv) if total_mv > 0 else 0.0,
        "dollar": float(impact_dollar),
    }


def _position_variance_contribution(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Approximate each position's contribution to portfolio variance (assuming avg correlation)."""
    df = enriched_df.copy()
    w     = df["weight"].fillna(0).to_numpy()
    sigma = df["sigma"].fillna(0).to_numpy()
    # Variance contribution ~ w_i * sigma_i * (sum_j w_j * sigma_j * rho_ij)
    # Approximate cross terms with avg rho = 0.30 (reasonable for AI/power-heavy book)
    avg_rho = 0.30
    port_vol_proxy = float(np.sum(w * sigma) * np.sqrt(avg_rho)
                           + np.sqrt(np.sum((w * sigma) ** 2) * (1 - avg_rho)))
    contribs = w * sigma * (avg_rho * np.sum(w * sigma)
                            + (1 - avg_rho) * w * sigma) / max(port_vol_proxy ** 2, 1e-9)
    df["risk_contrib"] = contribs / max(contribs.sum(), 1e-9)
    return df.sort_values("risk_contrib", ascending=False)


# ---------------------------------------------------------------------------
# Deep analytics — chart helpers
# ---------------------------------------------------------------------------

def _composition_pie_fig(enriched_df: pd.DataFrame):
    df = enriched_df.copy()
    df["theme"] = df["ticker"].map(lambda t: THEMES.get(t, "Other"))
    grouped = df.groupby("theme")["mv"].sum().sort_values(ascending=False)
    palette = ["#1e3a5f", "#b8860b", "#2c7a4d", "#7d3c98", "#c0392b",
               "#2980b9", "#d35400", "#16a085", "#888888"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)
    wedges, _texts, autopcts = ax.pie(
        grouped.values, labels=grouped.index, autopct="%1.0f%%",
        colors=palette[:len(grouped)], startangle=90,
        textprops={"fontsize": 8, "color": "#222222"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.78,
    )
    for ap in autopcts:
        ap.set_color("white"); ap.set_fontweight("bold"); ap.set_fontsize(8)
    ax.set_title("Capital allocation by theme", fontsize=9, color=MPLNAVY,
                 fontweight="bold", pad=4)
    return fig


def _factor_bar_fig(fx: dict):
    labels = ["Market", "AI / Semis", "Power", "Idiosyncratic"]
    vals   = [fx["market"] * 100, fx["ai"] * 100, fx["power"] * 100, fx["idio"] * 100]
    palette = ["#4a90c4", "#1e3a5f", "#2c7a4d", "#888888"]
    fig, ax = plt.subplots(figsize=(7.5, 2.6), constrained_layout=True)
    bars = ax.barh(labels, vals, color=palette, edgecolor="white")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlabel("Share of portfolio variance", fontsize=8, color="#444444")
    ax.set_title("Factor decomposition of portfolio variance", fontsize=9,
                 color=MPLNAVY, fontweight="bold", pad=4)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=8, color="#222222")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.tick_params(labelsize=8, colors="#444444")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def _drawdown_dist_fig(max_dd: np.ndarray):
    fig, ax = plt.subplots(figsize=(7.5, 3.0), constrained_layout=True)
    pct = max_dd * 100
    ax.hist(pct, bins=60, color=MPLRED, alpha=0.65, edgecolor="white", linewidth=0.5)
    for q, c, lbl in [(10, MPLNAVY, "P10 (worst-decile)"),
                      (50, MPLGOLD, "P50 (median)"),
                      (90, "#27ae60", "P90 (mildest)")]:
        v = float(np.percentile(pct, q))
        ax.axvline(v, color=c, lw=1.6, label=f"{lbl}: {v:.0f}%")
    ax.set_xlabel("Maximum peak-to-trough drawdown (%)", fontsize=8, color="#444444")
    ax.set_ylabel("Frequency", fontsize=8, color="#444444")
    ax.set_title("Distribution of max drawdowns across simulated paths",
                 fontsize=9, color=MPLNAVY, fontweight="bold", pad=4)
    ax.legend(fontsize=7, framealpha=0.85, loc="upper left")
    ax.tick_params(labelsize=8, colors="#444444")
    ax.grid(True, alpha=0.2, linestyle=":", color="#aaaaaa", axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# ---------------------------------------------------------------------------
# Deep analytics — section flowables
# ---------------------------------------------------------------------------

def _executive_summary(book: dict, horizon_y: int, regime_name: str,
                       starting_mv: float, n_paths: int) -> list:
    stats   = book["stats"]
    df      = book["enriched_df"]
    cm      = _concentration_metrics(df)
    fx_table, fx = _factor_exposure_table(df)
    risks   = _identify_risks(df, stats)
    recs    = _recommend(df, stats)

    out = []
    out.append(_section_header("Executive Summary"))
    out.append(Spacer(1, 0.10 * inch))

    # Bottom-line line
    median_mult = stats["median"] / starting_mv if starting_mv > 0 else 0
    p10_mult    = stats["p10"]    / starting_mv if starting_mv > 0 else 0
    p90_mult    = stats["p90"]    / starting_mv if starting_mv > 0 else 0
    out.append(Paragraph(
        f"<b>Bottom line.</b> Starting from ${starting_mv:,.0f} across "
        f"{cm['n']} positions, the {horizon_y}-year median outcome is "
        f"<b>${stats['median']:,.0f}</b> ({median_mult:.1f}× starting capital). "
        f"The realistic range — P10 to P90 — runs from ${stats['p10']:,.0f} "
        f"({p10_mult:.1f}×) to ${stats['p90']:,.0f} ({p90_mult:.1f}×). "
        f"Probability of beating S&amp;P 500: <b>{stats['prob_beat_spx']*100:.0f}%</b>. "
        f"Probability of doubling capital: <b>{stats['prob_double']*100:.0f}%</b>. "
        f"Probability of ending below deposits: <b>{stats['prob_loss']*100:.1f}%</b>.",
        BODY
    ))

    # Three takeaways block
    out.append(Paragraph("Three things to know", H3))
    sys_share = fx["market"] + fx["ai"] + fx["power"]
    dom = max(("market beta", fx["market"]), ("AI/Semis cycle", fx["ai"]),
              ("power/nuclear cycle", fx["power"]),
              key=lambda x: x[1])
    takeaways = [
        f"<b>Concentration is structural.</b> Effective N is "
        f"<b>{cm['eff_n']:.1f}</b> across {cm['n']} positions; top 5 hold "
        f"<b>{cm['top5']*100:.0f}%</b> of the book. This is by design — a "
        f"satellite sleeve, not a diversified core.",
        f"<b>Your dominant factor is {dom[0]}</b> at <b>{dom[1]*100:.0f}%</b> of "
        f"variance. Systematic factors explain <b>{sys_share*100:.0f}%</b> of risk; "
        f"the remaining {fx['idio']*100:.0f}% is single-name event risk that no "
        f"factor model prices in advance.",
        f"<b>Drawdowns are baseline, not tail.</b> Median peak-to-trough is "
        f"<b>{abs(stats['dd_p50'])*100:.0f}%</b>; in the bad decile it reaches "
        f"<b>{abs(stats['dd_p10'])*100:.0f}%</b>. You will see a 30%+ drawdown "
        f"at some point during this horizon — budget for it emotionally.",
    ]
    for t in takeaways:
        out.append(Paragraph("•&nbsp; " + t, BODY))

    # Top risks (compressed)
    if risks:
        out.append(Paragraph("Top risks", H3))
        for title, _body, impact in risks[:3]:
            out.append(Paragraph(
                f"<b>[{impact}]</b> {title}", BODY
            ))

    # Action items (compressed)
    if recs:
        out.append(Paragraph("Action items", H3))
        for action, current, target, _rationale in recs[:4]:
            out.append(Paragraph(
                f"•&nbsp; <b>{action}</b> — {current} → {target}", BODY
            ))

    out.append(Spacer(1, 0.10 * inch))
    out.append(Paragraph(
        f"<i>Regime: {regime_name}  ·  Paths: {n_paths:,}  ·  Horizon: {horizon_y}y  ·  "
        f"Method: factor-model GBM with market / AI / power systematic factors plus "
        f"idiosyncratic Gaussian shocks.</i>",
        SMALL
    ))
    return out


def _concentration_table(cm: dict) -> Table:
    rows = [
        ["Metric", "Value", "Interpretation"],
        ["Number of positions",   f"{cm['n']}",
         "How many distinct names you hold."],
        ["Herfindahl-Hirschman Index (HHI)",  f"{cm['hhi']:.3f}",
         "Sum of squared weights. >0.18 is concentrated; <0.10 is diversified."],
        ["Effective N (1/HHI)",   f"{cm['eff_n']:.1f}",
         f"Equivalent equal-weight portfolio size. {cm['eff_n']:.1f} ≪ {cm['n']} means "
         f"a few names dominate."],
        ["Top-3 weight",          f"{cm['top3']*100:.1f}%",
         "Capital share of the three largest positions."],
        ["Top-5 weight",          f"{cm['top5']*100:.1f}%",
         "Capital share of the five largest positions."],
        ["Largest single position", f"{cm['max_ticker']} {cm['max_w']*100:.1f}%",
         "Highest-weight name. Above 8% creates real single-stock event risk."],
        ["Gini coefficient",      f"{cm['gini']:.2f}",
         "Inequality of weight distribution. 0 = equal-weight; 1 = winner-take-all."],
    ]
    ratios = [1.6, 0.9, 3.5]
    col_w = [c * CONTENT_W / sum(ratios) for c in ratios]
    body_style = ParagraphStyle("ConcBody", parent=_base["Normal"],
                                fontSize=8.5, fontName=SANS, leading=11)
    rows = [rows[0]] + [
        [r[0], r[1], Paragraph(r[2], body_style)] for r in rows[1:]
    ]
    t = Table(rows, colWidths=col_w)
    style = _table_style()
    style.add("ALIGN",    (0, 0), (-1, -1), "LEFT")
    style.add("ALIGN",    (1, 0), (1, -1),  "CENTER")
    style.add("FONTSIZE", (0, 1), (-1, -1), 8.5)
    style.add("VALIGN",   (0, 0), (-1, -1), "TOP")
    t.setStyle(style)
    return t


def _probability_ladder_table(ladder: dict) -> Table:
    years     = ladder["years"]
    multiples = ladder["multiples"]
    grid      = ladder["grid"]
    header = ["Multiple"] + [f"By Year {y}" for y in years]
    rows   = [header]
    for m in multiples:
        row = [f"{m}× starting"]
        for y in years:
            row.append(f"{grid[y][m]*100:.1f}%")
        rows.append(row)
    n_cols = len(header)
    col_w = [1.4 * inch] + [(CONTENT_W - 1.4 * inch) / (n_cols - 1)] * (n_cols - 1)
    t = Table(rows, colWidths=col_w)
    style = _table_style()
    style.add("ALIGN", (1, 0), (-1, -1), "CENTER")
    t.setStyle(style)
    return t


def _drawdown_table(dd: dict) -> Table:
    rows = [
        ["Statistic",                                      "Value"],
        ["Median max drawdown (typical)",                 f"{dd['p50_dd']*100:.1f}%"],
        ["Bad-decile max drawdown (P10)",                 f"{dd['p10_dd']*100:.1f}%"],
        ["Mild-decile max drawdown (P90)",                f"{dd['p90_dd']*100:.1f}%"],
        ["Worst observed (out of all paths)",             f"{dd['worst']*100:.1f}%"],
        ["Probability of seeing ≥30% drawdown",           f"{dd['frac_dd_30']*100:.1f}%"],
        ["Probability of seeing ≥50% drawdown",           f"{dd['frac_dd_50']*100:.1f}%"],
        ["Probability of seeing ≥70% drawdown",           f"{dd['frac_dd_70']*100:.1f}%"],
        ["Median months to recover from −25% drawdown",
         f"{dd['median_months_to_recover_25']:.0f} mo"
         if not np.isnan(dd['median_months_to_recover_25']) else "—"],
    ]
    ratios = [3.0, 1.0]
    col_w = [c * CONTENT_W / sum(ratios) for c in ratios]
    t = Table(rows, colWidths=col_w)
    style = _table_style()
    style.add("ALIGN", (0, 0), (0, -1), "LEFT")
    style.add("ALIGN", (1, 0), (1, -1), "RIGHT")
    t.setStyle(style)
    return t


def _stress_test_table(enriched_df: pd.DataFrame) -> Table:
    starting_mv = float(enriched_df["mv"].fillna(0).sum())
    rows = [["Scenario", "Description", "Portfolio %", "Portfolio $", "Resulting MV"]]
    body_style = ParagraphStyle("StressBody", parent=_base["Normal"],
                                fontSize=8, fontName=SANS, leading=11)
    color_rows = []
    for i, sc in enumerate(STRESS_SCENARIOS, start=1):
        impact = _apply_stress(enriched_df, sc["shocks"])
        new_mv = starting_mv * (1 + impact["pct"])
        rows.append([
            sc["name"],
            Paragraph(sc["desc"], body_style),
            f"{impact['pct']*100:+.1f}%",
            f"${impact['dollar']:+,.0f}",
            f"${new_mv:,.0f}",
        ])
        color_rows.append((i, impact["pct"]))
    ratios = [1.4, 2.6, 0.9, 1.0, 1.0]
    col_w = [c * CONTENT_W / sum(ratios) for c in ratios]
    t = Table(rows, colWidths=col_w)
    style = _table_style()
    style.add("ALIGN",   (2, 0), (-1, -1), "RIGHT")
    style.add("VALIGN",  (0, 0), (-1, -1), "MIDDLE")
    style.add("FONTSIZE",(0, 1), (-1, -1), 8.5)
    for i, pct in color_rows:
        c = colors.HexColor("#1f7a3a") if pct >= 0 else RED
        style.add("TEXTCOLOR", (2, i), (3, i), c)
    t.setStyle(style)
    return t


# Theme-based talking points used in the position deep-dives
_THEME_NOTES: dict[str, str] = {
    "AI / Semis":
        "AI infrastructure cycle exposure — moves with hyperscaler capex. Key risk: "
        "any guide-down from MSFT/META/GOOGL/AMZN datacenter spend triggers a sector-wide repricing.",
    "Nuclear / Power":
        "Long-duration thesis on electrification + nuclear renaissance + AI datacenter power demand. "
        "Key risk: project delays or policy reversals can stall the thesis for years.",
    "Defense / Industrial":
        "Defensive industrial exposure with secular tailwinds (defense budgets, electrical infrastructure). "
        "Lower beta, slower compounder. Key risk: capex slowdown.",
    "Software / Automation":
        "Productivity / automation play. Key risk: enterprise IT budget compression, AI-native competition disrupting the legacy moat.",
    "China / Mega-cap Tech":
        "Cheap megacap with China discount. Key risk: regulatory action, geopolitical escalation, ADR delisting risk.",
    "Fintech":
        "Direct play on retail trading & crypto sentiment. Highly cyclical earnings. Key risk: rate cuts compress NIM.",
    "Speculative / Other":
        "Asymmetric tail-bet position. Sized small intentionally. Key risk: total loss should be considered acceptable.",
}


def _position_deepdive_block(row: pd.Series) -> Table:
    ticker  = row["ticker"]
    weight  = row.get("weight", 0) * 100
    mv      = row.get("mv", 0)
    avg     = row.get("avg_cost", float("nan"))
    shares  = row.get("shares", 0)
    sigma   = row.get("sigma", 0) * 100
    mu      = row.get("mu", 0) * 100
    risk_c  = row.get("risk_contrib", 0) * 100 if pd.notna(row.get("risk_contrib")) else float("nan")
    theme   = THEMES.get(ticker, "Other")
    note    = row.get("thesis", "") or _THEME_NOTES.get(theme, "")
    if pd.notna(avg) and avg > 0:
        cost_basis = avg * shares
        unr   = mv - cost_basis
        unr_p = unr / cost_basis * 100 if cost_basis > 0 else 0
        pnl_str = (f"Cost basis ${cost_basis:,.0f}  ·  "
                   f"<b>Unrealized: {'+' if unr >= 0 else ''}${unr:,.0f} "
                   f"({unr_p:+.1f}%)</b>")
    else:
        pnl_str = ""

    title = f"{ticker}  ·  {weight:.1f}% weight  ·  {theme}"
    body_lines = [f"<b>Position:</b> {shares:.4f} shares  ·  Market value ${mv:,.0f}"]
    if pnl_str: body_lines.append(pnl_str)
    body_lines.append(f"<b>Risk profile:</b> assumed σ = {sigma:.0f}% annual  ·  "
                      f"expected µ = {mu:.0f}% annual  ·  contribution to portfolio risk ≈ "
                      f"{risk_c:.1f}%" if not np.isnan(risk_c) else "")
    body_lines.append(f"<b>Thesis:</b> {note}")

    cells = [[Paragraph(title, RISK_TITLE)],
             [Paragraph("<br/>".join(b for b in body_lines if b), RISK_BODY)]]
    t = Table(cells, colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 0.5, MID),
        ("BACKGROUND",    (0, 0), (-1,  0), GHOST),
        ("BACKGROUND",    (0, 1), (-1,  1), WHITE),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return t


# ---------------------------------------------------------------------------
# Cover page flowables
# ---------------------------------------------------------------------------

def _cover_flowables() -> list:
    # All cover content is drawn directly on canvas in _cover_canvas.
    # This just triggers the page break to move to the main template.
    return [NextPageTemplate("main"), PageBreak()]

# ---------------------------------------------------------------------------
# Per-book flowables
# ---------------------------------------------------------------------------

def _book_flowables(book: dict, horizon_y: int, regime_name: str,
                    n_paths: int, lump_sum: float, monthly: float,
                    run_date: str, risk_model_meta: dict | None = None) -> list:
    name        = book["name"]
    enriched_df = book["enriched_df"]
    port        = book["port"]
    spx         = book["spx"]
    stats       = book["stats"]
    starting_mv = float(enriched_df["mv"].sum()) if "mv" in enriched_df.columns else stats["deposits"]

    story = []
    cm     = _concentration_metrics(enriched_df)
    fx_t, fx = _factor_exposure_table(enriched_df)
    ladder = _probability_ladder(port, horizon_y, starting_mv)
    dd     = _drawdown_stats(port)
    risk_df = _position_variance_contribution(enriched_df)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Executive Summary
    # ════════════════════════════════════════════════════════════════════════
    story.extend(_executive_summary(book, horizon_y, regime_name, starting_mv, n_paths))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Headline Metrics (tear sheet)
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Headline Metrics"))
    story.append(Spacer(1, 0.10 * inch))
    contrib_note = ""
    if lump_sum: contrib_note += f"  ·  ${lump_sum:,.0f} lump sum"
    if monthly:  contrib_note += f"  ·  ${monthly:,.0f}/mo contributions"
    meta = (f"{name}  ·  {regime_name}  ·  {horizon_y}-Year Horizon  ·  "
            f"{n_paths:,} paths{contrib_note}")
    story.append(Paragraph(meta, SMALL))
    story.append(Spacer(1, 0.10 * inch))
    story.append(_metrics_grid(stats))
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph(
        f"Starting from <b>${starting_mv:,.0f}</b>, the most likely outcome after "
        f"{horizon_y} years is <b>${stats['median']:,.0f}</b> (median). "
        f"The right tail (P90) reaches <b>${stats['p90']:,.0f}</b>; the left tail "
        f"(P10) holds at <b>${stats['p10']:,.0f}</b>. "
        f"You will likely see a {abs(stats['dd_p50'])*100:.0f}% peak-to-trough "
        f"drawdown along the way — this is the median experience, not a tail event.",
        BODY
    ))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Portfolio Holdings (with P&L)
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Portfolio Holdings"))
    story.append(Spacer(1, 0.10 * inch))
    top3 = enriched_df.sort_values("weight", ascending=False).head(3)
    top3_str = "  ·  ".join(
        f"{r['ticker']} {r['weight']*100:.1f}%" for _, r in top3.iterrows()
        if pd.notna(r.get("weight"))
    )
    story.append(Paragraph(
        f"<b>{len(enriched_df)} positions  ·  "
        f"Total MV ${enriched_df['mv'].sum():,.2f}  ·  "
        f"Top 3:</b>  {top3_str}", SMALL
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_holdings_table(enriched_df))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Composition by Theme (table + pie)
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Composition by Theme"))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(
        "Capital grouped by investment thesis. Within-theme concentration means a "
        "sector-wide shock — AI capex pause, power demand softening, geopolitical "
        "escalation — moves multiple positions in the same direction simultaneously.",
        SMALL
    ))
    story.append(Spacer(1, 0.10 * inch))
    story.append(_theme_breakdown_table(enriched_df))
    story.append(Spacer(1, 0.16 * inch))
    story.append(_fig_to_image(_composition_pie_fig(enriched_df), 5.5 * inch, 3.4 * inch))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Concentration Metrics
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Concentration Profile"))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(
        "Standard concentration diagnostics. A book with HHI < 0.10 and effective N "
        "above 10 behaves close to a diversified index; a book above HHI 0.18 with "
        "effective N below 6 inherits substantial single-name risk.",
        SMALL
    ))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_concentration_table(cm))
    story.append(Spacer(1, 0.14 * inch))
    if cm['eff_n'] < 6:
        verdict = "<b>Verdict: highly concentrated.</b> This is a high-conviction satellite sleeve, not a diversified core. Position sizing matters more than picking."
    elif cm['eff_n'] < 10:
        verdict = "<b>Verdict: moderately concentrated.</b> Mid-ground between satellite and core; a few positions still dominate outcomes."
    else:
        verdict = "<b>Verdict: well diversified.</b> Single-name risk is muted; results will be driven by factor exposure rather than individual picks."
    story.append(Paragraph(verdict, BODY))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Factor Exposure (table + chart)
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Factor Exposure Decomposition"))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(
        "Portfolio variance decomposed across the three systematic factors and "
        "stock-specific (idiosyncratic) risk. Systematic exposure cannot be "
        "diversified away by adding more names within the same theme; it can only "
        "be reduced by allocating to factors that are genuinely uncorrelated.",
        SMALL
    ))
    story.append(Spacer(1, 0.10 * inch))
    story.append(fx_t)
    story.append(Spacer(1, 0.14 * inch))
    story.append(_fig_to_image(_factor_bar_fig(fx), CONTENT_W, 2.4 * inch))
    story.append(Spacer(1, 0.10 * inch))
    sys_share = fx["market"] + fx["ai"] + fx["power"]
    dom = max(("Market", fx["market"]), ("AI/Semis", fx["ai"]),
              ("Power", fx["power"]), ("Idiosyncratic", fx["idio"]),
              key=lambda x: x[1])
    story.append(Paragraph(
        f"<b>Read:</b> {dom[0]} is your largest exposure at {dom[1]*100:.0f}% of variance. "
        f"Systematic factors together explain {sys_share*100:.0f}% of risk; the remaining "
        f"{fx['idio']*100:.0f}% is idiosyncratic — company-specific events that no factor "
        f"model prices in advance. Idiosyncratic risk is uncompensated unless your picks "
        f"actually generate alpha; otherwise it is pure variance drag.",
        BODY
    ))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Monte Carlo Simulation Results
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Monte Carlo Simulation"))
    story.append(Spacer(1, 0.10 * inch))
    chart_w = CONTENT_W
    chart_h = 2.95 * inch
    fan_fig  = _fan_chart_fig(port, f"{name} — fan chart, {horizon_y}y", starting_mv)
    hist_fig = _hist_chart_fig(port, f"{name} — terminal value distribution (clipped at P99)", starting_mv)
    story.append(_fig_to_image(fan_fig, chart_w, chart_h))
    story.append(Spacer(1, 0.10 * inch))
    story.append(_fig_to_image(hist_fig, chart_w, chart_h))
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(
        "Fan chart: P10 / P25 / P50 / P75 / P90 bands across all simulated paths; "
        "red-shaded zone is below starting capital. Histogram is clipped at the 99th "
        "percentile so the bulk of the distribution is visible — the tail extends much further.",
        CAVEAT
    ))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 8 — Quantitative Summary (percentiles + milestones + ladder)
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Quantitative Summary"))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(
        f"Starting portfolio value: ${starting_mv:,.0f}  |  "
        f"Median end value: ${stats['median']:,.0f}  |  "
        f"Mean end value: ${stats['mean']:,.0f}",
        SUMMARY_LINE
    ))
    story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=MID, spaceAfter=10))
    story.append(Paragraph("Percentile Outcomes", H3))
    story.extend(_quant_tables(stats))
    story.append(Spacer(1, 0.16 * inch))
    story.append(Paragraph("Wealth Trajectory — Year-by-Year", H3))
    story.append(Paragraph(
        "Where the simulation lands at each milestone year. The multiple shows the "
        "median path's growth on starting capital.",
        SMALL
    ))
    story.append(Spacer(1, 0.06 * inch))
    story.append(_milestones_table(port, horizon_y, starting_mv))
    story.append(Spacer(1, 0.16 * inch))
    story.append(Paragraph("Probability Ladder", H3))
    story.append(Paragraph(
        "Probability of reaching each wealth multiple by year Y. Read across to see how "
        "different multiples compress as the horizon shortens.",
        SMALL
    ))
    story.append(Spacer(1, 0.06 * inch))
    story.append(_probability_ladder_table(ladder))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 9 — Drawdown Analysis
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Drawdown Analysis"))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(
        "Distribution of the worst peak-to-trough drawdown observed in each simulated "
        "path. Even a portfolio with strong terminal outcomes will travel through severe "
        "drawdowns along the way — sizing emotional capacity matters as much as sizing positions.",
        SMALL
    ))
    story.append(Spacer(1, 0.10 * inch))
    story.append(_fig_to_image(_drawdown_dist_fig(dd["max_dd_array"]), CONTENT_W, 2.8 * inch))
    story.append(Spacer(1, 0.14 * inch))
    story.append(_drawdown_table(dd))
    story.append(Spacer(1, 0.12 * inch))
    p30 = dd["frac_dd_30"] * 100
    p50 = dd["frac_dd_50"] * 100
    story.append(Paragraph(
        f"<b>Read:</b> {p30:.0f}% of paths see at least a 30% drawdown; {p50:.0f}% see 50%+. "
        f"The relevant question is not whether a deep drawdown happens, but whether you "
        f"hold through it — a forced sale near the bottom converts a paper loss into a "
        f"realized one and locks out the recovery.",
        BODY
    ))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 10 — Stress Test Scenarios
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Stress Test Scenarios"))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(
        "Instantaneous shock scenarios applied to current factor loadings. Each row "
        "shows the immediate mark-to-market impact if the named scenario hit today. "
        "These are not full-cycle simulations — they isolate sensitivity to specific "
        "factor moves so you can size for the bad day, not just the typical one.",
        SMALL
    ))
    story.append(Spacer(1, 0.10 * inch))
    story.append(_stress_test_table(enriched_df))
    story.append(Spacer(1, 0.12 * inch))
    worst = min(STRESS_SCENARIOS, key=lambda s: _apply_stress(enriched_df, s["shocks"])["pct"])
    worst_impact = _apply_stress(enriched_df, worst["shocks"])
    story.append(Paragraph(
        f"<b>Worst scenario:</b> <i>{worst['name']}</i> delivers an immediate "
        f"{worst_impact['pct']*100:.1f}% drawdown — roughly ${worst_impact['dollar']:+,.0f}. "
        f"Idiosyncratic risk is not modeled in these shocks; real outcomes can be worse "
        f"because individual names will react beyond their factor sensitivities.",
        BODY
    ))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 11 — Position Deep-Dives
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Position Deep-Dives — Top Holdings"))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(
        "Detailed read on the largest positions by weight. Risk contribution is an "
        "approximation using assumed within-portfolio correlation of 0.30.",
        SMALL
    ))
    story.append(Spacer(1, 0.10 * inch))
    top_n = min(7, len(risk_df))
    for i, (_, row) in enumerate(risk_df.sort_values("weight", ascending=False).head(top_n).iterrows()):
        if i > 0:
            story.append(Spacer(1, 0.08 * inch))
        story.append(_position_deepdive_block(row))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 12 — Risks, Recommendations & Action Plan
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Risks & Recommended Actions"))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph("How to read these results", H3))
    for para in _interpret_results(stats, horizon_y, starting_mv):
        story.append(Paragraph(para, BODY))
    story.append(Spacer(1, 0.10 * inch))

    risks = _identify_risks(enriched_df, stats)
    if risks:
        story.append(Paragraph("Key Risks", H3))
        for title, body, impact in risks:
            story.append(Spacer(1, 0.06 * inch))
            story.append(_risk_card(title, body, impact))

    recs = _recommend(enriched_df, stats)
    if recs:
        story.append(Spacer(1, 0.14 * inch))
        story.append(Paragraph("Recommended Actions", H3))
        story.append(Spacer(1, 0.06 * inch))
        story.append(_action_table(recs))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 13 — Methodology & Assumptions
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Methodology & Assumptions"))
    story.append(Spacer(1, 0.10 * inch))

    # ── How the model works (plain-English primer) ───────────────────────────
    story.append(Paragraph("How to read the factor model", H3))
    story.append(Paragraph(
        "Every position's monthly return is decomposed into four pieces: "
        "<b>(1) market move</b> (proxied by SPY) — what the broad equity market did; "
        "<b>(2) AI / semiconductor move</b> (SMH proxy) — the unique component "
        "of semis after stripping out market beta; "
        "<b>(3) power / nuclear move</b> (URA proxy) — unique to that theme after "
        "stripping market and AI; "
        "<b>(4) company-specific noise</b> — the idiosyncratic residual that no "
        "factor explains. Each ticker has a loading on each factor (the f-values: "
        "<i>f_market, f_ai, f_power</i>) — these are <b>variance fractions</b>, not "
        "betas, and they sum to ≤ 1. The residual is idiosyncratic risk. "
        "Simulated paths are generated as Geometric Brownian Motion (GBM): each "
        "month, every position's price is multiplied by exp(drift + factor shocks + "
        "idio shock), where the shocks are correlated across positions via the "
        "shared factors. 10,000 paths give the percentile bands you see in the fan chart.",
        BODY
    ))
    story.append(Spacer(1, 0.10 * inch))

    # ── Provenance: where σ and factor loadings came from ────────────────────
    story.append(Paragraph("Risk model provenance", H3))
    if risk_model_meta and risk_model_meta.get("use_hist"):
        period = risk_model_meta.get("hist_period", "2y")
        n_ovr = risk_model_meta.get("n_overridden")
        n_fb  = risk_model_meta.get("n_fallback")
        fb_tk = risk_model_meta.get("fallback_tickers", [])
        if n_ovr is not None:
            cov_str = f"{n_ovr} of {n_ovr + (n_fb or 0)} tickers"
        else:
            cov_str = "all tickers with sufficient history"
        fb_str = (f" Tickers with insufficient history fell back to hand-set defaults: "
                  f"{', '.join(fb_tk)}." if fb_tk else "")
        provenance_text = (
            f"<b>σ and factor loadings were estimated from {period} of daily price "
            f"history</b> ({cov_str}). Method: factor proxies SPY (market), SMH (AI/semis), "
            f"and URA (power) are Gram-Schmidt orthogonalized in that order, then each "
            f"ticker's daily log returns are regressed against the orthogonalized factor "
            f"set via OLS. Variance fractions are derived from the regression coefficients. "
            f"<b>Expected return (μ) remains hand-set</b> — historical mean returns are too "
            f"noisy at 2-year horizons to be useful as forward forecasts (the standard "
            f"error on a 2-year mean is roughly σ/√2, which dwarfs any plausible signal)."
            f"{fb_str}"
        )
    else:
        provenance_text = (
            "<b>σ and factor loadings are hand-set priors</b>, not estimated from "
            "price history. They reflect rough sector-level expectations and should be "
            "treated as plausible-order-of-magnitude inputs rather than calibrated values. "
            "Toggle 'Use historical estimates (2y)' in Portfolio Lab to override these "
            "with values regressed from real return history."
        )
    story.append(Paragraph(provenance_text, BODY))
    story.append(Spacer(1, 0.12 * inch))

    # ── Per-ticker factor model table ────────────────────────────────────────
    story.append(Paragraph("Per-ticker factor model (effective values)", H3))
    story.append(Paragraph(
        "Per-ticker annual expected return (μ) and volatility (σ). Factor loadings "
        "are variance fractions: f_market + f_ai + f_power ≤ 1; remainder is idiosyncratic.",
        SMALL
    ))
    story.append(Spacer(1, 0.06 * inch))
    story.append(_factor_table(enriched_df))
    story.append(Spacer(1, 0.18 * inch))

    # ── Caveats ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Model caveats", H3))
    story.append(Paragraph(
        "<b>Lognormal returns have no fat tails</b> — the real left tail is worse than what "
        "this model produces; black-swan scenarios are systematically understated. "
        "<b>Static factor correlations</b> (0.30 assumed in risk-contribution math) "
        "understate crash-regime co-movement; in real selloffs correlations spike toward 1. "
        "<b>No skill-alpha is assumed</b> — μ values are coarse priors, not forecasts. "
        "<b>No tax drag, dividend reinvestment, or rebalancing friction</b> is modeled. "
        "<b>Stress tests apply only the named factor shocks</b> — idiosyncratic events on "
        "top of factor moves can compound losses beyond what is shown.",
        BODY
    ))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 14 — Glossary
    # ════════════════════════════════════════════════════════════════════════
    story.append(_section_header("Glossary"))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(
        "Quick definitions for the terms used throughout this report.",
        SMALL
    ))
    story.append(Spacer(1, 0.10 * inch))
    glossary = [
        ("σ (sigma) — annualized volatility",
         "Standard deviation of returns, scaled to one year. A position with σ=50% is "
         "expected to move within ±50% of its trend in roughly two-thirds of years; in "
         "one year out of twenty it moves more than ±100%. Higher σ means wider outcome "
         "distribution, both up and down."),
        ("μ (mu) — annual expected return",
         "Drift term — the average annual return assumed for each position. Hand-set "
         "in this model; not derived from history. Treat as a plausible prior, not a "
         "forecast."),
        ("Factor loading (f_market, f_ai, f_power)",
         "Share of the position's variance explained by each systematic factor. "
         "f_market = 0.30 means 30% of the position's monthly variance comes from market-wide "
         "moves. The three values plus idiosyncratic must sum to 1."),
        ("Idiosyncratic risk",
         "The portion of variance not explained by any factor — pure company-specific "
         "noise. An earnings miss, CEO change, contract loss, or fraud reveal lives here. "
         "It can be diversified away by adding genuinely uncorrelated positions; it cannot "
         "be hedged with index ETFs."),
        ("R² (coefficient of determination)",
         "How much of a position's variance is explained by the factor regression. "
         "R² = 70% means 70% of the position's daily returns can be predicted from "
         "SPY/SMH/URA moves; the remaining 30% is idiosyncratic. Mature large-caps "
         "typically score 50–80%; recent IPOs and crypto often score below 20%."),
        ("HHI (Herfindahl-Hirschman Index)",
         "Sum of squared portfolio weights. 0 = perfect diversification (infinite "
         "positions, equal weight); 1 = single name. Above 0.18 is generally considered "
         "concentrated; 0.10 or below behaves index-like."),
        ("Effective N",
         "1 / HHI. The number of equally-weighted positions that would produce the same "
         "concentration as your actual book. Your stated count of positions overstates "
         "diversification; effective N is the honest number."),
        ("P10 / P50 / P90",
         "10th / 50th / 90th percentile outcomes from the simulation. P50 is the median "
         "(half of paths land above, half below). P10 is the 'bad decile' — only 10% of "
         "paths come out worse. P90 is the 'good decile.'"),
        ("Max drawdown",
         "The largest peak-to-trough decline observed within a single simulated path. "
         "A 50% max drawdown means at the worst point in that path, the portfolio was "
         "down 50% from its prior peak. Median max drawdown is what you should expect "
         "to live through, not a tail event."),
        ("GBM (Geometric Brownian Motion)",
         "The mathematical model used to generate simulated price paths. Each price "
         "follows a continuous random walk where percentage moves (not dollar moves) are "
         "normally distributed. Standard for equity modeling but understates extreme "
         "events — real markets have fatter tails than GBM produces."),
        ("Variance fraction",
         "Unlike a regression beta (which is a sensitivity coefficient), a variance "
         "fraction tells you what share of the position's total volatility comes from "
         "that factor. Variance fractions are bounded [0, 1] and sum to ≤ 1 across all "
         "factors — anything left over is idiosyncratic."),
        ("Stress test",
         "An instantaneous mark-to-market impact analysis under a scripted factor shock. "
         "Unlike the Monte Carlo simulation, stress tests are not probabilistic — they "
         "answer 'if X happened today, what would the portfolio do?' and ignore "
         "idiosyncratic moves on top of the factor shock."),
    ]
    glossary_rows = [["Term", "Meaning"]]
    body_style = ParagraphStyle("GlossBody", parent=_base["Normal"],
                                fontSize=8.5, fontName=SERIF, leading=11.5)
    title_style = ParagraphStyle("GlossTitle", parent=_base["Normal"],
                                 fontSize=8.5, fontName=SANS_B, leading=11)
    for term, defn in glossary:
        glossary_rows.append([Paragraph(term, title_style), Paragraph(defn, body_style)])
    ratios = [1.6, 4.0]
    col_w = [c * CONTENT_W / sum(ratios) for c in ratios]
    g_table = Table(glossary_rows, colWidths=col_w)
    g_style = _table_style()
    g_style.add("VALIGN", (0, 0), (-1, -1), "TOP")
    g_style.add("ALIGN",  (0, 0), (-1, -1), "LEFT")
    g_table.setStyle(g_style)
    story.append(g_table)
    return story

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_report(
    books: list[dict],
    horizon_y: int,
    regime_name: str,
    n_paths: int,
    lump_sum: float = 0.0,
    monthly: float = 0.0,
    risk_model_meta: dict | None = None,
) -> bytes:
    """
    books: list of {name, enriched_df, port, spx, stats}
    Returns raw PDF bytes for st.download_button.
    """
    buf      = io.BytesIO()
    run_date = datetime.now().strftime("%B %d, %Y")
    title    = f"Portfolio Lab — {', '.join(b['name'] for b in books)}"
    doc      = _make_doc(buf, title, run_date)

    # Populate cover data before building so the canvas callback can read it
    doc._covers = [
        {"name": b["name"], "horizon_y": horizon_y, "n_paths": n_paths,
         "regime_name": regime_name, "run_date": run_date}
        for b in books
    ]

    story = []
    for book in books:
        story.extend(_cover_flowables())
        story.extend(_book_flowables(
            book, horizon_y, regime_name, n_paths, lump_sum, monthly, run_date,
            risk_model_meta=risk_model_meta,
        ))

    if len(books) > 1:
        story.append(_section_header("Side-by-Side Comparison"))
        story.append(Spacer(1, 0.10 * inch))
        best_p10 = max(books, key=lambda b: b["stats"]["p10"])
        best_dd  = max(books, key=lambda b: b["stats"]["dd_p50"])
        story.append(Paragraph(
            f"<b>{best_p10['name']}</b> has the best P10 downside outcome. "
            f"<b>{best_dd['name']}</b> has the lowest median peak-to-trough drawdown. "
            f"Median end values typically cluster in a narrow range — the meaningful "
            f"difference between books is in the tails, not the middle.",
            BODY
        ))
        story.append(Spacer(1, 0.10 * inch))
        story.append(_comparison_table(books))

    doc.build(story)
    return buf.getvalue()
