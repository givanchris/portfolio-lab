import json
import time
from pathlib import Path

import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Portfolio News",
    page_icon="📰",
    layout="wide",
)

HOLDINGS_FP = Path(__file__).parent.parent / "holdings.json"
NAVY = "#1e3a5f"
GOLD = "#b8860b"

# ---------------------------------------------------------------------------
# Styles (mirrors main app)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background-color: #f0f2f6;
  }

  [data-testid="stSidebar"] { background: #1a2e4a !important; color: #cbd5e0; }
  [data-testid="stSidebar"] *:not(button):not(input):not(select):not(textarea):not(svg):not(path):not(h1):not(h2):not(h3):not(strong) {
    color: #cbd5e0 !important;
  }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] strong { color: #f0c040 !important; }

  .block-container { padding-top: 1.5rem; max-width: 1200px; }

  .page-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5282 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(30,58,95,0.25);
  }
  .page-header h1 { color: white !important; margin: 0; font-size: 1.8rem; font-weight: 700; }
  .page-header .subtitle { color: rgba(255,255,255,0.6); margin: 5px 0 0 0; font-size: 0.84rem; }

  .news-card {
    background: white;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid #1e3a5f;
    transition: box-shadow 0.15s ease;
  }
  .news-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
  .news-headline {
    font-size: 0.97rem;
    font-weight: 600;
    color: #1e3a5f;
    text-decoration: none;
    line-height: 1.4;
  }
  .news-headline:hover { color: #b8860b; }
  .news-meta {
    font-size: 0.75rem;
    color: #718096;
    margin-top: 8px;
    display: flex;
    gap: 10px;
    align-items: center;
  }
  .ticker-badge {
    background: #1e3a5f;
    color: white !important;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
  }
  .source-tag { color: #b8860b; font-weight: 500; }
  .time-tag { color: #a0aec0; }

  .section-label {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0 12px 0;
  }
  .section-label::before {
    content: '';
    width: 3px;
    height: 16px;
    background: #b8860b;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .section-label span {
    font-size: 0.77rem;
    font-weight: 600;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.09em;
  }
  h2, h3 { color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# News fetch — cached 5 minutes
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_portfolio_news(tickers: tuple[str, ...], max_per_ticker: int = 6) -> list[dict]:
    from datetime import datetime, timezone

    items = []
    seen_titles: set[str] = set()

    def _parse_ts(pub_date: str) -> int:
        try:
            return int(datetime.fromisoformat(pub_date.replace("Z", "+00:00")).timestamp())
        except Exception:
            return 0

    for t in tickers:
        try:
            raw = yf.Ticker(t).news or []
            for item in raw[:max_per_ticker]:
                # yfinance 1.2+ nests everything under "content"
                content = item.get("content") or {}
                title = (content.get("title") or item.get("title") or "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                link = (
                    (content.get("canonicalUrl") or {}).get("url")
                    or (content.get("clickThroughUrl") or {}).get("url")
                    or item.get("link", "")
                )
                publisher = (
                    (content.get("provider") or {}).get("displayName")
                    or item.get("publisher", "")
                )
                pub_date = content.get("pubDate", "")
                pub_time = (
                    _parse_ts(pub_date) if pub_date
                    else int(item.get("providerPublishTime") or 0)
                )
                items.append({
                    "ticker":    t,
                    "title":     title,
                    "publisher": publisher,
                    "link":      link,
                    "ts":        pub_time,
                })
        except Exception:
            pass

    items.sort(key=lambda x: x["ts"], reverse=True)
    return items


def _time_ago(ts: int) -> str:
    if not ts:
        return ""
    delta = int(time.time()) - ts
    if delta < 3600:
        m = max(delta // 60, 1)
        return f"{m}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _news_card_html(item: dict) -> str:
    title    = item["title"].replace('"', "&quot;")
    link     = item.get("link", "#") or "#"
    ticker   = item["ticker"]
    source   = item.get("publisher", "")
    ago      = _time_ago(item["ts"])
    return f"""
<div class="news-card">
  <a class="news-headline" href="{link}" target="_blank" rel="noopener noreferrer">{item['title']}</a>
  <div class="news-meta">
    <span class="ticker-badge">{ticker}</span>
    {'<span class="source-tag">' + source + '</span>' if source else ''}
    {'<span class="time-tag">' + ago + '</span>' if ago else ''}
  </div>
</div>"""

# ---------------------------------------------------------------------------
# Resolve tickers
# ---------------------------------------------------------------------------

books = st.session_state.get("books", {})
book_a = books.get("A — Current")

if book_a is not None and not book_a.empty and "ticker" in book_a.columns:
    tickers = tuple(book_a["ticker"].dropna().astype(str).str.upper().tolist())
elif HOLDINGS_FP.exists():
    try:
        data = json.loads(HOLDINGS_FP.read_text())
        tickers = tuple(h["ticker"].upper() for h in data.get("holdings", []))
    except Exception:
        tickers = ()
else:
    tickers = ()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

ticker_str = " · ".join(tickers) if tickers else "No holdings loaded"
st.markdown(f"""
<div class="page-header">
  <div>
    <h1>Portfolio News</h1>
    <p class="subtitle">{ticker_str}</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Ticker filter + refresh
# ---------------------------------------------------------------------------

if not tickers:
    st.info("No holdings found. Open Portfolio Lab first, or make sure holdings.json exists.")
    st.stop()

col_filter, col_refresh = st.columns([4, 1])
with col_filter:
    selected = st.multiselect(
        "Filter by ticker",
        options=list(tickers),
        default=list(tickers),
        key="news_ticker_filter",
    )
with col_refresh:
    st.write("")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

active_tickers = tuple(selected) if selected else tickers

# ---------------------------------------------------------------------------
# Fetch & render
# ---------------------------------------------------------------------------

with st.spinner("Fetching headlines…"):
    news = fetch_portfolio_news(active_tickers)

if not news:
    st.info("No recent news found for your holdings.")
    st.stop()

st.markdown(
    f'<div class="section-label"><span>{len(news)} headlines</span></div>',
    unsafe_allow_html=True,
)

for item in news:
    st.markdown(_news_card_html(item), unsafe_allow_html=True)
