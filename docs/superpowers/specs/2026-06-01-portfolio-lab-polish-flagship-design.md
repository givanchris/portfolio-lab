# Portfolio Lab — Polish & Flagship Pass

**Date:** 2026-06-01
**Goal:** Turn the Portfolio Lab Streamlit app into a portfolio-grade signal of financial-modeling skill for employers. Balanced pass weighted toward visual polish, with one flagship quant feature built well. Remove any path that exposes the owner's real holdings.

## Decisions
- **Aesthetic:** Dark fintech terminal. Reuse/refine existing dark CSS (bg `#07101e`, blue structure `#2e74e8`), introduce signature teal data-accent `#00e0a4` for KPI values and the flagship chart. Green/red reserved for P&L.
- **Flagship:** Efficient Frontier optimizer (mean-variance).
- **Out of scope (this pass):** VaR/CVaR dashboard, risk decomposition (MCTR/component VaR). Deferred.

## Work items

### 1. Privacy / cleanup
- `HOLDINGS_FILE` default fallback changes from `holdings.json` → `holdings_demo.json` (portfolio_lab.py:42). Real holdings load only when env var is explicitly set locally. Real file already gitignored + never committed.
- Rename the "My Portfolio" book label → "Sample Portfolio" everywhere it appears (book_names, REGIMES/desc dicts, comments).
- Finish & commit the in-progress ticker-universe edit (~400 searchable symbols). Public betas added to DEFAULT_RISK_MODEL (XRP, HUBB, TCEHY, MOG-A, MIR) are risk params, not holdings — keep.

### 2. Visual polish — commit to dark terminal
- `.streamlit/config.toml`: switch `base` from `light` → `dark` with colors matching the CSS (`backgroundColor=#07101e`, `secondaryBackgroundColor=#0c1828`, `primaryColor=#2e74e8`, `textColor` light). Fixes widget/dropdown/slider inconsistencies.
- Consolidate palette into a single source of truth; add teal `#00e0a4` accent token. KPI values use JetBrains Mono. Unify section labels, tab styling, divider.
- One shared chart palette/helper so every matplotlib figure matches (fan chart, hist, donut, factor bar, frontier).
- Header with live "as of" timestamp.

### 3. Flagship: Efficient Frontier Optimizer (new tab)
- **Covariance:** factor-structured Σ = B·Σ_f·Bᵀ + diag(idiosyncratic), built from existing market/AI/power loadings + per-ticker σ. Idiosyncratic variance = max(σ_total² − factor-explained var, floor).
- **Expected returns:** per-ticker μ from the risk model (hand-set, per house rules).
- **Frontier:** N random long-only Dirichlet portfolios scatter (vol vs return, colored by Sharpe); analytical efficient frontier overlay; mark Max-Sharpe (tangency) and Min-Variance; plot current portfolio as a star.
- **Optimize:** `scipy.optimize` SLSQP — max-Sharpe and min-variance, constraints: weights ≥ 0, sum = 1, optional max-weight cap. Risk-free input (default 4.5%).
- **Output:** optimal vs current weights diff table; frontier chart; KPI cards (current Sharpe vs max-Sharpe). All cached.
- **Degrade gracefully:** if a ticker lacks loadings/σ, fall back to GENERIC; if <2 valid tickers, show an info message instead of erroring.

### 4. Smoothness
- Audit yfinance failure handling so the demo book never throws end-to-end. Optimizer guards against empty/NaN inputs.

## Success criteria
- App boots on demo data with zero exceptions; no real-holdings path on default.
- Every view renders in the dark terminal theme with consistent components.
- Optimizer tab produces a frontier, optimal weights, and a current-vs-optimal comparison for the demo portfolio.

## Notes / house rules respected
- portfolio_lab.py stays self-contained (one file).
- Expensive work (yfinance, MC, regression, optimization) cached.
- μ stays hand-set; σ and loadings may use historical when that mode is on.
- No rebalancing/tax/skill-alpha added beyond the optimizer's mean-variance suggestion (which is clearly framed as theoretical).
