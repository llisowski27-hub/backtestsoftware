from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest import (
    _fill_position,
    _fill_position_bidirectional,
    atr,
    compute_analytics,
    detect_timeframe,
    ema,
    ibs,
    load_ohlcv,
    momentum,
    range_filter,
    rsi,
    sma,
    stdev_bands,
    run_backtest,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Backtest Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { color: #8b949e; font-size: 0.78rem; margin-bottom: 4px; }
    .metric-value { color: #e6edf3; font-size: 1.45rem; font-weight: 600; }
    .metric-pos   { color: #3fb950; }
    .metric-neg   { color: #f85149; }
    .metric-neu   { color: #e6edf3; }
    [data-testid="stSidebar"] { background: #0d1117; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PARAMETRIC STRATEGY BUILDERS
# ---------------------------------------------------------------------------

def build_ibs_ma(params: dict):
    def strategy(df: pd.DataFrame) -> pd.Series:
        ibs_val = ibs(df["high"], df["low"], df["close"])
        ma_fn = sma if params["ma_type"] == "SMA" else ema
        ma_val = ma_fn(df["close"], params["ma_period"])
        raw = pd.Series(0, index=df.index, dtype=int)
        raw[(ibs_val < params["ibs_entry"]) & (df["close"] > ma_val)] = 1
        raw[ibs_val > params["ibs_exit"]] = -1
        return _fill_position(raw)
    return strategy


def build_rsi_mean_reversion(params: dict):
    def strategy(df: pd.DataFrame) -> pd.Series:
        rsi_val = rsi(df["close"], params["rsi_period"])
        raw = pd.Series(0, index=df.index, dtype=int)
        raw[rsi_val < params["oversold"]] = 1
        raw[rsi_val > params["exit_level"]] = -1
        return _fill_position(raw)
    return strategy


def build_momentum_trend(params: dict):
    def strategy(df: pd.DataFrame) -> pd.Series:
        mom = momentum(df["close"], params["mom_period"])
        ma_val = sma(df["close"], params["ma_period"])
        raw = pd.Series(0, index=df.index, dtype=int)
        raw[(mom > params["entry_thresh"]) & (df["close"] > ma_val)] = 1
        raw[(mom < -params["entry_thresh"]) & (df["close"] < ma_val)] = -1
        raw[mom.abs() < params["exit_thresh"]] = -2
        fill = _fill_position_bidirectional if params["bidirectional"] else _fill_position
        return fill(raw)
    return strategy


def build_range_filter_trend(params: dict):
    def strategy(df: pd.DataFrame) -> pd.Series:
        atr_val = atr(df["high"], df["low"], df["close"], params["atr_period"])
        rf = range_filter(df["close"], atr_val, params["multiplier"])
        raw = pd.Series(0, index=df.index, dtype=int)
        cross_up = (df["close"] > rf) & (df["close"].shift(1) <= rf.shift(1))
        cross_dn = (df["close"] < rf) & (df["close"].shift(1) >= rf.shift(1))
        raw[cross_up] = 1
        raw[cross_dn] = -1
        fill = _fill_position_bidirectional if params["bidirectional"] else _fill_position
        return fill(raw)
    return strategy


def build_stdev_bands(params: dict):
    def strategy(df: pd.DataFrame) -> pd.Series:
        upper, lower = stdev_bands(df["close"], params["period"], params["num_std"])
        raw = pd.Series(0, index=df.index, dtype=int)
        # Buy when close crosses below lower band (mean reversion long)
        raw[df["close"] < lower] = 1
        raw[df["close"] > upper] = -1
        return _fill_position(raw)
    return strategy


STRATEGY_BUILDERS = {
    "IBS + Moving Average":   build_ibs_ma,
    "RSI Mean Reversion":     build_rsi_mean_reversion,
    "Momentum Trend":         build_momentum_trend,
    "ATR Range Filter":       build_range_filter_trend,
    "Std Dev Bands":          build_stdev_bands,
}

# ---------------------------------------------------------------------------
# PLOTLY CHART
# ---------------------------------------------------------------------------

def make_chart(
    equity: pd.Series,
    drawdown: pd.Series,
    trades: pd.DataFrame,
    df: pd.DataFrame,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity.values,
            name="Equity", line=dict(color="#2196F3", width=1.5),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Equity: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=1.0, line=dict(color="#30363d", width=1, dash="dash"), row=1, col=1)

    # Drawdown (filled area)
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values,
            name="Drawdown", fill="tozeroy",
            line=dict(color="#f85149", width=0.8),
            fillcolor="rgba(248,81,73,0.25)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>DD: %{y:.2%}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Trade entry/exit markers on equity curve
    if len(trades) > 0:
        entries = trades["entry_time"]
        entry_equity = equity.reindex(entries, method="nearest")
        exits = trades["exit_time"]
        exit_equity = equity.reindex(exits, method="nearest")

        fig.add_trace(
            go.Scatter(
                x=entries, y=entry_equity.values,
                mode="markers", name="Entry",
                marker=dict(color="#3fb950", size=5, symbol="triangle-up"),
                hovertemplate="Entry<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exits, y=exit_equity.values,
                mode="markers", name="Exit",
                marker=dict(color="#f85149", size=5, symbol="triangle-down"),
                hovertemplate="Exit<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
            ),
            row=1, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
        height=520,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#21262d",
        tickformat=".2%", row=2, col=1,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#21262d", row=1, col=1)
    fig.update_xaxes(showgrid=False, rangeslider_visible=False)

    return fig


# ---------------------------------------------------------------------------
# METRIC CARD HELPER
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, colour_class: str = "neu") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value metric-{colour_class}">{value}</div>'
        f'</div>'
    )


def colour(val: float) -> str:
    return "pos" if val > 0 else ("neg" if val < 0 else "neu")


# ---------------------------------------------------------------------------
# SIDEBAR — DATA
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 Data")

    # File upload
    uploaded = st.file_uploader(
        "Upload TSV / CSV file",
        type=["csv", "tsv"],
        help="Tab-separated OHLCV file. Columns: DateTime, Open, High, Low, Close, Volume",
    )

    # Or pick an existing file from the project folder
    existing = sorted(Path(".").glob("*_data.csv"))
    existing_names = [f.name for f in existing]
    selected_file = None

    if existing_names and not uploaded:
        selected_file = st.selectbox(
            "Or use a local file",
            options=existing_names,
            index=0,
        )

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # SIDEBAR — DATE RANGE
    # ---------------------------------------------------------------------------
    st.markdown("## 📅 Date Range")
    use_date_filter = st.checkbox("Filter date range", value=False)
    date_from = st.date_input("From", value=None, disabled=not use_date_filter)
    date_to   = st.date_input("To",   value=None, disabled=not use_date_filter)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # SIDEBAR — STRATEGY
    # ---------------------------------------------------------------------------
    st.markdown("## 🧠 Strategy")
    strategy_name = st.selectbox("Select strategy", list(STRATEGY_BUILDERS.keys()))
    st.markdown("**Parameters**")

    params: dict = {}

    if strategy_name == "IBS + Moving Average":
        params["ibs_entry"] = st.slider("IBS Entry threshold <", 0.05, 0.50, 0.20, 0.01,
                                         help="Enter long when IBS is below this value")
        params["ibs_exit"]  = st.slider("IBS Exit threshold >",  0.50, 0.95, 0.80, 0.01,
                                         help="Exit when IBS exceeds this value")
        params["ma_type"]   = st.radio("MA type", ["SMA", "EMA"], horizontal=True)
        params["ma_period"] = st.selectbox("MA period", [20, 50, 100, 200], index=3,
                                             help="Close must be above this MA to take longs")

    elif strategy_name == "RSI Mean Reversion":
        params["rsi_period"]  = st.slider("RSI period",    5,  30, 14, 1)
        params["oversold"]    = st.slider("Oversold (entry <)", 10, 45, 30, 1)
        params["exit_level"]  = st.slider("Exit (RSI >)",  45, 75, 50, 1)

    elif strategy_name == "Momentum Trend":
        params["mom_period"]    = st.slider("Momentum period (bars)", 5, 60, 20, 1)
        params["entry_thresh"]  = st.slider("Entry threshold (%)",   0.5, 8.0, 2.0, 0.1)
        params["exit_thresh"]   = st.slider("Exit flat zone (%)",    0.0, 3.0, 0.5, 0.1)
        params["ma_period"]     = st.selectbox("Trend filter MA period", [20, 50, 100, 200], index=2)
        params["bidirectional"] = st.checkbox("Enable short trades", value=True)

    elif strategy_name == "ATR Range Filter":
        params["atr_period"]    = st.slider("ATR period",      5, 50, 14, 1)
        params["multiplier"]    = st.slider("ATR multiplier",  0.5, 5.0, 2.0, 0.1)
        params["bidirectional"] = st.checkbox("Enable short trades", value=False)

    elif strategy_name == "Std Dev Bands":
        params["period"]  = st.slider("Lookback period", 10, 200, 20, 5)
        params["num_std"] = st.slider("Std dev width",   0.5, 4.0, 2.0, 0.1)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # SIDEBAR — EXECUTION
    # ---------------------------------------------------------------------------
    st.markdown("## ⚙️ Execution")
    costs_bps = st.number_input("Transaction costs (bps)", 0.0, 100.0, 5.0, 0.5,
                                 help="Applied per entry and exit event")
    slippage  = st.number_input("Slippage (price units)",  0.0, 50.0,  0.0, 0.5)

    st.markdown("---")
    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# MAIN — HEADER
# ---------------------------------------------------------------------------

st.markdown("# 📈 Backtest Engine")
st.caption("Vectorized · No look-ahead bias · Signals execute on next bar's Open")

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_load(path: str | None, content: bytes | None) -> pd.DataFrame:
    if content is not None:
        return load_ohlcv(StringIO(content.decode()))
    if path:
        return load_ohlcv(path)
    return pd.DataFrame()


if run_btn or ("results" in st.session_state):
    if run_btn:
        # Load data
        with st.spinner("Loading data…"):
            try:
                if uploaded:
                    df_full = cached_load(None, uploaded.getvalue())
                elif selected_file:
                    df_full = cached_load(selected_file, None)
                else:
                    st.error("Please upload a file or select an existing dataset.")
                    st.stop()
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                st.stop()

        # Apply date filter
        df = df_full.copy()
        if use_date_filter:
            if date_from:
                df = df[df.index >= pd.Timestamp(date_from)]
            if date_to:
                df = df[df.index <= pd.Timestamp(date_to)]

        if len(df) < 210:
            st.warning("Not enough bars after filtering (need at least 210). Adjust date range.")
            st.stop()

        # Build strategy & run
        strategy_fn = STRATEGY_BUILDERS[strategy_name](params)
        with st.spinner("Running backtest…"):
            results = run_backtest(
                df=df,
                strategy_fn=strategy_fn,
                transaction_costs_bps=costs_bps,
                slippage=slippage,
            )

        metrics = compute_analytics(
            equity_curve=results["equity_curve"],
            net_return=results["net_return"],
            trades=results["trades"],
            df=df,
        )

        running_max = results["equity_curve"].cummax()
        drawdown = (results["equity_curve"] - running_max) / running_max

        st.session_state["results"]  = results
        st.session_state["metrics"]  = metrics
        st.session_state["drawdown"] = drawdown
        st.session_state["df"]       = df
        st.session_state["strategy"] = strategy_name
        st.session_state["params"]   = params

    # Pull from session state
    results  = st.session_state["results"]
    metrics  = st.session_state["metrics"]
    drawdown = st.session_state["drawdown"]
    df       = st.session_state["df"]

    # ---------------------------------------------------------------------------
    # DATA INFO BAR
    # ---------------------------------------------------------------------------
    tf = detect_timeframe(df)
    st.markdown(
        f"**{st.session_state['strategy']}** &nbsp;·&nbsp; "
        f"`{len(df):,}` bars &nbsp;·&nbsp; "
        f"Timeframe: `{tf}` &nbsp;·&nbsp; "
        f"`{df.index[0].date()}` → `{df.index[-1].date()}`"
    )

    # ---------------------------------------------------------------------------
    # METRICS GRID
    # ---------------------------------------------------------------------------
    cols = st.columns(4)
    m = metrics

    pf_str = f"{m['profit_factor']:.3f}" if m["profit_factor"] != float("inf") else "∞"

    cards_row1 = [
        ("Total Return",  f"{m['total_return']:+.2%}", colour(m["total_return"])),
        ("CAGR",          f"{m['cagr']:+.2%}",         colour(m["cagr"])),
        ("Max Drawdown",  f"{m['max_drawdown']:+.2%}",  colour(m["max_drawdown"])),
        ("Sharpe Ratio",  f"{m['sharpe']:.3f}",         colour(m["sharpe"])),
    ]
    cards_row2 = [
        ("Win Rate",       f"{m['win_rate']:.2%}",      colour(m["win_rate"] - 0.5)),
        ("Profit Factor",  pf_str,                       colour(m["profit_factor"] - 1)),
        ("# Trades",       f"{m['n_trades']:,}",         "neu"),
        ("Avg Hold (bars)",f"{m['avg_bars']:.1f}",       "neu"),
    ]

    for col, (label, val, cls) in zip(cols, cards_row1):
        col.markdown(metric_card(label, val, cls), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    for col, (label, val, cls) in zip(cols2, cards_row2):
        col.markdown(metric_card(label, val, cls), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # PLOTLY CHART
    # ---------------------------------------------------------------------------
    fig = make_chart(results["equity_curve"], drawdown, results["trades"], df)
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------------------
    # ANNUAL RETURNS TABLE
    # ---------------------------------------------------------------------------
    with st.expander("📅 Annual Returns"):
        ann_ret = (
            results["net_return"]
            .resample("YE")
            .apply(lambda r: (1 + r).prod() - 1)
        )
        ann_df = ann_ret.to_frame("Return")
        ann_df.index = ann_df.index.year
        ann_df.index.name = "Year"
        ann_df["Return"] = ann_df["Return"].map(lambda x: f"{x:+.2%}")
        st.dataframe(ann_df, use_container_width=True)

    # ---------------------------------------------------------------------------
    # TRADE LOG
    # ---------------------------------------------------------------------------
    with st.expander(f"📋 Trade Log ({m['n_trades']:,} trades)"):
        trades_df = results["trades"].copy()
        if len(trades_df) > 0:
            trades_df["trade_return"] = trades_df["trade_return"].map(lambda x: f"{x:+.4%}")
            trades_df["direction"]    = trades_df["direction"].map({1: "Long", -1: "Short"})
            trades_df.columns         = ["Entry Time", "Exit Time", "Direction", "Return", "Bars"]
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades generated.")

else:
    # Landing state — no backtest run yet
    st.info("👈 Configure your strategy in the sidebar and click **Run Backtest**.")
    st.markdown("""
    **Quick start:**
    1. Select a local dataset (e.g. `1h_data.csv`) or upload your own TSV file
    2. Pick a strategy and adjust its parameters
    3. Set transaction costs and slippage
    4. Click **▶ Run Backtest**

    **Adding your own strategies:** Open `backtest.py`, define a new function,
    and register a corresponding builder in `app.py` → `STRATEGY_BUILDERS`.
    """)
