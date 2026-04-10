from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# TYPE ALIAS
# ---------------------------------------------------------------------------
StrategyFn = Callable[[pd.DataFrame], pd.Series]

# ---------------------------------------------------------------------------
# SECTION 1 — DATA LAYER
# ---------------------------------------------------------------------------

BARS_PER_YEAR: dict[str, float] = {
    "1min":  525_600.0,
    "5min":  105_120.0,
    "15min":  35_040.0,
    "1h":     8_766.0,
    "1d":       252.0,
}


def load_ohlcv(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep="\t",
        parse_dates=["DateTime"],
        date_format="%Y.%m.%d %H:%M:%S",
    )
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def detect_timeframe(df: pd.DataFrame) -> str:
    """Infer timeframe from median bar gap; returns key for BARS_PER_YEAR."""
    gaps = df.index.to_series().diff().dropna()
    median_min = gaps.median().total_seconds() / 60
    if median_min <= 1.5:
        return "1min"
    if median_min <= 7.5:
        return "5min"
    if median_min <= 30:
        return "15min"
    if median_min <= 90:
        return "1h"
    return "1d"


def annual_factor(df: pd.DataFrame) -> float:
    return BARS_PER_YEAR.get(detect_timeframe(df), 252.0)


# ---------------------------------------------------------------------------
# SECTION 2 — INDICATOR SUITE (fully vectorized)
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    denom = (high - low).replace(0, np.nan)
    return (close - low) / denom


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def range_filter(close: pd.Series, atr_series: pd.Series, multiplier: float) -> pd.Series:
    """
    Recurrence filter: the filter line only steps by > band from its prior value.
    Requires a numpy loop over bars (inherent recurrence — not purely vectorizable).
    Operates on numpy arrays for speed; O(n) in C via numpy scalar ops.
    """
    band = (atr_series * multiplier).to_numpy()
    c = close.to_numpy()
    out = np.full(len(c), np.nan)

    first = np.argmax(~np.isnan(c) & ~np.isnan(band))
    out[first] = c[first]

    for i in range(first + 1, len(c)):
        if np.isnan(band[i]) or np.isnan(c[i]):
            out[i] = out[i - 1]
            continue
        prev = out[i - 1]
        if c[i] > prev + band[i]:
            out[i] = c[i] - band[i]
        elif c[i] < prev - band[i]:
            out[i] = c[i] + band[i]
        else:
            out[i] = prev

    return pd.Series(out, index=close.index)


def momentum(close: pd.Series, period: int) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stdev_bands(
    close: pd.Series, period: int, num_std: float
) -> tuple[pd.Series, pd.Series]:
    mid = sma(close, period)
    std = close.rolling(period).std(ddof=1)
    return mid + num_std * std, mid - num_std * std


# ---------------------------------------------------------------------------
# SECTION 3 — STRATEGY PATTERN
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, StrategyFn] = {}


def register_strategy(name: str) -> Callable[[StrategyFn], StrategyFn]:
    def decorator(fn: StrategyFn) -> StrategyFn:
        STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


@register_strategy("ibs_sma200")
def strategy_ibs_sma200(df: pd.DataFrame) -> pd.Series:
    """Long when IBS < 0.2 AND close > SMA(200). Exit when IBS > 0.8."""
    ibs_val = ibs(df["high"], df["low"], df["close"])
    sma200 = sma(df["close"], 200)
    signal = pd.Series(0, index=df.index, dtype=int)
    long_entry = (ibs_val < 0.2) & (df["close"] > sma200)
    long_exit = ibs_val > 0.8
    # State machine: propagate position forward, vectorized via cumsum grouping
    raw = pd.Series(0, index=df.index, dtype=int)
    raw[long_entry] = 1
    raw[long_exit] = -1
    pos = _fill_position(raw)
    signal[:] = pos
    return signal


@register_strategy("ibs_ema50")
def strategy_ibs_ema50(df: pd.DataFrame) -> pd.Series:
    """Long when IBS < 0.25 AND close > EMA(50). Exit when IBS > 0.75."""
    ibs_val = ibs(df["high"], df["low"], df["close"])
    ema50 = ema(df["close"], 50)
    raw = pd.Series(0, index=df.index, dtype=int)
    raw[(ibs_val < 0.25) & (df["close"] > ema50)] = 1
    raw[ibs_val > 0.75] = -1
    return _fill_position(raw)


@register_strategy("momentum_trend")
def strategy_momentum_trend(df: pd.DataFrame) -> pd.Series:
    """Long when 20-bar momentum > 2% AND close > SMA(100). Short when < -2%."""
    mom = momentum(df["close"], 20)
    sma100 = sma(df["close"], 100)
    raw = pd.Series(0, index=df.index, dtype=int)
    raw[(mom > 2) & (df["close"] > sma100)] = 1
    raw[(mom < -2) & (df["close"] < sma100)] = -1
    raw[mom.abs() < 0.5] = -2  # exit marker
    return _fill_position(raw)


@register_strategy("mean_reversion_rsi")
def strategy_mean_reversion_rsi(df: pd.DataFrame) -> pd.Series:
    """Long when RSI(14) < 30. Exit when RSI > 50."""
    rsi_val = rsi(df["close"], 14)
    raw = pd.Series(0, index=df.index, dtype=int)
    raw[rsi_val < 30] = 1
    raw[rsi_val > 50] = -1
    return _fill_position(raw)


@register_strategy("range_filter_trend")
def strategy_range_filter_trend(df: pd.DataFrame) -> pd.Series:
    """Trend-follow using ATR-based range filter crossovers."""
    atr14 = atr(df["high"], df["low"], df["close"], 14)
    rf = range_filter(df["close"], atr14, 2.0)
    raw = pd.Series(0, index=df.index, dtype=int)
    # Long when close crosses above filter; short when below
    cross_up = (df["close"] > rf) & (df["close"].shift(1) <= rf.shift(1))
    cross_dn = (df["close"] < rf) & (df["close"].shift(1) >= rf.shift(1))
    raw[cross_up] = 1
    raw[cross_dn] = -1
    return _fill_position(raw)


def _fill_position(raw: pd.Series) -> pd.Series:
    """
    Convert raw entry/exit markers to a held long-only position series.
    raw: 1=enter long, -1=exit to flat, -2=exit to flat, 0=hold.
    """
    pos = np.zeros(len(raw), dtype=int)
    arr = raw.to_numpy()
    current = 0
    for i in range(len(arr)):
        if arr[i] == 1:
            current = 1
        elif arr[i] in (-1, -2):
            current = 0
        pos[i] = current
    return pd.Series(pos, index=raw.index, dtype=int)


def _fill_position_bidirectional(raw: pd.Series) -> pd.Series:
    """Fill position for strategies that use both 1 (long) and -1 (short)."""
    arr = raw.to_numpy()
    pos = np.zeros(len(arr), dtype=int)
    current = 0
    for i in range(len(arr)):
        if arr[i] == 1:
            current = 1
        elif arr[i] == -1:
            current = -1
        elif arr[i] == -2:
            current = 0
        pos[i] = current
    return pd.Series(pos, index=raw.index, dtype=int)


# Override momentum strategy to use bidirectional fill
@register_strategy("momentum_trend")
def strategy_momentum_trend(df: pd.DataFrame) -> pd.Series:  # noqa: F811
    """Long when 20-bar momentum > 2% AND close > SMA(100). Short when < -2%."""
    mom = momentum(df["close"], 20)
    sma100 = sma(df["close"], 100)
    raw = pd.Series(0, index=df.index, dtype=int)
    raw[(mom > 2) & (df["close"] > sma100)] = 1
    raw[(mom < -2) & (df["close"] < sma100)] = -1
    raw[mom.abs() < 0.5] = -2
    return _fill_position_bidirectional(raw)


# ---------------------------------------------------------------------------
# SECTION 4 — BACKTESTING ENGINE
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    transaction_costs_bps: float = 5.0,
    slippage: float = 0.0,
    initial_capital: float = 100_000.0,
) -> dict:
    """
    Execute backtest. Enters on next bar's Open after signal fires on Close.
    Returns dict with equity_curve, position, net_return, gross_return, trades.
    """
    signal = strategy_fn(df)

    # Shift signal by 1: act on next bar. This is the core look-ahead prevention.
    position = signal.shift(1).fillna(0).astype(int)

    prev_pos = position.shift(1).fillna(0).astype(int)
    is_entry = (position != 0) & (prev_pos == 0)
    is_exit  = (position == 0) & (prev_pos != 0)
    is_flip  = (position != 0) & (prev_pos != 0) & (position != prev_pos)

    # Bar returns — three segments per trade leg:
    #   entry bar  : open → close  (entered at open this bar)
    #   middle bars: prev_close → close
    #   exit bar   : prev_close → open  (exited at open this bar)
    open_ret  = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    close_ret = df["close"].pct_change()
    oc_ret    = (df["close"] - df["open"]) / df["open"]   # open→close of entry bar

    # Base: every held bar earns close-to-close
    bar_ret = position * close_ret

    # Correct entry bars: earned open-to-close, not prev_close-to-close
    bar_ret[is_entry] = position[is_entry] * oc_ret[is_entry]

    # Correct exit bars: earned prev_close-to-open (exit at open), prev_pos held
    bar_ret[is_exit] = prev_pos[is_exit] * open_ret[is_exit]

    # Flips (e.g. long→short): prev_close→open for old leg + open→close for new leg
    bar_ret[is_flip] = (
        prev_pos[is_flip] * open_ret[is_flip]
        + position[is_flip] * oc_ret[is_flip]
    )

    # Transaction costs & slippage on every open (entry, exit, flip)
    trade_events = (is_entry | is_exit | is_flip).astype(float)
    # Slippage as fraction of open price
    slip_frac = slippage / df["open"].replace(0, np.nan)
    cost_per_bar = trade_events * (transaction_costs_bps / 10_000 + slip_frac.fillna(0))

    net_ret = bar_ret - cost_per_bar
    gross_ret = bar_ret.copy()

    equity = (1.0 + net_ret).cumprod()

    trades = _extract_trades(position, net_ret, df)

    return {
        "equity_curve": equity,
        "position":     position,
        "signal":       signal,
        "net_return":   net_ret,
        "gross_return": gross_ret,
        "trades":       trades,
        "df":           df,
    }


def _extract_trades(
    position: pd.Series, net_return: pd.Series, df: pd.DataFrame
) -> pd.DataFrame:
    """Group consecutive identical non-zero positions into trade records."""
    if (position != 0).sum() == 0:
        return pd.DataFrame(
            columns=["entry_time", "exit_time", "direction", "trade_return", "n_bars"]
        )

    # Assign a group ID to each run of consecutive equal values
    run_id = (position != position.shift(1)).cumsum()
    records = []

    for gid, grp in position[position != 0].groupby(run_id[position != 0]):
        entry_time = grp.index[0]
        exit_time  = grp.index[-1]
        direction  = int(grp.iloc[0])
        trade_ret  = net_return.loc[grp.index].sum()
        records.append(
            {
                "entry_time":   entry_time,
                "exit_time":    exit_time,
                "direction":    direction,
                "trade_return": trade_ret,
                "n_bars":       len(grp),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# SECTION 5 — ANALYTICS MODULE
# ---------------------------------------------------------------------------

def compute_analytics(
    equity_curve: pd.Series,
    net_return: pd.Series,
    trades: pd.DataFrame,
    df: pd.DataFrame,
) -> dict[str, float | int]:
    n_years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
    ann = annual_factor(df)

    total_return = float(equity_curve.iloc[-1] - 1.0)
    cagr = float(equity_curve.iloc[-1] ** (1.0 / n_years) - 1.0)

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min())

    std = net_return.std(ddof=1)
    sharpe = float(net_return.mean() / std * np.sqrt(ann)) if std > 0 else 0.0

    n_trades = len(trades)
    if n_trades > 0:
        wins = (trades["trade_return"] > 0).sum()
        win_rate = float(wins / n_trades)
        gross_profit = trades.loc[trades["trade_return"] > 0, "trade_return"].sum()
        gross_loss   = trades.loc[trades["trade_return"] < 0, "trade_return"].abs().sum()
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        avg_bars = float(trades["n_bars"].mean())
    else:
        win_rate = profit_factor = avg_bars = 0.0

    return {
        "total_return":  total_return,
        "cagr":          cagr,
        "max_drawdown":  max_dd,
        "sharpe":        sharpe,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "n_trades":      n_trades,
        "avg_bars":      avg_bars,
    }


def print_summary(metrics: dict, strategy_name: str, filepath: str) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Strategy : {strategy_name}")
    print(f"  Data     : {filepath}")
    print(sep)
    print(f"  Total Return     : {metrics['total_return']:>+10.2%}")
    print(f"  CAGR             : {metrics['cagr']:>+10.2%}")
    print(f"  Max Drawdown     : {metrics['max_drawdown']:>+10.2%}")
    print(f"  Sharpe Ratio     : {metrics['sharpe']:>10.3f}")
    print(f"  Win Rate         : {metrics['win_rate']:>10.2%}")
    pf = metrics["profit_factor"]
    pf_str = f"{pf:>10.3f}" if pf != float("inf") else "       inf"
    print(f"  Profit Factor    : {pf_str}")
    print(f"  # Trades         : {metrics['n_trades']:>10d}")
    print(f"  Avg Trade (bars) : {metrics['avg_bars']:>10.1f}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# SECTION 6 — VISUALIZATION
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity_curve: pd.Series,
    df: pd.DataFrame,
    output_path: str | Path,
    strategy_name: str,
    show: bool = False,
) -> None:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#30363d")

    # Equity curve
    ax0 = axes[0]
    ax0.plot(equity_curve.index, equity_curve.values, color="#2196F3", linewidth=1.2)
    ax0.axhline(1.0, color="#30363d", linewidth=0.8, linestyle="--")
    ax0.set_ylabel("Equity (normalised)", color="#8b949e")
    ax0.set_title(f"Strategy: {strategy_name}", color="#e6edf3", fontsize=11, pad=8)
    ax0.yaxis.label.set_color("#8b949e")

    # Drawdown
    ax1 = axes[1]
    ax1.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.5)
    ax1.set_ylabel("Drawdown", color="#8b949e")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8b949e")

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vectorized backtesting engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to TSV OHLCV file")
    parser.add_argument(
        "--strategy",
        default="ibs_sma200",
        choices=sorted(STRATEGY_REGISTRY.keys()),
        help="Strategy to run",
    )
    parser.add_argument("--costs-bps", type=float, default=5.0, metavar="BPS",
                        help="Round-trip transaction costs in basis points")
    parser.add_argument("--slippage", type=float, default=0.0,
                        help="Slippage in price units per trade")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Initial capital (for display only)")
    parser.add_argument("--output", default="equity_curve.png",
                        help="Output path for equity curve PNG")
    parser.add_argument("--show-plot", action="store_true",
                        help="Display plot interactively after saving")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {data_path} …")
    df = load_ohlcv(data_path)
    tf = detect_timeframe(df)
    print(f"  {len(df):,} bars | timeframe={tf} | "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    strategy_fn = STRATEGY_REGISTRY[args.strategy]

    print(f"Running strategy '{args.strategy}' …")
    results = run_backtest(
        df=df,
        strategy_fn=strategy_fn,
        transaction_costs_bps=args.costs_bps,
        slippage=args.slippage,
        initial_capital=args.capital,
    )

    metrics = compute_analytics(
        equity_curve=results["equity_curve"],
        net_return=results["net_return"],
        trades=results["trades"],
        df=df,
    )
    print_summary(metrics, strategy_name=args.strategy, filepath=str(data_path))

    out = Path(args.output)
    plot_equity_curve(
        equity_curve=results["equity_curve"],
        df=df,
        output_path=out,
        strategy_name=args.strategy,
        show=args.show_plot,
    )
    print(f"Equity curve saved → {out.resolve()}")


if __name__ == "__main__":
    main()
