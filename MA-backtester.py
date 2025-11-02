import math
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

#loading prices in a clean and organized manner
def load_prices(
    ticker: str,
    start: str,
    end: str,
    auto_adjust: bool = True,
    warmup_bdays: int = 0
) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    start_fetch = (start_dt - BDay(warmup_bdays)).date().isoformat() if warmup_bdays > 0 else start

    df = yf.download(
        ticker,
        start=start_fetch,
        end=end,
        auto_adjust=auto_adjust,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}.")

    df = df.dropna().copy() #cleaning, remove any empty values
    df.index = pd.to_datetime(df.index)
    return df

#moving averages
def add_mas(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    price_col: str = "Close",
    min_periods_fast: Optional[int] = None,
    min_periods_slow: Optional[int] = None
) -> pd.DataFrame:
    df = df.copy()

    if min_periods_fast is None:
        min_periods_fast = fast
    if min_periods_slow is None:
        min_periods_slow = slow

    df[f"MA_{fast}"] = df[price_col].rolling(window=fast, min_periods=min_periods_fast).mean()
    df[f"MA_{slow}"] = df[price_col].rolling(window=slow, min_periods=min_periods_slow).mean()

    return df


def build_signals(df: pd.DataFrame, fast: int, slow: int, price_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    out["pos_signal"] = (out[f"MA_{fast}"] > out[f"MA_{slow}"]).astype(int)
    out["pos_next"] = out["pos_signal"].shift(1).fillna(0).astype(int)
    return out


@dataclass
class Costs:
    commission_per_trade: float = 0.0  # fixed currency per entry/exit
    slippage_bps_per_side: float = 0.0  # eg 5 = 0.05% per side


@dataclass
class Result:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    prices: pd.DataFrame


def run_backtest(
        prices: pd.DataFrame,
        fast: int,
        slow: int,
        initial_cash: float = 10_000,
        costs: Costs = Costs(),
        price_col_close: str = "Close",
        price_col_open: str = "Open",
        focus_start: str | None = None,
        focus_end: str | None = None,
) -> Result:
    df = add_mas(prices, fast=fast, slow=slow, price_col=price_col_close)
    df = build_signals(df, fast=fast, slow=slow, price_col=price_col_close)

    if focus_start or focus_end:
        df = df.loc[
            (pd.to_datetime(focus_start) if focus_start else df.index.min()):
            (pd.to_datetime(focus_end) if focus_end else df.index.max())
        ].copy()

    close = df.loc[:, price_col_close]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    ret = close.pct_change().fillna(0.0)

    pos = pd.to_numeric(df["pos_next"], errors="coerce").fillna(0.0)
    strat_ret_gross = (pos * ret)

    pos_change = pos.diff().fillna(pos.iloc[0])

    slip_r = -(costs.slippage_bps_per_side / 10_000.0) * pos_change.abs()
    slip_r = slip_r.astype(float)

    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = initial_cash
    last_equity = initial_cash

    trades = []
    for i in range(1, len(df)):
        date = df.index[i]
        gross = float(strat_ret_gross.iat[i])
        slip = float(slip_r.iat[i])
        did_trade = abs(pos_change.iat[i]) > 0

        commission_r = 0.0
        if did_trade and costs.commission_per_trade > 0:
            commission_r = -(costs.commission_per_trade / last_equity)

        day_r = gross + slip + commission_r
        equity.iloc[i] = last_equity * (1.0 + day_r)
        last_equity = equity.iloc[i]

        if did_trade:
            trades.append({
                "date": date,
                "action": "BUY" if pos_change.iat[i] > 0 else "SELL",
                "pos_after": int(pos.iat[i]),
                "equity": float(equity.iat[i]),
            })

    trades_df = (
        pd.DataFrame(trades).set_index("date")
        if trades else pd.DataFrame(columns=["action", "pos_after", "equity"])
    )

    eq = equity.dropna()
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    days = len(eq)
    daily_ret = eq.pct_change().dropna()
    cagr = float((1.0 + total_return) ** (252.0 / max(1, days)) - 1.0)
    sharpe = float(np.sqrt(252) * (daily_ret.mean() / (daily_ret.std() + 1e-12)))
    roll_max = eq.cummax()
    max_dd = float((eq / roll_max - 1.0).min())
    in_pos_days = daily_ret[df["pos_next"].iloc[1:] == 1]
    win_rate = float((in_pos_days > 0).mean()) if len(in_pos_days) else np.nan

    metrics = {
        "Initial Cash": float(initial_cash),
        "Final Equity": float(eq.iloc[-1]),
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe (daily)": sharpe,
        "Max Drawdown": max_dd,
        "Days": int(days),
        "Trades": int(len(trades_df)),
        "Win Rate (in-position days)": win_rate,
        "Fast MA": fast,
        "Slow MA": slow,
    }

    return Result(
        equity_curve=eq,
        trades=trades_df,
        metrics=metrics,
        prices=df,
    )

print("=== Moving Average Backtester ===")
ticker = input("Enter a stock ticker (e.g. AAPL, SPY, PLTR): ").upper().strip() or "SPY"
focus_start = input("Enter start date (YYYY-MM-DD) [default: 2024-01-01]: ").strip() or "2024-01-01"
focus_end   = input("Enter end date (YYYY-MM-DD) [default: 2025-01-01]: ").strip() or "2025-01-01"
fast = int(input("Enter fast MA window [default: 10]: ") or 10)
slow = int(input("Enter slow MA window [default: 50]: ") or 50)

print(f"\nRunning backtest for {ticker} ({focus_start} → {focus_end}), MAs {fast}/{slow}...\n")

px_full = load_prices(ticker, focus_start, focus_end, auto_adjust=True, warmup_bdays=slow+10)
res = run_backtest(
    prices=px_full,
    fast=fast, slow=slow,
    initial_cash=10_000,
    costs=Costs(commission_per_trade=0.00, slippage_bps_per_side=2.0),
    focus_start=focus_start, focus_end=focus_end,
)

print("Metrics:")
for k, v in res.metrics.items():
    print(f"{k}: {v}")

# Build aligned frame first (as you already did)
dfp = res.prices.loc[focus_start:focus_end, ["Close", f"MA_{fast}", f"MA_{slow}", "pos_next"]].copy()

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(dfp.index, dfp["Close"], label="Close Price", linewidth=1.5, color="black")
ax.plot(dfp.index, dfp[f"MA_{fast}"], label=f"{fast}-Day Moving Average", linewidth=1.3)
ax.plot(dfp.index, dfp[f"MA_{slow}"], label=f"{slow}-Day Moving Average", linewidth=1.3)

# --- Shade in-position periods across the FULL height (easiest: axvspan) ---
pos = dfp["pos_next"].astype(bool).to_numpy()
idx = dfp.index.to_numpy()

spans = []
start = None
for i, v in enumerate(pos):
    if v and start is None:
        start = idx[i]
    if (not v) and (start is not None):
        spans.append((start, idx[i]))
        start = None
if start is not None:
    spans.append((start, idx[-1]))

first = True
for s, e in spans:
    ax.axvspan(s, e, color="tab:blue", alpha=0.10, zorder=0,
               label="In position" if first else None)
    first = False

ax.set_title(f"{ticker} {focus_start} → {focus_end}  (MA{fast}/{slow})")
ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()