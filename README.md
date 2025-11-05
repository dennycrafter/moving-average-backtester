# Moving Average Crossover Backtester

A minimal Python backtester for MA-based strategies.

## Features
- Clean, modular functions
- Real yfinance data
- Long/flat crossover logic (no look-ahead)
- Performance metrics: CAGR, Sharpe, Max Drawdown
- Visual signals & equity curve

## Example
### $AAPL (2023-01-01 -> 2024-01-01)
#### MA10/MA50 strategy
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/c9ba62ce-2842-416a-9e5a-cfe98a7e359d" />

#### MA Strategy vs Buy/Hold Performance

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/10358427-3faa-443a-8867-ba5531ea5f6e" />

#### Results

| Metric               | MA Strategy | Buy & Hold |
|-----------------------|-------------|-------------|
| **Final Equity ($)**  | 12,324.95   | 15,479.82   |
| **Total Return**      | 0.2325      | 0.5480      |
| **CAGR**              | 0.2346      | 0.5534      |
| **Sharpe (daily ann.)** | 1.4020   | 2.3173      |
| **Max Drawdown**      | -0.1236     | -0.1493     |
| **Days**              | 250         | 250         |
| **Trades**            | 5           | 1           |

## Development
- Import dataclasses, typing, numpy, pandas, yfinance, matplotlib, Bday
- Load prices with yfinance and organize with pandas
- Create rolling MAs functions with fast and slow windows
- Create signals for buy/sell
- Add in metrics, comission, slippage, and other misc things using classes
- Run backtests, create scoring system for number of trades, cash, final cash, returns, CAGR, Sharpe ratio, max drawdown, days, & WR
- Create user input section for customization and usability
- Build buy/hold function
- Output and compare
