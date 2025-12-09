# MT5 Automated Trading Bot

A comprehensive Python-based automated trading bot for MetaTrader 5 with modular architecture, risk management, performance tracking, and web dashboard.

## Features

- **MT5 Platform Integration**: Direct connection to MetaTrader 5 terminal via official Python API
- **Modular Strategy System**: Pluggable strategy modules (starting with Moving Average Crossover)
- **Risk Management**: Position sizing, daily loss limits, kill switch, and exposure controls
- **Performance Tracking**: Real-time metrics (Win Rate, ROI, Drawdown) with file-based logging
- **Backtesting Engine**: Historical strategy validation with realistic transaction costs
- **Web Dashboard**: FastAPI + Plotly Dash dashboard for live monitoring
- **CLI Interface**: Command-line interface for bot control

## Installation

1. Install Python 3.8 or higher
2. Install MetaTrader 5 terminal
3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure MT5 connection in `config/default_config.json` or use environment variables

## Configuration

Edit `config/default_config.json` to set:
- MT5 connection credentials
- Strategy parameters
- Risk management rules
- Trading symbols

## Usage

### CLI Commands

```bash
# Start the bot
python main.py start --config config/default_config.json

# Run backtest
python main.py backtest --config config/default_config.json --start-date 2023-01-01 --end-date 2023-12-31

# Check status
python main.py status

# Stop the bot
python main.py stop
```

### Web Dashboard

```bash
# Start the dashboard (runs on http://localhost:8050)
python main.py dashboard --port 8050
```

## Project Structure

- `src/platform/` - MT5 connection and data feed
- `src/strategy/` - Trading strategy modules
- `src/risk/` - Risk management system
- `src/performance/` - Metrics and logging
- `src/backtest/` - Backtesting engine
- `src/core/` - Main bot orchestrator
- `src/ui/` - CLI and web dashboard
- `config/` - Configuration files
- `logs/` - Log files

## License

MIT

