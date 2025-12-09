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

1. Install Python 3.8-3.11 (MetaTrader5 supports Python 3.8-3.11)
2. Install MetaTrader 5 terminal from [MetaQuotes website](https://www.metatrader5.com/en/download)
3. Install MetaTrader5 Python package:
   ```bash
   # Option 1: Try installing from PyPI (may not always work)
   pip install MetaTrader5

   # Option 2: If PyPI fails, download the wheel file from MetaQuotes
   # Visit: https://www.mql5.com/en/docs/integration/python_metatrader5
   # Or install directly from the MT5 terminal directory:
   pip install <path_to_mt5_terminal>/MQL5/Scripts/include/python/
   ```
4. Install other dependencies:

```bash
pip install -r requirements.txt
```

**Note**: If you encounter "Could not find a version that satisfies the requirement MetaTrader5", try:

- Ensure you're using Python 3.8-3.11 (not 3.12+)
- Download the MetaTrader5 wheel file directly from MetaQuotes
- Install MetaTrader 5 terminal first, then try installing the package again

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
python main.py backtest --config config/default_config.json --start-date 2025-08-01 --end-date 2025-12-01

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
