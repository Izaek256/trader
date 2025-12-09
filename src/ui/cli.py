"""Command-line interface for the trading bot"""

import click
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

from src.utils.config import ConfigLoader
from src.strategy.ma_crossover import MACrossoverStrategy
from src.core.bot import TradingBot
from src.backtest.engine import BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global bot instance
bot_instance: TradingBot = None


@click.group()
def cli():
    """MT5 Automated Trading Bot CLI"""
    pass


@cli.command()
@click.option('--config', '-c', default='config/default_config.json', help='Path to config file')
def start(config: str):
    """Start the trading bot"""
    global bot_instance
    
    try:
        # Load configuration
        config_loader = ConfigLoader(config)
        config_dict = config_loader.to_dict()
        
        # Load strategy configuration
        strategy_config_path = config_dict.get('strategy', {}).get('config_file')
        if strategy_config_path:
            strategy_config_loader = ConfigLoader(strategy_config_path)
            strategy_config = strategy_config_loader.to_dict()
        else:
            strategy_config = {}
        
        # Create strategy
        strategy_name = config_dict.get('strategy', {}).get('name', 'ma_crossover')
        if strategy_name == 'ma_crossover':
            strategy = MACrossoverStrategy(strategy_config)
        else:
            click.echo(f"Unknown strategy: {strategy_name}", err=True)
            sys.exit(1)
        
        # Create and start bot
        bot_instance = TradingBot(config_dict, strategy)
        bot_instance.start()
        
        click.echo("Trading bot started. Press Ctrl+C to stop.")
        
        # Keep running
        try:
            while bot_instance.running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo("\nStopping bot...")
            bot_instance.stop()
            click.echo("Bot stopped.")
    
    except Exception as e:
        click.echo(f"Error starting bot: {e}", err=True)
        logger.exception("Error in start command")
        sys.exit(1)


@cli.command()
def stop():
    """Stop the trading bot"""
    global bot_instance
    
    if bot_instance is None or not bot_instance.running:
        click.echo("Bot is not running", err=True)
        sys.exit(1)
    
    bot_instance.stop()
    click.echo("Bot stopped")
    bot_instance = None


@cli.command()
def status():
    """Show bot status"""
    global bot_instance
    
    if bot_instance is None:
        click.echo("Bot is not running")
        return
    
    status_dict = bot_instance.get_status()
    
    click.echo("\n=== Bot Status ===")
    click.echo(f"Running: {status_dict.get('running', False)}")
    click.echo(f"Connected: {status_dict.get('connected', False)}")
    click.echo(f"Strategy: {status_dict.get('strategy', 'N/A')}")
    click.echo(f"Open Positions: {status_dict.get('positions', 0)}")
    click.echo(f"Open Trades: {status_dict.get('open_trades', 0)}")
    
    if 'metrics' in status_dict:
        metrics = status_dict['metrics']
        click.echo("\n=== Performance Metrics ===")
        click.echo(f"Total Trades: {metrics.get('total_trades', 0)}")
        click.echo(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
        click.echo(f"ROI: {metrics.get('roi', 0):.2f}%")
        click.echo(f"Total Profit: {metrics.get('total_profit', 0):.2f}")
        click.echo(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        click.echo(f"Current Equity: {metrics.get('current_equity', 0):.2f}")


@cli.command()
@click.option('--config', '-c', default='config/default_config.json', help='Path to config file')
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--symbol', default='EURUSD', help='Symbol to backtest')
def backtest(config: str, start_date: str, end_date: str, symbol: str):
    """Run backtest on historical data"""
    try:
        # Load configuration
        config_loader = ConfigLoader(config)
        config_dict = config_loader.to_dict()
        
        # Load strategy configuration
        strategy_config_path = config_dict.get('strategy', {}).get('config_file')
        if strategy_config_path:
            strategy_config_loader = ConfigLoader(strategy_config_path)
            strategy_config = strategy_config_loader.to_dict()
        else:
            strategy_config = {}
        
        # Create strategy
        strategy_name = config_dict.get('strategy', {}).get('name', 'ma_crossover')
        if strategy_name == 'ma_crossover':
            strategy = MACrossoverStrategy(strategy_config)
        else:
            click.echo(f"Unknown strategy: {strategy_name}", err=True)
            sys.exit(1)
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get timeframe from config
        timeframe = config_dict.get('trading', {}).get('timeframe', 'M15')
        
        # Initialize MT5 (for data access)
        import MetaTrader5 as mt5
        if not mt5.initialize():
            click.echo(f"Failed to initialize MT5: {mt5.last_error()}", err=True)
            sys.exit(1)
        
        try:
            # Create backtest engine
            engine = BacktestEngine(strategy, config_dict)
            
            # Run backtest
            click.echo(f"Running backtest: {symbol} from {start_date} to {end_date}")
            results = engine.run(symbol, start_dt, end_dt, timeframe)
            
            # Display results
            click.echo("\n=== Backtest Results ===")
            click.echo(json.dumps(results, indent=2, default=str))
            
        finally:
            mt5.shutdown()
    
    except Exception as e:
        click.echo(f"Error running backtest: {e}", err=True)
        logger.exception("Error in backtest command")
        sys.exit(1)


@cli.command()
@click.option('--port', default=8050, help='Dashboard port')
@click.option('--host', default='0.0.0.0', help='Dashboard host')
def dashboard(port: int, host: str):
    """Start the web dashboard"""
    from src.ui.dashboard.app import create_app
    
    app = create_app()
    
    click.echo(f"Starting dashboard on http://{host}:{port}")
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    cli()

