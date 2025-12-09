"""Performance logger that tracks trades and computes metrics"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

from src.performance.metrics import PerformanceMetrics
from src.utils.file_logger import FileLogger

logger = logging.getLogger(__name__)


class PerformanceLogger:
    """Tracks trades and computes performance metrics in real-time"""
    
    def __init__(self, config: Dict[str, Any], initial_capital: float = 10000.0):
        """
        Initialize performance logger
        
        Args:
            config: Configuration dictionary
            initial_capital: Initial account capital
        """
        self.config = config
        self.metrics = PerformanceMetrics(initial_capital)
        self.file_logger = FileLogger(
            log_dir=config.get('logging', {}).get('log_dir', 'logs'),
            trade_log_file=config.get('logging', {}).get('trade_log_file', 'trades.csv'),
            metrics_log_file=config.get('logging', {}).get('metrics_log_file', 'metrics.json')
        )
        self.open_trades: Dict[int, Dict[str, Any]] = {}  # Track open trades by ticket
    
    def log_trade_entry(self, ticket: int, symbol: str, entry_price: float, volume: float, 
                        strategy_name: str, order_type: str) -> None:
        """
        Log trade entry
        
        Args:
            ticket: Position ticket
            symbol: Symbol name
            entry_price: Entry price
            volume: Position volume
            strategy_name: Strategy name
            order_type: Order type ('BUY' or 'SELL')
        """
        self.open_trades[ticket] = {
            'ticket': ticket,
            'symbol': symbol,
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'volume': volume,
            'strategy_name': strategy_name,
            'type': order_type,
            'commission': 0.0,
            'swap': 0.0
        }
        logger.info(f"Trade entry logged: {symbol} {order_type} @ {entry_price}")
    
    def log_trade_exit(self, ticket: int, exit_price: float, profit: float, 
                      commission: float = 0.0, swap: float = 0.0) -> None:
        """
        Log trade exit and compute metrics
        
        Args:
            ticket: Position ticket
            exit_price: Exit price
            profit: Trade profit/loss
            commission: Commission paid
            swap: Swap paid/received
        """
        if ticket not in self.open_trades:
            logger.warning(f"Trade {ticket} not found in open trades")
            return
        
        trade = self.open_trades.pop(ticket)
        trade['exit_time'] = datetime.now()
        trade['exit_price'] = exit_price
        trade['profit'] = profit
        trade['commission'] = commission
        trade['swap'] = swap
        
        # Add to metrics
        self.metrics.add_trade(trade)
        
        # Log to file
        self.file_logger.log_trade(trade)
        
        # Log metrics periodically (after each trade)
        summary = self.metrics.get_summary()
        self.file_logger.log_metrics(summary)
        
        logger.info(f"Trade exit logged: {trade['symbol']} P&L: {profit:.2f}")
    
    def update_trade_profit(self, ticket: int, current_profit: float) -> None:
        """
        Update current profit for open trade (for display purposes)
        
        Args:
            ticket: Position ticket
            current_profit: Current unrealized profit
        """
        if ticket in self.open_trades:
            self.open_trades[ticket]['current_profit'] = current_profit
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Dictionary with current metrics
        """
        return self.metrics.get_summary()
    
    def get_open_trades(self) -> Dict[int, Dict[str, Any]]:
        """
        Get currently open trades
        
        Returns:
            Dictionary of open trades by ticket
        """
        return self.open_trades.copy()
    
    def reset(self) -> None:
        """Reset all metrics and logs"""
        self.metrics.reset()
        self.open_trades.clear()

