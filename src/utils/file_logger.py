"""File-based logging system"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FileLogger:
    """Handles file-based logging for trades and metrics"""
    
    def __init__(self, log_dir: str = "logs", trade_log_file: str = "trades.csv", metrics_log_file: str = "metrics.json"):
        """
        Initialize file logger
        
        Args:
            log_dir: Directory for log files
            trade_log_file: Name of trade log CSV file
            metrics_log_file: Name of metrics log JSON file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.trade_log_path = self.log_dir / trade_log_file
        self.metrics_log_path = self.log_dir / metrics_log_file
        
        # Initialize trade log CSV with headers if new file
        self._init_trade_log()
    
    def _init_trade_log(self) -> None:
        """Initialize trade log CSV file with headers"""
        if not self.trade_log_path.exists():
            with open(self.trade_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'strategy', 'entry_time', 'exit_time',
                    'entry_price', 'exit_price', 'volume', 'profit', 'commission',
                    'swap', 'type', 'duration_seconds'
                ])
    
    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Log a completed trade to CSV
        
        Args:
            trade: Trade dictionary with required fields
        """
        try:
            with open(self.trade_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                
                entry_time = trade.get('entry_time', '')
                exit_time = trade.get('exit_time', datetime.now())
                
                # Calculate duration
                duration = 0
                if entry_time and exit_time:
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time)
                    if isinstance(exit_time, str):
                        exit_time = datetime.fromisoformat(exit_time)
                    duration = (exit_time - entry_time).total_seconds()
                
                writer.writerow([
                    datetime.now().isoformat(),
                    trade.get('symbol', ''),
                    trade.get('strategy_name', ''),
                    entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time,
                    exit_time.isoformat() if isinstance(exit_time, datetime) else exit_time,
                    trade.get('entry_price', 0),
                    trade.get('exit_price', 0),
                    trade.get('volume', 0),
                    trade.get('profit', 0),
                    trade.get('commission', 0),
                    trade.get('swap', 0),
                    trade.get('type', ''),
                    duration
                ])
        except Exception as e:
            logger.error(f"Error logging trade: {e}", exc_info=True)
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log performance metrics to JSON
        
        Args:
            metrics: Metrics dictionary
        """
        try:
            # Add timestamp
            metrics_with_time = {
                'timestamp': datetime.now().isoformat(),
                **metrics
            }
            
            # Read existing metrics if file exists
            existing_metrics = []
            if self.metrics_log_path.exists():
                try:
                    with open(self.metrics_log_path, 'r') as f:
                        existing_metrics = json.load(f)
                        if not isinstance(existing_metrics, list):
                            existing_metrics = [existing_metrics]
                except json.JSONDecodeError:
                    existing_metrics = []
            
            # Append new metrics
            existing_metrics.append(metrics_with_time)
            
            # Keep only last 1000 entries
            if len(existing_metrics) > 1000:
                existing_metrics = existing_metrics[-1000:]
            
            # Write back
            with open(self.metrics_log_path, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error logging metrics: {e}", exc_info=True)
    
    def read_trades(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Read trades from log file
        
        Args:
            limit: Maximum number of trades to return (None for all)
            
        Returns:
            List of trade dictionaries
        """
        if not self.trade_log_path.exists():
            return []
        
        trades = []
        try:
            with open(self.trade_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trades.append(row)
            
            if limit:
                trades = trades[-limit:]
        except Exception as e:
            logger.error(f"Error reading trades: {e}", exc_info=True)
        
        return trades
    
    def read_metrics(self) -> List[Dict[str, Any]]:
        """
        Read metrics from log file
        
        Returns:
            List of metrics dictionaries
        """
        if not self.metrics_log_path.exists():
            return []
        
        try:
            with open(self.metrics_log_path, 'r') as f:
                metrics = json.load(f)
                if isinstance(metrics, list):
                    return metrics
                return [metrics]
        except Exception as e:
            logger.error(f"Error reading metrics: {e}", exc_info=True)
            return []

