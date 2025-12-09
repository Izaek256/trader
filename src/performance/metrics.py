"""Performance metrics calculations"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np


class PerformanceMetrics:
    """Calculates trading performance metrics"""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize performance metrics tracker
        
        Args:
            initial_capital: Initial account capital
        """
        self.initial_capital = initial_capital
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.current_equity = initial_capital
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Add a completed trade
        
        Args:
            trade: Trade dictionary with keys:
                - symbol: Symbol name
                - entry_time: Entry datetime
                - exit_time: Exit datetime
                - entry_price: Entry price
                - exit_price: Exit price
                - volume: Position volume
                - profit: Trade profit/loss
                - commission: Commission paid
                - swap: Swap paid/received
                - strategy_name: Strategy name
        """
        self.trades.append(trade)
        
        # Update equity curve
        if trade.get('profit') is not None:
            self.current_equity += trade['profit']
            self.equity_curve.append({
                'time': trade.get('exit_time', datetime.now()),
                'equity': self.current_equity,
                'balance': self.current_equity  # Simplified: assume equity = balance
            })
    
    def calculate_win_rate(self) -> float:
        """
        Calculate win rate percentage
        
        Returns:
            Win rate as percentage (0-100)
        """
        if len(self.trades) == 0:
            return 0.0
        
        profitable_trades = sum(1 for t in self.trades if t.get('profit', 0) > 0)
        return (profitable_trades / len(self.trades)) * 100.0
    
    def calculate_loss_rate(self) -> float:
        """
        Calculate loss rate percentage
        
        Returns:
            Loss rate as percentage (0-100)
        """
        if len(self.trades) == 0:
            return 0.0
        
        losing_trades = sum(1 for t in self.trades if t.get('profit', 0) < 0)
        return (losing_trades / len(self.trades)) * 100.0
    
    def calculate_roi(self) -> float:
        """
        Calculate Return on Investment percentage
        
        Returns:
            ROI as percentage
        """
        if self.initial_capital == 0:
            return 0.0
        
        total_profit = sum(t.get('profit', 0) for t in self.trades)
        roi = (total_profit / self.initial_capital) * 100.0
        return roi
    
    def calculate_total_profit(self) -> float:
        """
        Calculate total net profit
        
        Returns:
            Total profit/loss
        """
        return sum(t.get('profit', 0) for t in self.trades)
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown (peak-to-trough)
        
        Returns:
            Maximum drawdown as percentage
        """
        if len(self.equity_curve) == 0:
            return 0.0
        
        df = pd.DataFrame(self.equity_curve)
        if len(df) == 0:
            return 0.0
        
        df = df.sort_values('time')
        df['peak'] = df['equity'].expanding().max()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak'] * 100
        
        max_drawdown = df['drawdown'].min()
        return abs(max_drawdown) if max_drawdown < 0 else 0.0
    
    def calculate_avg_win(self) -> float:
        """
        Calculate average winning trade
        
        Returns:
            Average profit of winning trades
        """
        winning_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        if len(winning_trades) == 0:
            return 0.0
        return sum(t['profit'] for t in winning_trades) / len(winning_trades)
    
    def calculate_avg_loss(self) -> float:
        """
        Calculate average losing trade
        
        Returns:
            Average loss of losing trades
        """
        losing_trades = [t for t in self.trades if t.get('profit', 0) < 0]
        if len(losing_trades) == 0:
            return 0.0
        return sum(t['profit'] for t in losing_trades) / len(losing_trades)
    
    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        Returns:
            Profit factor (1.0 = breakeven, >1.0 = profitable)
        """
        gross_profit = sum(t['profit'] for t in self.trades if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t['profit'] for t in self.trades if t.get('profit', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_metrics_by_symbol(self) -> Dict[str, Dict[str, float]]:
        """
        Get metrics grouped by symbol
        
        Returns:
            Dictionary mapping symbol to metrics
        """
        symbols = set(t.get('symbol') for t in self.trades if t.get('symbol'))
        symbol_metrics = {}
        
        for symbol in symbols:
            symbol_trades = [t for t in self.trades if t.get('symbol') == symbol]
            if len(symbol_trades) == 0:
                continue
            
            profitable = sum(1 for t in symbol_trades if t.get('profit', 0) > 0)
            total_profit = sum(t.get('profit', 0) for t in symbol_trades)
            
            symbol_metrics[symbol] = {
                'total_trades': len(symbol_trades),
                'win_rate': (profitable / len(symbol_trades)) * 100 if len(symbol_trades) > 0 else 0,
                'total_profit': total_profit,
                'avg_profit': total_profit / len(symbol_trades) if len(symbol_trades) > 0 else 0
            }
        
        return symbol_metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Dictionary with all key metrics
        """
        return {
            'total_trades': len(self.trades),
            'win_rate': self.calculate_win_rate(),
            'loss_rate': self.calculate_loss_rate(),
            'roi': self.calculate_roi(),
            'total_profit': self.calculate_total_profit(),
            'max_drawdown': self.calculate_max_drawdown(),
            'avg_win': self.calculate_avg_win(),
            'avg_loss': self.calculate_avg_loss(),
            'profit_factor': self.calculate_profit_factor(),
            'current_equity': self.current_equity,
            'initial_capital': self.initial_capital,
            'symbol_metrics': self.get_metrics_by_symbol()
        }
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.trades.clear()
        self.equity_curve.clear()
        self.current_equity = self.initial_capital

