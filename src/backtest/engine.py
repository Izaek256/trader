"""Backtesting engine"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

from src.strategy.base_strategy import BaseStrategy
from src.backtest.simulator import OrderSimulator
from src.performance.metrics import PerformanceMetrics
from src.core.events import BarEvent, SignalEvent, OrderEvent

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Runs backtests on historical data"""
    
    def __init__(self, strategy: BaseStrategy, config: Dict[str, Any]):
        """
        Initialize backtest engine
        
        Args:
            strategy: Strategy instance to backtest
            config: Configuration dictionary
        """
        self.strategy = strategy
        self.config = config
        self.backtest_config = config.get('backtest', {})
        self.simulator = OrderSimulator(self.backtest_config)
        self.metrics = PerformanceMetrics(self.backtest_config.get('initial_capital', 10000.0))
        
        self.current_balance = self.backtest_config.get('initial_capital', 10000.0)
        self.open_positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position dict
        self.trade_history: List[Dict[str, Any]] = []
    
    def run(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = 'M15'
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            symbol: Symbol to backtest
            start_date: Start date
            end_date: End date
            timeframe: Timeframe string (e.g., 'M15', 'H1')
            
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info(f"Starting backtest: {symbol} from {start_date} to {end_date}")
        
        # Parse timeframe
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        mt5_timeframe = timeframe_map.get(timeframe.upper(), mt5.TIMEFRAME_M15)
        
        # Fetch historical data
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            logger.error(f"No historical data found for {symbol}")
            return {'error': 'No historical data'}
        
        logger.info(f"Loaded {len(rates)} bars for backtest")
        
        # Reset strategy
        self.strategy.reset()
        self.current_balance = self.backtest_config.get('initial_capital', 10000.0)
        self.open_positions.clear()
        self.trade_history.clear()
        self.metrics.reset()
        
        # Process each bar
        for i, rate in enumerate(rates):
            bar_time = datetime.fromtimestamp(rate[0])
            
            # Create bar event
            bar_event = BarEvent(
                symbol=symbol,
                timeframe=mt5_timeframe,
                time=bar_time,
                open=rate[1],
                high=rate[2],
                low=rate[3],
                close=rate[4],
                tick_volume=int(rate[5]),
                spread=int(rate[6]),
                real_volume=int(rate[7]) if len(rate) > 7 else 0
            )
            
            # Process bar through strategy
            signal = self.strategy.on_bar(bar_event)
            
            # Check for stop loss/take profit on open positions
            self._check_exits(bar_event)
            
            # Process signal if generated
            if signal:
                self._process_signal(signal, bar_event)
        
        # Close any remaining positions at end
        self._close_all_positions(rates[-1])
        
        # Calculate final metrics
        summary = self.metrics.get_summary()
        summary['final_balance'] = self.current_balance
        summary['total_trades'] = len(self.trade_history)
        
        logger.info(f"Backtest completed: {summary['total_trades']} trades, ROI: {summary['roi']:.2f}%")
        
        return summary
    
    def _process_signal(self, signal: SignalEvent, bar_event: BarEvent) -> None:
        """Process a trading signal"""
        symbol = signal.symbol
        
        # Check if we already have a position
        if symbol in self.open_positions:
            # Close existing position if opposite signal
            existing_pos = self.open_positions[symbol]
            if (signal.signal_type == 'BUY' and existing_pos['type'] == 'SELL') or \
               (signal.signal_type == 'SELL' and existing_pos['type'] == 'BUY'):
                self._close_position(symbol, bar_event)
        
        # Open new position
        if signal.signal_type in ['BUY', 'SELL']:
            self._open_position(signal, bar_event)
    
    def _open_position(self, signal: SignalEvent, bar_event: BarEvent) -> None:
        """Open a new position"""
        symbol = signal.symbol
        
        # Calculate position size (simplified: 1% of balance)
        position_size = (self.current_balance * 0.01) / 100000  # Approximate lot size
        position_size = max(0.01, min(position_size, 1.0))  # Clamp between 0.01 and 1.0
        
        # Get entry price
        if signal.signal_type == 'BUY':
            entry_price = bar_event.close
            order_type = 'BUY'
        else:
            entry_price = bar_event.close
            order_type = 'SELL'
        
        # Simulate fill with costs
        bar_data = {
            'close': bar_event.close,
            'ask': bar_event.close,
            'bid': bar_event.close
        }
        fill_price, entry_cost, commission = self.simulator.simulate_fill(
            symbol, order_type, entry_price, position_size, bar_data
        )
        
        # Use SL/TP from strategy metadata if available, otherwise use defaults
        metadata = signal.metadata or {}
        if 'sl_price' in metadata and metadata['sl_price'] > 0:
            sl = metadata['sl_price']
        else:
            # Default: 1% SL
            if order_type == 'BUY':
                sl = fill_price * 0.99
            else:
                sl = fill_price * 1.01
        
        if 'tp_price' in metadata and metadata['tp_price'] > 0:
            tp = metadata['tp_price']
        else:
            # Default: 2% TP
            if order_type == 'BUY':
                tp = fill_price * 1.02
            else:
                tp = fill_price * 0.98
        
        # Record position
        self.open_positions[symbol] = {
            'symbol': symbol,
            'type': order_type,
            'entry_time': bar_event.time,
            'entry_price': fill_price,
            'volume': position_size,
            'sl': sl,
            'tp': tp,
            'entry_cost': entry_cost,
            'commission': commission,
            'strategy_name': signal.strategy_name
        }
        
        # Deduct entry cost
        self.current_balance -= entry_cost
    
    def _check_exits(self, bar_event: BarEvent) -> None:
        """Check if any positions should be closed (SL/TP)"""
        symbol = bar_event.symbol
        
        if symbol not in self.open_positions:
            return
        
        pos = self.open_positions[symbol]
        bar_high = bar_event.high
        bar_low = bar_event.low
        bar_close = bar_event.close
        
        # Check stop loss first (priority)
        if pos['type'] == 'BUY':
            # For long positions: SL hit if low touches or goes below SL
            if bar_low <= pos['sl']:
                # Use SL price (or low, whichever is higher to avoid overshoot)
                exit_price = max(pos['sl'], bar_low)
                self._close_position(symbol, bar_event, exit_reason='SL', exit_price=exit_price)
                return
            # TP hit if high touches or goes above TP
            elif bar_high >= pos['tp']:
                # Use TP price (or high, whichever is lower)
                exit_price = min(pos['tp'], bar_high)
                self._close_position(symbol, bar_event, exit_reason='TP', exit_price=exit_price)
                return
        else:  # SELL
            # For short positions: SL hit if high touches or goes above SL
            if bar_high >= pos['sl']:
                # Use SL price (or high, whichever is lower)
                exit_price = min(pos['sl'], bar_high)
                self._close_position(symbol, bar_event, exit_reason='SL', exit_price=exit_price)
                return
            # TP hit if low touches or goes below TP
            elif bar_low <= pos['tp']:
                # Use TP price (or low, whichever is higher)
                exit_price = max(pos['tp'], bar_low)
                self._close_position(symbol, bar_event, exit_reason='TP', exit_price=exit_price)
                return
    
    def _close_position(self, symbol: str, bar_event: BarEvent, exit_reason: str = 'Signal', exit_price: Optional[float] = None) -> None:
        """Close an open position"""
        if symbol not in self.open_positions:
            return
        
        pos = self.open_positions.pop(symbol)
        if exit_price is None:
            exit_price = bar_event.close
        
        # Simulate exit with costs
        final_exit_price, exit_cost = self.simulator.simulate_exit(
            symbol, pos['type'], pos['entry_price'], exit_price, pos['volume']
        )
        
        # Calculate profit
        if pos['type'] == 'BUY':
            gross_profit = (final_exit_price - pos['entry_price']) * pos['volume'] * 100000
        else:
            gross_profit = (pos['entry_price'] - final_exit_price) * pos['volume'] * 100000
        
        net_profit = gross_profit - pos['entry_cost'] - exit_cost
        
        # Update balance
        self.current_balance += net_profit
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_time': pos['entry_time'],
            'exit_time': bar_event.time,
            'entry_price': pos['entry_price'],
            'exit_price': final_exit_price,
            'volume': pos['volume'],
            'profit': net_profit,
            'commission': pos['commission'],
            'swap': 0.0,
            'strategy_name': pos['strategy_name'],
            'type': pos['type'],
            'exit_reason': exit_reason
        }
        
        self.trade_history.append(trade)
        self.metrics.add_trade(trade)
    
    def _close_all_positions(self, last_bar: Any) -> None:
        """Close all remaining positions at end of backtest"""
        bar_event = BarEvent(
            symbol=list(self.open_positions.keys())[0] if self.open_positions else '',
            timeframe=0,
            time=datetime.fromtimestamp(last_bar[0]),
            open=last_bar[1],
            high=last_bar[2],
            low=last_bar[3],
            close=last_bar[4],
            tick_volume=int(last_bar[5]),
            spread=int(last_bar[6]),
            real_volume=0
        )
        
        for symbol in list(self.open_positions.keys()):
            bar_event.symbol = symbol
            self._close_position(symbol, bar_event, exit_reason='End')

