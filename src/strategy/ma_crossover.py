"""Moving Average Crossover Strategy"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy
    
    Generates buy signal when fast MA crosses above slow MA.
    Generates sell signal when fast MA crosses below slow MA.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MA Crossover strategy
        
        Args:
            config: Strategy configuration with:
                - fast_period: Fast MA period (default: 10)
                - slow_period: Slow MA period (default: 50)
                - ma_type: 'SMA' or 'EMA' (default: 'SMA')
                - signal_threshold: Minimum crossover threshold (default: 0.0001)
        """
        super().__init__("MA_Crossover", config)
        self.fast_period = config.get('fast_period', 10)
        self.slow_period = config.get('slow_period', 50)
        self.ma_type = config.get('ma_type', 'SMA').upper()
        self.signal_threshold = config.get('signal_threshold', 0.0001)
        
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """
        Process new bar and generate signal if crossover occurs
        
        Args:
            bar_event: Bar event data
            
        Returns:
            SignalEvent if crossover detected, None otherwise
        """
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Need at least slow_period bars to calculate MAs
        bars_df = self.get_bars(symbol)
        if len(bars_df) < self.slow_period:
            return None
        
        # Calculate moving averages
        if self.ma_type == 'EMA':
            fast_ma = bars_df['close'].ewm(span=self.fast_period, adjust=False).mean()
            slow_ma = bars_df['close'].ewm(span=self.slow_period, adjust=False).mean()
        else:  # SMA
            fast_ma = bars_df['close'].rolling(window=self.fast_period).mean()
            slow_ma = bars_df['close'].rolling(window=self.slow_period).mean()
        
        # Get current and previous values
        if len(fast_ma) < 2:
            return None
        
        current_fast = fast_ma.iloc[-1]
        previous_fast = fast_ma.iloc[-2]
        current_slow = slow_ma.iloc[-1]
        previous_slow = slow_ma.iloc[-2]
        
        # Check for crossover
        current_signal = self.current_signals.get(symbol)
        
        # Bullish crossover: fast MA crosses above slow MA
        if previous_fast <= previous_slow and current_fast > current_slow:
            if abs(current_fast - current_slow) >= self.signal_threshold:
                signal = SignalEvent(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=abs(current_fast - current_slow) / current_slow,
                    timestamp=bar_event.time,
                    strategy_name=self.name,
                    metadata={
                        'fast_ma': float(current_fast),
                        'slow_ma': float(current_slow),
                        'price': float(bar_event.close)
                    }
                )
                self.current_signals[symbol] = 'BUY'
                return signal
        
        # Bearish crossover: fast MA crosses below slow MA
        elif previous_fast >= previous_slow and current_fast < current_slow:
            if abs(current_fast - current_slow) >= self.signal_threshold:
                signal = SignalEvent(
                    symbol=symbol,
                    signal_type='SELL',
                    strength=abs(current_fast - current_slow) / current_slow,
                    timestamp=bar_event.time,
                    strategy_name=self.name,
                    metadata={
                        'fast_ma': float(current_fast),
                        'slow_ma': float(current_slow),
                        'price': float(bar_event.close)
                    }
                )
                self.current_signals[symbol] = 'SELL'
                return signal
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'ma_type': self.ma_type,
            'signal_threshold': self.signal_threshold
        }

