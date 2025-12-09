"""Base strategy abstract class"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

from src.core.events import BarEvent, TickEvent, SignalEvent


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize strategy
        
        Args:
            name: Strategy name
            config: Strategy configuration dictionary
        """
        self.name = name
        self.config = config
        self.bars: Dict[str, pd.DataFrame] = {}  # Store historical bars per symbol
        self.current_signals: Dict[str, Optional[str]] = {}  # Current signal per symbol
    
    @abstractmethod
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """
        Called when a new bar arrives
        
        Args:
            bar_event: Bar event data
            
        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        pass
    
    def on_tick(self, tick_event: TickEvent) -> Optional[SignalEvent]:
        """
        Called when a new tick arrives (optional override)
        
        Args:
            tick_event: Tick event data
            
        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        return None
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters
        
        Returns:
            Dictionary of strategy parameters
        """
        pass
    
    def add_bar(self, bar_event: BarEvent) -> None:
        """
        Add a bar to the historical data
        
        Args:
            bar_event: Bar event to add
        """
        symbol = bar_event.symbol
        
        if symbol not in self.bars:
            self.bars[symbol] = pd.DataFrame()
        
        new_row = pd.DataFrame([{
            'time': bar_event.time,
            'open': bar_event.open,
            'high': bar_event.high,
            'low': bar_event.low,
            'close': bar_event.close,
            'volume': bar_event.tick_volume
        }])
        
        self.bars[symbol] = pd.concat([self.bars[symbol], new_row], ignore_index=True)
        
        # Keep only recent bars (e.g., last 1000)
        if len(self.bars[symbol]) > 1000:
            self.bars[symbol] = self.bars[symbol].tail(1000).reset_index(drop=True)
    
    def get_bars(self, symbol: str, count: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical bars for a symbol
        
        Args:
            symbol: Symbol name
            count: Number of bars to return (None for all)
            
        Returns:
            DataFrame with bar data
        """
        if symbol not in self.bars:
            return pd.DataFrame()
        
        df = self.bars[symbol]
        if count is not None:
            return df.tail(count)
        return df
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.bars.clear()
        self.current_signals.clear()

