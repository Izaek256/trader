"""Event definitions for event-driven architecture"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import MetaTrader5 as mt5


@dataclass
class TickEvent:
    """Tick data event"""
    symbol: str
    time: datetime
    bid: float
    ask: float
    volume: int
    flags: int
    time_msc: int
    time_update_msc: int


@dataclass
class BarEvent:
    """Bar/OHLC data event"""
    symbol: str
    timeframe: int
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int
    real_volume: int


@dataclass
class SignalEvent:
    """Trading signal event"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'EXIT'
    strength: float = 1.0
    timestamp: datetime = None
    strategy_name: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OrderEvent:
    """Order submission event"""
    symbol: str
    order_type: int  # mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL
    volume: float
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 20
    magic: int = 0
    comment: str = ""
    strategy_name: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class FillEvent:
    """Order fill event"""
    symbol: str
    order_id: int
    ticket: int
    order_type: int
    volume: float
    price: float
    sl: float
    tp: float
    profit: float
    commission: float
    swap: float
    time: datetime
    strategy_name: str = ""


@dataclass
class PositionEvent:
    """Position update event"""
    symbol: str
    ticket: int
    type: int  # mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    time: datetime
    strategy_name: str = ""

