"""Live market data feed from MT5"""

import MetaTrader5 as mt5
import threading
import time
from datetime import datetime, timedelta
from typing import List, Callable, Optional, Dict
from queue import Queue
import logging

from src.core.events import TickEvent, BarEvent

logger = logging.getLogger(__name__)


class DataFeed:
    """Subscribes to live market data and emits events"""
    
    def __init__(self, symbols: List[str], timeframe: str, event_queue: Queue):
        """
        Initialize data feed
        
        Args:
            symbols: List of symbols to subscribe to
            timeframe: Timeframe string (e.g., 'M15', 'H1', 'D1')
            event_queue: Queue to emit events to
        """
        self.symbols = symbols
        self.timeframe = self._parse_timeframe(timeframe)
        self.event_queue = event_queue
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_bar_times: Dict[str, datetime] = {}
        self.market_book_subscriptions: Dict[str, bool] = {}
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to MT5 constant"""
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1,
        }
        return timeframe_map.get(timeframe.upper(), mt5.TIMEFRAME_M15)
    
    def start(self) -> None:
        """Start the data feed thread"""
        if self.running:
            logger.warning("Data feed already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Data feed started for symbols: {self.symbols}")
    
    def stop(self) -> None:
        """Stop the data feed"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Data feed stopped")
    
    def _run(self) -> None:
        """Main data feed loop"""
        # Initialize last bar times
        for symbol in self.symbols:
            rates = mt5.copy_rates_from(symbol, self.timeframe, datetime.now(), 1)
            if rates is not None and len(rates) > 0:
                self.last_bar_times[symbol] = datetime.fromtimestamp(rates[-1][0])
        
        while self.running:
            try:
                for symbol in self.symbols:
                    # Check for new bars
                    self._check_new_bars(symbol)
                    
                    # Get latest tick
                    self._get_latest_tick(symbol)
                
                time.sleep(0.1)  # Small delay to avoid excessive CPU usage
            except Exception as e:
                logger.error(f"Error in data feed loop: {e}", exc_info=True)
                time.sleep(1)
    
    def _check_new_bars(self, symbol: str) -> None:
        """Check for new bars and emit BarEvent"""
        try:
            # Get latest bar
            rates = mt5.copy_rates_from(symbol, self.timeframe, datetime.now(), 1)
            if rates is None or len(rates) == 0:
                return
            
            latest_bar = rates[-1]
            bar_time = datetime.fromtimestamp(latest_bar[0])
            
            # Check if this is a new bar
            last_time = self.last_bar_times.get(symbol)
            if last_time is None or bar_time > last_time:
                bar_event = BarEvent(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    time=bar_time,
                    open=latest_bar[1],
                    high=latest_bar[2],
                    low=latest_bar[3],
                    close=latest_bar[4],
                    tick_volume=int(latest_bar[5]),
                    spread=int(latest_bar[6]),
                    real_volume=int(latest_bar[7]) if len(latest_bar) > 7 else 0
                )
                self.event_queue.put(bar_event)
                self.last_bar_times[symbol] = bar_time
        except Exception as e:
            logger.error(f"Error checking new bars for {symbol}: {e}")
    
    def _get_latest_tick(self, symbol: str) -> None:
        """Get latest tick and emit TickEvent"""
        try:
            ticks = mt5.copy_ticks_from(symbol, datetime.now(), 1, mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                return
            
            latest_tick = ticks[-1]
            tick_event = TickEvent(
                symbol=symbol,
                time=datetime.fromtimestamp(latest_tick[0]),
                bid=latest_tick[1],
                ask=latest_tick[2],
                volume=latest_tick[3],
                flags=latest_tick[4],
                time_msc=latest_tick[5],
                time_update_msc=latest_tick[6]
            )
            self.event_queue.put(tick_event)
        except Exception as e:
            logger.error(f"Error getting latest tick for {symbol}: {e}")
    
    def subscribe_market_book(self, symbol: str) -> bool:
        """
        Subscribe to market depth (Level II) data
        
        Args:
            symbol: Symbol to subscribe to
            
        Returns:
            True if successful
        """
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not in subscription list")
            return False
        
        if mt5.market_book_add(symbol):
            self.market_book_subscriptions[symbol] = True
            logger.info(f"Subscribed to market book for {symbol}")
            return True
        else:
            logger.error(f"Failed to subscribe to market book for {symbol}: {mt5.last_error()}")
            return False
    
    def unsubscribe_market_book(self, symbol: str) -> None:
        """Unsubscribe from market depth data"""
        if mt5.market_book_release(symbol):
            self.market_book_subscriptions.pop(symbol, None)
            logger.info(f"Unsubscribed from market book for {symbol}")

