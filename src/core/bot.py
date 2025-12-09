"""Main bot orchestrator with event-driven architecture"""

import threading
import time
from queue import Queue, Empty
from typing import Dict, Any, Optional
import logging

from src.platform.mt5_connector import MT5Connector
from src.platform.data_feed import DataFeed
from src.platform.order_manager import OrderManager
from src.strategy.base_strategy import BaseStrategy
from src.risk.risk_manager import RiskManager
from src.performance.logger import PerformanceLogger
from src.core.events import BarEvent, TickEvent, SignalEvent, OrderEvent, FillEvent, PositionEvent
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config: Dict[str, Any], strategy: BaseStrategy):
        """
        Initialize trading bot
        
        Args:
            config: Configuration dictionary
            strategy: Strategy instance
        """
        self.config = config
        self.strategy = strategy
        self.running = False
        self.event_queue = Queue()
        
        # Initialize components
        self.connector = MT5Connector(config)
        self.data_feed: Optional[DataFeed] = None
        self.order_manager: Optional[OrderManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.performance_logger: Optional[PerformanceLogger] = None
        
        # Threading
        self.event_thread: Optional[threading.Thread] = None
    
    def initialize(self) -> bool:
        """Initialize bot components"""
        try:
            # Connect to MT5
            if not self.connector.connect():
                logger.error("Failed to connect to MT5")
                return False
            
            # Get account info for initial capital
            account_info = self.connector.get_account_info()
            initial_capital = account_info.balance if account_info else 10000.0
            
            # Initialize components
            trading_config = self.config.get('trading', {})
            symbols = trading_config.get('symbols', ['EURUSD'])
            timeframe = trading_config.get('timeframe', 'M15')
            
            self.data_feed = DataFeed(symbols, timeframe, self.event_queue)
            self.order_manager = OrderManager(self.config)
            self.risk_manager = RiskManager(self.config, self.order_manager)
            self.performance_logger = PerformanceLogger(self.config, initial_capital)
            
            logger.info("Bot initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing bot: {e}", exc_info=True)
            return False
    
    def start(self) -> None:
        """Start the bot"""
        if self.running:
            logger.warning("Bot is already running")
            return
        
        if not self.initialize():
            logger.error("Failed to initialize bot")
            return
        
        self.running = True
        
        # Start data feed
        if self.data_feed:
            self.data_feed.start()
        
        # Start event processing thread
        self.event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self.event_thread.start()
        
        logger.info("Trading bot started")
    
    def stop(self) -> None:
        """Stop the bot"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop data feed
        if self.data_feed:
            self.data_feed.stop()
        
        # Wait for event thread
        if self.event_thread:
            self.event_thread.join(timeout=5)
        
        # Disconnect from MT5
        self.connector.disconnect()
        
        logger.info("Trading bot stopped")
    
    def _event_loop(self) -> None:
        """Main event processing loop"""
        while self.running:
            try:
                # Get event from queue (non-blocking with timeout)
                try:
                    event = self.event_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Process event
                if isinstance(event, BarEvent):
                    self._handle_bar_event(event)
                elif isinstance(event, TickEvent):
                    self._handle_tick_event(event)
                elif isinstance(event, SignalEvent):
                    self._handle_signal_event(event)
                elif isinstance(event, FillEvent):
                    self._handle_fill_event(event)
                elif isinstance(event, PositionEvent):
                    self._handle_position_event(event)
                
                # Periodically update positions (every 10 events)
                if self.event_queue.qsize() % 10 == 0:
                    self.update_positions()
                
            except Exception as e:
                logger.error(f"Error in event loop: {e}", exc_info=True)
                time.sleep(1)
    
    def _handle_bar_event(self, event: BarEvent) -> None:
        """Handle bar event from data feed"""
        # Pass to strategy
        signal = self.strategy.on_bar(event)
        
        if signal:
            # Put signal event in queue
            self.event_queue.put(signal)
    
    def _handle_tick_event(self, event: TickEvent) -> None:
        """Handle tick event from data feed"""
        # Pass to strategy (if it handles ticks)
        signal = self.strategy.on_tick(event)
        
        if signal:
            self.event_queue.put(signal)
    
    def _handle_signal_event(self, event: SignalEvent) -> None:
        """Handle signal event from strategy"""
        # Validate signal with risk manager
        if not self.risk_manager.validate_signal(event):
            logger.warning(f"Signal rejected by risk manager: {event.signal_type} {event.symbol}")
            return
        
        # Get current price
        symbol_info = mt5.symbol_info_tick(event.symbol)
        if symbol_info is None:
            logger.error(f"Cannot get price for {event.symbol}")
            return
        
        # Determine entry price
        if event.signal_type == 'BUY':
            entry_price = symbol_info.ask
        elif event.signal_type == 'SELL':
            entry_price = symbol_info.bid
        else:
            logger.warning(f"Unknown signal type: {event.signal_type}")
            return
        
        # Create order event
        order_event = self.risk_manager.create_order_event(event, entry_price)
        
        # Execute order
        fill_event = self.order_manager.execute_order(order_event)
        
        if fill_event:
            # Log trade entry
            self.performance_logger.log_trade_entry(
                ticket=fill_event.ticket,
                symbol=fill_event.symbol,
                entry_price=fill_event.price,
                volume=fill_event.volume,
                strategy_name=fill_event.strategy_name,
                order_type='BUY' if fill_event.order_type == mt5.ORDER_TYPE_BUY else 'SELL'
            )
    
    def _handle_fill_event(self, event: FillEvent) -> None:
        """Handle fill event (order executed)"""
        # This is handled in _handle_signal_event after order execution
        pass
    
    def _handle_position_event(self, event: PositionEvent) -> None:
        """Handle position update event"""
        # Update performance logger with current profit
        self.performance_logger.update_trade_profit(event.ticket, event.profit)
    
    def update_positions(self) -> None:
        """Update open positions and check for exits"""
        if not self.order_manager or not self.performance_logger:
            return
        
        # Get current open positions
        current_positions = self.order_manager.get_positions()
        current_tickets = {pos.ticket for pos in current_positions}
        
        # Get previously tracked open trades
        tracked_trades = self.performance_logger.get_open_trades()
        tracked_tickets = set(tracked_trades.keys())
        
        # Find closed positions
        closed_tickets = tracked_tickets - current_tickets
        
        # Log closed positions
        for ticket in closed_tickets:
            trade_info = tracked_trades.get(ticket)
            if trade_info:
                # Get deal history from MT5
                deal_info = self.order_manager.get_deal_history(ticket)
                if deal_info:
                    self.performance_logger.log_trade_exit(
                        ticket=ticket,
                        exit_price=deal_info['exit_price'],
                        profit=deal_info['profit'],
                        commission=deal_info['commission'],
                        swap=deal_info['swap']
                    )
                else:
                    # Fallback if deal history not available
                    self.performance_logger.log_trade_exit(
                        ticket=ticket,
                        exit_price=trade_info.get('entry_price', 0),
                        profit=0.0,
                        commission=0.0,
                        swap=0.0
                    )
        
        # Update current positions
        for pos in current_positions:
            # Update performance logger
            self.performance_logger.update_trade_profit(pos.ticket, pos.profit)
            
            # Check for trailing stops (if configured)
            # This could be enhanced with trailing stop logic
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        status = {
            'running': self.running,
            'connected': self.connector.is_connected() if self.connector else False,
            'strategy': self.strategy.name if self.strategy else None,
        }
        
        if self.performance_logger:
            status['metrics'] = self.performance_logger.get_current_metrics()
            status['open_trades'] = len(self.performance_logger.get_open_trades())
        
        if self.order_manager:
            positions = self.order_manager.get_positions()
            status['positions'] = len(positions)
        
        return status

