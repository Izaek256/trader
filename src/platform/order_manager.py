"""Order execution and position management"""

import MetaTrader5 as mt5
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from src.core.events import OrderEvent, FillEvent, PositionEvent

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order execution and position tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize order manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.magic_number = config.get('trading', {}).get('magic_number', 0)
        self.open_positions: Dict[int, PositionEvent] = {}
        self.order_history: List[FillEvent] = []
    
    def execute_order(self, order_event: OrderEvent) -> Optional[FillEvent]:
        """
        Execute an order via MT5
        
        Args:
            order_event: Order event to execute
            
        Returns:
            FillEvent if successful, None otherwise
        """
        symbol = order_event.symbol
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None
        
        # Prepare order request
        if order_event.order_type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        elif order_event.order_type == mt5.ORDER_TYPE_SELL:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            logger.error(f"Invalid order type: {order_event.order_type}")
            return None
        
        # Normalize volume
        volume = self._normalize_volume(symbol, order_event.volume)
        if volume is None:
            return None
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price if order_event.price == 0.0 else order_event.price,
            "deviation": order_event.deviation,
            "magic": self.magic_number,
            "comment": order_event.comment or f"{order_event.strategy_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add stop loss and take profit
        if order_event.sl > 0:
            request["sl"] = order_event.sl
        if order_event.tp > 0:
            request["tp"] = order_event.tp
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order rejected: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Order executed: {symbol} {order_type} {volume} @ {result.price}")
        
        # Create fill event
        fill_event = FillEvent(
            symbol=symbol,
            order_id=result.order,
            ticket=result.deal,
            order_type=order_type,
            volume=volume,
            price=result.price,
            sl=order_event.sl,
            tp=order_event.tp,
            profit=0.0,
            commission=result.commission,
            swap=0.0,
            time=datetime.fromtimestamp(result.time),
            strategy_name=order_event.strategy_name
        )
        
        self.order_history.append(fill_event)
        return fill_event
    
    def _normalize_volume(self, symbol: str, volume: float) -> Optional[float]:
        """Normalize volume according to symbol requirements"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None
        
        min_volume = symbol_info.volume_min
        max_volume = symbol_info.volume_max
        volume_step = symbol_info.volume_step
        
        # Round to step
        volume = round(volume / volume_step) * volume_step
        
        # Clamp to min/max
        volume = max(min_volume, min(max_volume, volume))
        
        return volume
    
    def get_positions(self, symbol: Optional[str] = None) -> List[PositionEvent]:
        """
        Get open positions
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of PositionEvent objects
        """
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        position_events = []
        current_tickets = set()
        for pos in positions:
            if pos.magic == self.magic_number:
                pos_event = PositionEvent(
                    symbol=pos.symbol,
                    ticket=pos.ticket,
                    type=pos.type,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    profit=pos.profit,
                    swap=pos.swap,
                    time=datetime.fromtimestamp(pos.time),
                    strategy_name=pos.comment
                )
                position_events.append(pos_event)
                self.open_positions[pos.ticket] = pos_event
                current_tickets.add(pos.ticket)
        
        # Remove closed positions from tracking
        closed_tickets = set(self.open_positions.keys()) - current_tickets
        for ticket in closed_tickets:
            self.open_positions.pop(ticket, None)
        
        return position_events
    
    def get_deal_history(self, ticket: int) -> Optional[Dict[str, Any]]:
        """
        Get deal history for a position ticket
        
        Args:
            ticket: Position ticket number
            
        Returns:
            Dictionary with deal information or None
        """
        deals = mt5.history_deals_get(ticket=ticket)
        if deals is None or len(deals) == 0:
            return None
        
        # Get entry and exit deals
        entry_deal = None
        exit_deal = None
        
        for deal in deals:
            if deal.entry == mt5.DEAL_ENTRY_IN:
                entry_deal = deal
            elif deal.entry == mt5.DEAL_ENTRY_OUT:
                exit_deal = deal
        
        if entry_deal and exit_deal:
            return {
                'entry_price': entry_deal.price,
                'exit_price': exit_deal.price,
                'profit': exit_deal.profit,
                'commission': exit_deal.commission,
                'swap': exit_deal.swap,
                'entry_time': datetime.fromtimestamp(entry_deal.time),
                'exit_time': datetime.fromtimestamp(exit_deal.time)
            }
        
        return None
    
    def close_position(self, ticket: int) -> bool:
        """
        Close a position by ticket
        
        Args:
            ticket: Position ticket number
            
        Returns:
            True if successful
        """
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return False
        
        pos = position[0]
        symbol = pos.symbol
        
        # Determine order type for closing
        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position {ticket}: {mt5.last_error()}")
            return False
        
        logger.info(f"Position {ticket} closed")
        self.open_positions.pop(ticket, None)
        return True
    
    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
        """
        Modify stop loss and/or take profit of a position
        
        Args:
            ticket: Position ticket number
            sl: New stop loss price (None to keep current)
            tp: New take profit price (None to keep current)
            
        Returns:
            True if successful
        """
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return False
        
        pos = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": sl if sl is not None else pos.sl,
            "tp": tp if tp is not None else pos.tp,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to modify position {ticket}: {mt5.last_error()}")
            return False
        
        logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
        return True
    
    def update_trailing_stop(self, ticket: int, trailing_pips: float, min_profit_pips: float) -> bool:
        """
        Update trailing stop for a position
        
        Args:
            ticket: Position ticket number
            trailing_pips: Trailing stop distance in pips
            min_profit_pips: Minimum profit in pips before trailing activates
            
        Returns:
            True if stop was updated
        """
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return False
        
        pos = position[0]
        symbol = pos.symbol
        symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            return False
        
        point = symbol_info.point
        pip_value = point * (10 if symbol_info.digits == 3 or symbol_info.digits == 5 else 1)
        
        current_price = pos.price_current
        current_sl = pos.sl
        
        if pos.type == mt5.ORDER_TYPE_BUY:
            # For long positions, trail stop upward
            profit_pips = (current_price - pos.price_open) / pip_value
            
            if profit_pips >= min_profit_pips:
                new_sl = current_price - (trailing_pips * pip_value)
                if new_sl > current_sl:
                    return self.modify_position(ticket, sl=new_sl)
        
        elif pos.type == mt5.ORDER_TYPE_SELL:
            # For short positions, trail stop downward
            profit_pips = (pos.price_open - current_price) / pip_value
            
            if profit_pips >= min_profit_pips:
                new_sl = current_price + (trailing_pips * pip_value)
                if new_sl < current_sl or current_sl == 0:
                    return self.modify_position(ticket, sl=new_sl)
        
        return False

