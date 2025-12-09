"""Order execution simulator with transaction costs"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OrderSimulator:
    """Simulates order execution with realistic transaction costs"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize order simulator
        
        Args:
            config: Backtest configuration with:
                - spread_pips: Spread in pips
                - commission_per_lot: Commission per lot
                - slippage_pips: Slippage in pips
        """
        self.config = config
        self.spread_pips = config.get('spread_pips', 2)
        self.commission_per_lot = config.get('commission_per_lot', 0.0)
        self.slippage_pips = config.get('slippage_pips', 1)
    
    def calculate_spread(self, symbol: str, price: float) -> float:
        """
        Calculate spread cost
        
        Args:
            symbol: Symbol name
            price: Current price
            
        Returns:
            Spread in price units
        """
        # Convert pips to price units
        # For most pairs, 1 pip = 0.0001, but for JPY pairs it's 0.01
        pip_value = 0.0001
        if 'JPY' in symbol:
            pip_value = 0.01
        
        return (self.spread_pips * pip_value) * price
    
    def calculate_slippage(self, symbol: str, price: float, order_type: str) -> float:
        """
        Calculate slippage (adverse price move)
        
        Args:
            symbol: Symbol name
            price: Entry price
            order_type: 'BUY' or 'SELL'
            
        Returns:
            Slippage amount (positive for buy, negative for sell)
        """
        pip_value = 0.0001
        if 'JPY' in symbol:
            pip_value = 0.01
        
        slippage = self.slippage_pips * pip_value * price
        
        # Slippage is adverse: buy at higher price, sell at lower price
        if order_type == 'BUY':
            return slippage
        else:
            return -slippage
    
    def calculate_commission(self, volume: float) -> float:
        """
        Calculate commission
        
        Args:
            volume: Position volume in lots
            
        Returns:
            Commission amount
        """
        return self.commission_per_lot * volume
    
    def simulate_fill(
        self,
        symbol: str,
        order_type: str,
        requested_price: float,
        volume: float,
        current_bar: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """
        Simulate order fill with transaction costs
        
        Args:
            symbol: Symbol name
            order_type: 'BUY' or 'SELL'
            requested_price: Requested entry price
            volume: Position volume
            current_bar: Current bar data with OHLC
            
        Returns:
            Tuple of (fill_price, total_cost, commission)
        """
        # Use bar price for realistic fill
        if order_type == 'BUY':
            base_price = current_bar.get('ask', current_bar.get('close', requested_price))
        else:  # SELL
            base_price = current_bar.get('bid', current_bar.get('close', requested_price))
        
        # Apply slippage
        slippage = self.calculate_slippage(symbol, base_price, order_type)
        fill_price = base_price + slippage
        
        # Calculate costs
        spread_cost = self.calculate_spread(symbol, fill_price) * volume
        commission = self.calculate_commission(volume)
        total_cost = spread_cost + commission
        
        return fill_price, total_cost, commission
    
    def simulate_exit(
        self,
        symbol: str,
        order_type: str,
        entry_price: float,
        exit_price: float,
        volume: float
    ) -> Tuple[float, float]:
        """
        Simulate position exit with transaction costs
        
        Args:
            symbol: Symbol name
            order_type: Original order type ('BUY' or 'SELL')
            entry_price: Entry price
            exit_price: Requested exit price
            volume: Position volume
            
        Returns:
            Tuple of (final_exit_price, exit_cost)
        """
        # Apply slippage on exit (opposite direction)
        exit_order_type = 'SELL' if order_type == 'BUY' else 'BUY'
        slippage = self.calculate_slippage(symbol, exit_price, exit_order_type)
        final_exit_price = exit_price + slippage
        
        # Calculate exit costs
        spread_cost = self.calculate_spread(symbol, final_exit_price) * volume
        commission = self.calculate_commission(volume)
        exit_cost = spread_cost + commission
        
        return final_exit_price, exit_cost

