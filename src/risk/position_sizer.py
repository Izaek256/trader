"""Position sizing calculations"""

import MetaTrader5 as mt5
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculates position sizes based on different methods"""
    
    @staticmethod
    def calculate_volume(
        method: str,
        account_balance: float,
        risk_percent: Optional[float] = None,
        fixed_lot: Optional[float] = None,
        symbol: Optional[str] = None,
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        atr_multiplier: Optional[float] = None,
        atr_value: Optional[float] = None
    ) -> float:
        """
        Calculate position volume based on sizing method
        
        Args:
            method: 'fixed_fraction', 'fixed_lot', or 'volatility_based'
            account_balance: Current account balance
            risk_percent: Risk percentage for fixed_fraction (e.g., 0.02 for 2%)
            fixed_lot: Fixed lot size for fixed_lot method
            symbol: Symbol name (required for volatility_based)
            entry_price: Entry price (required for volatility_based)
            stop_loss_price: Stop loss price (required for volatility_based)
            atr_multiplier: ATR multiplier for volatility_based
            atr_value: ATR value (required for volatility_based)
            
        Returns:
            Position volume in lots
        """
        if method == 'fixed_fraction':
            if risk_percent is None:
                raise ValueError("risk_percent required for fixed_fraction method")
            risk_amount = account_balance * risk_percent
            return risk_amount / 100000  # Approximate: 1 lot = $100k for major pairs
        
        elif method == 'fixed_lot':
            if fixed_lot is None:
                raise ValueError("fixed_lot required for fixed_lot method")
            return fixed_lot
        
        elif method == 'volatility_based':
            if symbol is None or entry_price is None or stop_loss_price is None:
                raise ValueError("symbol, entry_price, and stop_loss_price required for volatility_based method")
            if atr_multiplier is None:
                atr_multiplier = 2.0
            if atr_value is None:
                # Calculate ATR if not provided
                atr_value = PositionSizer._calculate_atr(symbol, 14)
            
            # Calculate risk per unit based on ATR
            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit == 0:
                risk_per_unit = atr_value * atr_multiplier
            
            # Risk 2% of account
            risk_amount = account_balance * 0.02
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Symbol info not found for {symbol}, using default")
                return risk_amount / (risk_per_unit * 100000)
            
            # Calculate contract size
            contract_size = symbol_info.trade_contract_size
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size
            
            # Calculate volume
            risk_per_lot = (risk_per_unit / tick_size) * tick_value * contract_size
            if risk_per_lot > 0:
                volume = risk_amount / risk_per_lot
            else:
                volume = risk_amount / 100000  # Fallback
            
            return max(0.01, min(volume, 100.0))  # Clamp between 0.01 and 100 lots
        
        else:
            raise ValueError(f"Unknown position sizing method: {method}")
    
    @staticmethod
    def _calculate_atr(symbol: str, period: int = 14) -> float:
        """
        Calculate Average True Range for a symbol
        
        Args:
            symbol: Symbol name
            period: ATR period
            
        Returns:
            ATR value
        """
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta
        
        # Get historical data
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_D1, datetime.now() - timedelta(days=period+10), period+10)
        if rates is None or len(rates) < period:
            return 0.001  # Default fallback
        
        import pandas as pd
        df = pd.DataFrame(rates)
        df['high'] = df['high']
        df['low'] = df['low']
        df['close'] = df['close']
        
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        atr = df['tr'].tail(period).mean()
        return float(atr)

