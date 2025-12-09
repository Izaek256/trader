"""Risk management and position sizing"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from src.core.events import SignalEvent, OrderEvent
from src.risk.position_sizer import PositionSizer
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk rules and position sizing"""
    
    def __init__(self, config: Dict[str, Any], order_manager):
        """
        Initialize risk manager
        
        Args:
            config: Configuration dictionary with risk settings
            order_manager: OrderManager instance for position queries
        """
        self.config = config
        self.order_manager = order_manager
        self.risk_config = config.get('risk', {})
        
        # Position sizing config
        self.position_sizing = self.risk_config.get('position_sizing', {})
        self.sizing_method = self.position_sizing.get('method', 'fixed_fraction')
        self.sizing_value = self.position_sizing.get('value', 0.02)
        
        # Risk limits
        self.max_daily_loss_percent = self.risk_config.get('max_daily_loss_percent', 5.0)
        self.max_daily_drawdown_percent = self.risk_config.get('max_daily_drawdown_percent', 10.0)
        self.max_consecutive_losses = self.risk_config.get('max_consecutive_losses', 5)
        self.max_exposure_per_symbol_percent = self.risk_config.get('max_exposure_per_symbol_percent', 30.0)
        
        # Default stop loss/take profit
        self.default_stop_loss_percent = self.risk_config.get('default_stop_loss_percent', 1.0)
        self.default_take_profit_percent = self.risk_config.get('default_take_profit_percent', 2.0)
        
        # Tracking
        self.daily_start_balance: Optional[float] = None
        self.daily_start_equity: Optional[float] = None
        self.consecutive_losses = 0
        self.last_reset_date = datetime.now().date()
        self.trading_suspended = False
        
        # Get account info to initialize daily tracking
        self._reset_daily_tracking()
    
    def _reset_daily_tracking(self) -> None:
        """Reset daily tracking at start of new day"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            account_info = mt5.account_info()
            if account_info:
                self.daily_start_balance = account_info.balance
                self.daily_start_equity = account_info.equity
                self.last_reset_date = today
                self.trading_suspended = False
                logger.info("Daily risk tracking reset")
    
    def validate_signal(self, signal: SignalEvent) -> bool:
        """
        Validate if a signal should be executed based on risk rules
        
        Args:
            signal: Signal event to validate
            
        Returns:
            True if signal is valid, False if blocked by risk rules
        """
        self._reset_daily_tracking()
        
        # Check if trading is suspended
        if self.trading_suspended:
            logger.warning("Trading suspended due to risk limits")
            return False
        
        # Check daily loss limit
        if not self._check_daily_loss_limit():
            self.trading_suspended = True
            logger.error("Daily loss limit exceeded - trading suspended")
            return False
        
        # Check daily drawdown limit
        if not self._check_daily_drawdown_limit():
            self.trading_suspended = True
            logger.error("Daily drawdown limit exceeded - trading suspended")
            return False
        
        # Check consecutive losses (kill switch)
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.trading_suspended = True
            logger.error(f"Kill switch activated: {self.consecutive_losses} consecutive losses")
            return False
        
        # Check exposure per symbol
        if not self._check_symbol_exposure(signal.symbol):
            logger.warning(f"Symbol exposure limit reached for {signal.symbol}")
            return False
        
        return True
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        if self.daily_start_balance is None:
            return True
        
        account_info = mt5.account_info()
        if account_info is None:
            return True
        
        current_balance = account_info.balance
        loss = self.daily_start_balance - current_balance
        loss_percent = (loss / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0
        
        return loss_percent < self.max_daily_loss_percent
    
    def _check_daily_drawdown_limit(self) -> bool:
        """Check if daily drawdown limit is exceeded"""
        if self.daily_start_equity is None:
            return True
        
        account_info = mt5.account_info()
        if account_info is None:
            return True
        
        current_equity = account_info.equity
        drawdown = self.daily_start_equity - current_equity
        drawdown_percent = (drawdown / self.daily_start_equity) * 100 if self.daily_start_equity > 0 else 0
        
        return drawdown_percent < self.max_daily_drawdown_percent
    
    def _check_symbol_exposure(self, symbol: str) -> bool:
        """Check if symbol exposure limit is exceeded"""
        account_info = mt5.account_info()
        if account_info is None:
            return True
        
        equity = account_info.equity
        
        # Get open positions for this symbol
        positions = self.order_manager.get_positions(symbol=symbol)
        
        # Calculate total exposure
        total_exposure = 0.0
        for pos in positions:
            # Approximate exposure as position value
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                contract_size = symbol_info.trade_contract_size
                total_exposure += pos.volume * contract_size * pos.price_open
        
        exposure_percent = (total_exposure / equity) * 100 if equity > 0 else 0
        
        return exposure_percent < self.max_exposure_per_symbol_percent
    
    def calculate_position_size(
        self,
        signal: SignalEvent,
        entry_price: float,
        stop_loss_price: Optional[float] = None
    ) -> float:
        """
        Calculate position size for a signal
        
        Args:
            signal: Signal event
            entry_price: Entry price
            stop_loss_price: Stop loss price (optional)
            
        Returns:
            Position volume in lots
        """
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Cannot get account info for position sizing")
            return 0.01  # Minimum lot size
        
        account_balance = account_info.balance
        
        # Calculate stop loss if not provided
        if stop_loss_price is None:
            if signal.signal_type == 'BUY':
                stop_loss_price = entry_price * (1 - self.default_stop_loss_percent / 100)
            else:
                stop_loss_price = entry_price * (1 + self.default_stop_loss_percent / 100)
        
        # Calculate position size
        try:
            if self.sizing_method == 'fixed_fraction':
                volume = PositionSizer.calculate_volume(
                    method='fixed_fraction',
                    account_balance=account_balance,
                    risk_percent=self.sizing_value
                )
            elif self.sizing_method == 'fixed_lot':
                volume = PositionSizer.calculate_volume(
                    method='fixed_lot',
                    fixed_lot=self.sizing_value
                )
            elif self.sizing_method == 'volatility_based':
                volume = PositionSizer.calculate_volume(
                    method='volatility_based',
                    account_balance=account_balance,
                    symbol=signal.symbol,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price
                )
            else:
                logger.warning(f"Unknown sizing method: {self.sizing_method}, using fixed_fraction")
                volume = PositionSizer.calculate_volume(
                    method='fixed_fraction',
                    account_balance=account_balance,
                    risk_percent=0.02
                )
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            volume = 0.01  # Minimum fallback
        
        return max(0.01, volume)  # Ensure minimum lot size
    
    def create_order_event(
        self,
        signal: SignalEvent,
        entry_price: float,
        volume: Optional[float] = None
    ) -> OrderEvent:
        """
        Create an order event from a signal with risk management applied
        
        Args:
            signal: Signal event
            entry_price: Entry price
            volume: Optional volume (if None, calculated by risk manager)
            
        Returns:
            OrderEvent with stop loss and take profit
        """
        import MetaTrader5 as mt5
        
        # Calculate volume if not provided
        if volume is None:
            stop_loss_price = None
            if signal.signal_type == 'BUY':
                stop_loss_price = entry_price * (1 - self.default_stop_loss_percent / 100)
            elif signal.signal_type == 'SELL':
                stop_loss_price = entry_price * (1 + self.default_stop_loss_percent / 100)
            
            volume = self.calculate_position_size(signal, entry_price, stop_loss_price)
        
        # Determine order type
        if signal.signal_type == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
        elif signal.signal_type == 'SELL':
            order_type = mt5.ORDER_TYPE_SELL
        else:
            raise ValueError(f"Invalid signal type: {signal.signal_type}")
        
        # Calculate stop loss and take profit
        if signal.signal_type == 'BUY':
            sl = entry_price * (1 - self.default_stop_loss_percent / 100)
            tp = entry_price * (1 + self.default_take_profit_percent / 100)
        else:  # SELL
            sl = entry_price * (1 + self.default_stop_loss_percent / 100)
            tp = entry_price * (1 - self.default_take_profit_percent / 100)
        
        return OrderEvent(
            symbol=signal.symbol,
            order_type=order_type,
            volume=volume,
            price=entry_price,
            sl=sl,
            tp=tp,
            strategy_name=signal.strategy_name,
            comment=f"{signal.strategy_name} {signal.signal_type}"
        )
    
    def record_trade_result(self, profit: float) -> None:
        """
        Record trade result for consecutive loss tracking
        
        Args:
            profit: Trade profit (negative for loss)
        """
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        logger.info(f"Consecutive losses: {self.consecutive_losses}/{self.max_consecutive_losses}")

