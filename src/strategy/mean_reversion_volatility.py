"""Mean-Reversion Volatility Compression Strategy

This strategy trades mean reversion during low volatility periods,
entering when price compresses and exiting on volatility expansion.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class MeanReversionVolatilityStrategy(BaseStrategy):
    """
    Mean-Reversion Volatility Compression Strategy
    
    Entry Rules:
    - Bollinger Bands width < 20-period average (compression)
    - Price touches lower band (for long) or upper band (for short)
    - RSI confirms oversold/overbought (< 30 for long, > 70 for short)
    - Volume is below average (quiet before storm)
    - ATR is below 20-period average (low volatility)
    
    Exit Rules:
    - Take profit: Price reaches opposite band or 1.5R
    - Stop loss: Beyond band + 0.5x ATR buffer
    - Exit if volatility expands (BB width > 1.5x average)
    
    Filters:
    - Volatility: BB width must be < 80% of 20-period average
    - Volume: Entry volume < 90% of average
    - Time: Best during Asian session (low volatility)
    - Market Regime: Avoid during trending markets (ADX < 20)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MeanReversionVolatility", config)
        
        # Bollinger Bands parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.bb_compression_threshold = config.get('bb_compression_threshold', 0.8)
        self.bb_expansion_threshold = config.get('bb_expansion_threshold', 1.5)
        
        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        
        # ATR parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_low_threshold = config.get('atr_low_threshold', 0.9)
        self.atr_sl_buffer = config.get('atr_sl_buffer', 0.5)
        
        # Volume parameters
        self.volume_period = config.get('volume_period', 20)
        self.volume_threshold = config.get('volume_threshold', 0.9)
        
        # ADX filter (avoid trending markets)
        self.adx_period = config.get('adx_period', 14)
        self.adx_max = config.get('adx_max', 20.0)
        
        # Risk/Reward
        self.risk_reward_ratio = config.get('risk_reward_ratio', 1.5)
        
        # Time filter
        self.preferred_session = config.get('preferred_session', 'asian')  # asian, london, ny
        
    def _calculate_bb(self, series: pd.Series, period: int, std: float) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        width = (upper - lower) / sma  # Normalized width
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'width': width
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _is_preferred_session(self, bar_time: datetime) -> bool:
        """Check if current time is in preferred trading session"""
        hour = bar_time.hour
        
        if self.preferred_session == 'asian':
            # Asian session: 00:00-09:00 UTC
            return 0 <= hour < 9
        elif self.preferred_session == 'london':
            # London session: 08:00-17:00 UTC
            return 8 <= hour < 17
        elif self.preferred_session == 'ny':
            # NY session: 13:00-22:00 UTC
            return 13 <= hour < 22
        
        return True  # No filter if not specified
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Time filter
        if not self._is_preferred_session(bar_event.time):
            return None
        
        # Get bars
        bars = self.get_bars(symbol, max(self.bb_period, self.atr_period, self.rsi_period, self.volume_period, self.adx_period) + 10)
        if len(bars) < max(self.bb_period, self.atr_period, self.rsi_period, self.volume_period, self.adx_period):
            return None
        
        # Calculate indicators
        bb = self._calculate_bb(bars['close'], self.bb_period, self.bb_std)
        atr = self._calculate_atr(bars, self.atr_period)
        rsi = self._calculate_rsi(bars['close'], self.rsi_period)
        volume_avg = bars['volume'].rolling(window=self.volume_period).mean()
        adx = self._calculate_adx(bars, self.adx_period)
        
        if len(bb['width']) < 2 or len(atr) < 2 or len(rsi) < 1:
            return None
        
        current_price = bar_event.close
        current_bb_upper = bb['upper'].iloc[-1]
        current_bb_lower = bb['lower'].iloc[-1]
        current_bb_middle = bb['middle'].iloc[-1]
        current_bb_width = bb['width'].iloc[-1]
        avg_bb_width = bb['width'].tail(20).mean()
        
        current_atr = atr.iloc[-1]
        avg_atr = atr.tail(self.atr_period).mean()
        current_rsi = rsi.iloc[-1]
        current_volume = bars['volume'].iloc[-1]
        avg_volume = volume_avg.iloc[-1]
        current_adx = adx.iloc[-1]
        
        # Market regime filter: avoid trending markets
        if not pd.isna(current_adx) and current_adx > self.adx_max:
            return None
        
        # Volatility compression filter
        if pd.isna(current_bb_width) or pd.isna(avg_bb_width):
            return None
        
        if current_bb_width > avg_bb_width * self.bb_compression_threshold:
            return None  # Not compressed enough
        
        # ATR filter: must be low volatility
        if pd.isna(current_atr) or pd.isna(avg_atr):
            return None
        
        if current_atr > avg_atr * self.atr_low_threshold:
            return None  # Volatility too high
        
        # Volume filter
        if pd.isna(current_volume) or pd.isna(avg_volume):
            return None
        
        if current_volume > avg_volume * self.volume_threshold:
            return None  # Volume too high (not quiet)
        
        # Entry logic: LONG (price at lower band, RSI oversold)
        if (current_price <= current_bb_lower * 1.001 and  # Within 0.1% of lower band
            not pd.isna(current_rsi) and current_rsi < self.rsi_oversold):
            
            # Calculate SL and TP
            sl_price = current_bb_lower - current_atr * self.atr_sl_buffer
            risk = current_price - sl_price
            tp_price = current_bb_middle  # Target middle band, or 1.5R
            
            # Use better of middle band or 1.5R
            tp_from_rr = current_price + risk * self.risk_reward_ratio
            if tp_from_rr < current_bb_middle:
                tp_price = tp_from_rr
            
            return SignalEvent(
                symbol=symbol,
                signal_type='BUY',
                strength=(self.rsi_oversold - current_rsi) / self.rsi_oversold,  # Normalize
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(tp_price - current_price),
                    'bb_width_ratio': float(current_bb_width / avg_bb_width),
                    'atr_ratio': float(current_atr / avg_atr),
                    'rsi': float(current_rsi),
                    'adx': float(current_adx) if not pd.isna(current_adx) else 0.0
                }
            )
        
        # Entry logic: SHORT (price at upper band, RSI overbought)
        elif (current_price >= current_bb_upper * 0.999 and  # Within 0.1% of upper band
              not pd.isna(current_rsi) and current_rsi > self.rsi_overbought):
            
            # Calculate SL and TP
            sl_price = current_bb_upper + current_atr * self.atr_sl_buffer
            risk = sl_price - current_price
            tp_price = current_bb_middle  # Target middle band, or 1.5R
            
            # Use better of middle band or 1.5R
            tp_from_rr = current_price - risk * self.risk_reward_ratio
            if tp_from_rr > current_bb_middle:
                tp_price = tp_from_rr
            
            return SignalEvent(
                symbol=symbol,
                signal_type='SELL',
                strength=(current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(current_price - tp_price),
                    'bb_width_ratio': float(current_bb_width / avg_bb_width),
                    'atr_ratio': float(current_atr / avg_atr),
                    'rsi': float(current_rsi),
                    'adx': float(current_adx) if not pd.isna(current_adx) else 0.0
                }
            )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'bb_compression_threshold': self.bb_compression_threshold,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'atr_period': self.atr_period,
            'atr_low_threshold': self.atr_low_threshold,
            'risk_reward_ratio': self.risk_reward_ratio,
            'adx_max': self.adx_max
        }

