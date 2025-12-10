"""Balanced Trend Following Strategy

This strategy balances win rate and R:R for consistent profitability.
Focuses on quality entries in trends with proper risk management.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class BalancedTrendFollowingStrategy(BaseStrategy):
    """
    Balanced Trend Following Strategy
    
    Entry Rules:
    - Clear trend (EMA alignment + ADX > 20)
    - Price pulls back to EMA support/resistance
    - RSI shows pullback (not extreme, recovering)
    - MACD histogram confirms direction
    - Volume confirms (lower on pullback)
    - Moderate R:R (2.0-2.5:1) for better win rate
    
    Exit Rules:
    - Take profit: 2.0-2.5R
    - Stop loss: 1.0-1.5x ATR or swing structure
    - Exit on trend reversal
    
    Filters:
    - Trend: Moderate trend (ADX > 20, not too extreme)
    - Volatility: Normal ATR range
    - Volume: Confirmation required
    - Time: Avoid low liquidity
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BalancedTrendFollowing", config)
        
        # EMA parameters
        self.fast_ema = config.get('fast_ema', 21)
        self.slow_ema = config.get('slow_ema', 50)
        self.trend_ema = config.get('trend_ema', 100)  # Medium-term trend
        
        # ADX parameters
        self.adx_period = config.get('adx_period', 14)
        self.adx_min = config.get('adx_min', 20.0)  # Moderate trend
        self.adx_max = config.get('adx_max', 50.0)  # Avoid extreme trends
        
        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 35)  # Less extreme
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.rsi_recovery = config.get('rsi_recovery', 3)  # Must be recovering
        
        # MACD parameters
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        
        # ATR parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 1.2)  # Moderate stop
        self.atr_tp_multiplier = config.get('atr_tp_multiplier', 2.5)  # Moderate target
        
        # Volume parameters
        self.volume_period = config.get('volume_period', 20)
        self.volume_pullback_threshold = config.get('volume_pullback_threshold', 0.95)
        
        # Risk/Reward
        self.min_rr = config.get('min_rr', 2.0)  # Minimum 2:1
        self.target_rr = config.get('target_rr', 2.5)  # Target 2.5:1
        
        # Swing detection
        self.swing_lookback = config.get('swing_lookback', 10)
        
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return series.ewm(span=period, adjust=False).mean()
    
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
    
    def _calculate_macd(self, series: pd.Series, fast: int, slow: int, signal: int) -> tuple:
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> tuple:
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
        
        return adx, plus_di, minus_di
    
    def _find_swing_points(self, df: pd.DataFrame, lookback: int) -> Dict[str, float]:
        """Find recent swing high and low"""
        if len(df) < lookback * 2:
            return {'high': None, 'low': None}
        
        recent = df.tail(lookback * 2)
        swing_high = recent['high'].tail(lookback).max()
        swing_low = recent['low'].tail(lookback).min()
        
        return {
            'high': float(swing_high),
            'low': float(swing_low)
        }
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Get bars
        min_bars = max(self.trend_ema, self.slow_ema, self.atr_period, 
                     self.rsi_period, self.macd_slow, self.adx_period, 
                     self.volume_period) + 20
        bars = self.get_bars(symbol, min_bars)
        if len(bars) < min_bars:
            return None
        
        # Calculate indicators
        close = bars['close']
        fast_ema = self._calculate_ema(close, self.fast_ema)
        slow_ema = self._calculate_ema(close, self.slow_ema)
        trend_ema = self._calculate_ema(close, self.trend_ema)
        atr = self._calculate_atr(bars, self.atr_period)
        rsi = self._calculate_rsi(close, self.rsi_period)
        macd_line, signal_line, macd_hist = self._calculate_macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        adx, plus_di, minus_di = self._calculate_adx(bars, self.adx_period)
        volume_avg = bars['volume'].rolling(window=self.volume_period).mean()
        
        if len(fast_ema) < 2 or len(atr) < 1 or len(rsi) < 2 or len(macd_hist) < 1 or len(adx) < 1:
            return None
        
        current_price = bar_event.close
        current_fast = fast_ema.iloc[-1]
        prev_fast = fast_ema.iloc[-2]
        current_slow = slow_ema.iloc[-1]
        current_trend = trend_ema.iloc[-1]
        current_atr = atr.iloc[-1]
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        current_macd_hist = macd_hist.iloc[-1]
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_volume = bar_event.tick_volume
        avg_volume = volume_avg.iloc[-1]
        
        # Trend filter: Moderate trend required
        if pd.isna(current_adx) or current_adx < self.adx_min or current_adx > self.adx_max:
            return None
        
        # Volatility filter: ATR must be reasonable
        if pd.isna(current_atr) or current_atr <= 0:
            return None
        
        # Volume filter
        if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
            return None
        
        volume_ratio = current_volume / avg_volume
        
        # Find swing points
        swings = self._find_swing_points(bars, self.swing_lookback)
        
        # Entry logic: BULLISH (uptrend pullback)
        if (current_fast > current_slow and  # Fast above slow
            current_price > current_trend and  # Above trend EMA
            current_plus_di > current_minus_di and  # Uptrend confirmed
            current_price <= current_fast * 1.002 and  # Near or below fast EMA (pullback)
            current_price >= current_slow * 0.998):  # Above slow EMA (not too deep)
            
            # RSI should show pullback (oversold but recovering)
            if pd.isna(current_rsi) or current_rsi < self.rsi_oversold:
                return None
            
            # RSI should be recovering (rising)
            if current_rsi < prev_rsi:
                return None
            
            # MACD should be positive or recovering
            if pd.isna(current_macd_hist) or current_macd_hist < -0.0001:
                return None
            
            # Volume should be lower on pullback
            if volume_ratio > self.volume_pullback_threshold:
                return None
            
            # Calculate stop loss
            swing_low = swings['low'] if swings['low'] else current_price - current_atr * 2
            sl_price = min(swing_low, current_price - current_atr * self.atr_sl_multiplier)
            risk = current_price - sl_price
            
            if risk <= 0:
                return None
            
            # Calculate take profit (2.5R)
            tp_price = current_price + risk * self.target_rr
            
            # Verify minimum R:R
            if (tp_price - current_price) / risk < self.min_rr:
                return None
            
            return SignalEvent(
                symbol=symbol,
                signal_type='BUY',
                strength=min(1.0, current_adx / 40.0),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(tp_price - current_price),
                    'rr_ratio': float((tp_price - current_price) / risk),
                    'adx': float(current_adx),
                    'rsi': float(current_rsi),
                    'macd_hist': float(current_macd_hist),
                    'atr': float(current_atr)
                }
            )
        
        # Entry logic: BEARISH (downtrend pullback)
        elif (current_fast < current_slow and  # Fast below slow
              current_price < current_trend and  # Below trend EMA
              current_minus_di > current_plus_di and  # Downtrend confirmed
              current_price >= current_fast * 0.998 and  # Near or above fast EMA (pullback)
              current_price <= current_slow * 1.002):  # Below slow EMA (not too deep)
            
            # RSI should show pullback (overbought but recovering downward)
            if pd.isna(current_rsi) or current_rsi > self.rsi_overbought:
                return None
            
            # RSI should be recovering (falling)
            if current_rsi > prev_rsi:
                return None
            
            # MACD should be negative or recovering
            if pd.isna(current_macd_hist) or current_macd_hist > 0.0001:
                return None
            
            # Volume should be lower on pullback
            if volume_ratio > self.volume_pullback_threshold:
                return None
            
            # Calculate stop loss
            swing_high = swings['high'] if swings['high'] else current_price + current_atr * 2
            sl_price = max(swing_high, current_price + current_atr * self.atr_sl_multiplier)
            risk = sl_price - current_price
            
            if risk <= 0:
                return None
            
            # Calculate take profit (2.5R)
            tp_price = current_price - risk * self.target_rr
            
            # Verify minimum R:R
            if (current_price - tp_price) / risk < self.min_rr:
                return None
            
            return SignalEvent(
                symbol=symbol,
                signal_type='SELL',
                strength=min(1.0, current_adx / 40.0),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(current_price - tp_price),
                    'rr_ratio': float((current_price - tp_price) / risk),
                    'adx': float(current_adx),
                    'rsi': float(current_rsi),
                    'macd_hist': float(current_macd_hist),
                    'atr': float(current_atr)
                }
            )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'adx_min': self.adx_min,
            'atr_sl_multiplier': self.atr_sl_multiplier,
            'atr_tp_multiplier': self.atr_tp_multiplier,
            'min_rr': self.min_rr,
            'target_rr': self.target_rr
        }

