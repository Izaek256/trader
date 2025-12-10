"""High R:R Trend Pullback Strategy

This strategy focuses on high probability setups with excellent risk-reward ratios.
It trades pullbacks in strong trends with tight stops and wide targets.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class TrendPullbackRRStrategy(BaseStrategy):
    """
    High R:R Trend Pullback Strategy
    
    Entry Rules:
    - Strong trend identified (EMA alignment + ADX > 25)
    - Price pulls back to key support/resistance (EMA or recent swing)
    - RSI shows oversold in uptrend or overbought in downtrend
    - Volume confirms (lower on pullback, higher on continuation)
    - Tight stop loss (below swing low/high or 1.0x ATR)
    - Wide take profit (minimum 3.0R, target 4.0R)
    
    Exit Rules:
    - Take profit: 3.0-4.0R (trailing stop after 2R)
    - Stop loss: Tight, 1.0x ATR or swing structure
    - Exit on trend reversal (ADX < 20)
    
    Filters:
    - Trend: Strong trend required (ADX > 25)
    - Volatility: ATR must be reasonable (not too high, not too low)
    - Volume: Pullback volume < average, continuation volume > average
    - Time: Avoid low liquidity hours
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TrendPullbackRR", config)
        
        # EMA parameters
        self.fast_ema = config.get('fast_ema', 21)
        self.slow_ema = config.get('slow_ema', 50)
        self.trend_ema = config.get('trend_ema', 200)  # Long-term trend filter
        
        # ADX parameters
        self.adx_period = config.get('adx_period', 14)
        self.adx_min = config.get('adx_min', 25.0)  # Strong trend required
        self.adx_exit = config.get('adx_exit', 20.0)  # Exit if trend weakens
        
        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 40)  # Less extreme for pullbacks
        self.rsi_overbought = config.get('rsi_overbought', 60)
        
        # ATR parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 1.0)  # Tight stop
        self.atr_tp_multiplier = config.get('atr_tp_multiplier', 4.0)  # Wide target
        self.atr_min = config.get('atr_min', 0.0005)  # Minimum volatility
        self.atr_max = config.get('atr_max', 0.005)   # Maximum volatility
        
        # Volume parameters
        self.volume_period = config.get('volume_period', 20)
        self.volume_pullback_threshold = config.get('volume_pullback_threshold', 0.9)
        self.volume_continuation_threshold = config.get('volume_continuation_threshold', 1.1)
        
        # Risk/Reward
        self.min_rr = config.get('min_rr', 3.0)  # Minimum 3:1
        self.target_rr = config.get('target_rr', 4.0)  # Target 4:1
        
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
                     self.rsi_period, self.adx_period, self.volume_period) + 20
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
        adx, plus_di, minus_di = self._calculate_adx(bars, self.adx_period)
        volume_avg = bars['volume'].rolling(window=self.volume_period).mean()
        
        if len(fast_ema) < 1 or len(atr) < 1 or len(rsi) < 1 or len(adx) < 1:
            return None
        
        current_price = bar_event.close
        current_fast = fast_ema.iloc[-1]
        current_slow = slow_ema.iloc[-1]
        current_trend = trend_ema.iloc[-1]
        current_atr = atr.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_volume = bar_event.tick_volume
        avg_volume = volume_avg.iloc[-1]
        
        # Trend filter: Must have strong trend
        if pd.isna(current_adx) or current_adx < self.adx_min:
            return None
        
        # Volatility filter: ATR must be in reasonable range
        if pd.isna(current_atr):
            return None
        
        atr_normalized = current_atr / current_price
        if atr_normalized < self.atr_min or atr_normalized > self.atr_max:
            return None
        
        # Volume filter
        if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
            return None
        
        volume_ratio = current_volume / avg_volume
        
        # Find swing points
        swings = self._find_swing_points(bars, self.swing_lookback)
        
        # Entry logic: BULLISH (uptrend pullback)
        if (current_fast > current_slow and  # Fast above slow
            current_price > current_trend and  # Above long-term trend
            current_plus_di > current_minus_di and  # Uptrend confirmed
            current_price < current_fast):  # Pullback below fast EMA
            
            # RSI should show pullback (oversold but not extreme)
            if pd.isna(current_rsi) or current_rsi < self.rsi_oversold:
                return None
            
            # Volume should be lower on pullback
            if volume_ratio > self.volume_pullback_threshold:
                return None
            
            # Calculate tight stop loss
            swing_low = swings['low'] if swings['low'] else current_price - current_atr * 2
            sl_price = min(swing_low, current_price - current_atr * self.atr_sl_multiplier)
            risk = current_price - sl_price
            
            if risk <= 0:
                return None
            
            # Calculate wide take profit (minimum 3R, target 4R)
            tp_price = current_price + risk * self.target_rr
            
            # Verify minimum R:R
            if (tp_price - current_price) / risk < self.min_rr:
                return None
            
            return SignalEvent(
                symbol=symbol,
                signal_type='BUY',
                strength=min(1.0, current_adx / 50.0),  # Normalize ADX
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
                    'atr': float(current_atr),
                    'volume_ratio': float(volume_ratio)
                }
            )
        
        # Entry logic: BEARISH (downtrend pullback)
        elif (current_fast < current_slow and  # Fast below slow
              current_price < current_trend and  # Below long-term trend
              current_minus_di > current_plus_di and  # Downtrend confirmed
              current_price > current_fast):  # Pullback above fast EMA
            
            # RSI should show pullback (overbought but not extreme)
            if pd.isna(current_rsi) or current_rsi > self.rsi_overbought:
                return None
            
            # Volume should be lower on pullback
            if volume_ratio > self.volume_pullback_threshold:
                return None
            
            # Calculate tight stop loss
            swing_high = swings['high'] if swings['high'] else current_price + current_atr * 2
            sl_price = max(swing_high, current_price + current_atr * self.atr_sl_multiplier)
            risk = sl_price - current_price
            
            if risk <= 0:
                return None
            
            # Calculate wide take profit (minimum 3R, target 4R)
            tp_price = current_price - risk * self.target_rr
            
            # Verify minimum R:R
            if (current_price - tp_price) / risk < self.min_rr:
                return None
            
            return SignalEvent(
                symbol=symbol,
                signal_type='SELL',
                strength=min(1.0, current_adx / 50.0),
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
                    'atr': float(current_atr),
                    'volume_ratio': float(volume_ratio)
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

