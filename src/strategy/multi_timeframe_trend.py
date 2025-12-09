"""Multi-Timeframe Trend Continuation Strategy

This strategy identifies strong trends across multiple timeframes and enters
on pullbacks in the direction of the higher timeframe trend.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, time

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class MultiTimeframeTrendStrategy(BaseStrategy):
    """
    Multi-Timeframe Trend Continuation Strategy
    
    Entry Rules:
    - Higher timeframe (HTF) must show clear trend (EMA alignment + ADX > 25)
    - Lower timeframe (LTF) shows pullback to key support/resistance
    - RSI on LTF shows oversold (for longs) or overbought (for shorts) but recovering
    - Volume confirms pullback (lower volume on pullback, higher on continuation)
    
    Exit Rules:
    - Take profit: 2.5R or when HTF trend weakens (ADX < 20)
    - Stop loss: Below/above recent swing low/high or 1.5x ATR
    
    Filters:
    - Trend: HTF EMA alignment (fast > slow for uptrend)
    - Volatility: ATR must be above 20-period average (avoid low vol)
    - Volume: Pullback volume < 80% of average, continuation volume > 120%
    - Time: Avoid 30min before/after major news (configurable)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MultiTimeframeTrend", config)
        
        # Timeframe configuration
        self.htf_multiplier = config.get('htf_multiplier', 4)  # HTF = LTF * 4
        self.ltf_periods = config.get('ltf_periods', 50)  # Bars to keep for LTF
        
        # EMA parameters
        self.fast_ema = config.get('fast_ema', 21)
        self.slow_ema = config.get('slow_ema', 50)
        
        # ADX parameters
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 25.0)
        self.adx_exit_threshold = config.get('adx_exit_threshold', 20.0)
        
        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 35)
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.rsi_recovery = config.get('rsi_recovery', 5)  # RSI must recover by this amount
        
        # ATR parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 1.5)
        self.atr_volatility_filter = config.get('atr_volatility_filter', 0.8)  # ATR must be > 80% of avg
        
        # Volume parameters
        self.volume_period = config.get('volume_period', 20)
        self.volume_pullback_threshold = config.get('volume_pullback_threshold', 0.8)
        self.volume_continuation_threshold = config.get('volume_continuation_threshold', 1.2)
        
        # Risk/Reward
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.5)
        
        # Time filters
        self.avoid_news_hours = config.get('avoid_news_hours', True)
        self.news_buffer_minutes = config.get('news_buffer_minutes', 30)
        
        # State tracking
        self.htf_bars: Dict[str, pd.DataFrame] = {}
        self.last_htf_update: Dict[str, datetime] = {}
        
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
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / atr
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def _get_htf_trend(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get higher timeframe trend direction and strength"""
        if symbol not in self.htf_bars or len(self.htf_bars[symbol]) < self.slow_ema:
            return None
        
        htf_df = self.htf_bars[symbol]
        
        # Calculate indicators
        fast_ema = self._calculate_ema(htf_df['close'], self.fast_ema)
        slow_ema = self._calculate_ema(htf_df['close'], self.slow_ema)
        adx, plus_di, minus_di = self._calculate_adx(htf_df, self.adx_period)
        
        if len(fast_ema) < 2:
            return None
        
        current_fast = fast_ema.iloc[-1]
        current_slow = slow_ema.iloc[-1]
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        
        # Determine trend
        if pd.isna(current_adx) or current_adx < self.adx_threshold:
            return None
        
        trend = None
        if current_fast > current_slow and current_plus_di > current_minus_di:
            trend = 'BULLISH'
        elif current_fast < current_slow and current_minus_di > current_plus_di:
            trend = 'BEARISH'
        
        return {
            'trend': trend,
            'adx': float(current_adx),
            'plus_di': float(current_plus_di),
            'minus_di': float(current_minus_di),
            'fast_ema': float(current_fast),
            'slow_ema': float(current_slow)
        }
    
    def _update_htf(self, bar_event: BarEvent) -> None:
        """Update higher timeframe bars (aggregate LTF bars)"""
        symbol = bar_event.symbol
        
        if symbol not in self.htf_bars:
            self.htf_bars[symbol] = pd.DataFrame()
            self.last_htf_update[symbol] = bar_event.time
        
        # Simple aggregation: every N bars, create HTF bar
        # In production, use proper timeframe conversion
        if symbol not in self.last_htf_update:
            self.last_htf_update[symbol] = bar_event.time
        
        # For simplicity, aggregate every N bars
        ltf_bars = self.get_bars(symbol, self.htf_multiplier)
        if len(ltf_bars) >= self.htf_multiplier:
            # Create HTF bar from last N LTF bars
            htf_bar = {
                'time': bar_event.time,
                'open': ltf_bars.iloc[-self.htf_multiplier]['open'],
                'high': ltf_bars['high'].tail(self.htf_multiplier).max(),
                'low': ltf_bars['low'].tail(self.htf_multiplier).min(),
                'close': bar_event.close,
                'volume': ltf_bars['volume'].tail(self.htf_multiplier).sum()
            }
            
            new_row = pd.DataFrame([htf_bar])
            self.htf_bars[symbol] = pd.concat([self.htf_bars[symbol], new_row], ignore_index=True)
            
            # Keep only recent HTF bars
            if len(self.htf_bars[symbol]) > 200:
                self.htf_bars[symbol] = self.htf_bars[symbol].tail(200).reset_index(drop=True)
            
            self.last_htf_update[symbol] = bar_event.time
    
    def _is_valid_time(self, bar_time: datetime) -> bool:
        """Check if current time is valid for trading (avoid news)"""
        if not self.avoid_news_hours:
            return True
        
        # Avoid trading 30min before/after major news (8:30, 10:00, 14:00, 16:00 EST)
        # Simplified: avoid 8:00-9:00, 9:30-10:30, 13:30-14:30, 15:30-16:30 UTC
        hour = bar_time.hour
        minute = bar_time.minute
        
        # Major news times (UTC, adjust for your broker)
        avoid_windows = [
            (8, 0, 9, 0),   # 8:00-9:00
            (9, 30, 10, 30), # 9:30-10:30
            (13, 30, 14, 30), # 13:30-14:30
            (15, 30, 16, 30)  # 15:30-16:30
        ]
        
        for start_h, start_m, end_h, end_m in avoid_windows:
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m
            current_minutes = hour * 60 + minute
            
            if start_minutes <= current_minutes <= end_minutes:
                return False
        
        return True
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Update HTF
        self._update_htf(bar_event)
        
        # Get HTF trend
        htf_trend = self._get_htf_trend(symbol)
        if not htf_trend or not htf_trend['trend']:
            return None
        
        # Time filter
        if not self._is_valid_time(bar_event.time):
            return None
        
        # Get LTF bars
        ltf_bars = self.get_bars(symbol, self.ltf_periods)
        if len(ltf_bars) < max(self.slow_ema, self.atr_period, self.rsi_period, self.volume_period):
            return None
        
        # Calculate LTF indicators
        atr = self._calculate_atr(ltf_bars, self.atr_period)
        rsi = self._calculate_rsi(ltf_bars['close'], self.rsi_period)
        volume_avg = ltf_bars['volume'].rolling(window=self.volume_period).mean()
        
        if len(atr) < 2 or len(rsi) < 2:
            return None
        
        current_atr = atr.iloc[-1]
        avg_atr = atr.tail(self.atr_period).mean()
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        current_volume = ltf_bars['volume'].iloc[-1]
        avg_volume = volume_avg.iloc[-1]
        
        # Volatility filter
        if pd.isna(current_atr) or pd.isna(avg_atr) or current_atr < avg_atr * self.atr_volatility_filter:
            return None
        
        # Volume filter
        if pd.isna(current_volume) or pd.isna(avg_volume):
            return None
        
        # Entry logic for BULLISH trend
        if htf_trend['trend'] == 'BULLISH':
            # Look for pullback: price below fast EMA but above slow EMA
            fast_ema_ltf = self._calculate_ema(ltf_bars['close'], self.fast_ema)
            slow_ema_ltf = self._calculate_ema(ltf_bars['close'], self.slow_ema)
            
            if len(fast_ema_ltf) < 1:
                return None
            
            current_price = bar_event.close
            current_fast_ema = fast_ema_ltf.iloc[-1]
            current_slow_ema = slow_ema_ltf.iloc[-1]
            
            # Pullback condition: price near slow EMA, below fast EMA
            pullback_zone = current_slow_ema * 0.998  # Within 0.2% of slow EMA
            
            if (pullback_zone <= current_price <= current_slow_ema * 1.002 and
                current_price < current_fast_ema and
                current_rsi < self.rsi_oversold + self.rsi_recovery and
                current_rsi > prev_rsi and  # RSI recovering
                current_volume < avg_volume * self.volume_pullback_threshold):
                
                # Calculate SL and TP
                swing_low = ltf_bars['low'].tail(20).min()
                sl_price = min(swing_low, current_price - current_atr * self.atr_sl_multiplier)
                risk = current_price - sl_price
                tp_price = current_price + risk * self.risk_reward_ratio
                
                return SignalEvent(
                    symbol=symbol,
                    signal_type='BUY',
                    strength=htf_trend['adx'] / 50.0,  # Normalize ADX
                    timestamp=bar_event.time,
                    strategy_name=self.name,
                    metadata={
                        'entry_price': float(current_price),
                        'sl_price': float(sl_price),
                        'tp_price': float(tp_price),
                        'risk': float(risk),
                        'reward': float(risk * self.risk_reward_ratio),
                        'htf_trend': htf_trend['trend'],
                        'htf_adx': htf_trend['adx'],
                        'rsi': float(current_rsi),
                        'atr': float(current_atr)
                    }
                )
        
        # Entry logic for BEARISH trend
        elif htf_trend['trend'] == 'BEARISH':
            fast_ema_ltf = self._calculate_ema(ltf_bars['close'], self.fast_ema)
            slow_ema_ltf = self._calculate_ema(ltf_bars['close'], self.slow_ema)
            
            if len(fast_ema_ltf) < 1:
                return None
            
            current_price = bar_event.close
            current_fast_ema = fast_ema_ltf.iloc[-1]
            current_slow_ema = slow_ema_ltf.iloc[-1]
            
            # Pullback condition: price near slow EMA, above fast EMA
            pullback_zone = current_slow_ema * 1.002
            
            if (current_slow_ema * 0.998 <= current_price <= pullback_zone and
                current_price > current_fast_ema and
                current_rsi > self.rsi_overbought - self.rsi_recovery and
                current_rsi < prev_rsi and  # RSI recovering downward
                current_volume < avg_volume * self.volume_pullback_threshold):
                
                # Calculate SL and TP
                swing_high = ltf_bars['high'].tail(20).max()
                sl_price = max(swing_high, current_price + current_atr * self.atr_sl_multiplier)
                risk = sl_price - current_price
                tp_price = current_price - risk * self.risk_reward_ratio
                
                return SignalEvent(
                    symbol=symbol,
                    signal_type='SELL',
                    strength=htf_trend['adx'] / 50.0,
                    timestamp=bar_event.time,
                    strategy_name=self.name,
                    metadata={
                        'entry_price': float(current_price),
                        'sl_price': float(sl_price),
                        'tp_price': float(tp_price),
                        'risk': float(risk),
                        'reward': float(risk * self.risk_reward_ratio),
                        'htf_trend': htf_trend['trend'],
                        'htf_adx': htf_trend['adx'],
                        'rsi': float(current_rsi),
                        'atr': float(current_atr)
                    }
                )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'htf_multiplier': self.htf_multiplier,
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'adx_period': self.adx_period,
            'adx_threshold': self.adx_threshold,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'atr_period': self.atr_period,
            'atr_sl_multiplier': self.atr_sl_multiplier,
            'risk_reward_ratio': self.risk_reward_ratio
        }

