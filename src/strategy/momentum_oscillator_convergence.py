"""Momentum Oscillator Convergence Strategy

This strategy uses multiple momentum oscillators (RSI, Stochastic, MACD, Williams %R)
and enters when they converge in the same direction, indicating strong momentum.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class MomentumOscillatorConvergenceStrategy(BaseStrategy):
    """
    Momentum Oscillator Convergence Strategy
    
    Entry Rules:
    - Multiple oscillators align in same direction (RSI, Stochastic, MACD, Williams %R)
    - At least 3 out of 4 oscillators must agree
    - Price must be above/below key moving average (trend filter)
    - Volume confirms (above average)
    - ATR indicates sufficient volatility
    
    Exit Rules:
    - Take profit: 2.5R or when oscillators diverge (2+ change direction)
    - Stop loss: 1.5x ATR or below/above recent swing
    
    Filters:
    - Oscillator Convergence: Minimum 3/4 agreement
    - Trend: Price above/below EMA(50) for long/short
    - Volatility: ATR > 80% of average
    - Volume: Entry volume > 120% of average
    - Time: Best during active sessions (London/NY)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MomentumOscillatorConvergence", config)
        
        # Oscillator parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        
        self.stoch_k_period = config.get('stoch_k_period', 14)
        self.stoch_d_period = config.get('stoch_d_period', 3)
        self.stoch_oversold = config.get('stoch_oversold', 20)
        self.stoch_overbought = config.get('stoch_overbought', 80)
        
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        
        self.williams_period = config.get('williams_period', 14)
        self.williams_oversold = config.get('williams_oversold', -80)
        self.williams_overbought = config.get('williams_overbought', -20)
        
        # Trend filter
        self.trend_ema_period = config.get('trend_ema_period', 50)
        
        # Convergence requirement
        self.min_oscillators_agree = config.get('min_oscillators_agree', 3)
        
        # ATR parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 1.5)
        self.atr_volatility_filter = config.get('atr_volatility_filter', 0.8)
        
        # Volume parameters
        self.volume_period = config.get('volume_period', 20)
        self.volume_threshold = config.get('volume_threshold', 1.2)
        
        # Risk/Reward
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.5)
        
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def _calculate_macd(self, series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
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
    
    def _check_oscillator_convergence(self, rsi: float, stoch_k: float, macd_hist: float, 
                                     williams_r: float) -> Tuple[bool, bool, int]:
        """Check if oscillators converge (bullish or bearish)"""
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI
        if not pd.isna(rsi):
            if rsi > 50:
                bullish_signals += 1
            elif rsi < 50:
                bearish_signals += 1
        
        # Stochastic
        if not pd.isna(stoch_k):
            if stoch_k > 50:
                bullish_signals += 1
            elif stoch_k < 50:
                bearish_signals += 1
        
        # MACD Histogram
        if not pd.isna(macd_hist):
            if macd_hist > 0:
                bullish_signals += 1
            elif macd_hist < 0:
                bearish_signals += 1
        
        # Williams %R
        if not pd.isna(williams_r):
            if williams_r > -50:
                bullish_signals += 1
            elif williams_r < -50:
                bearish_signals += 1
        
        bullish_convergence = bullish_signals >= self.min_oscillators_agree
        bearish_convergence = bearish_signals >= self.min_oscillators_agree
        
        return bullish_convergence, bearish_convergence, max(bullish_signals, bearish_signals)
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Get bars
        min_bars = max(self.rsi_period, self.stoch_k_period, self.macd_slow, 
                     self.williams_period, self.trend_ema_period, self.atr_period, 
                     self.volume_period) + 10
        bars = self.get_bars(symbol, min_bars)
        if len(bars) < min_bars:
            return None
        
        # Calculate all oscillators
        close = bars['close']
        rsi = self._calculate_rsi(close, self.rsi_period)
        stoch_k, stoch_d = self._calculate_stochastic(bars, self.stoch_k_period, self.stoch_d_period)
        macd_line, signal_line, macd_hist = self._calculate_macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        williams_r = self._calculate_williams_r(bars, self.williams_period)
        trend_ema = close.ewm(span=self.trend_ema_period, adjust=False).mean()
        atr = self._calculate_atr(bars, self.atr_period)
        volume_avg = bars['volume'].rolling(window=self.volume_period).mean()
        
        if len(rsi) < 1 or len(stoch_k) < 1 or len(macd_hist) < 1 or len(williams_r) < 1:
            return None
        
        current_price = bar_event.close
        current_rsi = rsi.iloc[-1]
        current_stoch_k = stoch_k.iloc[-1]
        current_macd_hist = macd_hist.iloc[-1]
        current_williams_r = williams_r.iloc[-1]
        current_trend_ema = trend_ema.iloc[-1]
        current_atr = atr.iloc[-1]
        avg_atr = atr.tail(self.atr_period).mean()
        current_volume = bar_event.tick_volume
        avg_volume = volume_avg.iloc[-1]
        
        # Volatility filter
        if pd.isna(current_atr) or pd.isna(avg_atr) or current_atr < avg_atr * self.atr_volatility_filter:
            return None
        
        # Volume filter
        if pd.isna(current_volume) or pd.isna(avg_volume) or current_volume < avg_volume * self.volume_threshold:
            return None
        
        # Check oscillator convergence
        bullish_conv, bearish_conv, agreement_count = self._check_oscillator_convergence(
            current_rsi, current_stoch_k, current_macd_hist, current_williams_r
        )
        
        # Entry logic: BULLISH CONVERGENCE
        if bullish_conv and current_price > current_trend_ema:
            # Calculate SL and TP
            swing_low = bars['low'].tail(20).min()
            sl_price = min(swing_low, current_price - current_atr * self.atr_sl_multiplier)
            risk = current_price - sl_price
            tp_price = current_price + risk * self.risk_reward_ratio
            
            return SignalEvent(
                symbol=symbol,
                signal_type='BUY',
                strength=min(1.0, agreement_count / 4.0),  # Strength based on agreement
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(risk * self.risk_reward_ratio),
                    'oscillator_agreement': agreement_count,
                    'rsi': float(current_rsi),
                    'stochastic_k': float(current_stoch_k),
                    'macd_histogram': float(current_macd_hist),
                    'williams_r': float(current_williams_r),
                    'atr': float(current_atr)
                }
            )
        
        # Entry logic: BEARISH CONVERGENCE
        elif bearish_conv and current_price < current_trend_ema:
            # Calculate SL and TP
            swing_high = bars['high'].tail(20).max()
            sl_price = max(swing_high, current_price + current_atr * self.atr_sl_multiplier)
            risk = sl_price - current_price
            tp_price = current_price - risk * self.risk_reward_ratio
            
            return SignalEvent(
                symbol=symbol,
                signal_type='SELL',
                strength=min(1.0, agreement_count / 4.0),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(risk * self.risk_reward_ratio),
                    'oscillator_agreement': agreement_count,
                    'rsi': float(current_rsi),
                    'stochastic_k': float(current_stoch_k),
                    'macd_histogram': float(current_macd_hist),
                    'williams_r': float(current_williams_r),
                    'atr': float(current_atr)
                }
            )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'rsi_period': self.rsi_period,
            'stoch_k_period': self.stoch_k_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'min_oscillators_agree': self.min_oscillators_agree,
            'risk_reward_ratio': self.risk_reward_ratio
        }

