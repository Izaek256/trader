"""VIX / Volatility Index Specific Strategy

This strategy is specifically designed for volatility indices (VIX, VXX, etc.)
which have unique characteristics: mean-reverting, contango/backwardation effects.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class VIXVolatilityIndexStrategy(BaseStrategy):
    """
    VIX / Volatility Index Strategy
    
    Entry Rules:
    - VIX spikes above 2 standard deviations from mean (fear extreme) -> SELL
    - VIX drops below -2 standard deviations (complacency) -> BUY (if mean-reverting)
    - RSI confirms extreme (< 25 for buy, > 75 for sell)
    - Volume spike confirms the move
    - Contango/Backwardation filter (if VIX futures data available)
    
    Exit Rules:
    - Take profit: VIX returns to mean (Z-score < 0.5) or 2.0R
    - Stop loss: VIX continues extreme move (Z-score > 3.0 or < -3.0)
    - Time decay: Close positions before expiration (if applicable)
    
    Filters:
    - Volatility: VIX must be in extreme territory (Z-score > 2.0)
    - Volume: Spike in volume confirms the extreme
    - Time: Avoid trading near VIX expiration dates
    - Market Regime: Works best in mean-reverting volatility regimes
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("VIXVolatilityIndex", config)
        
        # Z-score parameters
        self.zscore_period = config.get('zscore_period', 60)  # Longer period for VIX
        self.zscore_entry = config.get('zscore_entry', 2.0)
        self.zscore_exit = config.get('zscore_exit', 0.5)
        self.zscore_stop = config.get('zscore_stop', 3.0)
        
        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 25)
        self.rsi_overbought = config.get('rsi_overbought', 75)
        
        # Volume parameters
        self.volume_period = config.get('volume_period', 20)
        self.volume_spike_threshold = config.get('volume_spike_threshold', 1.5)
        
        # Mean reversion parameters
        self.mean_reversion_period = config.get('mean_reversion_period', 100)
        self.mean_reversion_strength = config.get('mean_reversion_strength', 0.7)  # How strong mean reversion
        
        # ATR parameters (for stop loss)
        self.atr_period = config.get('atr_period', 14)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 2.0)
        
        # Risk/Reward
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.0)
        
        # VIX-specific: expiration avoidance
        self.avoid_expiration_days = config.get('avoid_expiration_days', 3)
        
        # State tracking
        self.entry_zscore: Dict[str, float] = {}
        
    def _calculate_zscore(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Z-score"""
        mean = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return (series - mean) / std
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
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
    
    def _test_mean_reversion(self, series: pd.Series, period: int) -> float:
        """Test strength of mean reversion using half-life"""
        if len(series) < period:
            return 0.0
        
        spread = series - series.rolling(window=period).mean()
        spread_diff = spread.diff().dropna()
        spread_lag = spread.shift(1).dropna()
        
        if len(spread_diff) < 10 or len(spread_lag) < 10:
            return 0.0
        
        aligned = pd.concat([spread_diff, spread_lag], axis=1).dropna()
        if len(aligned) < 10:
            return 0.0
        
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                aligned.iloc[:, 1], aligned.iloc[:, 0]
            )
            # Negative slope indicates mean reversion
            # Return absolute value of slope as strength (higher = stronger)
            return abs(slope) if slope < 0 else 0.0
        except:
            return 0.0
    
    def _is_near_expiration(self, bar_time: datetime) -> bool:
        """Check if we're near VIX expiration (3rd Wednesday of month)"""
        # VIX expires on 3rd Wednesday of month
        # Simplified: avoid last 3 days of month
        day = bar_time.day
        days_in_month = (bar_time.replace(month=bar_time.month % 12 + 1, day=1) - 
                        pd.Timedelta(days=1)).day
        
        return day > (days_in_month - self.avoid_expiration_days)
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Expiration filter
        if self._is_near_expiration(bar_event.time):
            return None
        
        # Get bars
        min_bars = max(self.zscore_period, self.rsi_period, self.volume_period, 
                     self.mean_reversion_period, self.atr_period) + 10
        bars = self.get_bars(symbol, min_bars)
        if len(bars) < min_bars:
            return None
        
        # Calculate indicators
        zscore = self._calculate_zscore(bars['close'], self.zscore_period)
        rsi = self._calculate_rsi(bars['close'], self.rsi_period)
        atr = self._calculate_atr(bars, self.atr_period)
        volume_avg = bars['volume'].rolling(window=self.volume_period).mean()
        
        if len(zscore) < 1 or len(rsi) < 1 or len(atr) < 1:
            return None
        
        current_price = bar_event.close
        current_zscore = zscore.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]
        current_volume = bar_event.tick_volume
        avg_volume = volume_avg.iloc[-1]
        
        if pd.isna(current_zscore) or pd.isna(current_rsi):
            return None
        
        # Test mean reversion strength
        mean_rev_strength = self._test_mean_reversion(bars['close'], self.mean_reversion_period)
        if mean_rev_strength < self.mean_reversion_strength:
            return None  # Not strong enough mean reversion
        
        # Check for existing position
        existing_entry_zscore = self.entry_zscore.get(symbol)
        
        # Exit logic
        if existing_entry_zscore is not None:
            current_signal = self.current_signals.get(symbol)
            
            # Exit on mean reversion
            if abs(current_zscore) < self.zscore_exit:
                self.entry_zscore.pop(symbol, None)
                self.current_signals.pop(symbol, None)
                return SignalEvent(
                    symbol=symbol,
                    signal_type='EXIT',
                    strength=1.0,
                    timestamp=bar_event.time,
                    strategy_name=self.name,
                    metadata={
                        'exit_reason': 'mean_reversion',
                        'entry_zscore': existing_entry_zscore,
                        'exit_zscore': float(current_zscore)
                    }
                )
            
            # Stop loss
            if (current_signal == 'BUY' and current_zscore < -self.zscore_stop) or \
               (current_signal == 'SELL' and current_zscore > self.zscore_stop):
                self.entry_zscore.pop(symbol, None)
                self.current_signals.pop(symbol, None)
                return SignalEvent(
                    symbol=symbol,
                    signal_type='EXIT',
                    strength=1.0,
                    timestamp=bar_event.time,
                    strategy_name=self.name,
                    metadata={
                        'exit_reason': 'stop_loss',
                        'entry_zscore': existing_entry_zscore,
                        'exit_zscore': float(current_zscore)
                    }
                )
            
            return None
        
        # Volume confirmation
        if pd.isna(current_volume) or pd.isna(avg_volume):
            return None
        
        volume_spike = current_volume > avg_volume * self.volume_spike_threshold
        
        # Entry logic: SELL VIX (VIX spike, expect mean reversion down)
        if (current_zscore > self.zscore_entry and
            not pd.isna(current_rsi) and current_rsi > self.rsi_overbought and
            volume_spike):
            
            # Calculate SL and TP
            price_mean = bars['close'].tail(self.zscore_period).mean()
            price_std = bars['close'].tail(self.zscore_period).std()
            
            sl_price = current_price + price_std * self.zscore_stop
            risk = sl_price - current_price
            tp_price = price_mean  # Target mean, or 2.0R
            
            tp_from_rr = current_price - risk * self.risk_reward_ratio
            if tp_from_rr > price_mean:
                tp_price = tp_from_rr
            
            self.entry_zscore[symbol] = float(current_zscore)
            self.current_signals[symbol] = 'SELL'
            
            return SignalEvent(
                symbol=symbol,
                signal_type='SELL',
                strength=min(1.0, (current_zscore - self.zscore_entry) / (self.zscore_stop - self.zscore_entry)),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(current_price - tp_price),
                    'zscore': float(current_zscore),
                    'rsi': float(current_rsi),
                    'mean_reversion_strength': float(mean_rev_strength),
                    'volume_ratio': float(current_volume / avg_volume)
                }
            )
        
        # Entry logic: BUY VIX (VIX crash, expect mean reversion up)
        # Note: This is riskier, VIX can stay low for extended periods
        elif (current_zscore < -self.zscore_entry and
              not pd.isna(current_rsi) and current_rsi < self.rsi_oversold and
              volume_spike):
            
            price_mean = bars['close'].tail(self.zscore_period).mean()
            price_std = bars['close'].tail(self.zscore_period).std()
            
            sl_price = max(0, current_price - price_std * self.zscore_stop)
            risk = current_price - sl_price
            tp_price = price_mean
            
            tp_from_rr = current_price + risk * self.risk_reward_ratio
            if tp_from_rr < price_mean:
                tp_price = tp_from_rr
            
            self.entry_zscore[symbol] = float(current_zscore)
            self.current_signals[symbol] = 'BUY'
            
            return SignalEvent(
                symbol=symbol,
                signal_type='BUY',
                strength=min(1.0, abs(current_zscore - (-self.zscore_entry)) / (self.zscore_stop - self.zscore_entry)),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(tp_price - current_price),
                    'zscore': float(current_zscore),
                    'rsi': float(current_rsi),
                    'mean_reversion_strength': float(mean_rev_strength),
                    'volume_ratio': float(current_volume / avg_volume)
                }
            )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'zscore_period': self.zscore_period,
            'zscore_entry': self.zscore_entry,
            'rsi_period': self.rsi_period,
            'risk_reward_ratio': self.risk_reward_ratio,
            'mean_reversion_strength': self.mean_reversion_strength
        }

