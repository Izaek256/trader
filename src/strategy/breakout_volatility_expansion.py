"""Breakout Volatility Expansion Strategy

This strategy trades breakouts when volatility expands after compression,
entering on confirmed breakouts with volume.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class BreakoutVolatilityExpansionStrategy(BaseStrategy):
    """
    Breakout Volatility Expansion Strategy
    
    Entry Rules:
    - Price breaks above/below recent consolidation range (last N bars)
    - Volatility expansion: ATR increases > 20% from 20-period average
    - Volume confirmation: Breakout volume > 150% of average
    - Bollinger Bands expanding (width > 1.2x average)
    - Price closes outside previous consolidation range
    
    Exit Rules:
    - Take profit: 2.0R or when volatility contracts (ATR < 80% of peak)
    - Stop loss: Below/above consolidation range or 1.5x ATR
    
    Filters:
    - Volatility: Must have compression before expansion (BB width < 0.7x avg in last 10 bars)
    - Volume: Breakout must have volume > 150% of average
    - Time: Avoid low liquidity hours (best during London/NY overlap)
    - Market Regime: Works in both trending and ranging markets
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BreakoutVolatilityExpansion", config)
        
        # Consolidation detection
        self.consolidation_period = config.get('consolidation_period', 20)
        self.consolidation_threshold = config.get('consolidation_threshold', 0.005)  # 0.5% range
        
        # Volatility expansion parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_expansion_threshold = config.get('atr_expansion_threshold', 1.2)
        self.atr_compression_threshold = config.get('atr_compression_threshold', 0.7)
        self.atr_compression_lookback = config.get('atr_compression_lookback', 10)
        
        # Bollinger Bands parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.bb_expansion_threshold = config.get('bb_expansion_threshold', 1.2)
        self.bb_compression_threshold = config.get('bb_compression_threshold', 0.7)
        
        # Volume parameters
        self.volume_period = config.get('volume_period', 20)
        self.volume_breakout_threshold = config.get('volume_breakout_threshold', 1.5)
        
        # Risk/Reward
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.0)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 1.5)
        
        # Breakout confirmation
        self.breakout_confirmation_bars = config.get('breakout_confirmation_bars', 2)
        
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
    
    def _calculate_bb(self, series: pd.Series, period: int, std: float) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        width = (upper - lower) / sma
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'width': width
        }
    
    def _detect_consolidation(self, df: pd.DataFrame, period: int, threshold: float) -> Optional[Dict[str, float]]:
        """Detect consolidation range"""
        if len(df) < period:
            return None
        
        recent = df.tail(period)
        high_max = recent['high'].max()
        low_min = recent['low'].min()
        range_size = (high_max - low_min) / recent['close'].mean()
        
        if range_size < threshold:
            return {
                'high': float(high_max),
                'low': float(low_min),
                'range_size': float(range_size),
                'mid': float((high_max + low_min) / 2)
            }
        
        return None
    
    def _has_volatility_compression(self, atr: pd.Series, bb_width: pd.Series, lookback: int) -> bool:
        """Check if there was volatility compression recently"""
        if len(atr) < lookback or len(bb_width) < lookback:
            return False
        
        recent_atr = atr.tail(lookback)
        recent_bb = bb_width.tail(lookback)
        avg_atr = atr.tail(self.atr_period).mean()
        avg_bb = bb_width.tail(self.bb_period).mean()
        
        if pd.isna(avg_atr) or pd.isna(avg_bb):
            return False
        
        # Check if recent ATR/BB were compressed
        compressed_atr = (recent_atr < avg_atr * self.atr_compression_threshold).any()
        compressed_bb = (recent_bb < avg_bb * self.bb_compression_threshold).any()
        
        return compressed_atr or compressed_bb
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Get bars
        min_bars = max(self.consolidation_period, self.atr_period, self.bb_period, self.volume_period) + 10
        bars = self.get_bars(symbol, min_bars)
        if len(bars) < min_bars:
            return None
        
        # Calculate indicators
        atr = self._calculate_atr(bars, self.atr_period)
        bb = self._calculate_bb(bars['close'], self.bb_period, self.bb_std)
        volume_avg = bars['volume'].rolling(window=self.volume_period).mean()
        
        if len(atr) < 2 or len(bb['width']) < 2:
            return None
        
        current_price = bar_event.close
        current_high = bar_event.high
        current_low = bar_event.low
        current_volume = bar_event.tick_volume
        
        current_atr = atr.iloc[-1]
        avg_atr = atr.tail(self.atr_period).mean()
        current_bb_width = bb['width'].iloc[-1]
        avg_bb_width = bb['width'].tail(self.bb_period).mean()
        avg_volume = volume_avg.iloc[-1]
        
        # Check for volatility compression in recent past
        if not self._has_volatility_compression(atr, bb['width'], self.atr_compression_lookback):
            return None  # No compression, no valid breakout setup
        
        # Check for volatility expansion
        if pd.isna(current_atr) or pd.isna(avg_atr) or pd.isna(current_bb_width) or pd.isna(avg_bb_width):
            return None
        
        atr_expanding = current_atr > avg_atr * self.atr_expansion_threshold
        bb_expanding = current_bb_width > avg_bb_width * self.bb_expansion_threshold
        
        if not (atr_expanding or bb_expanding):
            return None  # Volatility not expanding
        
        # Volume confirmation
        if pd.isna(current_volume) or pd.isna(avg_volume):
            return None
        
        if current_volume < avg_volume * self.volume_breakout_threshold:
            return None  # Insufficient volume
        
        # Detect consolidation
        consolidation = self._detect_consolidation(bars, self.consolidation_period, self.consolidation_threshold)
        if not consolidation:
            return None  # No clear consolidation
        
        # Check for breakout: price closes above/below consolidation
        breakout_up = (current_price > consolidation['high'] and 
                      bar_event.close > bar_event.open)  # Bullish candle
        
        breakout_down = (current_price < consolidation['low'] and 
                        bar_event.close < bar_event.open)  # Bearish candle
        
        # Additional confirmation: check last N bars closed outside
        if breakout_up:
            recent_closes = bars['close'].tail(self.breakout_confirmation_bars)
            if not (recent_closes > consolidation['high']).all():
                breakout_up = False
        
        if breakout_down:
            recent_closes = bars['close'].tail(self.breakout_confirmation_bars)
            if not (recent_closes < consolidation['low']).all():
                breakout_down = False
        
        # Entry logic: BULLISH BREAKOUT
        if breakout_up:
            # Calculate SL and TP
            sl_price = max(consolidation['low'], current_price - current_atr * self.atr_sl_multiplier)
            risk = current_price - sl_price
            tp_price = current_price + risk * self.risk_reward_ratio
            
            return SignalEvent(
                symbol=symbol,
                signal_type='BUY',
                strength=min(1.0, current_volume / (avg_volume * self.volume_breakout_threshold)),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(risk * self.risk_reward_ratio),
                    'consolidation_high': consolidation['high'],
                    'consolidation_low': consolidation['low'],
                    'atr_ratio': float(current_atr / avg_atr),
                    'bb_width_ratio': float(current_bb_width / avg_bb_width),
                    'volume_ratio': float(current_volume / avg_volume)
                }
            )
        
        # Entry logic: BEARISH BREAKOUT
        elif breakout_down:
            # Calculate SL and TP
            sl_price = min(consolidation['high'], current_price + current_atr * self.atr_sl_multiplier)
            risk = sl_price - current_price
            tp_price = current_price - risk * self.risk_reward_ratio
            
            return SignalEvent(
                symbol=symbol,
                signal_type='SELL',
                strength=min(1.0, current_volume / (avg_volume * self.volume_breakout_threshold)),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(risk * self.risk_reward_ratio),
                    'consolidation_high': consolidation['high'],
                    'consolidation_low': consolidation['low'],
                    'atr_ratio': float(current_atr / avg_atr),
                    'bb_width_ratio': float(current_bb_width / avg_bb_width),
                    'volume_ratio': float(current_volume / avg_volume)
                }
            )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'consolidation_period': self.consolidation_period,
            'consolidation_threshold': self.consolidation_threshold,
            'atr_period': self.atr_period,
            'atr_expansion_threshold': self.atr_expansion_threshold,
            'volume_breakout_threshold': self.volume_breakout_threshold,
            'risk_reward_ratio': self.risk_reward_ratio,
            'atr_sl_multiplier': self.atr_sl_multiplier
        }

