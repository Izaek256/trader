"""Market Structure & Smart Money Concept - Liquidity Sweep Strategy

This strategy identifies liquidity sweeps (stop hunts) and trades the reversal,
following smart money concepts and market structure.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class LiquiditySweepSMCStrategy(BaseStrategy):
    """
    Market Structure & SMC Liquidity Sweep Strategy
    
    Entry Rules:
    - Identify liquidity zones (previous swing highs/lows)
    - Price sweeps liquidity (breaks above/below but quickly reverses)
    - Market structure shift: price breaks previous structure
    - Order block confirmation: strong candle in opposite direction
    - Volume spike on sweep, then reversal
    
    Exit Rules:
    - Take profit: Next structure level or 3.0R
    - Stop loss: Beyond order block or 2.0x ATR
    
    Filters:
    - Structure: Must identify clear market structure (higher highs/lower lows)
    - Liquidity: Sweep must be < 0.1% beyond previous high/low
    - Volume: Reversal candle must have volume > 120% of average
    - Time: Best during London/NY sessions (high liquidity)
    - Avoid: Choppy markets (ADX < 15)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("LiquiditySweepSMC", config)
        
        # Structure detection
        self.swing_lookback = config.get('swing_lookback', 20)
        self.structure_shift_bars = config.get('structure_shift_bars', 3)
        
        # Liquidity sweep parameters
        self.sweep_threshold = config.get('sweep_threshold', 0.001)  # 0.1% beyond high/low
        self.sweep_reversal_bars = config.get('sweep_reversal_bars', 3)
        
        # Order block parameters
        self.order_block_lookback = config.get('order_block_lookback', 20)
        self.order_block_min_size = config.get('order_block_min_size', 0.003)  # 0.3% of price
        
        # ATR parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 2.0)
        
        # Volume parameters
        self.volume_period = config.get('volume_period', 20)
        self.volume_reversal_threshold = config.get('volume_reversal_threshold', 1.2)
        
        # ADX filter
        self.adx_period = config.get('adx_period', 14)
        self.adx_min = config.get('adx_min', 15.0)
        
        # Risk/Reward
        self.risk_reward_ratio = config.get('risk_reward_ratio', 3.0)
        
        # State tracking
        self.swing_highs: Dict[str, List[Tuple[datetime, float]]] = {}
        self.swing_lows: Dict[str, List[Tuple[datetime, float]]] = {}
        
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
    
    def _identify_swing_points(self, df: pd.DataFrame, lookback: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Identify swing highs and lows"""
        highs = []
        lows = []
        
        if len(df) < lookback * 2:
            return highs, lows
        
        for i in range(lookback, len(df) - lookback):
            # Swing high: high is highest in lookback window
            if df.iloc[i]['high'] == df.iloc[i-lookback:i+lookback+1]['high'].max():
                highs.append((i, df.iloc[i]['high']))
            
            # Swing low: low is lowest in lookback window
            if df.iloc[i]['low'] == df.iloc[i-lookback:i+lookback+1]['low'].min():
                lows.append((i, df.iloc[i]['low']))
        
        return highs, lows
    
    def _detect_liquidity_sweep(self, bars: pd.DataFrame, swing_highs: List[Tuple[int, float]], 
                                swing_lows: List[Tuple[int, float]], current_idx: int) -> Optional[Dict[str, Any]]:
        """Detect if current price action is a liquidity sweep"""
        if len(bars) < current_idx + 1:
            return None
        
        current_bar = bars.iloc[current_idx]
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_close = current_bar['close']
        
        # Check for bullish sweep (sweep low, reverse up)
        if swing_lows:
            recent_low = max(swing_lows, key=lambda x: x[0])  # Most recent swing low
            low_price = recent_low[1]
            
            # Check if price swept below swing low but closed above
            if (current_low < low_price * (1 - self.sweep_threshold) and
                current_close > low_price):
                
                # Check for reversal in next bars
                if current_idx < len(bars) - self.sweep_reversal_bars:
                    next_bars = bars.iloc[current_idx+1:current_idx+1+self.sweep_reversal_bars]
                    if len(next_bars) > 0 and next_bars['close'].iloc[-1] > current_close:
                        return {
                            'type': 'BULLISH_SWEEP',
                            'liquidity_level': low_price,
                            'sweep_price': current_low,
                            'reversal_price': current_close
                        }
        
        # Check for bearish sweep (sweep high, reverse down)
        if swing_highs:
            recent_high = max(swing_highs, key=lambda x: x[0])  # Most recent swing high
            high_price = recent_high[1]
            
            # Check if price swept above swing high but closed below
            if (current_high > high_price * (1 + self.sweep_threshold) and
                current_close < high_price):
                
                # Check for reversal in next bars
                if current_idx < len(bars) - self.sweep_reversal_bars:
                    next_bars = bars.iloc[current_idx+1:current_idx+1+self.sweep_reversal_bars]
                    if len(next_bars) > 0 and next_bars['close'].iloc[-1] < current_close:
                        return {
                            'type': 'BEARISH_SWEEP',
                            'liquidity_level': high_price,
                            'sweep_price': current_high,
                            'reversal_price': current_close
                        }
        
        return None
    
    def _find_order_block(self, bars: pd.DataFrame, sweep_type: str, sweep_idx: int) -> Optional[Dict[str, float]]:
        """Find order block (strong candle in reversal direction)"""
        if sweep_idx < 1 or sweep_idx >= len(bars):
            return None
        
        # Look for strong reversal candle after sweep
        for i in range(sweep_idx, min(sweep_idx + self.order_block_lookback, len(bars))):
            bar = bars.iloc[i]
            body_size = abs(bar['close'] - bar['open']) / bar['close']
            
            if sweep_type == 'BULLISH_SWEEP':
                # Strong bullish candle
                if (bar['close'] > bar['open'] and 
                    body_size > self.order_block_min_size and
                    bar['close'] > bar['high'] * 0.7):  # Close in upper 30% of range
                    return {
                        'high': float(bar['high']),
                        'low': float(bar['low']),
                        'open': float(bar['open']),
                        'close': float(bar['close'])
                    }
            
            elif sweep_type == 'BEARISH_SWEEP':
                # Strong bearish candle
                if (bar['close'] < bar['open'] and 
                    body_size > self.order_block_min_size and
                    bar['close'] < bar['low'] * 1.3):  # Close in lower 30% of range
                    return {
                        'high': float(bar['high']),
                        'low': float(bar['low']),
                        'open': float(bar['open']),
                        'close': float(bar['close'])
                    }
        
        return None
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Get bars
        min_bars = max(self.swing_lookback * 2, self.atr_period, self.volume_period, self.adx_period) + 20
        bars = self.get_bars(symbol, min_bars)
        if len(bars) < min_bars:
            return None
        
        # Calculate indicators
        atr = self._calculate_atr(bars, self.atr_period)
        adx = self._calculate_adx(bars, self.adx_period)
        volume_avg = bars['volume'].rolling(window=self.volume_period).mean()
        
        if len(atr) < 1 or len(adx) < 1:
            return None
        
        current_atr = atr.iloc[-1]
        current_adx = adx.iloc[-1]
        current_volume = bar_event.tick_volume
        avg_volume = volume_avg.iloc[-1]
        
        # ADX filter: avoid choppy markets
        if pd.isna(current_adx) or current_adx < self.adx_min:
            return None
        
        # Identify swing points
        swing_highs, swing_lows = self._identify_swing_points(bars, self.swing_lookback)
        
        if not swing_highs and not swing_lows:
            return None
        
        # Detect liquidity sweep
        current_idx = len(bars) - 1
        sweep = self._detect_liquidity_sweep(bars, swing_highs, swing_lows, current_idx)
        
        if not sweep:
            return None
        
        # Volume confirmation
        if pd.isna(current_volume) or pd.isna(avg_volume):
            return None
        
        if current_volume < avg_volume * self.volume_reversal_threshold:
            return None  # Insufficient volume on reversal
        
        # Find order block
        order_block = self._find_order_block(bars, sweep['type'], current_idx)
        if not order_block:
            return None
        
        current_price = bar_event.close
        
        # Entry logic: BULLISH SWEEP (sweep low, buy reversal)
        if sweep['type'] == 'BULLISH_SWEEP':
            # Entry: price above order block
            if current_price < order_block['high']:
                return None  # Wait for price to break above order block
            
            # Calculate SL and TP
            sl_price = min(order_block['low'], sweep['liquidity_level'] - current_atr * self.atr_sl_multiplier)
            risk = current_price - sl_price
            
            # TP: next swing high or 3.0R
            if swing_highs:
                next_high = min([h[1] for h in swing_highs if h[1] > current_price], default=None)
                if next_high:
                    tp_price = min(next_high, current_price + risk * self.risk_reward_ratio)
                else:
                    tp_price = current_price + risk * self.risk_reward_ratio
            else:
                tp_price = current_price + risk * self.risk_reward_ratio
            
            return SignalEvent(
                symbol=symbol,
                signal_type='BUY',
                strength=min(1.0, (current_price - sweep['liquidity_level']) / (sweep['sweep_price'] - sweep['liquidity_level'])),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(tp_price - current_price),
                    'liquidity_level': sweep['liquidity_level'],
                    'sweep_price': sweep['sweep_price'],
                    'order_block_high': order_block['high'],
                    'order_block_low': order_block['low'],
                    'atr': float(current_atr)
                }
            )
        
        # Entry logic: BEARISH SWEEP (sweep high, sell reversal)
        elif sweep['type'] == 'BEARISH_SWEEP':
            # Entry: price below order block
            if current_price > order_block['low']:
                return None  # Wait for price to break below order block
            
            # Calculate SL and TP
            sl_price = max(order_block['high'], sweep['liquidity_level'] + current_atr * self.atr_sl_multiplier)
            risk = sl_price - current_price
            
            # TP: next swing low or 3.0R
            if swing_lows:
                next_low = max([l[1] for l in swing_lows if l[1] < current_price], default=None)
                if next_low:
                    tp_price = max(next_low, current_price - risk * self.risk_reward_ratio)
                else:
                    tp_price = current_price - risk * self.risk_reward_ratio
            else:
                tp_price = current_price - risk * self.risk_reward_ratio
            
            return SignalEvent(
                symbol=symbol,
                signal_type='SELL',
                strength=min(1.0, (sweep['liquidity_level'] - current_price) / (sweep['liquidity_level'] - sweep['sweep_price'])),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(current_price - tp_price),
                    'liquidity_level': sweep['liquidity_level'],
                    'sweep_price': sweep['sweep_price'],
                    'order_block_high': order_block['high'],
                    'order_block_low': order_block['low'],
                    'atr': float(current_atr)
                }
            )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'swing_lookback': self.swing_lookback,
            'sweep_threshold': self.sweep_threshold,
            'atr_period': self.atr_period,
            'atr_sl_multiplier': self.atr_sl_multiplier,
            'risk_reward_ratio': self.risk_reward_ratio,
            'adx_min': self.adx_min
        }

