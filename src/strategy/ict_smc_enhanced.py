"""Enhanced ICT/SMC Strategy

This strategy combines multiple ICT (Inner Circle Trader) and SMC (Smart Money Concepts) concepts:
- Liquidity Sweeps
- Order Blocks
- Fair Value Gaps (FVG)
- Market Structure (BOS/CHoCH)
- Breaker Blocks
- Mitigation Zones
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class ICTSMCEnhancedStrategy(BaseStrategy):
    """
    Enhanced ICT/SMC Strategy
    
    Entry Rules:
    - Market Structure: Identify BOS (Break of Structure) or CHoCH (Change of Character)
    - Liquidity Sweep: Price sweeps previous high/low then reverses
    - Order Block: Strong candle in reversal direction
    - Fair Value Gap: Three-candle pattern creating imbalance
    - Breaker Block: Previous order block that gets broken (becomes support/resistance)
    - Mitigation: Price returns to fill FVG or test breaker block
    
    Exit Rules:
    - Take profit: Next structure level, opposite order block, or 3.0R
    - Stop loss: Beyond order block or 2.0x ATR
    
    Filters:
    - Structure: Clear market structure required
    - Time: London/NY sessions preferred
    - Volume: Confirmation on reversal
    - ADX: Avoid choppy markets (< 15)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ICTSMCEnhanced", config)
        
        # Structure detection
        self.swing_lookback = config.get('swing_lookback', 20)
        self.structure_shift_bars = config.get('structure_shift_bars', 3)
        
        # Liquidity sweep parameters
        self.sweep_threshold = config.get('sweep_threshold', 0.001)  # 0.1% beyond high/low
        self.sweep_reversal_bars = config.get('sweep_reversal_bars', 3)
        
        # Order block parameters
        self.order_block_lookback = config.get('order_block_lookback', 20)
        self.order_block_min_size = config.get('order_block_min_size', 0.003)  # 0.3% of price
        
        # Fair Value Gap (FVG) parameters
        self.fvg_lookback = config.get('fvg_lookback', 50)
        self.fvg_min_size = config.get('fvg_min_size', 0.001)  # 0.1% minimum gap
        
        # Breaker block parameters
        self.breaker_lookback = config.get('breaker_lookback', 100)
        
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
        self.swing_highs: Dict[str, List[Tuple[int, float]]] = {}
        self.swing_lows: Dict[str, List[Tuple[int, float]]] = {}
        self.order_blocks: Dict[str, List[Dict[str, Any]]] = {}
        self.fair_value_gaps: Dict[str, List[Dict[str, Any]]] = {}
        self.breaker_blocks: Dict[str, List[Dict[str, Any]]] = {}
        self.market_structure: Dict[str, str] = {}  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    
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
            if df.iloc[i]['high'] == df.iloc[i-lookback:i+lookback+1]['high'].max():
                highs.append((i, df.iloc[i]['high']))
            if df.iloc[i]['low'] == df.iloc[i-lookback:i+lookback+1]['low'].min():
                lows.append((i, df.iloc[i]['low']))
        
        return highs, lows
    
    def _detect_fair_value_gap(self, bars: pd.DataFrame, idx: int) -> Optional[Dict[str, Any]]:
        """Detect Fair Value Gap (FVG) - three candle pattern with gap"""
        if idx < 2 or idx >= len(bars) - 1:
            return None
        
        candle1 = bars.iloc[idx - 1]
        candle2 = bars.iloc[idx]
        candle3 = bars.iloc[idx + 1]
        
        # Bullish FVG: candle1 high < candle3 low
        if candle1['high'] < candle3['low']:
            gap_high = candle3['low']
            gap_low = candle1['high']
            gap_size = (gap_high - gap_low) / candle2['close']
            
            if gap_size >= self.fvg_min_size:
                return {
                    'type': 'BULLISH_FVG',
                    'high': float(gap_high),
                    'low': float(gap_low),
                    'size': float(gap_size),
                    'idx': idx,
                    'filled': False
                }
        
        # Bearish FVG: candle1 low > candle3 high
        elif candle1['low'] > candle3['high']:
            gap_high = candle1['low']
            gap_low = candle3['high']
            gap_size = (gap_high - gap_low) / candle2['close']
            
            if gap_size >= self.fvg_min_size:
                return {
                    'type': 'BEARISH_FVG',
                    'high': float(gap_high),
                    'low': float(gap_low),
                    'size': float(gap_size),
                    'idx': idx,
                    'filled': False
                }
        
        return None
    
    def _detect_market_structure(self, bars: pd.DataFrame, swing_highs: List[Tuple[int, float]], 
                                 swing_lows: List[Tuple[int, float]]) -> str:
        """Detect market structure (BOS/CHoCH)"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'NEUTRAL'
        
        # Get recent swings
        recent_highs = sorted(swing_highs, key=lambda x: x[0], reverse=True)[:2]
        recent_lows = sorted(swing_lows, key=lambda x: x[0], reverse=True)[:2]
        
        # BOS (Break of Structure): Higher high in uptrend or lower low in downtrend
        # CHoCH (Change of Character): Structure shift
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            hh = recent_highs[0][1] > recent_highs[1][1]  # Higher high
            ll = recent_lows[0][1] > recent_lows[1][1]  # Higher low
            
            if hh and ll:
                return 'BULLISH'
            elif not hh and not ll:
                return 'BEARISH'
        
        return 'NEUTRAL'
    
    def _find_order_block(self, bars: pd.DataFrame, sweep_type: str, sweep_idx: int) -> Optional[Dict[str, float]]:
        """Find order block (strong candle in reversal direction)"""
        if sweep_idx < 1 or sweep_idx >= len(bars):
            return None
        
        for i in range(sweep_idx, min(sweep_idx + self.order_block_lookback, len(bars))):
            bar = bars.iloc[i]
            body_size = abs(bar['close'] - bar['open']) / bar['close']
            
            if sweep_type == 'BULLISH_SWEEP':
                if (bar['close'] > bar['open'] and 
                    body_size > self.order_block_min_size and
                    bar['close'] > bar['high'] * 0.7):
                    return {
                        'high': float(bar['high']),
                        'low': float(bar['low']),
                        'open': float(bar['open']),
                        'close': float(bar['close']),
                        'type': 'BULLISH',
                        'idx': i
                    }
            
            elif sweep_type == 'BEARISH_SWEEP':
                if (bar['close'] < bar['open'] and 
                    body_size > self.order_block_min_size and
                    bar['close'] < bar['low'] * 1.3):
                    return {
                        'high': float(bar['high']),
                        'low': float(bar['low']),
                        'open': float(bar['open']),
                        'close': float(bar['close']),
                        'type': 'BEARISH',
                        'idx': i
                    }
        
        return None
    
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
            recent_low = max(swing_lows, key=lambda x: x[0])
            low_price = recent_low[1]
            
            if (current_low < low_price * (1 - self.sweep_threshold) and
                current_close > low_price):
                
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
            recent_high = max(swing_highs, key=lambda x: x[0])
            high_price = recent_high[1]
            
            if (current_high > high_price * (1 + self.sweep_threshold) and
                current_close < high_price):
                
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
    
    def _check_breaker_block(self, bars: pd.DataFrame, current_price: float, order_blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check if price is at a breaker block (broken order block)"""
        for ob in order_blocks:
            # Breaker: order block was broken, now acts as support/resistance
            if ob['type'] == 'BULLISH' and ob.get('broken', False):
                if ob['low'] <= current_price <= ob['high']:
                    return ob
            elif ob['type'] == 'BEARISH' and ob.get('broken', False):
                if ob['low'] <= current_price <= ob['high']:
                    return ob
        return None
    
    def _check_fvg_mitigation(self, bars: pd.DataFrame, current_price: float, fvgs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check if price is mitigating (filling) a Fair Value Gap"""
        for fvg in fvgs:
            if not fvg.get('filled', False):
                if fvg['low'] <= current_price <= fvg['high']:
                    return fvg
        return None
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Get bars
        min_bars = max(self.swing_lookback * 2, self.atr_period, self.volume_period, self.adx_period, self.fvg_lookback) + 20
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
        
        # ADX filter
        if pd.isna(current_adx) or current_adx < self.adx_min:
            return None
        
        # Identify swing points
        swing_highs, swing_lows = self._identify_swing_points(bars, self.swing_lookback)
        
        if not swing_highs and not swing_lows:
            return None
        
        # Detect market structure
        structure = self._detect_market_structure(bars, swing_highs, swing_lows)
        self.market_structure[symbol] = structure
        
        # Detect Fair Value Gaps
        if symbol not in self.fair_value_gaps:
            self.fair_value_gaps[symbol] = []
        
        # Scan for FVGs in recent bars
        for i in range(max(0, len(bars) - self.fvg_lookback), len(bars) - 2):
            fvg = self._detect_fair_value_gap(bars, i)
            if fvg:
                # Check if FVG already exists
                exists = any(abs(f['high'] - fvg['high']) < 0.0001 for f in self.fair_value_gaps[symbol])
                if not exists:
                    self.fair_value_gaps[symbol].append(fvg)
        
        # Detect liquidity sweep
        current_idx = len(bars) - 1
        sweep = self._detect_liquidity_sweep(bars, swing_highs, swing_lows, current_idx)
        
        # Check for FVG mitigation
        current_price = bar_event.close
        fvg_mitigation = self._check_fvg_mitigation(bars, current_price, self.fair_value_gaps.get(symbol, []))
        
        # Check for breaker block
        breaker = self._check_breaker_block(bars, current_price, self.order_blocks.get(symbol, []))
        
        # Entry logic: BULLISH (sweep low, buy reversal)
        if sweep and sweep['type'] == 'BULLISH_SWEEP':
            # Volume confirmation
            if pd.isna(current_volume) or pd.isna(avg_volume) or current_volume < avg_volume * self.volume_reversal_threshold:
                return None
            
            # Find order block
            order_block = self._find_order_block(bars, sweep['type'], current_idx)
            if not order_block:
                return None
            
            # Entry: price above order block
            if current_price < order_block['high']:
                return None
            
            # Additional confluence: FVG mitigation or breaker block
            confluence_score = 1.0
            if fvg_mitigation and fvg_mitigation['type'] == 'BULLISH_FVG':
                confluence_score += 0.5
            if breaker and breaker['type'] == 'BULLISH':
                confluence_score += 0.5
            if structure == 'BULLISH':
                confluence_score += 0.3
            
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
                strength=min(1.0, confluence_score / 2.0),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(tp_price - current_price),
                    'liquidity_level': sweep['liquidity_level'],
                    'order_block_high': order_block['high'],
                    'order_block_low': order_block['low'],
                    'market_structure': structure,
                    'fvg_mitigation': fvg_mitigation is not None,
                    'breaker_block': breaker is not None,
                    'confluence_score': float(confluence_score),
                    'atr': float(current_atr)
                }
            )
        
        # Entry logic: BEARISH (sweep high, sell reversal)
        elif sweep and sweep['type'] == 'BEARISH_SWEEP':
            # Volume confirmation
            if pd.isna(current_volume) or pd.isna(avg_volume) or current_volume < avg_volume * self.volume_reversal_threshold:
                return None
            
            # Find order block
            order_block = self._find_order_block(bars, sweep['type'], current_idx)
            if not order_block:
                return None
            
            # Entry: price below order block
            if current_price > order_block['low']:
                return None
            
            # Additional confluence
            confluence_score = 1.0
            if fvg_mitigation and fvg_mitigation['type'] == 'BEARISH_FVG':
                confluence_score += 0.5
            if breaker and breaker['type'] == 'BEARISH':
                confluence_score += 0.5
            if structure == 'BEARISH':
                confluence_score += 0.3
            
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
                strength=min(1.0, confluence_score / 2.0),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(current_price - tp_price),
                    'liquidity_level': sweep['liquidity_level'],
                    'order_block_high': order_block['high'],
                    'order_block_low': order_block['low'],
                    'market_structure': structure,
                    'fvg_mitigation': fvg_mitigation is not None,
                    'breaker_block': breaker is not None,
                    'confluence_score': float(confluence_score),
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
            'adx_min': self.adx_min,
            'fvg_min_size': self.fvg_min_size
        }

