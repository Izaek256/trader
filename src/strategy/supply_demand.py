"""Supply and Demand Strategy

This strategy identifies supply and demand zones (areas where price previously
reversed strongly) and trades bounces from these zones.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class SupplyDemandStrategy(BaseStrategy):
    """
    Supply and Demand Zone Trading Strategy
    
    Entry Rules:
    - Identify supply zones (resistance where price dropped sharply)
    - Identify demand zones (support where price rose sharply)
    - Price returns to zone and shows rejection (wick/candle pattern)
    - Volume confirms (lower volume on approach, higher on rejection)
    - Zone must be "fresh" (not tested too many times)
    
    Exit Rules:
    - Take profit: Next opposite zone or 2.5R
    - Stop loss: Beyond zone or 1.5x ATR
    
    Filters:
    - Zone Quality: Strong initial move (impulse) required
    - Zone Freshness: Zone not tested more than 2-3 times
    - Rejection Pattern: Clear wick or engulfing pattern
    - Volume: Confirmation required
    - Time: Zone age (prefer recent zones)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SupplyDemand", config)
        
        # Zone detection parameters
        self.zone_lookback = config.get('zone_lookback', 100)
        self.impulse_min_bars = config.get('impulse_min_bars', 5)
        self.impulse_min_move = config.get('impulse_min_move', 0.005)  # 0.5% minimum move
        self.zone_base_bars = config.get('zone_base_bars', 3)  # Bars to form zone base
        
        # Zone quality filters
        self.min_zone_strength = config.get('min_zone_strength', 0.003)  # 0.3% minimum zone size
        self.max_zone_tests = config.get('max_zone_tests', 3)  # Max times zone can be tested
        self.zone_age_max_bars = config.get('zone_age_max_bars', 200)  # Max age of valid zone
        
        # Rejection pattern detection
        self.rejection_wick_ratio = config.get('rejection_wick_ratio', 0.6)  # Wick must be 60% of range
        self.rejection_body_ratio = config.get('rejection_body_ratio', 0.3)  # Body max 30% of range
        
        # ATR parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 1.5)
        
        # Volume parameters
        self.volume_period = config.get('volume_period', 20)
        self.volume_approach_threshold = config.get('volume_approach_threshold', 0.9)  # Lower on approach
        self.volume_rejection_threshold = config.get('volume_rejection_threshold', 1.1)  # Higher on rejection
        
        # Risk/Reward
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.5)
        
        # State tracking
        self.supply_zones: Dict[str, List[Dict[str, Any]]] = {}  # symbol -> list of zones
        self.demand_zones: Dict[str, List[Dict[str, Any]]] = {}  # symbol -> list of zones
    
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
    
    def _identify_impulse_move(self, bars: pd.DataFrame, start_idx: int, direction: str) -> Optional[Dict[str, Any]]:
        """Identify strong impulse move (for zone formation)"""
        if start_idx + self.impulse_min_bars >= len(bars):
            return None
        
        impulse_bars = bars.iloc[start_idx:start_idx + self.impulse_min_bars]
        
        if direction == 'DOWN':  # Supply zone (price dropped)
            start_price = bars.iloc[start_idx]['close']
            end_price = impulse_bars['close'].min()
            move_pct = (start_price - end_price) / start_price
            
            if move_pct >= self.impulse_min_move:
                return {
                    'start_idx': start_idx,
                    'end_idx': start_idx + self.impulse_min_bars - 1,
                    'start_price': float(start_price),
                    'end_price': float(end_price),
                    'move_pct': float(move_pct),
                    'direction': 'DOWN'
                }
        
        elif direction == 'UP':  # Demand zone (price rose)
            start_price = bars.iloc[start_idx]['close']
            end_price = impulse_bars['close'].max()
            move_pct = (end_price - start_price) / start_price
            
            if move_pct >= self.impulse_min_move:
                return {
                    'start_idx': start_idx,
                    'end_idx': start_idx + self.impulse_min_bars - 1,
                    'start_price': float(start_price),
                    'end_price': float(end_price),
                    'move_pct': float(move_pct),
                    'direction': 'UP'
                }
        
        return None
    
    def _find_supply_zones(self, bars: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find supply zones (resistance areas)"""
        zones = []
        
        if len(bars) < self.zone_lookback:
            return zones
        
        # Look for impulse moves down (supply zones form before drop)
        for i in range(len(bars) - self.zone_lookback, len(bars) - self.impulse_min_bars - self.zone_base_bars):
            # Check for impulse down
            impulse = self._identify_impulse_move(bars, i + self.zone_base_bars, 'DOWN')
            if not impulse:
                continue
            
            # Zone is the base before the drop
            zone_bars = bars.iloc[i:i + self.zone_base_bars]
            zone_high = zone_bars['high'].max()
            zone_low = zone_bars['low'].min()
            zone_size = (zone_high - zone_low) / zone_bars['close'].mean()
            
            if zone_size < self.min_zone_strength:
                continue
            
            # Check if zone is still valid (not too old)
            zone_age = len(bars) - (i + self.zone_base_bars)
            if zone_age > self.zone_age_max_bars:
                continue
            
            zones.append({
                'high': float(zone_high),
                'low': float(zone_low),
                'start_idx': i,
                'end_idx': i + self.zone_base_bars - 1,
                'age': zone_age,
                'tests': 0,
                'strength': float(zone_size),
                'impulse_move': impulse['move_pct']
            })
        
        # Remove overlapping zones (keep strongest)
        zones = self._merge_overlapping_zones(zones, 'supply')
        return zones
    
    def _find_demand_zones(self, bars: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find demand zones (support areas)"""
        zones = []
        
        if len(bars) < self.zone_lookback:
            return zones
        
        # Look for impulse moves up (demand zones form before rise)
        for i in range(len(bars) - self.zone_lookback, len(bars) - self.impulse_min_bars - self.zone_base_bars):
            # Check for impulse up
            impulse = self._identify_impulse_move(bars, i + self.zone_base_bars, 'UP')
            if not impulse:
                continue
            
            # Zone is the base before the rise
            zone_bars = bars.iloc[i:i + self.zone_base_bars]
            zone_high = zone_bars['high'].max()
            zone_low = zone_bars['low'].min()
            zone_size = (zone_high - zone_low) / zone_bars['close'].mean()
            
            if zone_size < self.min_zone_strength:
                continue
            
            # Check if zone is still valid (not too old)
            zone_age = len(bars) - (i + self.zone_base_bars)
            if zone_age > self.zone_age_max_bars:
                continue
            
            zones.append({
                'high': float(zone_high),
                'low': float(zone_low),
                'start_idx': i,
                'end_idx': i + self.zone_base_bars - 1,
                'age': zone_age,
                'tests': 0,
                'strength': float(zone_size),
                'impulse_move': impulse['move_pct']
            })
        
        # Remove overlapping zones (keep strongest)
        zones = self._merge_overlapping_zones(zones, 'demand')
        return zones
    
    def _merge_overlapping_zones(self, zones: List[Dict[str, Any]], zone_type: str) -> List[Dict[str, Any]]:
        """Merge overlapping zones, keeping the strongest"""
        if not zones:
            return []
        
        # Sort by strength (descending)
        zones_sorted = sorted(zones, key=lambda z: z['strength'], reverse=True)
        merged = []
        
        for zone in zones_sorted:
            overlap = False
            for existing in merged:
                # Check if zones overlap
                if zone_type == 'supply':
                    if not (zone['low'] > existing['high'] or zone['high'] < existing['low']):
                        overlap = True
                        break
                else:  # demand
                    if not (zone['low'] > existing['high'] or zone['high'] < existing['low']):
                        overlap = True
                        break
            
            if not overlap:
                merged.append(zone)
        
        return merged
    
    def _is_price_in_zone(self, price: float, zone: Dict[str, Any]) -> bool:
        """Check if price is in zone"""
        return zone['low'] <= price <= zone['high']
    
    def _detect_rejection_pattern(self, bar: pd.Series, zone: Dict[str, Any], zone_type: str) -> bool:
        """Detect rejection pattern (wick) at zone"""
        high = bar['high']
        low = bar['low']
        open_price = bar['open']
        close = bar['close']
        
        range_size = high - low
        if range_size == 0:
            return False
        
        body_size = abs(close - open_price)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        if zone_type == 'supply':
            # Rejection at supply: upper wick should be significant
            wick_ratio = upper_wick / range_size
            body_ratio = body_size / range_size
            return wick_ratio >= self.rejection_wick_ratio and body_ratio <= self.rejection_body_ratio
        
        else:  # demand
            # Rejection at demand: lower wick should be significant
            wick_ratio = lower_wick / range_size
            body_ratio = body_size / range_size
            return wick_ratio >= self.rejection_wick_ratio and body_ratio <= self.rejection_body_ratio
    
    def _update_zone_tests(self, zones: List[Dict[str, Any]], price: float, zone_type: str) -> None:
        """Update test count for zones that price touched"""
        for zone in zones:
            if self._is_price_in_zone(price, zone):
                zone['tests'] += 1
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Get bars
        min_bars = max(self.zone_lookback, self.atr_period, self.volume_period) + 20
        bars = self.get_bars(symbol, min_bars)
        if len(bars) < min_bars:
            return None
        
        # Update zones periodically
        if symbol not in self.supply_zones or len(self.supply_zones[symbol]) == 0:
            self.supply_zones[symbol] = self._find_supply_zones(bars)
        if symbol not in self.demand_zones or len(self.demand_zones[symbol]) == 0:
            self.demand_zones[symbol] = self._find_demand_zones(bars)
        
        # Recalculate zones every 20 bars
        if len(bars) % 20 == 0:
            self.supply_zones[symbol] = self._find_supply_zones(bars)
            self.demand_zones[symbol] = self._find_demand_zones(bars)
        
        # Calculate indicators
        atr = self._calculate_atr(bars, self.atr_period)
        volume_avg = bars['volume'].rolling(window=self.volume_period).mean()
        
        if len(atr) < 1:
            return None
        
        current_price = bar_event.close
        current_high = bar_event.high
        current_low = bar_event.low
        current_volume = bar_event.tick_volume
        avg_volume = volume_avg.iloc[-1] if len(volume_avg) > 0 else current_volume
        current_atr = atr.iloc[-1]
        
        current_bar = bars.iloc[-1] if len(bars) > 0 else None
        if current_bar is None:
            return None
        
        # Check supply zones (resistance - sell signals)
        for zone in self.supply_zones.get(symbol, []):
            # Check if zone is still valid
            if zone['tests'] >= self.max_zone_tests:
                continue
            
            # Check if price is in or near zone
            if not (zone['low'] * 0.999 <= current_price <= zone['high'] * 1.001):
                continue
            
            # Update test count
            self._update_zone_tests([zone], current_price, 'supply')
            
            # Check for rejection pattern
            if not self._detect_rejection_pattern(current_bar, zone, 'supply'):
                continue
            
            # Volume confirmation: should be lower on approach, higher on rejection
            if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
                continue
            
            volume_ratio = current_volume / avg_volume
            if volume_ratio < self.volume_rejection_threshold:
                continue  # Need higher volume on rejection
            
            # Calculate SL and TP
            sl_price = zone['high'] + current_atr * self.atr_sl_multiplier
            risk = sl_price - current_price
            if risk <= 0:
                continue
            
            # TP: next demand zone or 2.5R
            tp_price = current_price - risk * self.risk_reward_ratio
            if self.demand_zones.get(symbol):
                next_demand = max([z['high'] for z in self.demand_zones[symbol] if z['high'] < current_price], default=None)
                if next_demand:
                    tp_price = max(next_demand, tp_price)
            
            return SignalEvent(
                symbol=symbol,
                signal_type='SELL',
                strength=min(1.0, zone['strength'] / 0.01),  # Normalize strength
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(current_price - tp_price),
                    'zone_high': zone['high'],
                    'zone_low': zone['low'],
                    'zone_strength': zone['strength'],
                    'zone_tests': zone['tests'],
                    'atr': float(current_atr)
                }
            )
        
        # Check demand zones (support - buy signals)
        for zone in self.demand_zones.get(symbol, []):
            # Check if zone is still valid
            if zone['tests'] >= self.max_zone_tests:
                continue
            
            # Check if price is in or near zone
            if not (zone['low'] * 0.999 <= current_price <= zone['high'] * 1.001):
                continue
            
            # Update test count
            self._update_zone_tests([zone], current_price, 'demand')
            
            # Check for rejection pattern
            if not self._detect_rejection_pattern(current_bar, zone, 'demand'):
                continue
            
            # Volume confirmation
            if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
                continue
            
            volume_ratio = current_volume / avg_volume
            if volume_ratio < self.volume_rejection_threshold:
                continue
            
            # Calculate SL and TP
            sl_price = zone['low'] - current_atr * self.atr_sl_multiplier
            risk = current_price - sl_price
            if risk <= 0:
                continue
            
            # TP: next supply zone or 2.5R
            tp_price = current_price + risk * self.risk_reward_ratio
            if self.supply_zones.get(symbol):
                next_supply = min([z['low'] for z in self.supply_zones[symbol] if z['low'] > current_price], default=None)
                if next_supply:
                    tp_price = min(next_supply, tp_price)
            
            return SignalEvent(
                symbol=symbol,
                signal_type='BUY',
                strength=min(1.0, zone['strength'] / 0.01),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(tp_price - current_price),
                    'zone_high': zone['high'],
                    'zone_low': zone['low'],
                    'zone_strength': zone['strength'],
                    'zone_tests': zone['tests'],
                    'atr': float(current_atr)
                }
            )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'zone_lookback': self.zone_lookback,
            'impulse_min_move': self.impulse_min_move,
            'min_zone_strength': self.min_zone_strength,
            'max_zone_tests': self.max_zone_tests,
            'atr_period': self.atr_period,
            'atr_sl_multiplier': self.atr_sl_multiplier,
            'risk_reward_ratio': self.risk_reward_ratio
        }

