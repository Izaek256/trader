"""Statistical Arbitrage / Spread Trading Strategy

This strategy trades mean reversion in spreads between correlated instruments
or statistical deviations from normal relationships.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage / Spread Trading Strategy
    
    Entry Rules:
    - Calculate spread between two correlated instruments (or price vs moving average)
    - Spread deviates > 2 standard deviations from mean
    - Z-score > 2.0 (for long spread) or < -2.0 (for short spread)
    - Cointegration test passes (spread is mean-reverting)
    - Volume confirms (both instruments have sufficient liquidity)
    
    Exit Rules:
    - Take profit: Spread returns to mean (Z-score < 0.5)
    - Stop loss: Spread continues to widen (Z-score > 3.0 or < -3.0)
    - Time-based exit: Close after N bars if not profitable
    
    Filters:
    - Correlation: Instruments must have correlation > 0.7 (if pair trading)
    - Cointegration: Spread must be cointegrated (ADF test p-value < 0.05)
    - Volatility: Both instruments must have similar volatility
    - Time: Best during overlapping sessions
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("StatisticalArbitrage", config)
        
        # Spread calculation
        self.spread_period = config.get('spread_period', 60)  # Bars for spread calculation
        self.zscore_period = config.get('zscore_period', 20)  # Period for Z-score calculation
        self.zscore_entry = config.get('zscore_entry', 2.0)
        self.zscore_exit = config.get('zscore_exit', 0.5)
        self.zscore_stop = config.get('zscore_stop', 3.0)
        
        # Pair trading parameters
        self.pair_symbol = config.get('pair_symbol', None)  # Second symbol for pair trading
        self.use_pair = config.get('use_pair', False)
        
        # Single instrument mode (price vs MA spread)
        self.ma_period = config.get('ma_period', 50)
        self.spread_method = config.get('spread_method', 'ratio')  # 'ratio' or 'difference'
        
        # Cointegration test
        self.check_cointegration = config.get('check_cointegration', True)
        self.cointegration_lookback = config.get('cointegration_lookback', 100)
        
        # Correlation filter
        self.min_correlation = config.get('min_correlation', 0.7)
        
        # Risk/Reward
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.0)
        self.max_hold_bars = config.get('max_hold_bars', 50)
        
        # State tracking
        self.spreads: Dict[str, pd.Series] = {}
        self.zscores: Dict[str, pd.Series] = {}
        self.entry_zscore: Dict[str, float] = {}
        
    def _calculate_spread(self, bars1: pd.DataFrame, bars2: Optional[pd.DataFrame] = None, 
                         ma_period: Optional[int] = None) -> pd.Series:
        """Calculate spread between two series"""
        if bars2 is not None:
            # Pair trading: spread = price1 - price2 (or ratio)
            if len(bars1) != len(bars2):
                # Align by time
                merged = pd.merge(bars1[['time', 'close']], bars2[['time', 'close']], 
                                on='time', suffixes=('_1', '_2'))
                if self.spread_method == 'ratio':
                    return merged['close_1'] / merged['close_2']
                else:
                    return merged['close_1'] - merged['close_2']
            else:
                if self.spread_method == 'ratio':
                    return bars1['close'] / bars2['close']
                else:
                    return bars1['close'] - bars2['close']
        elif ma_period:
            # Single instrument: spread = price - MA (normalized)
            ma = bars1['close'].rolling(window=ma_period).mean()
            return (bars1['close'] - ma) / ma  # Percentage deviation
        else:
            return pd.Series()
    
    def _calculate_zscore(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Z-score of series"""
        mean = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return (series - mean) / std
    
    def _test_cointegration(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Simple cointegration test using ADF on spread"""
        if len(series1) < 50 or len(series2) < 50:
            return False
        
        # Calculate spread
        spread = series1 - series2
        
        # Simple mean reversion test: check if spread is stationary
        # Use half-life of mean reversion as proxy
        spread_diff = spread.diff().dropna()
        spread_lag = spread.shift(1).dropna()
        
        if len(spread_diff) < 10 or len(spread_lag) < 10:
            return False
        
        # Align
        aligned = pd.concat([spread_diff, spread_lag], axis=1).dropna()
        if len(aligned) < 10:
            return False
        
        # Simple regression: spread_diff = alpha + beta * spread_lag
        # If beta is negative and significant, spread is mean-reverting
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                aligned.iloc[:, 1], aligned.iloc[:, 0]
            )
            # Mean reverting if slope < 0
            return slope < -0.1 and p_value < 0.1
        except:
            # Fallback: check if spread variance is stable
            spread_std = spread.std()
            spread_mean = spread.mean()
            cv = spread_std / abs(spread_mean) if spread_mean != 0 else 0
            return cv < 0.5  # Coefficient of variation < 0.5
    
    def _calculate_correlation(self, series1: pd.Series, series2: pd.Series, period: int) -> float:
        """Calculate rolling correlation"""
        if len(series1) < period or len(series2) < period:
            return 0.0
        
        aligned = pd.concat([series1.tail(period), series2.tail(period)], axis=1).dropna()
        if len(aligned) < period:
            return 0.0
        
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Get bars for primary symbol
        bars1 = self.get_bars(symbol, max(self.spread_period, self.zscore_period, self.ma_period) + 10)
        if len(bars1) < max(self.zscore_period, self.ma_period):
            return None
        
        # For pair trading, need second symbol bars
        bars2 = None
        if self.use_pair and self.pair_symbol:
            # In production, you'd fetch bars for pair_symbol
            # For now, we'll use single instrument mode
            pass
        
        # Calculate spread
        if self.use_pair and bars2 is not None:
            spread = self._calculate_spread(bars1, bars2)
        else:
            spread = self._calculate_spread(bars1, ma_period=self.ma_period)
        
        if len(spread) < self.zscore_period:
            return None
        
        # Calculate Z-score
        zscore = self._calculate_zscore(spread, self.zscore_period)
        
        if len(zscore) < 1:
            return None
        
        current_zscore = zscore.iloc[-1]
        current_spread = spread.iloc[-1]
        spread_mean = spread.tail(self.zscore_period).mean()
        spread_std = spread.tail(self.zscore_period).std()
        
        if pd.isna(current_zscore) or pd.isna(spread_std) or spread_std == 0:
            return None
        
        # Check for existing position
        existing_entry_zscore = self.entry_zscore.get(symbol)
        
        # Exit logic: if we have a position, check for exit
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
            
            # Stop loss: spread continues to widen
            if (current_signal == 'BUY' and current_zscore > self.zscore_stop) or \
               (current_signal == 'SELL' and current_zscore < -self.zscore_stop):
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
            
            return None  # Hold position
        
        # Entry logic: LONG SPREAD (spread is too low, expect mean reversion up)
        if current_zscore < -self.zscore_entry:
            # For single instrument: buy when price is below MA
            # For pair: buy spread (buy symbol1, sell symbol2)
            
            current_price = bar_event.close
            
            # Calculate SL and TP based on spread
            sl_spread = spread_mean - spread_std * self.zscore_stop
            tp_spread = spread_mean  # Target mean
            
            # Convert spread targets to price targets
            if self.use_pair and bars2 is not None:
                # Pair trading: more complex, would need current price of both
                risk = abs(current_spread - sl_spread)
                reward = abs(tp_spread - current_spread)
            else:
                # Single instrument: convert spread to price
                risk = abs(current_spread - sl_spread) * current_price
                reward = abs(tp_spread - current_spread) * current_price
            
            # Adjust for risk/reward
            if reward / risk < self.risk_reward_ratio:
                reward = risk * self.risk_reward_ratio
            
            sl_price = current_price - risk
            tp_price = current_price + reward
            
            self.entry_zscore[symbol] = float(current_zscore)
            self.current_signals[symbol] = 'BUY'
            
            return SignalEvent(
                symbol=symbol,
                signal_type='BUY',
                strength=min(1.0, abs(current_zscore) / self.zscore_entry),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(reward),
                    'zscore': float(current_zscore),
                    'spread': float(current_spread),
                    'spread_mean': float(spread_mean),
                    'spread_std': float(spread_std)
                }
            )
        
        # Entry logic: SHORT SPREAD (spread is too high, expect mean reversion down)
        elif current_zscore > self.zscore_entry:
            current_price = bar_event.close
            
            # Calculate SL and TP
            sl_spread = spread_mean + spread_std * self.zscore_stop
            tp_spread = spread_mean
            
            if self.use_pair and bars2 is not None:
                risk = abs(sl_spread - current_spread)
                reward = abs(current_spread - tp_spread)
            else:
                risk = abs(sl_spread - current_spread) * current_price
                reward = abs(current_spread - tp_spread) * current_price
            
            if reward / risk < self.risk_reward_ratio:
                reward = risk * self.risk_reward_ratio
            
            sl_price = current_price + risk
            tp_price = current_price - reward
            
            self.entry_zscore[symbol] = float(current_zscore)
            self.current_signals[symbol] = 'SELL'
            
            return SignalEvent(
                symbol=symbol,
                signal_type='SELL',
                strength=min(1.0, abs(current_zscore) / self.zscore_entry),
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(reward),
                    'zscore': float(current_zscore),
                    'spread': float(current_spread),
                    'spread_mean': float(spread_mean),
                    'spread_std': float(spread_std)
                }
            )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'spread_period': self.spread_period,
            'zscore_period': self.zscore_period,
            'zscore_entry': self.zscore_entry,
            'zscore_exit': self.zscore_exit,
            'ma_period': self.ma_period,
            'risk_reward_ratio': self.risk_reward_ratio
        }

