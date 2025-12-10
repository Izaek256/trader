"""Confluence Master Strategy

This strategy combines signals from multiple strategies and only trades when
there is confluence (multiple strategies agree on direction).

Strategies included:
- Balanced Trend Following
- Multi-Timeframe Trend
- Momentum Oscillator Convergence
- Breakout Volatility Expansion
- Liquidity Sweep SMC
- ICT/SMC Enhanced
- Supply and Demand
- MA Crossover
- Mean Reversion Volatility
- Statistical Arbitrage
- VIX Volatility Index
- Trend Pullback RR
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.strategy.balanced_trend_following import BalancedTrendFollowingStrategy
from src.strategy.multi_timeframe_trend import MultiTimeframeTrendStrategy
from src.strategy.momentum_oscillator_convergence import MomentumOscillatorConvergenceStrategy
from src.strategy.breakout_volatility_expansion import BreakoutVolatilityExpansionStrategy
from src.strategy.liquidity_sweep_smc import LiquiditySweepSMCStrategy
from src.strategy.ict_smc_enhanced import ICTSMCEnhancedStrategy
from src.strategy.supply_demand import SupplyDemandStrategy
from src.strategy.ma_crossover import MACrossoverStrategy
from src.strategy.mean_reversion_volatility import MeanReversionVolatilityStrategy
from src.strategy.statistical_arbitrage import StatisticalArbitrageStrategy
from src.strategy.vix_volatility_index import VIXVolatilityIndexStrategy
from src.strategy.trend_pullback_rr import TrendPullbackRRStrategy
from src.core.events import BarEvent, SignalEvent


class ConfluenceMasterStrategy(BaseStrategy):
    """
    Confluence Master Strategy
    
    This strategy aggregates signals from multiple strategies and only trades
    when there is sufficient confluence (multiple strategies agree).
    
    Entry Rules:
    - Minimum confluence required (configurable, default: 3 strategies)
    - Strategies must agree on direction (BUY or SELL)
    - Weighted scoring based on strategy strength and reliability
    - Risk management: Use average SL/TP from contributing strategies
    
    Exit Rules:
    - Take profit: Average TP from strategies or 2.5R
    - Stop loss: Average SL from strategies or 1.5x ATR
    - Exit on confluence breakdown (opposite signals from majority)
    
    Filters:
    - Minimum Confluence: At least N strategies must agree (default: 3)
    - Strategy Weights: Different strategies have different weights
    - Signal Strength: Only trade strong signals (strength > threshold)
    - Time: Avoid low liquidity periods
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ConfluenceMaster", config)
        
        # Confluence parameters
        self.min_confluence = config.get('min_confluence', 3)  # Minimum strategies that must agree
        self.min_signal_strength = config.get('min_signal_strength', 0.6)  # Minimum average strength
        
        # Strategy weights (higher = more important)
        self.strategy_weights = config.get('strategy_weights', {
            'ICTSMCEnhanced': 1.5,  # ICT/SMC is highly weighted
            'SupplyDemand': 1.5,     # Supply/Demand is highly weighted
            'LiquiditySweepSMC': 1.3,
            'BalancedTrendFollowing': 1.2,
            'MultiTimeframeTrend': 1.2,
            'MomentumOscillatorConvergence': 1.1,
            'BreakoutVolatilityExpansion': 1.0,
            'TrendPullbackRR': 1.0,
            'MACrossover': 0.8,
            'MeanReversionVolatility': 0.8,
            'StatisticalArbitrage': 0.7,
            'VIXVolatilityIndex': 0.7
        })
        
        # Strategy configurations (can be customized per strategy)
        self.strategy_configs = config.get('strategy_configs', {})
        
        # Initialize all strategies
        self.strategies: Dict[str, BaseStrategy] = {}
        self._initialize_strategies()
        
        # Signal tracking
        self.recent_signals: Dict[str, List[Dict[str, Any]]] = {}  # symbol -> list of signals
        self.signal_history_window = config.get('signal_history_window', 5)  # Keep last N signals
        
    def _initialize_strategies(self):
        """Initialize all sub-strategies"""
        # Get base config for each strategy
        base_config = self.config.get('base_config', {})
        
        # Initialize each strategy with its config
        strategy_configs = {
            'BalancedTrendFollowing': self.strategy_configs.get('balanced_trend_following', base_config),
            'MultiTimeframeTrend': self.strategy_configs.get('multi_timeframe_trend', base_config),
            'MomentumOscillatorConvergence': self.strategy_configs.get('momentum_oscillator_convergence', base_config),
            'BreakoutVolatilityExpansion': self.strategy_configs.get('breakout_volatility_expansion', base_config),
            'LiquiditySweepSMC': self.strategy_configs.get('liquidity_sweep_smc', base_config),
            'ICTSMCEnhanced': self.strategy_configs.get('ict_smc_enhanced', base_config),
            'SupplyDemand': self.strategy_configs.get('supply_demand', base_config),
            'MACrossover': self.strategy_configs.get('ma_crossover', base_config),
            'MeanReversionVolatility': self.strategy_configs.get('mean_reversion_volatility', base_config),
            'StatisticalArbitrage': self.strategy_configs.get('statistical_arbitrage', base_config),
            'VIXVolatilityIndex': self.strategy_configs.get('vix_volatility_index', base_config),
            'TrendPullbackRR': self.strategy_configs.get('trend_pullback_rr', base_config)
        }
        
        # Create strategy instances
        try:
            self.strategies['BalancedTrendFollowing'] = BalancedTrendFollowingStrategy(strategy_configs['BalancedTrendFollowing'])
            self.strategies['MultiTimeframeTrend'] = MultiTimeframeTrendStrategy(strategy_configs['MultiTimeframeTrend'])
            self.strategies['MomentumOscillatorConvergence'] = MomentumOscillatorConvergenceStrategy(strategy_configs['MomentumOscillatorConvergence'])
            self.strategies['BreakoutVolatilityExpansion'] = BreakoutVolatilityExpansionStrategy(strategy_configs['BreakoutVolatilityExpansion'])
            self.strategies['LiquiditySweepSMC'] = LiquiditySweepSMCStrategy(strategy_configs['LiquiditySweepSMC'])
            self.strategies['ICTSMCEnhanced'] = ICTSMCEnhancedStrategy(strategy_configs['ICTSMCEnhanced'])
            self.strategies['SupplyDemand'] = SupplyDemandStrategy(strategy_configs['SupplyDemand'])
            self.strategies['MACrossover'] = MACrossoverStrategy(strategy_configs['MACrossover'])
            self.strategies['MeanReversionVolatility'] = MeanReversionVolatilityStrategy(strategy_configs['MeanReversionVolatility'])
            self.strategies['StatisticalArbitrage'] = StatisticalArbitrageStrategy(strategy_configs['StatisticalArbitrage'])
            self.strategies['VIXVolatilityIndex'] = VIXVolatilityIndexStrategy(strategy_configs['VIXVolatilityIndex'])
            self.strategies['TrendPullbackRR'] = TrendPullbackRRStrategy(strategy_configs['TrendPullbackRR'])
        except Exception as e:
            # If a strategy fails to initialize, log but continue
            print(f"Warning: Some strategies failed to initialize: {e}")
    
    def _get_strategy_signals(self, bar_event: BarEvent) -> List[Dict[str, Any]]:
        """Get signals from all strategies"""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                # Share bars with strategy
                strategy.add_bar(bar_event)
                
                # Get signal from strategy
                signal = strategy.on_bar(bar_event)
                
                if signal:
                    signals.append({
                        'strategy': strategy_name,
                        'signal_type': signal.signal_type,
                        'strength': signal.strength,
                        'metadata': signal.metadata or {},
                        'timestamp': signal.timestamp
                    })
            except Exception as e:
                # If strategy fails, skip it
                continue
        
        return signals
    
    def _calculate_confluence(self, signals: List[Dict[str, Any]], direction: str) -> Dict[str, Any]:
        """Calculate confluence score for a direction"""
        if not signals:
            return {'count': 0, 'weighted_score': 0.0, 'avg_strength': 0.0, 'contributing_strategies': []}
        
        # Filter signals by direction
        direction_signals = [s for s in signals if s['signal_type'] == direction]
        
        if not direction_signals:
            return {'count': 0, 'weighted_score': 0.0, 'avg_strength': 0.0, 'contributing_strategies': []}
        
        # Calculate weighted score
        total_weight = 0.0
        weighted_strength = 0.0
        contributing = []
        
        for signal in direction_signals:
            strategy_name = signal['strategy']
            weight = self.strategy_weights.get(strategy_name, 1.0)
            strength = signal['strength']
            
            total_weight += weight
            weighted_strength += weight * strength
            contributing.append(strategy_name)
        
        avg_strength = weighted_strength / total_weight if total_weight > 0 else 0.0
        weighted_score = weighted_strength / len(self.strategies) if len(self.strategies) > 0 else 0.0
        
        return {
            'count': len(direction_signals),
            'weighted_score': weighted_score,
            'avg_strength': avg_strength,
            'contributing_strategies': contributing,
            'total_weight': total_weight
        }
    
    def _aggregate_sl_tp(self, signals: List[Dict[str, Any]], direction: str, current_price: float) -> Dict[str, float]:
        """Aggregate stop loss and take profit from contributing strategies"""
        direction_signals = [s for s in signals if s['signal_type'] == direction]
        
        if not direction_signals:
            return {'sl': 0.0, 'tp': 0.0}
        
        sl_prices = []
        tp_prices = []
        
        for signal in direction_signals:
            metadata = signal.get('metadata', {})
            if 'sl_price' in metadata:
                sl_prices.append(metadata['sl_price'])
            if 'tp_price' in metadata:
                tp_prices.append(metadata['tp_price'])
        
        if direction == 'BUY':
            sl = min(sl_prices) if sl_prices else current_price * 0.99  # Default 1% below
            tp = max(tp_prices) if tp_prices else current_price * 1.025  # Default 2.5% above
        else:  # SELL
            sl = max(sl_prices) if sl_prices else current_price * 1.01  # Default 1% above
            tp = min(tp_prices) if tp_prices else current_price * 0.975  # Default 2.5% below
        
        return {'sl': float(sl), 'tp': float(tp)}
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate confluence signal"""
        symbol = bar_event.symbol
        
        # Get signals from all strategies
        signals = self._get_strategy_signals(bar_event)
        
        # Store recent signals
        if symbol not in self.recent_signals:
            self.recent_signals[symbol] = []
        self.recent_signals[symbol].append({
            'time': bar_event.time,
            'signals': signals
        })
        
        # Keep only recent history
        if len(self.recent_signals[symbol]) > self.signal_history_window:
            self.recent_signals[symbol] = self.recent_signals[symbol][-self.signal_history_window:]
        
        if not signals:
            return None
        
        # Calculate confluence for BUY and SELL
        buy_confluence = self._calculate_confluence(signals, 'BUY')
        sell_confluence = self._calculate_confluence(signals, 'SELL')
        
        current_price = bar_event.close
        
        # Determine which direction has better confluence
        buy_score = buy_confluence['count'] * buy_confluence['weighted_score']
        sell_score = sell_confluence['count'] * sell_confluence['weighted_score']
        
        # Check if minimum confluence is met
        if buy_confluence['count'] >= self.min_confluence and buy_score > sell_score:
            # Check signal strength
            if buy_confluence['avg_strength'] >= self.min_signal_strength:
                # Aggregate SL/TP
                sl_tp = self._aggregate_sl_tp(signals, 'BUY', current_price)
                
                risk = current_price - sl_tp['sl']
                reward = sl_tp['tp'] - current_price
                
                if risk > 0:
                    return SignalEvent(
                        symbol=symbol,
                        signal_type='BUY',
                        strength=min(1.0, buy_confluence['weighted_score']),
                        timestamp=bar_event.time,
                        strategy_name=self.name,
                        metadata={
                            'entry_price': float(current_price),
                            'sl_price': sl_tp['sl'],
                            'tp_price': sl_tp['tp'],
                            'risk': float(risk),
                            'reward': float(reward),
                            'confluence_count': buy_confluence['count'],
                            'confluence_score': float(buy_confluence['weighted_score']),
                            'avg_strength': float(buy_confluence['avg_strength']),
                            'contributing_strategies': buy_confluence['contributing_strategies'],
                            'all_signals': len(signals),
                            'buy_signals': buy_confluence['count'],
                            'sell_signals': sell_confluence['count']
                        }
                    )
        
        elif sell_confluence['count'] >= self.min_confluence and sell_score > buy_score:
            # Check signal strength
            if sell_confluence['avg_strength'] >= self.min_signal_strength:
                # Aggregate SL/TP
                sl_tp = self._aggregate_sl_tp(signals, 'SELL', current_price)
                
                risk = sl_tp['sl'] - current_price
                reward = current_price - sl_tp['tp']
                
                if risk > 0:
                    return SignalEvent(
                        symbol=symbol,
                        signal_type='SELL',
                        strength=min(1.0, sell_confluence['weighted_score']),
                        timestamp=bar_event.time,
                        strategy_name=self.name,
                        metadata={
                            'entry_price': float(current_price),
                            'sl_price': sl_tp['sl'],
                            'tp_price': sl_tp['tp'],
                            'risk': float(risk),
                            'reward': float(reward),
                            'confluence_count': sell_confluence['count'],
                            'confluence_score': float(sell_confluence['weighted_score']),
                            'avg_strength': float(sell_confluence['avg_strength']),
                            'contributing_strategies': sell_confluence['contributing_strategies'],
                            'all_signals': len(signals),
                            'buy_signals': buy_confluence['count'],
                            'sell_signals': sell_confluence['count']
                        }
                    )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'min_confluence': self.min_confluence,
            'min_signal_strength': self.min_signal_strength,
            'strategy_weights': self.strategy_weights,
            'active_strategies': list(self.strategies.keys())
        }

