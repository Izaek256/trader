"""Strategy Comparison Summary

This module provides a comparison table of all trading strategies
including expected performance metrics and characteristics.
"""

from typing import List, Dict, Any


def get_strategy_comparison() -> List[Dict[str, Any]]:
    """
    Returns comparison table of all strategies
    
    Columns:
    - Strategy Name
    - Market Type
    - Market Regime
    - Expected Win Rate
    - Expected R:R
    - Volatility Regime
    - Best Timeframe
    - Key Features
    """
    
    strategies = [
        {
            'name': 'Multi-Timeframe Trend Continuation',
            'market_type': 'Forex, Indices',
            'market_regime': 'Trending',
            'win_rate_target': '45-55%',
            'rr_target': '2.5:1',
            'volatility_regime': 'Medium-High',
            'best_timeframe': 'H1, H4',
            'key_features': 'HTF trend filter, pullback entries, ADX confirmation'
        },
        {
            'name': 'Mean-Reversion Volatility Compression',
            'market_type': 'Forex, Indices',
            'market_regime': 'Ranging/Mean-Reverting',
            'win_rate_target': '55-65%',
            'rr_target': '1.5:1',
            'volatility_regime': 'Low (compression)',
            'best_timeframe': 'M15, M30',
            'key_features': 'BB compression, RSI extremes, low volume filter'
        },
        {
            'name': 'Breakout Volatility Expansion',
            'market_type': 'Forex, Indices, Commodities',
            'market_regime': 'Breakout/Expansion',
            'win_rate_target': '40-50%',
            'rr_target': '2.0:1',
            'volatility_regime': 'Expanding (after compression)',
            'best_timeframe': 'H1, H4',
            'key_features': 'Consolidation detection, volume confirmation, ATR expansion'
        },
        {
            'name': 'Liquidity Sweep SMC',
            'market_type': 'Forex, Indices',
            'market_regime': 'Any (structure-based)',
            'win_rate_target': '50-60%',
            'rr_target': '3.0:1',
            'volatility_regime': 'Medium-High',
            'best_timeframe': 'H1, H4, D1',
            'key_features': 'Liquidity zones, order blocks, market structure shifts'
        },
        {
            'name': 'Statistical Arbitrage',
            'market_type': 'Forex (pairs), Indices',
            'market_regime': 'Mean-Reverting',
            'win_rate_target': '60-70%',
            'rr_target': '2.0:1',
            'volatility_regime': 'Low-Medium',
            'best_timeframe': 'H1, H4',
            'key_features': 'Z-score mean reversion, cointegration, spread trading'
        },
        {
            'name': 'VIX Volatility Index',
            'market_type': 'Volatility Indices (VIX, VXX)',
            'market_regime': 'Mean-Reverting (extreme)',
            'win_rate_target': '55-65%',
            'rr_target': '2.0:1',
            'volatility_regime': 'Extreme (spikes)',
            'best_timeframe': 'D1, W1',
            'key_features': 'Z-score extremes, RSI confirmation, expiration avoidance'
        },
        {
            'name': 'ML-Assisted Entry Filter',
            'market_type': 'Forex, Indices',
            'market_regime': 'Adaptive (ML-learned)',
            'win_rate_target': '50-60%',
            'rr_target': '2.0:1',
            'volatility_regime': 'Medium',
            'best_timeframe': 'H1, H4',
            'key_features': 'ML probability filter, feature engineering, adaptive thresholds'
        },
        {
            'name': 'Momentum Oscillator Convergence',
            'market_type': 'Forex, Indices',
            'market_regime': 'Trending/Momentum',
            'win_rate_target': '50-55%',
            'rr_target': '2.5:1',
            'volatility_regime': 'Medium-High',
            'best_timeframe': 'H1, H4',
            'key_features': 'Multi-oscillator alignment, RSI+Stoch+MACD+Williams %R'
        }
    ]
    
    return strategies


def print_comparison_table():
    """Print formatted comparison table"""
    strategies = get_strategy_comparison()
    
    # Header
    print("\n" + "="*120)
    print("TRADING STRATEGY COMPARISON TABLE")
    print("="*120)
    print(f"{'Strategy Name':<35} {'Market':<15} {'Regime':<20} {'Win%':<10} {'R:R':<8} {'Vol':<12} {'TF':<8}")
    print("-"*120)
    
    # Rows
    for s in strategies:
        print(f"{s['name']:<35} {s['market_type']:<15} {s['market_regime']:<20} "
              f"{s['win_rate_target']:<10} {s['rr_target']:<8} {s['volatility_regime']:<12} {s['best_timeframe']:<8}")
    
    print("="*120)
    print("\nKEY FEATURES:")
    print("-"*120)
    for s in strategies:
        print(f"{s['name']:<35}: {s['key_features']}")
    
    print("\n" + "="*120)
    print("PERFORMANCE EXPECTATIONS:")
    print("-"*120)
    print("Win Rate Target: Expected win rate range after optimization")
    print("R:R Target: Risk:Reward ratio target")
    print("Volatility Regime: Optimal volatility conditions")
    print("Best Timeframe: Recommended timeframes for strategy")
    print("="*120 + "\n")


if __name__ == '__main__':
    print_comparison_table()

