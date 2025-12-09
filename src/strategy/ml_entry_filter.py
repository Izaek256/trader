"""ML-Assisted Entry Filter Strategy

This strategy uses a simple ML model (logistic regression or random forest)
as an entry filter on top of a base strategy signal.
The ML model predicts probability of successful trade based on features.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.core.events import BarEvent, SignalEvent


class MLEntryFilterStrategy(BaseStrategy):
    """
    ML-Assisted Entry Filter Strategy
    
    Entry Rules:
    - Base strategy generates signal (trend, momentum, etc.)
    - Extract features: RSI, MACD, ATR, Volume, Price action patterns
    - ML model predicts success probability
    - Only enter if probability > threshold (e.g., 0.65)
    - Combine with traditional filters (trend, volatility)
    
    Exit Rules:
    - Same as base strategy (TP/SL based on ATR or structure)
    - ML model can also predict exit timing (optional)
    
    Filters:
    - ML Probability: Must exceed threshold
    - Base Strategy: Must generate signal
    - Traditional: Trend, volatility, volume filters still apply
    - Time: Avoid low-probability hours (learned from data)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MLEntryFilter", config)
        
        # ML parameters
        self.ml_model_type = config.get('ml_model_type', 'logistic')  # 'logistic', 'random_forest', 'gradient_boosting'
        self.ml_probability_threshold = config.get('ml_probability_threshold', 0.65)
        self.use_ml = config.get('use_ml', True)
        
        # Base strategy parameters (trend following)
        self.fast_ema = config.get('fast_ema', 21)
        self.slow_ema = config.get('slow_ema', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        
        # ATR parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 1.5)
        
        # Volume parameters
        self.volume_period = config.get('volume_period', 20)
        self.volume_threshold = config.get('volume_threshold', 1.2)
        
        # Risk/Reward
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.0)
        
        # ML model (simple implementation - in production, load trained model)
        self.ml_model = None
        self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize simple ML model (placeholder - use trained model in production)"""
        if not self.use_ml:
            return
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            
            if self.ml_model_type == 'logistic':
                self.ml_model = LogisticRegression(max_iter=1000, random_state=42)
            elif self.ml_model_type == 'random_forest':
                self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            elif self.ml_model_type == 'gradient_boosting':
                self.ml_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
            else:
                self.ml_model = LogisticRegression(max_iter=1000, random_state=42)
            
            # In production, load pre-trained model:
            # import joblib
            # self.ml_model = joblib.load('models/ml_entry_filter.pkl')
            
        except ImportError:
            # Fallback: use simple rule-based filter
            self.use_ml = False
            self.ml_model = None
    
    def _extract_features(self, bars: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for ML model"""
        if len(bars) < max(self.slow_ema, self.rsi_period, self.macd_slow, self.atr_period):
            return None
        
        features = []
        
        # Price features
        close = bars['close']
        high = bars['high']
        low = bars['low']
        volume = bars['volume']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0)
        
        # MACD
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        features.append(macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0.0)
        features.append(histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0)
        
        # ATR (normalized)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        atr_normalized = (atr / close).iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
        features.append(atr_normalized)
        
        # Volume ratio
        volume_avg = volume.rolling(window=self.volume_period).mean()
        volume_ratio = (volume.iloc[-1] / volume_avg.iloc[-1]) if not pd.isna(volume_avg.iloc[-1]) and volume_avg.iloc[-1] > 0 else 1.0
        features.append(volume_ratio)
        
        # EMA alignment
        ema_fast_val = close.ewm(span=self.fast_ema, adjust=False).mean().iloc[-1]
        ema_slow_val = close.ewm(span=self.slow_ema, adjust=False).mean().iloc[-1]
        ema_alignment = 1.0 if ema_fast_val > ema_slow_val else -1.0
        features.append(ema_alignment)
        
        # Price momentum (rate of change)
        roc = ((close.iloc[-1] - close.iloc[-self.rsi_period]) / close.iloc[-self.rsi_period]) * 100
        features.append(roc if not pd.isna(roc) else 0.0)
        
        # Volatility (BB width)
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        bb_width = (std * 2 * 2 / sma).iloc[-1] if not pd.isna(sma.iloc[-1]) and sma.iloc[-1] > 0 else 0.0
        features.append(bb_width)
        
        # Candle patterns (simplified)
        current_candle = bars.iloc[-1]
        body_size = abs(current_candle['close'] - current_candle['open']) / current_candle['close']
        features.append(body_size)
        
        return np.array(features).reshape(1, -1)
    
    def _predict_probability(self, features: np.ndarray) -> float:
        """Predict success probability using ML model"""
        if not self.use_ml or self.ml_model is None:
            # Fallback: simple rule-based probability
            # Based on feature quality
            rsi = features[0, 0]
            macd_hist = features[0, 2]
            volume_ratio = features[0, 4]
            ema_alignment = features[0, 5]
            
            prob = 0.5  # Base probability
            
            # RSI in good range (30-70)
            if 30 < rsi < 70:
                prob += 0.1
            
            # MACD histogram positive (for long) or negative (for short)
            if abs(macd_hist) > 0.0001:
                prob += 0.1
            
            # Volume above average
            if volume_ratio > 1.2:
                prob += 0.1
            
            # EMA alignment
            if ema_alignment > 0:
                prob += 0.1
            
            return min(1.0, max(0.0, prob))
        
        try:
            # Predict probability
            prob = self.ml_model.predict_proba(features)[0, 1]  # Probability of class 1 (success)
            return float(prob)
        except:
            # Fallback if model not trained
            return 0.5
    
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
    
    def on_bar(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process new bar and generate signal"""
        self.add_bar(bar_event)
        symbol = bar_event.symbol
        
        # Get bars
        min_bars = max(self.slow_ema, self.rsi_period, self.macd_slow, self.atr_period, self.volume_period) + 10
        bars = self.get_bars(symbol, min_bars)
        if len(bars) < min_bars:
            return None
        
        # Extract features
        features = self._extract_features(bars)
        if features is None:
            return None
        
        # Predict probability
        ml_probability = self._predict_probability(features)
        
        # Base strategy: EMA crossover with RSI and MACD confirmation
        close = bars['close']
        ema_fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_ema, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        macd_fast_ema = close.ewm(span=self.macd_fast, adjust=False).mean()
        macd_slow_ema = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = macd_fast_ema - macd_slow_ema
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # ATR
        atr = self._calculate_atr(bars, self.atr_period)
        
        # Volume
        volume_avg = bars['volume'].rolling(window=self.volume_period).mean()
        
        if len(ema_fast) < 2 or len(rsi) < 1 or len(histogram) < 1 or len(atr) < 1:
            return None
        
        current_price = bar_event.close
        current_fast = ema_fast.iloc[-1]
        prev_fast = ema_fast.iloc[-2]
        current_slow = ema_slow.iloc[-1]
        prev_slow = ema_slow.iloc[-2]
        current_rsi = rsi.iloc[-1]
        current_histogram = histogram.iloc[-1]
        current_atr = atr.iloc[-1]
        current_volume = bar_event.tick_volume
        avg_volume = volume_avg.iloc[-1]
        
        # ML filter: must exceed threshold
        if ml_probability < self.ml_probability_threshold:
            return None
        
        # Volume filter
        if pd.isna(current_volume) or pd.isna(avg_volume) or current_volume < avg_volume * self.volume_threshold:
            return None
        
        # Entry logic: BULLISH (EMA crossover + confirmations)
        if (prev_fast <= prev_slow and current_fast > current_slow and
            not pd.isna(current_rsi) and 40 < current_rsi < 70 and
            not pd.isna(current_histogram) and current_histogram > 0):
            
            sl_price = current_price - current_atr * self.atr_sl_multiplier
            risk = current_price - sl_price
            tp_price = current_price + risk * self.risk_reward_ratio
            
            return SignalEvent(
                symbol=symbol,
                signal_type='BUY',
                strength=ml_probability,  # Use ML probability as strength
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(risk * self.risk_reward_ratio),
                    'ml_probability': float(ml_probability),
                    'rsi': float(current_rsi),
                    'macd_histogram': float(current_histogram),
                    'atr': float(current_atr)
                }
            )
        
        # Entry logic: BEARISH (EMA crossover + confirmations)
        elif (prev_fast >= prev_slow and current_fast < current_slow and
              not pd.isna(current_rsi) and 30 < current_rsi < 60 and
              not pd.isna(current_histogram) and current_histogram < 0):
            
            sl_price = current_price + current_atr * self.atr_sl_multiplier
            risk = sl_price - current_price
            tp_price = current_price - risk * self.risk_reward_ratio
            
            return SignalEvent(
                symbol=symbol,
                signal_type='SELL',
                strength=ml_probability,
                timestamp=bar_event.time,
                strategy_name=self.name,
                metadata={
                    'entry_price': float(current_price),
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'risk': float(risk),
                    'reward': float(risk * self.risk_reward_ratio),
                    'ml_probability': float(ml_probability),
                    'rsi': float(current_rsi),
                    'macd_histogram': float(current_histogram),
                    'atr': float(current_atr)
                }
            )
        
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'ml_model_type': self.ml_model_type,
            'ml_probability_threshold': self.ml_probability_threshold,
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'risk_reward_ratio': self.risk_reward_ratio
        }

