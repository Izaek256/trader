"""Configuration loader for JSON/YAML files"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigLoader:
    """Loads and manages configuration from JSON or YAML files"""
    
    def __init__(self, config_path: str):
        """
        Initialize config loader
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                self.config = yaml.safe_load(f) or {}
            else:
                self.config = json.load(f)
        
        # Override with environment variables if present
        self._load_env_overrides()
    
    def _load_env_overrides(self) -> None:
        """Override config values with environment variables"""
        from dotenv import load_dotenv
        load_dotenv()
        
        # MT5 credentials from environment
        if 'MT5_LOGIN' in os.environ:
            self.config.setdefault('mt5', {})['login'] = int(os.environ['MT5_LOGIN'])
        if 'MT5_PASSWORD' in os.environ:
            self.config.setdefault('mt5', {})['password'] = os.environ['MT5_PASSWORD']
        if 'MT5_SERVER' in os.environ:
            self.config.setdefault('mt5', {})['server'] = os.environ['MT5_SERVER']
        if 'MT5_PATH' in os.environ:
            self.config.setdefault('mt5', {})['path'] = os.environ['MT5_PATH']
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary"""
        return self.config.copy()

