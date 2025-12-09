"""MT5 Connection and Session Management"""

import MetaTrader5 as mt5
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MT5Connector:
    """Handles MT5 initialization, login, and connection state"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MT5 connector
        
        Args:
            config: Configuration dictionary with MT5 settings
        """
        self.config = config
        self.connected = False
        self.account_info = None
        self.terminal_info = None
    
    def initialize(self) -> bool:
        """
        Initialize MT5 connection
        
        Returns:
            True if successful, False otherwise
        """
        mt5_config = self.config.get('mt5', {})
        path = mt5_config.get('path')
        timeout = mt5_config.get('timeout', 60000)
        
        # Handle null/None path from JSON
        if path is None or path == 'null' or path == '':
            path = None
        
        # Try to initialize with path if provided
        if path:
            # Check if path exists
            import os
            if os.path.exists(path):
                if not mt5.initialize(path=path):
                    error = mt5.last_error()
                    logger.error(f"MT5 initialization failed with path '{path}': {error}")
                    logger.info("Attempting to initialize without path (auto-detect)...")
                    # Fall through to auto-detect
                else:
                    logger.info(f"MT5 initialized successfully from path: {path}")
                    return True
            else:
                logger.warning(f"MT5 path does not exist: {path}")
                logger.info("Attempting to initialize without path (auto-detect)...")
        
        # Try auto-detect (initialize without path)
        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed (auto-detect): {error}")
            logger.error("Please ensure MetaTrader 5 is installed and:")
            logger.error("  1. Set the correct path in config file (mt5.path), OR")
            logger.error("  2. Set MT5_PATH environment variable, OR")
            logger.error("  3. Ensure MT5 is in default installation location")
            logger.error("Common MT5 paths:")
            logger.error("  - Windows: C:\\Program Files\\MetaTrader 5\\terminal64.exe")
            logger.error("  - Windows (x86): C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe")
            return False
        
        logger.info("MT5 initialized successfully (auto-detected)")
        return True
    
    def login(self) -> bool:
        """
        Login to MT5 account
        
        Returns:
            True if successful, False otherwise
        """
        mt5_config = self.config.get('mt5', {})
        login = mt5_config.get('login')
        password = mt5_config.get('password')
        server = mt5_config.get('server')
        
        if not login or not password or not server:
            logger.error("MT5 login credentials missing in config")
            return False
        
        if not mt5.login(login, password=password, server=server):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            return False
        
        self.connected = True
        self.account_info = mt5.account_info()
        self.terminal_info = mt5.terminal_info()
        
        if self.account_info:
            logger.info(f"Logged in to account {self.account_info.login}")
            logger.info(f"Server: {self.account_info.server}")
            logger.info(f"Balance: {self.account_info.balance}")
        
        return True
    
    def connect(self) -> bool:
        """
        Initialize and login to MT5
        
        Returns:
            True if successful, False otherwise
        """
        if not self.initialize():
            return False
        return self.login()
    
    def disconnect(self) -> None:
        """Disconnect from MT5"""
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")
    
    def get_account_info(self) -> Optional[Any]:
        """Get account information"""
        if self.connected:
            self.account_info = mt5.account_info()
        return self.account_info
    
    def get_terminal_info(self) -> Optional[Any]:
        """Get terminal information"""
        if self.connected:
            self.terminal_info = mt5.terminal_info()
        return self.terminal_info
    
    def is_connected(self) -> bool:
        """Check if connected to MT5"""
        return self.connected and mt5.terminal_info() is not None
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

