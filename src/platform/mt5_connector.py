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
        
        if path:
            if not mt5.initialize(path=path):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
        else:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
        
        logger.info("MT5 initialized successfully")
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

