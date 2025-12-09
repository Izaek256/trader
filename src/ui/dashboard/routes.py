"""API routes for dashboard"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Global bot instance (set by app)
bot_instance: Optional[Any] = None
performance_logger: Optional[Any] = None


def set_bot_instance(bot: Any, perf_logger: Any):
    """Set the bot instance for API access"""
    global bot_instance, performance_logger
    bot_instance = bot
    performance_logger = perf_logger


@router.get("/api/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get current performance metrics"""
    if performance_logger is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")
    
    try:
        metrics = performance_logger.get_current_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/trades")
async def get_trades(limit: int = 100) -> Dict[str, Any]:
    """Get recent trade history"""
    if performance_logger is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")
    
    try:
        file_logger = performance_logger.file_logger
        trades = file_logger.read_trades(limit=limit)
        return {"trades": trades, "count": len(trades)}
    except Exception as e:
        logger.error(f"Error getting trades: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/positions")
async def get_positions() -> Dict[str, Any]:
    """Get open positions"""
    if bot_instance is None or bot_instance.order_manager is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")
    
    try:
        positions = bot_instance.order_manager.get_positions()
        return {
            "positions": [
                {
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "type": "BUY" if p.type == 0 else "SELL",
                    "volume": p.volume,
                    "price_open": p.price_open,
                    "price_current": p.price_current,
                    "profit": p.profit,
                    "time": p.time.isoformat()
                }
                for p in positions
            ],
            "count": len(positions)
        }
    except Exception as e:
        logger.error(f"Error getting positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/equity")
async def get_equity() -> Dict[str, Any]:
    """Get equity curve data"""
    if performance_logger is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")
    
    try:
        file_logger = performance_logger.file_logger
        metrics_history = file_logger.read_metrics()
        
        equity_data = []
        for metric in metrics_history:
            equity_data.append({
                "time": metric.get("timestamp", ""),
                "equity": metric.get("current_equity", 0),
                "balance": metric.get("current_equity", 0)
            })
        
        return {"equity_curve": equity_data}
    except Exception as e:
        logger.error(f"Error getting equity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/status")
async def get_status() -> Dict[str, Any]:
    """Get bot status"""
    if bot_instance is None:
        return {
            "running": False,
            "connected": False,
            "message": "Bot not initialized"
        }
    
    try:
        status = bot_instance.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

