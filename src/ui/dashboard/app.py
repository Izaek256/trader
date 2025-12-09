"""FastAPI application with Plotly Dash dashboard"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware
import logging

from src.ui.dashboard.routes import router, set_bot_instance
from src.ui.dashboard.dash_app import create_dash_app

logger = logging.getLogger(__name__)


def create_app(bot_instance=None, performance_logger=None):
    """Create FastAPI application with Dash dashboard"""
    
    # Create FastAPI app
    app = FastAPI(title="MT5 Trading Bot Dashboard")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Set bot instance for API routes
    if bot_instance and performance_logger:
        set_bot_instance(bot_instance, performance_logger)
    
    # Include API routes
    app.include_router(router)
    
    # Create and mount Dash app with WSGI middleware
    dash_app = create_dash_app()
    app.mount("/dash", WSGIMiddleware(dash_app.server))
    
    @app.get("/")
    async def root():
        return {"message": "MT5 Trading Bot Dashboard API", "dashboard": "/dash"}
    
    return app

