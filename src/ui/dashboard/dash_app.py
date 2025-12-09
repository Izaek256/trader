"""Plotly Dash dashboard application"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
import requests
import time
import logging

logger = logging.getLogger(__name__)


def create_dash_app():
    """Create and configure Dash application"""
    
    app = dash.Dash(__name__, requests_pathname_prefix='/dash/')
    
    app.layout = html.Div([
        html.H1("MT5 Trading Bot Dashboard", style={'textAlign': 'center'}),
        
        # Status and Key Metrics
        html.Div(id='status-container', style={'margin': '20px'}),
        
        # Metrics Cards
        html.Div(id='metrics-cards', style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
        
        # Charts
        html.Div([
            dcc.Graph(id='equity-chart'),
            dcc.Graph(id='drawdown-chart'),
        ], style={'margin': '20px'}),
        
        # Trade Log Table
        html.Div([
            html.H3("Recent Trades"),
            html.Div(id='trades-table')
        ], style={'margin': '20px'}),
        
        # Auto-refresh
        dcc.Interval(
            id='interval-component',
            interval=1000,  # Update every second
            n_intervals=0
        )
    ])
    
    @app.callback(
        [Output('status-container', 'children'),
         Output('metrics-cards', 'children'),
         Output('equity-chart', 'figure'),
         Output('drawdown-chart', 'figure'),
         Output('trades-table', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        """Update dashboard components"""
        try:
            base_url = "http://localhost:8050"
            
            # Get status
            try:
                status_resp = requests.get(f"{base_url}/api/status", timeout=1)
                status = status_resp.json() if status_resp.status_code == 200 else {}
            except:
                status = {"running": False, "connected": False}
            
            # Get metrics
            try:
                metrics_resp = requests.get(f"{base_url}/api/metrics", timeout=1)
                metrics = metrics_resp.json() if metrics_resp.status_code == 200 else {}
            except:
                metrics = {}
            
            # Get equity curve
            try:
                equity_resp = requests.get(f"{base_url}/api/equity", timeout=1)
                equity_data = equity_resp.json().get('equity_curve', []) if equity_resp.status_code == 200 else []
            except:
                equity_data = []
            
            # Get trades
            try:
                trades_resp = requests.get(f"{base_url}/api/trades", timeout=1, params={'limit': 20})
                trades = trades_resp.json().get('trades', []) if trades_resp.status_code == 200 else []
            except:
                trades = []
            
            # Status container
            status_color = "green" if status.get('running') else "red"
            status_container = html.Div([
                html.H3(f"Status: ", style={'display': 'inline'}),
                html.Span(
                    "Running" if status.get('running') else "Stopped",
                    style={'color': status_color, 'fontWeight': 'bold'}
                ),
                html.P(f"Connected: {status.get('connected', False)}"),
                html.P(f"Strategy: {status.get('strategy', 'N/A')}"),
                html.P(f"Open Positions: {status.get('positions', 0)}")
            ])
            
            # Metrics cards
            metrics_cards = [
                html.Div([
                    html.H4("Win Rate"),
                    html.H2(f"{metrics.get('win_rate', 0):.2f}%")
                ], style={'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'center'}),
                html.Div([
                    html.H4("ROI"),
                    html.H2(f"{metrics.get('roi', 0):.2f}%")
                ], style={'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'center'}),
                html.Div([
                    html.H4("Total Profit"),
                    html.H2(f"${metrics.get('total_profit', 0):.2f}")
                ], style={'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'center'}),
                html.Div([
                    html.H4("Max Drawdown"),
                    html.H2(f"{metrics.get('max_drawdown', 0):.2f}%")
                ], style={'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'center'}),
            ]
            
            # Equity chart
            if equity_data:
                df = pd.DataFrame(equity_data)
                if 'time' in df.columns and 'equity' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    equity_fig = {
                        'data': [go.Scatter(
                            x=df['time'],
                            y=df['equity'],
                            mode='lines',
                            name='Equity',
                            line=dict(color='blue')
                        )],
                        'layout': go.Layout(
                            title='Equity Curve',
                            xaxis={'title': 'Time'},
                            yaxis={'title': 'Equity ($)'}
                        )
                    }
                else:
                    equity_fig = {'data': [], 'layout': {'title': 'Equity Curve - No Data'}}
            else:
                equity_fig = {'data': [], 'layout': {'title': 'Equity Curve - No Data'}}
            
            # Drawdown chart
            if equity_data:
                df = pd.DataFrame(equity_data)
                if 'time' in df.columns and 'equity' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.sort_values('time')
                    df['peak'] = df['equity'].expanding().max()
                    df['drawdown'] = ((df['equity'] - df['peak']) / df['peak']) * 100
                    
                    drawdown_fig = {
                        'data': [go.Scatter(
                            x=df['time'],
                            y=df['drawdown'],
                            mode='lines',
                            name='Drawdown',
                            fill='tozeroy',
                            line=dict(color='red')
                        )],
                        'layout': go.Layout(
                            title='Drawdown',
                            xaxis={'title': 'Time'},
                            yaxis={'title': 'Drawdown (%)'}
                        )
                    }
                else:
                    drawdown_fig = {'data': [], 'layout': {'title': 'Drawdown - No Data'}}
            else:
                drawdown_fig = {'data': [], 'layout': {'title': 'Drawdown - No Data'}}
            
            # Trades table
            if trades:
                df_trades = pd.DataFrame(trades[-20:])  # Last 20 trades
                trades_table = html.Table([
                    html.Thead([
                        html.Tr([html.Th(col) for col in ['Symbol', 'Entry Time', 'Exit Time', 'Profit', 'Strategy']])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(str(row.get('symbol', ''))),
                            html.Td(str(row.get('entry_time', ''))),
                            html.Td(str(row.get('exit_time', ''))),
                            html.Td(f"${float(row.get('profit', 0)):.2f}", style={'color': 'green' if float(row.get('profit', 0)) > 0 else 'red'}),
                            html.Td(str(row.get('strategy', '')))
                        ]) for _, row in df_trades.iterrows()
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse'})
            else:
                trades_table = html.P("No trades yet")
            
            return status_container, metrics_cards, equity_fig, drawdown_fig, trades_table
        
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}", exc_info=True)
            error_msg = html.Div(f"Error: {str(e)}", style={'color': 'red'})
            return error_msg, [], {'data': [], 'layout': {}}, {'data': [], 'layout': {}}, error_msg
    
    return app

