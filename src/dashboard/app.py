"""
Banking dashboard for fraud detection and monitoring.
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraud_detection.engine import FraudDetectionEngine
from fraud_detection.data_generator import TransactionDataGenerator
from risk_scoring.engine import RiskScoringEngine
from graph_analytics.network_analyzer import FraudNetworkAnalyzer

logger = logging.getLogger(__name__)

# Initialize components
fraud_engine = FraudDetectionEngine()
risk_engine = RiskScoringEngine()
network_analyzer = FraudNetworkAnalyzer()

# Generate sample data if needed
def get_sample_data():
    """Get or generate sample data for the dashboard."""
    try:
        # Try to load existing data
        df = pd.read_csv('data/sample_transactions.csv')
        logger.info("Loaded existing sample data")
    except:
        # Generate new data
        logger.info("Generating new sample data...")
        generator = TransactionDataGenerator()
        df = generator.generate_dataset(n_legitimate=1000, n_fraudulent=200)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/sample_transactions.csv', index=False)
        logger.info("Generated and saved sample data")
    
    return df

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Fraud Detection Dashboard"

# Load data and train models
sample_data = get_sample_data()

# Train fraud detection model
try:
    fraud_engine.train(sample_data)
    logger.info("Fraud detection models trained")
except Exception as e:
    logger.error(f"Error training models: {e}")

# Score transactions with risk engine
risk_scored_data = risk_engine.batch_score_transactions(sample_data)

# Build network
network_analyzer.build_transaction_network(sample_data)
fraud_rings = network_analyzer.detect_fraud_rings()
network_viz_data = network_analyzer.get_network_visualization_data()

# Define the layout
app.layout = html.Div([
    html.H1("Fraud Detection & Risk Scoring Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Summary cards
    html.Div([
        html.Div([
            html.H3("Total Transactions"),
            html.H2(f"{len(sample_data):,}", style={'color': '#3498db'})
        ], className='summary-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                           'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
        
        html.Div([
            html.H3("Fraudulent Transactions"),
            html.H2(f"{sample_data['is_fraud'].sum():,}", style={'color': '#e74c3c'})
        ], className='summary-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                           'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
        
        html.Div([
            html.H3("Fraud Rate"),
            html.H2(f"{sample_data['is_fraud'].mean():.1%}", style={'color': '#e67e22'})
        ], className='summary-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                           'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
        
        html.Div([
            html.H3("Fraud Rings Detected"),
            html.H2(f"{len(fraud_rings)}", style={'color': '#9b59b6'})
        ], className='summary-card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                           'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ], style={'marginBottom': 30}),
    
    # Tabs for different views
    dcc.Tabs(id='main-tabs', value='overview', children=[
        dcc.Tab(label='Overview', value='overview'),
        dcc.Tab(label='Risk Scoring', value='risk'),
        dcc.Tab(label='Network Analysis', value='network'),
        dcc.Tab(label='Real-time Monitoring', value='monitoring')
    ]),
    
    # Content area
    html.Div(id='tab-content')
])

@app.callback(Output('tab-content', 'children'),
              Input('main-tabs', 'value'))
def render_tab_content(active_tab):
    if active_tab == 'overview':
        return create_overview_tab()
    elif active_tab == 'risk':
        return create_risk_tab()
    elif active_tab == 'network':
        return create_network_tab()
    elif active_tab == 'monitoring':
        return create_monitoring_tab()

def create_overview_tab():
    """Create the overview tab with fraud statistics."""
    
    # Fraud by amount distribution
    fraud_by_amount = px.histogram(
        sample_data, 
        x='amount', 
        color='is_fraud',
        title='Transaction Amount Distribution by Fraud Status',
        labels={'is_fraud': 'Fraud Status', 'amount': 'Transaction Amount ($)'}
    )
    fraud_by_amount.update_layout(height=400)
    
    # Fraud by time
    sample_data['hour'] = pd.to_datetime(sample_data['timestamp']).dt.hour
    fraud_by_time = sample_data.groupby(['hour', 'is_fraud']).size().reset_index(name='count')
    
    time_chart = px.bar(
        fraud_by_time,
        x='hour',
        y='count',
        color='is_fraud',
        title='Fraud Transactions by Hour of Day',
        labels={'hour': 'Hour of Day', 'count': 'Number of Transactions', 'is_fraud': 'Fraud Status'}
    )
    time_chart.update_layout(height=400)
    
    # Fraud by merchant category
    merchant_fraud = sample_data.groupby('merchant_category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    merchant_fraud['fraud_rate'] = merchant_fraud['mean']
    merchant_fraud = merchant_fraud.sort_values('fraud_rate', ascending=False)
    
    merchant_chart = px.bar(
        merchant_fraud.head(10),
        x='merchant_category',
        y='fraud_rate',
        title='Fraud Rate by Merchant Category (Top 10)',
        labels={'merchant_category': 'Merchant Category', 'fraud_rate': 'Fraud Rate'}
    )
    merchant_chart.update_layout(height=400, xaxis_tickangle=-45)
    
    return html.Div([
        html.Div([
            dcc.Graph(figure=fraud_by_amount)
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=time_chart)
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=merchant_chart)
        ], style={'width': '100%', 'marginTop': 20})
    ])

def create_risk_tab():
    """Create the risk scoring tab."""
    
    # Risk score distribution
    risk_dist = px.histogram(
        risk_scored_data,
        x='risk_score',
        color='is_fraud',
        title='Risk Score Distribution',
        labels={'risk_score': 'Risk Score', 'is_fraud': 'Actual Fraud Status'}
    )
    risk_dist.update_layout(height=400)
    
    # Risk level distribution
    risk_level_counts = risk_scored_data['risk_level'].value_counts()
    risk_level_pie = px.pie(
        values=risk_level_counts.values,
        names=risk_level_counts.index,
        title='Risk Level Distribution'
    )
    risk_level_pie.update_layout(height=400)
    
    # High risk transactions table
    high_risk_transactions = risk_scored_data[risk_scored_data['risk_score'] >= 60].sort_values('risk_score', ascending=False).head(10)
    
    return html.Div([
        html.Div([
            dcc.Graph(figure=risk_dist)
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=risk_level_pie)
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("High Risk Transactions", style={'marginTop': 20}),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Transaction ID"),
                        html.Th("User ID"),
                        html.Th("Amount"),
                        html.Th("Risk Score"),
                        html.Th("Risk Level"),
                        html.Th("Actual Fraud")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(row['transaction_id']),
                        html.Td(row['user_id']),
                        html.Td(f"${row['amount']:.2f}"),
                        html.Td(f"{row['risk_score']:.1f}"),
                        html.Td(row['risk_level']),
                        html.Td("Yes" if row['is_fraud'] else "No")
                    ]) for _, row in high_risk_transactions.iterrows()
                ])
            ], style={'width': '100%', 'border': '1px solid #ddd'})
        ], style={'marginTop': 20})
    ])

def create_network_tab():
    """Create the network analysis tab."""
    
    # Network metrics
    network_metrics = network_analyzer.analyze_network_metrics()
    
    # Fraud rings summary
    rings_df = pd.DataFrame(fraud_rings) if fraud_rings else pd.DataFrame()
    
    return html.Div([
        html.H3("Network Analysis", style={'marginBottom': 20}),
        
        # Network metrics cards
        html.Div([
            html.Div([
                html.H4("Network Metrics"),
                html.P(f"Total Nodes: {network_metrics.get('total_nodes', 0)}"),
                html.P(f"Total Edges: {network_metrics.get('total_edges', 0)}"),
                html.P(f"Users: {network_metrics.get('user_count', 0)}"),
                html.P(f"Merchants: {network_metrics.get('merchant_count', 0)}"),
                html.P(f"Network Density: {network_metrics.get('density', 0):.4f}"),
                html.P(f"Connected Components: {network_metrics.get('connected_components', 0)}")
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px', 
                     'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '1%'}),
            
            html.Div([
                html.H4("Fraud Rings Detected"),
                html.P(f"Total Rings: {len(fraud_rings)}"),
                *[html.P(f"Ring {i+1}: {ring['size']} users, {ring['fraud_rate']:.1%} fraud rate, {ring['risk_level']} risk") 
                  for i, ring in enumerate(fraud_rings[:5])]
            ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px', 
                     'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '1%'})
        ]),
        
        # Network visualization placeholder
        html.Div([
            html.H4("Network Visualization"),
            html.P("Network visualization would be displayed here using a graph library like Cytoscape or D3.js"),
            html.P(f"Current network has {len(network_viz_data['nodes'])} nodes and {len(network_viz_data['edges'])} edges")
        ], style={'marginTop': 20, 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ])

def create_monitoring_tab():
    """Create the real-time monitoring tab."""
    
    # Recent high-risk transactions
    recent_high_risk = risk_scored_data[risk_scored_data['risk_score'] >= 50].sort_values('risk_score', ascending=False).head(20)
    
    # Alerts summary
    critical_alerts = len(risk_scored_data[risk_scored_data['risk_level'] == 'CRITICAL'])
    high_alerts = len(risk_scored_data[risk_scored_data['risk_level'] == 'HIGH'])
    medium_alerts = len(risk_scored_data[risk_scored_data['risk_level'] == 'MEDIUM'])
    
    return html.Div([
        html.H3("Real-time Monitoring", style={'marginBottom': 20}),
        
        # Alert summary
        html.Div([
            html.Div([
                html.H4("Critical Alerts", style={'color': '#e74c3c'}),
                html.H2(f"{critical_alerts}", style={'color': '#e74c3c'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px', 
                     'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '1%'}),
            
            html.Div([
                html.H4("High Risk Alerts", style={'color': '#e67e22'}),
                html.H2(f"{high_alerts}", style={'color': '#e67e22'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px', 
                     'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '1%'}),
            
            html.Div([
                html.H4("Medium Risk Alerts", style={'color': '#f39c12'}),
                html.H2(f"{medium_alerts}", style={'color': '#f39c12'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px', 
                     'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '1%'})
        ]),
        
        # Recent alerts table
        html.Div([
            html.H4("Recent High-Risk Transactions", style={'marginTop': 20}),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Time"),
                        html.Th("Transaction ID"),
                        html.Th("User ID"),
                        html.Th("Amount"),
                        html.Th("Risk Score"),
                        html.Th("Risk Level"),
                        html.Th("Risk Factors")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(row.get('timestamp', 'N/A')),
                        html.Td(row['transaction_id']),
                        html.Td(row['user_id']),
                        html.Td(f"${row['amount']:.2f}"),
                        html.Td(f"{row['risk_score']:.1f}"),
                        html.Td(row['risk_level'], 
                               style={'color': '#e74c3c' if row['risk_level'] == 'CRITICAL' 
                                     else '#e67e22' if row['risk_level'] == 'HIGH'
                                     else '#f39c12' if row['risk_level'] == 'MEDIUM'
                                     else '#27ae60'}),
                        html.Td(", ".join(row.get('risk_factors', [])))
                    ], style={'backgroundColor': '#fff5f5' if row['risk_level'] == 'CRITICAL' 
                             else '#fff8f0' if row['risk_level'] == 'HIGH'
                             else '#fffcf0' if row['risk_level'] == 'MEDIUM'
                             else 'white'}) for _, row in recent_high_risk.iterrows()
                ])
            ], style={'width': '100%', 'border': '1px solid #ddd'})
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)