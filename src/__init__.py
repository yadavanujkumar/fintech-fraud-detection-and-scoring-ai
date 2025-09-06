"""
Initialization file for the main source package.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Fraud Detection Team"
__description__ = "AI-powered Fraud Detection and Risk Scoring Platform"

# Import main components
from .fraud_detection import FraudDetectionEngine
from .risk_scoring import RiskScoringEngine
from .graph_analytics import FraudNetworkAnalyzer

__all__ = [
    'FraudDetectionEngine',
    'RiskScoringEngine', 
    'FraudNetworkAnalyzer'
]