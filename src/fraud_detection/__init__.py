"""
Fraud detection module initialization.
"""

# Import only data generator for now to avoid ML dependencies
from .data_generator import TransactionDataGenerator, create_sample_data

__all__ = ['TransactionDataGenerator', 'create_sample_data']

# Conditional import for ML components
try:
    from .engine import FraudDetectionEngine
    __all__.append('FraudDetectionEngine')
except ImportError:
    pass