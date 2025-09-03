"""
Test suite for fraud detection engine.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection.engine import FraudDetectionEngine
from fraud_detection.data_generator import TransactionDataGenerator


class TestFraudDetectionEngine:
    """Test cases for FraudDetectionEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = FraudDetectionEngine(model_dir='test_models')
        self.generator = TransactionDataGenerator(seed=42)
        
        # Create test data
        self.test_data = self.generator.generate_dataset(
            n_legitimate=100, n_fraudulent=20
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        # Remove test model files if they exist
        import shutil
        if os.path.exists('test_models'):
            shutil.rmtree('test_models')
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality."""
        processed_data = self.engine.preprocess_data(self.test_data)
        
        # Check that new features are created
        assert 'log_amount' in processed_data.columns
        assert 'amount_zscore' in processed_data.columns
        
        # Check that missing values are handled
        assert processed_data.isnull().sum().sum() == 0
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        processed_data = self.engine.preprocess_data(self.test_data)
        features = self.engine.extract_features(processed_data)
        
        # Check that features are extracted
        assert len(features.columns) > 0
        assert features.dtypes.apply(lambda x: x in ['int64', 'float64']).all()
    
    def test_model_training(self):
        """Test model training."""
        results = self.engine.train(self.test_data)
        
        # Check that training completed
        assert 'isolation_forest_trained' in results
        assert results['isolation_forest_trained'] is True
        assert 'xgb_trained' in results
        assert results['xgb_trained'] is True
        assert self.engine.is_trained is True
    
    def test_predictions(self):
        """Test fraud predictions."""
        # Train models first
        self.engine.train(self.test_data)
        
        # Test prediction on subset
        test_subset = self.test_data.head(10)
        predictions = self.engine.predict(test_subset)
        
        # Check prediction structure
        assert 'isolation_forest' in predictions
        assert 'xgboost' in predictions
        
        # Check prediction arrays
        iso_pred = predictions['isolation_forest']['predictions']
        xgb_pred = predictions['xgboost']['predictions']
        
        assert len(iso_pred) == 10
        assert len(xgb_pred) == 10
        assert iso_pred.dtype in ['int64', 'int32', 'bool']
        assert xgb_pred.dtype in ['int64', 'int32', 'bool']
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train models
        self.engine.train(self.test_data)
        
        # Create new engine and load models
        new_engine = FraudDetectionEngine(model_dir='test_models')
        new_engine.load_models()
        
        assert new_engine.is_trained is True
        
        # Test that loaded models work
        test_subset = self.test_data.head(5)
        predictions = new_engine.predict(test_subset)
        
        assert 'isolation_forest' in predictions
        assert 'xgboost' in predictions


class TestTransactionDataGenerator:
    """Test cases for TransactionDataGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = TransactionDataGenerator(seed=42)
    
    def test_legitimate_transaction_generation(self):
        """Test generation of legitimate transactions."""
        transactions = self.generator.generate_legitimate_transactions(50)
        
        assert len(transactions) == 50
        
        # Check transaction structure
        for txn in transactions:
            assert 'transaction_id' in txn
            assert 'user_id' in txn
            assert 'amount' in txn
            assert 'is_fraud' in txn
            assert txn['is_fraud'] == 0
    
    def test_fraudulent_transaction_generation(self):
        """Test generation of fraudulent transactions."""
        transactions = self.generator.generate_fraudulent_transactions(20)
        
        assert len(transactions) == 20
        
        # Check transaction structure
        for txn in transactions:
            assert 'transaction_id' in txn
            assert 'user_id' in txn
            assert 'amount' in txn
            assert 'is_fraud' in txn
            assert txn['is_fraud'] == 1
    
    def test_dataset_generation(self):
        """Test complete dataset generation."""
        df = self.generator.generate_dataset(n_legitimate=80, n_fraudulent=20)
        
        assert len(df) == 100
        assert df['is_fraud'].sum() == 20
        assert (df['is_fraud'] == 0).sum() == 80
        
        # Check required columns
        required_cols = ['transaction_id', 'user_id', 'amount', 'is_fraud']
        for col in required_cols:
            assert col in df.columns
    
    def test_data_quality(self):
        """Test data quality and consistency."""
        df = self.generator.generate_dataset(n_legitimate=100, n_fraudulent=50)
        
        # Check for missing values
        assert df.isnull().sum().sum() == 0
        
        # Check data types
        assert df['amount'].dtype in ['float64', 'int64']
        assert df['is_fraud'].dtype in ['int64', 'int32', 'bool']
        
        # Check amount range
        assert df['amount'].min() > 0
        assert df['amount'].max() < 50000  # Reasonable upper bound


if __name__ == '__main__':
    pytest.main([__file__])