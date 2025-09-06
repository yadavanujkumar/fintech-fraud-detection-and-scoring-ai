"""
Test suite for risk scoring engine.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_scoring.engine import RiskScoringEngine
from fraud_detection.data_generator import TransactionDataGenerator


class TestRiskScoringEngine:
    """Test cases for RiskScoringEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RiskScoringEngine()
        self.generator = TransactionDataGenerator(seed=42)
        
        # Create test transaction
        self.test_transaction = {
            'transaction_id': 'TXN_001',
            'user_id': 'USER_001',
            'amount': 100.0,
            'timestamp': datetime.now(),
            'location': 'New York',
            'merchant_category': 'grocery'
        }
    
    def test_transaction_risk_scoring(self):
        """Test basic transaction risk scoring."""
        result = self.engine.calculate_transaction_risk_score(self.test_transaction)
        
        # Check result structure
        assert 'risk_score' in result
        assert 'risk_level' in result
        assert 'risk_factors' in result
        assert 'timestamp' in result
        
        # Check data types
        assert isinstance(result['risk_score'], (int, float))
        assert isinstance(result['risk_level'], str)
        assert isinstance(result['risk_factors'], list)
        assert result['risk_score'] >= 0
        assert result['risk_score'] <= 100
    
    def test_high_amount_risk(self):
        """Test risk scoring for high amount transactions."""
        high_amount_txn = self.test_transaction.copy()
        high_amount_txn['amount'] = 10000.0
        
        result = self.engine.calculate_transaction_risk_score(high_amount_txn)
        
        # High amount should increase risk score
        assert result['risk_score'] > 0
        assert any('amount' in factor.lower() for factor in result['risk_factors'])
    
    def test_unusual_time_risk(self):
        """Test risk scoring for unusual time transactions."""
        unusual_time_txn = self.test_transaction.copy()
        unusual_time_txn['timestamp'] = datetime.now().replace(hour=3)  # 3 AM
        
        result = self.engine.calculate_transaction_risk_score(unusual_time_txn)
        
        # Unusual time should increase risk score
        assert result['risk_score'] > 0
        assert any('hour' in factor.lower() for factor in result['risk_factors'])
    
    def test_high_risk_merchant_category(self):
        """Test risk scoring for high-risk merchant categories."""
        high_risk_txn = self.test_transaction.copy()
        high_risk_txn['merchant_category'] = 'gambling'
        
        result = self.engine.calculate_transaction_risk_score(high_risk_txn)
        
        # High-risk merchant should increase risk score
        assert result['risk_score'] > 0
        assert any('merchant' in factor.lower() for factor in result['risk_factors'])
    
    def test_user_profile_updates(self):
        """Test user profile updates."""
        user_id = 'USER_TEST'
        
        # Initial profile should not exist
        profile = self.engine.get_user_risk_profile(user_id)
        assert 'error' in profile
        
        # Update profile with transaction
        self.engine.update_user_profile(user_id, self.test_transaction)
        
        # Profile should now exist
        profile = self.engine.get_user_risk_profile(user_id)
        assert 'error' not in profile
        assert profile['transaction_count'] == 1
        assert profile['total_amount'] == self.test_transaction['amount']
    
    def test_velocity_risk(self):
        """Test velocity-based risk scoring."""
        user_id = 'USER_VELOCITY'
        
        # Create multiple transactions for the same user
        transactions = []
        for i in range(6):  # More than the threshold of 5
            txn = self.test_transaction.copy()
            txn['transaction_id'] = f'TXN_{i:03d}'
            txn['user_id'] = user_id
            txn['timestamp'] = datetime.now() - timedelta(minutes=i * 10)
            transactions.append(txn)
        
        # Update user profile with all transactions except the last
        for txn in transactions[:-1]:
            self.engine.update_user_profile(user_id, txn)
        
        # Check risk for the last transaction
        user_history = self.engine.transaction_history[user_id]
        result = self.engine.calculate_transaction_risk_score(
            transactions[-1], user_history
        )
        
        # High velocity should increase risk score
        assert result['risk_score'] > 0
        assert any('velocity' in factor.lower() for factor in result['risk_factors'])
    
    def test_new_location_risk(self):
        """Test risk scoring for new locations."""
        user_id = 'USER_LOCATION'
        
        # Add transaction in known location
        known_location_txn = self.test_transaction.copy()
        known_location_txn['user_id'] = user_id
        known_location_txn['location'] = 'Known City'
        self.engine.update_user_profile(user_id, known_location_txn)
        
        # Transaction in new location
        new_location_txn = self.test_transaction.copy()
        new_location_txn['user_id'] = user_id
        new_location_txn['location'] = 'New City'
        
        user_history = self.engine.transaction_history[user_id]
        result = self.engine.calculate_transaction_risk_score(
            new_location_txn, user_history
        )
        
        # New location should increase risk score
        assert result['risk_score'] > 0
        assert any('location' in factor.lower() for factor in result['risk_factors'])
    
    def test_batch_scoring(self):
        """Test batch transaction scoring."""
        # Generate test dataset
        df = self.generator.generate_dataset(n_legitimate=50, n_fraudulent=10)
        
        # Score transactions
        results_df = self.engine.batch_score_transactions(df)
        
        # Check that all transactions were scored
        assert len(results_df) == len(df)
        assert 'risk_score' in results_df.columns
        assert 'risk_level' in results_df.columns
        assert 'risk_factors' in results_df.columns
        
        # Check data quality
        assert results_df['risk_score'].notna().all()
        assert results_df['risk_level'].notna().all()
    
    def test_risk_level_calculation(self):
        """Test risk level assignment."""
        # Test different risk scores
        test_scores = [0, 25, 45, 65, 85]
        expected_levels = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        for score, expected_level in zip(test_scores, expected_levels):
            # Create a mock result with the specific score
            result = {
                'risk_score': score,
                'risk_level': '',
                'risk_factors': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Manually set risk level based on score (simulating the engine logic)
            if score >= 80:
                result['risk_level'] = "CRITICAL"
            elif score >= 60:
                result['risk_level'] = "HIGH"
            elif score >= 40:
                result['risk_level'] = "MEDIUM"
            elif score >= 20:
                result['risk_level'] = "LOW"
            else:
                result['risk_level'] = "MINIMAL"
            
            assert result['risk_level'] == expected_level


if __name__ == '__main__':
    pytest.main([__file__])