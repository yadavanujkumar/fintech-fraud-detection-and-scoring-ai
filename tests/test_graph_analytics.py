"""
Test suite for graph analytics.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graph_analytics.network_analyzer import FraudNetworkAnalyzer
from fraud_detection.data_generator import TransactionDataGenerator


class TestFraudNetworkAnalyzer:
    """Test cases for FraudNetworkAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = FraudNetworkAnalyzer()
        self.generator = TransactionDataGenerator(seed=42)
        
        # Create test data with merchant_id column
        self.test_data = self.generator.generate_dataset(
            n_legitimate=100, n_fraudulent=20
        )
        # Add merchant_id column
        self.test_data['merchant_id'] = self.test_data['merchant_category'] + '_' + \
                                       pd.Series(range(len(self.test_data))).astype(str)
    
    def test_network_building(self):
        """Test transaction network building."""
        graph = self.analyzer.build_transaction_network(self.test_data)
        
        # Check that graph was created
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        
        # Check node types
        node_types = [data.get('node_type') for _, data in graph.nodes(data=True)]
        assert 'user' in node_types
        assert 'merchant' in node_types
    
    def test_network_metrics(self):
        """Test network metrics calculation."""
        self.analyzer.build_transaction_network(self.test_data)
        metrics = self.analyzer.analyze_network_metrics()
        
        # Check required metrics
        required_metrics = [
            'total_nodes', 'total_edges', 'density',
            'user_count', 'merchant_count', 'connected_components'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_fraud_ring_detection(self):
        """Test fraud ring detection."""
        self.analyzer.build_transaction_network(self.test_data)
        fraud_rings = self.analyzer.detect_fraud_rings(min_ring_size=2, min_fraud_rate=0.3)
        
        # Check result structure
        assert isinstance(fraud_rings, list)
        
        for ring in fraud_rings:
            assert 'ring_id' in ring
            assert 'users' in ring
            assert 'size' in ring
            assert 'fraud_rate' in ring
            assert 'risk_level' in ring
            assert ring['size'] >= 2
            assert 0 <= ring['fraud_rate'] <= 1
    
    def test_suspicious_patterns(self):
        """Test suspicious pattern detection."""
        self.analyzer.build_transaction_network(self.test_data)
        patterns = self.analyzer.find_suspicious_patterns()
        
        # Check result structure
        assert isinstance(patterns, list)
        
        for pattern in patterns:
            assert 'pattern_type' in pattern
            assert 'risk_level' in pattern
    
    def test_visualization_data(self):
        """Test network visualization data preparation."""
        self.analyzer.build_transaction_network(self.test_data)
        viz_data = self.analyzer.get_network_visualization_data()
        
        # Check structure
        assert 'nodes' in viz_data
        assert 'edges' in viz_data
        assert 'stats' in viz_data
        
        # Check nodes structure
        if viz_data['nodes']:
            node = viz_data['nodes'][0]
            assert 'id' in node
            assert 'type' in node
            assert 'transaction_count' in node
        
        # Check edges structure
        if viz_data['edges']:
            edge = viz_data['edges'][0]
            assert 'source' in edge
            assert 'target' in edge
            assert 'transaction_count' in edge
    
    def test_empty_network(self):
        """Test behavior with empty network."""
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        graph = self.analyzer.build_transaction_network(empty_df)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
        
        # Metrics should handle empty network
        metrics = self.analyzer.analyze_network_metrics()
        assert 'error' in metrics or metrics['total_nodes'] == 0
    
    def test_shared_merchant_rings(self):
        """Test detection of rings based on shared merchants."""
        # Create data with users sharing merchants
        test_data = []
        
        # Create a group of users that all use the same merchant
        shared_merchant = "merchant_shared_001"
        for i in range(3):
            test_data.append({
                'transaction_id': f'TXN_{i:03d}',
                'user_id': f'USER_{i:03d}',
                'amount': 1000 + i * 100,
                'merchant_category': 'gambling',
                'merchant_id': shared_merchant,
                'is_fraud': 1,  # All fraudulent
                'timestamp': '2023-01-01',
                'location': 'Test City'
            })
        
        df = pd.DataFrame(test_data)
        self.analyzer.build_transaction_network(df)
        
        fraud_rings = self.analyzer.detect_fraud_rings(min_ring_size=2, min_fraud_rate=0.5)
        
        # Should detect the shared merchant ring
        assert len(fraud_rings) > 0
        
        # Check for shared merchant ring type
        shared_rings = [ring for ring in fraud_rings if ring.get('type') == 'shared_merchant']
        assert len(shared_rings) > 0


if __name__ == '__main__':
    pytest.main([__file__])