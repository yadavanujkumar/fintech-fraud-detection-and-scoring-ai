"""
Graph analytics for fraud ring detection and network analysis.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)


class FraudNetworkAnalyzer:
    """
    Analyzes transaction networks to detect fraud rings and suspicious patterns.
    """
    
    def __init__(self):
        self.transaction_graph = nx.Graph()
        self.user_graph = nx.Graph()
        self.merchant_graph = nx.Graph()
        self.suspicious_patterns = []
    
    def build_transaction_network(self, transactions_df: pd.DataFrame) -> nx.Graph:
        """
        Build a network graph from transaction data.
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            NetworkX graph representing transaction relationships
        """
        logger.info("Building transaction network...")
        
        # Clear existing graph
        self.transaction_graph.clear()
        
        # Add nodes for users and merchants
        users = set(transactions_df['user_id'].unique())
        merchants = set(transactions_df.get('merchant_id', transactions_df.get('merchant_category', [])).unique())
        
        # Add user nodes
        for user in users:
            user_transactions = transactions_df[transactions_df['user_id'] == user]
            fraud_count = user_transactions['is_fraud'].sum() if 'is_fraud' in user_transactions.columns else 0
            
            self.transaction_graph.add_node(
                user,
                node_type='user',
                transaction_count=len(user_transactions),
                total_amount=user_transactions['amount'].sum(),
                fraud_count=fraud_count,
                fraud_rate=fraud_count / len(user_transactions) if len(user_transactions) > 0 else 0
            )
        
        # Add merchant nodes
        merchant_col = 'merchant_id' if 'merchant_id' in transactions_df.columns else 'merchant_category'
        for merchant in merchants:
            if pd.isna(merchant):
                continue
                
            merchant_transactions = transactions_df[transactions_df[merchant_col] == merchant]
            fraud_count = merchant_transactions['is_fraud'].sum() if 'is_fraud' in merchant_transactions.columns else 0
            
            self.transaction_graph.add_node(
                f"merchant_{merchant}",
                node_type='merchant',
                transaction_count=len(merchant_transactions),
                total_amount=merchant_transactions['amount'].sum(),
                fraud_count=fraud_count,
                fraud_rate=fraud_count / len(merchant_transactions) if len(merchant_transactions) > 0 else 0
            )
        
        # Add edges between users and merchants
        for _, transaction in transactions_df.iterrows():
            user = transaction['user_id']
            merchant = f"merchant_{transaction[merchant_col]}"
            amount = transaction['amount']
            is_fraud = transaction.get('is_fraud', 0)
            
            if self.transaction_graph.has_edge(user, merchant):
                # Update existing edge
                edge_data = self.transaction_graph[user][merchant]
                edge_data['transaction_count'] += 1
                edge_data['total_amount'] += amount
                edge_data['fraud_count'] += is_fraud
            else:
                # Add new edge
                self.transaction_graph.add_edge(
                    user, merchant,
                    transaction_count=1,
                    total_amount=amount,
                    fraud_count=is_fraud,
                    weight=amount
                )
        
        logger.info(f"Built network with {self.transaction_graph.number_of_nodes()} nodes and {self.transaction_graph.number_of_edges()} edges")
        return self.transaction_graph
    
    def detect_fraud_rings(self, min_ring_size: int = 3, min_fraud_rate: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect potential fraud rings using community detection and pattern analysis.
        
        Args:
            min_ring_size: Minimum number of users in a fraud ring
            min_fraud_rate: Minimum fraud rate to consider a ring suspicious
            
        Returns:
            List of detected fraud rings with details
        """
        logger.info("Detecting fraud rings...")
        
        fraud_rings = []
        
        # Find connected components of users (exclude merchants for ring detection)
        user_subgraph = self.transaction_graph.subgraph([
            node for node, data in self.transaction_graph.nodes(data=True)
            if data.get('node_type') == 'user'
        ])
        
        # Analyze each connected component
        for component in nx.connected_components(user_subgraph):
            if len(component) >= min_ring_size:
                # Calculate component statistics
                component_nodes = list(component)
                total_fraud_count = sum(
                    self.transaction_graph.nodes[node].get('fraud_count', 0)
                    for node in component_nodes
                )
                total_transactions = sum(
                    self.transaction_graph.nodes[node].get('transaction_count', 0)
                    for node in component_nodes
                )
                
                if total_transactions > 0:
                    fraud_rate = total_fraud_count / total_transactions
                    
                    if fraud_rate >= min_fraud_rate:
                        # This looks like a fraud ring
                        ring_info = {
                            'ring_id': f"ring_{len(fraud_rings) + 1}",
                            'users': component_nodes,
                            'size': len(component_nodes),
                            'total_transactions': total_transactions,
                            'fraud_transactions': total_fraud_count,
                            'fraud_rate': round(fraud_rate, 3),
                            'total_amount': sum(
                                self.transaction_graph.nodes[node].get('total_amount', 0)
                                for node in component_nodes
                            ),
                            'risk_level': self._calculate_ring_risk_level(fraud_rate, len(component_nodes))
                        }
                        
                        fraud_rings.append(ring_info)
        
        # Detect rings based on shared merchants
        shared_merchant_rings = self._detect_shared_merchant_rings(min_ring_size, min_fraud_rate)
        fraud_rings.extend(shared_merchant_rings)
        
        # Detect rings based on timing patterns
        timing_rings = self._detect_timing_based_rings(min_ring_size)
        fraud_rings.extend(timing_rings)
        
        logger.info(f"Detected {len(fraud_rings)} potential fraud rings")
        return fraud_rings
    
    def _detect_shared_merchant_rings(self, min_ring_size: int, min_fraud_rate: float) -> List[Dict[str, Any]]:
        """Detect fraud rings based on users sharing the same merchants."""
        rings = []
        
        # Group users by merchants they transact with
        merchant_users = defaultdict(set)
        
        for user, merchant, data in self.transaction_graph.edges(data=True):
            if self.transaction_graph.nodes[user].get('node_type') == 'user':
                merchant_users[merchant].add(user)
        
        # Find merchants with suspicious user groups
        for merchant, users in merchant_users.items():
            if len(users) >= min_ring_size:
                # Calculate fraud rate for this group
                total_fraud = sum(
                    self.transaction_graph.nodes[user].get('fraud_count', 0)
                    for user in users
                )
                total_transactions = sum(
                    self.transaction_graph.nodes[user].get('transaction_count', 0)
                    for user in users
                )
                
                if total_transactions > 0:
                    fraud_rate = total_fraud / total_transactions
                    
                    if fraud_rate >= min_fraud_rate:
                        ring_info = {
                            'ring_id': f"merchant_ring_{len(rings) + 1}",
                            'type': 'shared_merchant',
                            'users': list(users),
                            'shared_merchant': merchant,
                            'size': len(users),
                            'fraud_rate': round(fraud_rate, 3),
                            'risk_level': self._calculate_ring_risk_level(fraud_rate, len(users))
                        }
                        rings.append(ring_info)
        
        return rings
    
    def _detect_timing_based_rings(self, min_ring_size: int) -> List[Dict[str, Any]]:
        """Detect fraud rings based on synchronized transaction timing."""
        # This is a simplified implementation
        # In practice, you'd analyze transaction timestamps more thoroughly
        rings = []
        
        # For now, return empty list - can be enhanced with actual timing analysis
        return rings
    
    def _calculate_ring_risk_level(self, fraud_rate: float, ring_size: int) -> str:
        """Calculate risk level for a detected ring."""
        risk_score = fraud_rate * 100 + ring_size * 5
        
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        else:
            return "LOW"
    
    def analyze_network_metrics(self) -> Dict[str, Any]:
        """
        Analyze various network metrics to identify suspicious patterns.
        
        Returns:
            Dictionary with network analysis results
        """
        if self.transaction_graph.number_of_nodes() == 0:
            return {'error': 'No network data available'}
        
        metrics = {}
        
        # Basic network metrics
        metrics['total_nodes'] = self.transaction_graph.number_of_nodes()
        metrics['total_edges'] = self.transaction_graph.number_of_edges()
        metrics['density'] = nx.density(self.transaction_graph)
        
        # User and merchant counts
        user_nodes = [n for n, d in self.transaction_graph.nodes(data=True) if d.get('node_type') == 'user']
        merchant_nodes = [n for n, d in self.transaction_graph.nodes(data=True) if d.get('node_type') == 'merchant']
        
        metrics['user_count'] = len(user_nodes)
        metrics['merchant_count'] = len(merchant_nodes)
        
        # Degree analysis
        degrees = dict(self.transaction_graph.degree())
        if degrees:
            metrics['avg_degree'] = np.mean(list(degrees.values()))
            metrics['max_degree'] = max(degrees.values())
            metrics['min_degree'] = min(degrees.values())
        
        # Connected components
        components = list(nx.connected_components(self.transaction_graph))
        metrics['connected_components'] = len(components)
        metrics['largest_component_size'] = len(max(components, key=len)) if components else 0
        
        # Centrality measures for top nodes
        try:
            betweenness = nx.betweenness_centrality(self.transaction_graph)
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
            metrics['top_betweenness_nodes'] = [(node, round(score, 4)) for node, score in top_betweenness]
        except:
            metrics['top_betweenness_nodes'] = []
        
        try:
            closeness = nx.closeness_centrality(self.transaction_graph)
            top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
            metrics['top_closeness_nodes'] = [(node, round(score, 4)) for node, score in top_closeness]
        except:
            metrics['top_closeness_nodes'] = []
        
        return metrics
    
    def find_suspicious_patterns(self) -> List[Dict[str, Any]]:
        """
        Find various suspicious patterns in the transaction network.
        
        Returns:
            List of suspicious patterns detected
        """
        patterns = []
        
        # Pattern 1: Highly connected users (potential money mules)
        user_degrees = {
            node: degree for node, degree in self.transaction_graph.degree()
            if self.transaction_graph.nodes[node].get('node_type') == 'user'
        }
        
        if user_degrees:
            avg_degree = np.mean(list(user_degrees.values()))
            std_degree = np.std(list(user_degrees.values()))
            threshold = avg_degree + 2 * std_degree
            
            for user, degree in user_degrees.items():
                if degree > threshold:
                    fraud_rate = self.transaction_graph.nodes[user].get('fraud_rate', 0)
                    patterns.append({
                        'pattern_type': 'highly_connected_user',
                        'user': user,
                        'degree': degree,
                        'fraud_rate': fraud_rate,
                        'risk_level': 'HIGH' if fraud_rate > 0.3 else 'MEDIUM'
                    })
        
        # Pattern 2: Merchants with unusually high fraud rates
        merchant_nodes = [
            node for node, data in self.transaction_graph.nodes(data=True)
            if data.get('node_type') == 'merchant'
        ]
        
        for merchant in merchant_nodes:
            fraud_rate = self.transaction_graph.nodes[merchant].get('fraud_rate', 0)
            transaction_count = self.transaction_graph.nodes[merchant].get('transaction_count', 0)
            
            if fraud_rate > 0.5 and transaction_count >= 5:
                patterns.append({
                    'pattern_type': 'high_fraud_merchant',
                    'merchant': merchant,
                    'fraud_rate': fraud_rate,
                    'transaction_count': transaction_count,
                    'risk_level': 'CRITICAL' if fraud_rate > 0.8 else 'HIGH'
                })
        
        # Pattern 3: Isolated high-value transactions
        for node in self.transaction_graph.nodes():
            if self.transaction_graph.nodes[node].get('node_type') == 'user':
                degree = self.transaction_graph.degree(node)
                total_amount = self.transaction_graph.nodes[node].get('total_amount', 0)
                
                if degree == 1 and total_amount > 10000:  # Single large transaction
                    patterns.append({
                        'pattern_type': 'isolated_high_value',
                        'user': node,
                        'amount': total_amount,
                        'risk_level': 'MEDIUM'
                    })
        
        return patterns
    
    def get_network_visualization_data(self) -> Dict[str, Any]:
        """
        Prepare network data for visualization.
        
        Returns:
            Dictionary with nodes and edges data for visualization
        """
        nodes = []
        edges = []
        
        # Prepare nodes
        for node, data in self.transaction_graph.nodes(data=True):
            node_info = {
                'id': node,
                'type': data.get('node_type', 'unknown'),
                'transaction_count': data.get('transaction_count', 0),
                'total_amount': data.get('total_amount', 0),
                'fraud_count': data.get('fraud_count', 0),
                'fraud_rate': data.get('fraud_rate', 0),
                'size': min(max(data.get('transaction_count', 1), 5), 50)  # Node size for visualization
            }
            nodes.append(node_info)
        
        # Prepare edges
        for source, target, data in self.transaction_graph.edges(data=True):
            edge_info = {
                'source': source,
                'target': target,
                'transaction_count': data.get('transaction_count', 0),
                'total_amount': data.get('total_amount', 0),
                'fraud_count': data.get('fraud_count', 0),
                'weight': data.get('weight', 1)
            }
            edges.append(edge_info)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'user_nodes': len([n for n in nodes if n['type'] == 'user']),
                'merchant_nodes': len([n for n in nodes if n['type'] == 'merchant'])
            }
        }