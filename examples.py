"""
Example usage of the fraud detection API.
This script shows how to interact with the system programmatically.
"""

import sys
import os
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fraud_detection.data_generator import TransactionDataGenerator
from risk_scoring.engine import RiskScoringEngine
from graph_analytics.network_analyzer import FraudNetworkAnalyzer


def example_single_transaction_check():
    """Example: Check a single transaction for fraud risk."""
    print("=== Single Transaction Fraud Check ===")
    
    # Initialize risk engine
    risk_engine = RiskScoringEngine()
    
    # Example transaction
    transaction = {
        'transaction_id': 'TXN_EXAMPLE_001',
        'user_id': 'USER_12345',
        'amount': 2500.0,
        'timestamp': datetime.now(),
        'location': 'San Francisco',
        'merchant_category': 'online'
    }
    
    # Check risk
    result = risk_engine.calculate_transaction_risk_score(transaction)
    
    print(f"Transaction ID: {transaction['transaction_id']}")
    print(f"Amount: ${transaction['amount']:.2f}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Risk Factors: {', '.join(result['risk_factors']) if result['risk_factors'] else 'None'}")
    print()


def example_batch_processing():
    """Example: Process multiple transactions in batch."""
    print("=== Batch Transaction Processing ===")
    
    # Generate sample data
    generator = TransactionDataGenerator()
    df = generator.generate_dataset(n_legitimate=50, n_fraudulent=10)
    
    # Initialize engines
    risk_engine = RiskScoringEngine()
    
    # Process batch
    results = risk_engine.batch_score_transactions(df)
    
    # Show summary
    print(f"Processed {len(results)} transactions")
    print(f"High risk transactions: {len(results[results['risk_level'] == 'HIGH'])}")
    print(f"Critical risk transactions: {len(results[results['risk_level'] == 'CRITICAL'])}")
    
    # Show top 3 highest risk
    top_risk = results.nlargest(3, 'risk_score')
    print("\nTop 3 Highest Risk Transactions:")
    for _, txn in top_risk.iterrows():
        print(f"  {txn['transaction_id']}: ${txn['amount']:.2f} "
              f"(Risk: {txn['risk_score']:.1f})")
    print()


def example_network_analysis():
    """Example: Analyze transaction network for fraud rings."""
    print("=== Network Analysis for Fraud Rings ===")
    
    # Generate data with potential fraud rings
    generator = TransactionDataGenerator()
    df = generator.generate_dataset(n_legitimate=100, n_fraudulent=30)
    
    # Initialize network analyzer
    analyzer = FraudNetworkAnalyzer()
    
    # Build network
    graph = analyzer.build_transaction_network(df)
    
    # Get metrics
    metrics = analyzer.analyze_network_metrics()
    print(f"Network has {metrics['total_nodes']} nodes and {metrics['total_edges']} edges")
    print(f"Network density: {metrics['density']:.4f}")
    
    # Detect fraud rings
    fraud_rings = analyzer.detect_fraud_rings()
    print(f"Detected {len(fraud_rings)} potential fraud rings")
    
    for i, ring in enumerate(fraud_rings[:3], 1):
        print(f"  Ring {i}: {ring['size']} users, "
              f"{ring['fraud_rate']:.1%} fraud rate, "
              f"{ring['risk_level']} risk")
    print()


def example_user_profile_tracking():
    """Example: Track user behavior over time."""
    print("=== User Profile Tracking ===")
    
    risk_engine = RiskScoringEngine()
    user_id = "USER_EXAMPLE"
    
    # Simulate a series of transactions for a user
    transactions = [
        {'transaction_id': 'T1', 'user_id': user_id, 'amount': 50.0, 
         'location': 'New York', 'merchant_category': 'grocery'},
        {'transaction_id': 'T2', 'user_id': user_id, 'amount': 75.0, 
         'location': 'New York', 'merchant_category': 'gas'},
        {'transaction_id': 'T3', 'user_id': user_id, 'amount': 2000.0, 
         'location': 'Las Vegas', 'merchant_category': 'gambling'},  # Suspicious
    ]
    
    print(f"Tracking user {user_id} over {len(transactions)} transactions:")
    
    for i, txn in enumerate(transactions, 1):
        # Get current user history
        user_history = risk_engine.transaction_history.get(user_id, [])
        
        # Score transaction
        result = risk_engine.calculate_transaction_risk_score(txn, user_history)
        
        print(f"  Transaction {i}: ${txn['amount']:.2f} in {txn['location']}")
        print(f"    Risk Score: {result['risk_score']:.1f} ({result['risk_level']})")
        if result['risk_factors']:
            print(f"    Factors: {', '.join(result['risk_factors'])}")
        
        # Update profile
        risk_engine.update_user_profile(user_id, txn)
    
    # Show final user profile
    profile = risk_engine.get_user_risk_profile(user_id)
    print(f"\nFinal Profile for {user_id}:")
    print(f"  Total transactions: {profile['transaction_count']}")
    print(f"  Average amount: ${profile['avg_amount']:.2f}")
    print(f"  Locations visited: {len(profile['locations'])}")
    print(f"  User risk score: {profile['user_risk_score']}")
    print()


def example_real_time_monitoring():
    """Example: Real-time transaction monitoring."""
    print("=== Real-time Transaction Monitoring ===")
    
    risk_engine = RiskScoringEngine()
    
    # Simulate incoming transactions
    incoming_transactions = [
        {'transaction_id': 'RT1', 'user_id': 'USER_A', 'amount': 100.0, 
         'timestamp': datetime.now(), 'location': 'Boston', 'merchant_category': 'retail'},
        {'transaction_id': 'RT2', 'user_id': 'USER_B', 'amount': 5000.0, 
         'timestamp': datetime.now(), 'location': 'Unknown', 'merchant_category': 'cash_advance'},
        {'transaction_id': 'RT3', 'user_id': 'USER_C', 'amount': 25.0, 
         'timestamp': datetime.now(), 'location': 'Chicago', 'merchant_category': 'grocery'},
    ]
    
    alerts = []
    
    print("Processing incoming transactions:")
    for txn in incoming_transactions:
        user_history = risk_engine.transaction_history.get(txn['user_id'], [])
        result = risk_engine.calculate_transaction_risk_score(txn, user_history)
        
        status = "🚨 ALERT" if result['risk_score'] >= 40 else "✅ OK"
        print(f"  {status} {txn['transaction_id']}: ${txn['amount']:.2f} "
              f"-> Risk: {result['risk_score']:.1f}")
        
        if result['risk_score'] >= 40:
            alerts.append({
                'transaction': txn,
                'risk_result': result
            })
        
        # Update user profile
        risk_engine.update_user_profile(txn['user_id'], txn)
    
    print(f"\nGenerated {len(alerts)} alerts for manual review")
    print()


def main():
    """Run all examples."""
    print("FRAUD DETECTION SYSTEM - USAGE EXAMPLES")
    print("=" * 50)
    print()
    
    example_single_transaction_check()
    example_batch_processing()
    example_network_analysis()
    example_user_profile_tracking()
    example_real_time_monitoring()
    
    print("All examples completed successfully!")
    print("\nFor more advanced usage, see:")
    print("  • Dashboard: python -m src.dashboard.app")
    print("  • API Server: python -m src.api.server")
    print("  • Full Demo: python demo.py")


if __name__ == "__main__":
    main()