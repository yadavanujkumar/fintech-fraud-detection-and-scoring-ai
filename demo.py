#!/usr/bin/env python3
"""
Comprehensive demo of the fraud detection and scoring AI platform.
This script demonstrates all major components working together.
"""

import sys
import os
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fraud_detection.data_generator import TransactionDataGenerator
from risk_scoring.engine import RiskScoringEngine
from graph_analytics.network_analyzer import FraudNetworkAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_banner(title):
    """Print a formatted banner."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def main():
    """Run the complete fraud detection system demo."""
    print_banner("FRAUD DETECTION & SCORING AI PLATFORM DEMO")
    
    # Step 1: Generate synthetic transaction data
    print_banner("1. GENERATING TRANSACTION DATA")
    generator = TransactionDataGenerator(seed=42)
    
    logger.info("Generating synthetic transaction dataset...")
    df = generator.generate_dataset(n_legitimate=500, n_fraudulent=100)
    
    print(f"📊 Generated {len(df)} transactions")
    print(f"💳 Legitimate transactions: {(df['is_fraud'] == 0).sum()}")
    print(f"🚨 Fraudulent transactions: {df['is_fraud'].sum()}")
    print(f"📈 Fraud rate: {df['is_fraud'].mean():.2%}")
    
    # Save data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/demo_transactions.csv', index=False)
    logger.info("Data saved to data/demo_transactions.csv")
    
    # Step 2: Risk Scoring Analysis
    print_banner("2. RISK SCORING ANALYSIS")
    
    risk_engine = RiskScoringEngine()
    logger.info("Performing risk scoring on transactions...")
    
    # Score all transactions
    risk_results = risk_engine.batch_score_transactions(df)
    
    # Analyze risk distribution
    risk_summary = risk_results['risk_level'].value_counts().sort_index()
    print("🎯 Risk Level Distribution:")
    for level, count in risk_summary.items():
        percentage = (count / len(risk_results)) * 100
        print(f"   {level}: {count} transactions ({percentage:.1f}%)")
    
    # Show high-risk transactions
    high_risk = risk_results[risk_results['risk_score'] >= 60].sort_values('risk_score', ascending=False)
    print(f"\n🔴 High-Risk Transactions: {len(high_risk)}")
    
    if len(high_risk) > 0:
        print("\nTop 5 Highest Risk Transactions:")
        for _, txn in high_risk.head(5).iterrows():
            print(f"   • {txn['transaction_id']}: ${txn['amount']:.2f} "
                  f"(Risk: {txn['risk_score']:.1f}, Level: {txn['risk_level']})")
            if txn['risk_factors']:
                print(f"     Factors: {', '.join(txn['risk_factors'])}")
    
    # Step 3: Network Analysis & Fraud Ring Detection
    print_banner("3. NETWORK ANALYSIS & FRAUD RING DETECTION")
    
    network_analyzer = FraudNetworkAnalyzer()
    logger.info("Building transaction network...")
    
    # Build network
    graph = network_analyzer.build_transaction_network(df)
    
    # Analyze network metrics
    metrics = network_analyzer.analyze_network_metrics()
    print("🕸️  Network Metrics:")
    print(f"   • Total Nodes: {metrics.get('total_nodes', 0)}")
    print(f"   • Total Edges: {metrics.get('total_edges', 0)}")
    print(f"   • Users: {metrics.get('user_count', 0)}")
    print(f"   • Merchants: {metrics.get('merchant_count', 0)}")
    print(f"   • Network Density: {metrics.get('density', 0):.4f}")
    print(f"   • Connected Components: {metrics.get('connected_components', 0)}")
    
    # Detect fraud rings
    logger.info("Detecting fraud rings...")
    fraud_rings = network_analyzer.detect_fraud_rings(min_ring_size=2, min_fraud_rate=0.4)
    
    print(f"\n🕵️  Fraud Ring Detection:")
    print(f"   • Fraud Rings Detected: {len(fraud_rings)}")
    
    for i, ring in enumerate(fraud_rings[:3], 1):
        print(f"   • Ring {i}: {ring['size']} users, "
              f"{ring['fraud_rate']:.1%} fraud rate, "
              f"{ring['risk_level']} risk")
    
    # Find suspicious patterns
    patterns = network_analyzer.find_suspicious_patterns()
    print(f"   • Suspicious Patterns: {len(patterns)}")
    
    for pattern in patterns[:3]:
        print(f"     - {pattern['pattern_type']}: {pattern['risk_level']} risk")
    
    # Step 4: Performance Analysis
    print_banner("4. FRAUD DETECTION PERFORMANCE ANALYSIS")
    
    # Compare actual fraud vs risk predictions
    actual_fraud = df['is_fraud'].sum()
    high_risk_count = len(risk_results[risk_results['risk_level'].isin(['HIGH', 'CRITICAL'])])
    
    print("📊 Detection Performance:")
    print(f"   • Actual Frauds: {actual_fraud}")
    print(f"   • High/Critical Risk Flagged: {high_risk_count}")
    
    # Calculate overlap
    high_risk_txns = risk_results[risk_results['risk_level'].isin(['HIGH', 'CRITICAL'])]
    true_positives = high_risk_txns['is_fraud'].sum()
    
    if high_risk_count > 0:
        precision = true_positives / high_risk_count
        print(f"   • Precision: {precision:.2%}")
    
    if actual_fraud > 0:
        recall = true_positives / actual_fraud
        print(f"   • Recall: {recall:.2%}")
    
    # Step 5: User Profile Analysis
    print_banner("5. USER PROFILE ANALYSIS")
    
    # Get some user profiles
    user_ids = df['user_id'].unique()[:5]
    print("👤 Sample User Risk Profiles:")
    
    for user_id in user_ids:
        profile = risk_engine.get_user_risk_profile(user_id)
        if 'error' not in profile:
            print(f"   • {user_id}: {profile['transaction_count']} transactions, "
                  f"${profile['avg_amount']:.2f} avg amount, "
                  f"Risk Score: {profile.get('user_risk_score', 0)}")
    
    # Step 6: Real-time Simulation
    print_banner("6. REAL-TIME FRAUD DETECTION SIMULATION")
    
    logger.info("Simulating real-time fraud detection...")
    
    # Create a few new transactions for real-time simulation
    new_transactions = [
        {
            'transaction_id': 'RT_001',
            'user_id': 'USER_9999',
            'amount': 5000.0,
            'timestamp': datetime.now(),
            'location': 'Unknown City',
            'merchant_category': 'cash_advance'
        },
        {
            'transaction_id': 'RT_002',
            'user_id': 'USER_0001',
            'amount': 25.0,
            'timestamp': datetime.now(),
            'location': 'New York',
            'merchant_category': 'grocery'
        }
    ]
    
    print("🚨 Real-time Transaction Analysis:")
    for txn in new_transactions:
        user_history = risk_engine.transaction_history.get(txn['user_id'], [])
        result = risk_engine.calculate_transaction_risk_score(txn, user_history)
        
        print(f"   • {txn['transaction_id']}: ${txn['amount']:.2f} -> "
              f"Risk: {result['risk_score']:.1f} ({result['risk_level']})")
        
        if result['risk_factors']:
            print(f"     Factors: {', '.join(result['risk_factors'])}")
        
        # Update user profile
        risk_engine.update_user_profile(txn['user_id'], txn)
    
    # Final Summary
    print_banner("DEMO SUMMARY")
    
    print("✅ Fraud Detection Platform Demo Completed Successfully!")
    print(f"📈 Processed {len(df)} transactions")
    print(f"🎯 Detected {len(fraud_rings)} fraud rings")
    print(f"⚠️  Identified {high_risk_count} high-risk transactions")
    print(f"👥 Analyzed {len(user_ids)} user profiles")
    print(f"📊 Generated comprehensive risk analysis")
    
    print("\n🚀 Next Steps:")
    print("   • Start the dashboard: python -m src.dashboard.app")
    print("   • Start the API server: python -m src.api.server")
    print("   • Run tests: python -m pytest tests/")
    print("   • View data: check data/demo_transactions.csv")
    
    print("\n📖 Documentation: See README.md for detailed usage instructions")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)