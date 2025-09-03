"""
Example API client for the fraud detection system.
This script demonstrates how to interact with the API endpoints.
"""

import requests
import json
from datetime import datetime


# API base URL (assuming server is running on localhost:5000)
BASE_URL = "http://localhost:5000"


def test_health_check():
    """Test the health check endpoint."""
    print("=== Health Check ===")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Service is healthy")
            print(f"Fraud Detection: {'✅' if data['services']['fraud_detection'] else '❌'}")
            print(f"Risk Scoring: {'✅' if data['services']['risk_scoring'] else '❌'}")
            print(f"Network Analysis: {'✅' if data['services']['network_analysis'] else '❌'}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Could not connect to API: {e}")
    print()


def test_single_fraud_check():
    """Test single transaction fraud check."""
    print("=== Single Transaction Fraud Check ===")
    
    transaction = {
        "transaction_id": "API_TEST_001",
        "user_id": "USER_API_001",
        "amount": 2500.0,
        "timestamp": datetime.now().isoformat(),
        "location": "New York",
        "merchant_category": "online"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/fraud-check", json=transaction)
        if response.status_code == 200:
            result = response.json()
            print("✅ Fraud check completed")
            print(f"Transaction: {result['transaction_id']}")
            print(f"Risk Score: {result['risk_assessment']['risk_score']}")
            print(f"Risk Level: {result['risk_assessment']['risk_level']}")
            if result['risk_assessment']['risk_factors']:
                print(f"Risk Factors: {', '.join(result['risk_assessment']['risk_factors'])}")
        else:
            print(f"❌ Fraud check failed: {response.status_code}")
            print(response.text)
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
    print()


def test_batch_fraud_check():
    """Test batch fraud checking."""
    print("=== Batch Fraud Check ===")
    
    transactions = [
        {
            "transaction_id": "BATCH_001",
            "user_id": "USER_BATCH_001",
            "amount": 100.0,
            "timestamp": datetime.now().isoformat(),
            "location": "Boston",
            "merchant_category": "grocery"
        },
        {
            "transaction_id": "BATCH_002",
            "user_id": "USER_BATCH_002",
            "amount": 5000.0,
            "timestamp": datetime.now().isoformat(),
            "location": "Unknown",
            "merchant_category": "cash_advance"
        }
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/batch-fraud-check", 
                               json={"transactions": transactions})
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch check completed")
            print(f"Processed: {result['summary']['total_transactions']} transactions")
            print(f"High Risk: {result['summary']['high_risk_count']} transactions")
            
            print("\nResults:")
            for res in result['results']:
                print(f"  {res['transaction_id']}: Risk {res['risk_assessment']['risk_score']} "
                      f"({res['risk_assessment']['risk_level']})")
        else:
            print(f"❌ Batch check failed: {response.status_code}")
            print(response.text)
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
    print()


def test_user_profile():
    """Test user profile retrieval."""
    print("=== User Profile ===")
    
    user_id = "USER_API_001"  # From previous test
    
    try:
        response = requests.get(f"{BASE_URL}/user-profile/{user_id}")
        if response.status_code == 200:
            result = response.json()
            profile = result['profile']
            print("✅ User profile retrieved")
            print(f"User: {result['user_id']}")
            print(f"Transactions: {profile['transaction_count']}")
            print(f"Average Amount: ${profile['avg_amount']:.2f}")
            print(f"Risk Score: {profile.get('user_risk_score', 0)}")
        elif response.status_code == 404:
            print("ℹ️  User profile not found (expected for new user)")
        else:
            print(f"❌ Profile check failed: {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
    print()


def test_network_analysis():
    """Test network analysis."""
    print("=== Network Analysis ===")
    
    try:
        response = requests.get(f"{BASE_URL}/network-analysis")
        if response.status_code == 200:
            result = response.json()
            metrics = result['network_metrics']
            
            print("✅ Network analysis completed")
            if 'error' not in metrics:
                print(f"Nodes: {metrics.get('total_nodes', 0)}")
                print(f"Edges: {metrics.get('total_edges', 0)}")
                print(f"Density: {metrics.get('density', 0):.4f}")
                print(f"Fraud Rings: {len(result.get('fraud_rings', []))}")
                print(f"Suspicious Patterns: {len(result.get('suspicious_patterns', []))}")
            else:
                print("ℹ️  No network data available yet")
        else:
            print(f"❌ Network analysis failed: {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
    print()


def test_model_training():
    """Test model training endpoint."""
    print("=== Model Training ===")
    
    try:
        response = requests.post(f"{BASE_URL}/train-models")
        if response.status_code == 200:
            result = response.json()
            print("✅ Model training completed")
            print(f"Message: {result['message']}")
            training_results = result.get('training_results', {})
            if training_results:
                print(f"Isolation Forest: {'✅' if training_results.get('isolation_forest_trained') else '❌'}")
                print(f"XGBoost: {'✅' if training_results.get('xgb_trained') else '❌'}")
        else:
            print(f"❌ Model training failed: {response.status_code}")
            print(response.text)
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
    print()


def main():
    """Run all API tests."""
    print("FRAUD DETECTION API - CLIENT EXAMPLES")
    print("=" * 50)
    print("Note: Make sure the API server is running with:")
    print("      python -m src.api.server")
    print()
    
    # Test all endpoints
    test_health_check()
    test_model_training()  # Train models first
    test_single_fraud_check()
    test_batch_fraud_check()
    test_user_profile()
    test_network_analysis()
    
    print("API examples completed!")
    print("\nTo start the API server:")
    print("  python -m src.api.server")


if __name__ == "__main__":
    main()