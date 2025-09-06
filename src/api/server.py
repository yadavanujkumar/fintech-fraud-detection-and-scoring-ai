"""
REST API server for fraud detection services.
"""

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraud_detection.engine import FraudDetectionEngine
from fraud_detection.data_generator import TransactionDataGenerator
from risk_scoring.engine import RiskScoringEngine
from graph_analytics.network_analyzer import FraudNetworkAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Initialize engines
fraud_engine = FraudDetectionEngine()
risk_engine = RiskScoringEngine()
network_analyzer = FraudNetworkAnalyzer()

# Try to load pre-trained models
try:
    fraud_engine.load_models()
    logger.info("Loaded pre-trained fraud detection models")
except:
    logger.warning("No pre-trained models found. Please train models first.")

class HealthCheck(Resource):
    """Health check endpoint."""
    
    def get(self):
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'fraud_detection': fraud_engine.is_trained,
                'risk_scoring': True,
                'network_analysis': True
            }
        }

class TransactionFraudCheck(Resource):
    """Check if a transaction is fraudulent."""
    
    def post(self):
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['transaction_id', 'user_id', 'amount']
            for field in required_fields:
                if field not in data:
                    return {'error': f'Missing required field: {field}'}, 400
            
            # Create DataFrame from single transaction
            df = pd.DataFrame([data])
            
            # Check if fraud detection models are trained
            if not fraud_engine.is_trained:
                return {'error': 'Fraud detection models not trained'}, 503
            
            # Get fraud predictions
            predictions = fraud_engine.predict(df)
            
            # Calculate risk score
            user_history = risk_engine.transaction_history.get(data['user_id'], [])
            risk_result = risk_engine.calculate_transaction_risk_score(data, user_history)
            
            # Update user profile
            risk_engine.update_user_profile(data['user_id'], data)
            
            # Prepare response
            result = {
                'transaction_id': data['transaction_id'],
                'fraud_prediction': {},
                'risk_assessment': risk_result,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add fraud predictions if available
            if 'isolation_forest' in predictions:
                iso_pred = predictions['isolation_forest']['predictions'][0]
                iso_score = predictions['isolation_forest']['anomaly_scores'][0]
                result['fraud_prediction']['isolation_forest'] = {
                    'is_fraud': bool(iso_pred),
                    'anomaly_score': float(iso_score)
                }
            
            if 'xgboost' in predictions:
                xgb_pred = predictions['xgboost']['predictions'][0]
                xgb_proba = predictions['xgboost']['fraud_probability'][0]
                result['fraud_prediction']['xgboost'] = {
                    'is_fraud': bool(xgb_pred),
                    'fraud_probability': float(xgb_proba)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fraud check: {e}")
            return {'error': str(e)}, 500

class BatchFraudCheck(Resource):
    """Check multiple transactions for fraud."""
    
    def post(self):
        try:
            data = request.get_json()
            
            if 'transactions' not in data:
                return {'error': 'Missing transactions field'}, 400
            
            transactions = data['transactions']
            if not isinstance(transactions, list):
                return {'error': 'Transactions must be a list'}, 400
            
            # Create DataFrame
            df = pd.DataFrame(transactions)
            
            # Check if fraud detection models are trained
            if not fraud_engine.is_trained:
                return {'error': 'Fraud detection models not trained'}, 503
            
            # Get fraud predictions
            predictions = fraud_engine.predict(df)
            
            # Calculate risk scores
            risk_results = []
            for _, transaction in df.iterrows():
                user_id = transaction.get('user_id')
                user_history = risk_engine.transaction_history.get(user_id, [])
                risk_result = risk_engine.calculate_transaction_risk_score(
                    transaction.to_dict(), user_history
                )
                risk_results.append(risk_result)
                
                # Update user profile
                risk_engine.update_user_profile(user_id, transaction.to_dict())
            
            # Prepare batch results
            results = []
            for i, transaction in enumerate(transactions):
                result = {
                    'transaction_id': transaction.get('transaction_id'),
                    'fraud_prediction': {},
                    'risk_assessment': risk_results[i]
                }
                
                # Add fraud predictions if available
                if 'isolation_forest' in predictions:
                    iso_pred = predictions['isolation_forest']['predictions'][i]
                    iso_score = predictions['isolation_forest']['anomaly_scores'][i]
                    result['fraud_prediction']['isolation_forest'] = {
                        'is_fraud': bool(iso_pred),
                        'anomaly_score': float(iso_score)
                    }
                
                if 'xgboost' in predictions:
                    xgb_pred = predictions['xgboost']['predictions'][i]
                    xgb_proba = predictions['xgboost']['fraud_probability'][i]
                    result['fraud_prediction']['xgboost'] = {
                        'is_fraud': bool(xgb_pred),
                        'fraud_probability': float(xgb_proba)
                    }
                
                results.append(result)
            
            return {
                'results': results,
                'summary': {
                    'total_transactions': len(results),
                    'high_risk_count': len([r for r in risk_results if r['risk_score'] >= 60]),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in batch fraud check: {e}")
            return {'error': str(e)}, 500

class UserRiskProfile(Resource):
    """Get user risk profile."""
    
    def get(self, user_id):
        try:
            profile = risk_engine.get_user_risk_profile(user_id)
            
            if 'error' in profile:
                return profile, 404
            
            return {
                'user_id': user_id,
                'profile': profile,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return {'error': str(e)}, 500

class NetworkAnalysis(Resource):
    """Network analysis endpoints."""
    
    def get(self):
        """Get network metrics and fraud rings."""
        try:
            metrics = network_analyzer.analyze_network_metrics()
            fraud_rings = network_analyzer.detect_fraud_rings()
            suspicious_patterns = network_analyzer.find_suspicious_patterns()
            
            return {
                'network_metrics': metrics,
                'fraud_rings': fraud_rings,
                'suspicious_patterns': suspicious_patterns,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in network analysis: {e}")
            return {'error': str(e)}, 500
    
    def post(self):
        """Update network with new transaction data."""
        try:
            data = request.get_json()
            
            if 'transactions' not in data:
                return {'error': 'Missing transactions field'}, 400
            
            # Create DataFrame and update network
            df = pd.DataFrame(data['transactions'])
            network_analyzer.build_transaction_network(df)
            
            # Get updated analysis
            metrics = network_analyzer.analyze_network_metrics()
            fraud_rings = network_analyzer.detect_fraud_rings()
            
            return {
                'message': 'Network updated successfully',
                'network_metrics': metrics,
                'fraud_rings_detected': len(fraud_rings),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating network: {e}")
            return {'error': str(e)}, 500

class TrainModels(Resource):
    """Train fraud detection models."""
    
    def post(self):
        try:
            data = request.get_json()
            
            if 'transactions' in data:
                # Use provided data
                df = pd.DataFrame(data['transactions'])
            else:
                # Generate sample data
                generator = TransactionDataGenerator()
                df = generator.generate_dataset()
                logger.info("Using generated sample data for training")
            
            # Train models
            results = fraud_engine.train(df)
            
            # Update network
            network_analyzer.build_transaction_network(df)
            
            return {
                'message': 'Models trained successfully',
                'training_results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'error': str(e)}, 500

# Add API endpoints
api.add_resource(HealthCheck, '/')
api.add_resource(TransactionFraudCheck, '/fraud-check')
api.add_resource(BatchFraudCheck, '/batch-fraud-check')
api.add_resource(UserRiskProfile, '/user-profile/<string:user_id>')
api.add_resource(NetworkAnalysis, '/network-analysis')
api.add_resource(TrainModels, '/train-models')

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Try to train models on startup if not already trained
    if not fraud_engine.is_trained:
        try:
            logger.info("Training models on startup...")
            generator = TransactionDataGenerator()
            sample_data = generator.generate_dataset(n_legitimate=1000, n_fraudulent=200)
            fraud_engine.train(sample_data)
            network_analyzer.build_transaction_network(sample_data)
            logger.info("Models trained successfully on startup")
        except Exception as e:
            logger.error(f"Failed to train models on startup: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)