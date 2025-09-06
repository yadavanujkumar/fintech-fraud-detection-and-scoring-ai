"""
Main module for fraud detection system.
"""

from .engine import FraudDetectionEngine
from .data_generator import TransactionDataGenerator, create_sample_data
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to demonstrate fraud detection system.
    """
    logger.info("Starting Fraud Detection System...")
    
    # Generate sample data
    logger.info("Generating sample transaction data...")
    generator = TransactionDataGenerator()
    df = generator.generate_dataset(n_legitimate=8000, n_fraudulent=2000)
    
    # Save sample data
    df.to_csv('data/sample_transactions.csv', index=False)
    logger.info("Sample data saved to data/sample_transactions.csv")
    
    # Initialize fraud detection engine
    engine = FraudDetectionEngine()
    
    # Train the models
    results = engine.train(df)
    logger.info("Training completed!")
    
    # Test predictions on a subset
    test_df = df.sample(n=100)
    predictions = engine.predict(test_df)
    
    # Display results
    logger.info("\n=== FRAUD DETECTION RESULTS ===")
    
    if 'isolation_forest' in predictions:
        iso_pred = predictions['isolation_forest']['predictions']
        logger.info(f"Isolation Forest detected {iso_pred.sum()} potential frauds out of {len(iso_pred)} transactions")
    
    if 'xgboost' in predictions:
        xgb_pred = predictions['xgboost']['predictions']
        xgb_proba = predictions['xgboost']['fraud_probability']
        logger.info(f"XGBoost detected {xgb_pred.sum()} potential frauds out of {len(xgb_pred)} transactions")
        logger.info(f"Average fraud probability: {xgb_proba.mean():.3f}")
    
    # Show some example predictions
    results_df = test_df.copy()
    if 'xgboost' in predictions:
        results_df['fraud_probability'] = predictions['xgboost']['fraud_probability']
        results_df['predicted_fraud'] = predictions['xgboost']['predictions']
        
        logger.info("\nTop 10 highest risk transactions:")
        top_risk = results_df.nlargest(10, 'fraud_probability')[['transaction_id', 'amount', 'fraud_probability', 'is_fraud', 'predicted_fraud']]
        print(top_risk.to_string(index=False))
    
    logger.info("Fraud detection demonstration completed!")


if __name__ == "__main__":
    main()