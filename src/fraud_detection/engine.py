"""
Core fraud detection engine with multiple ML algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
from typing import Tuple, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionEngine:
    """
    Main fraud detection engine that combines multiple ML algorithms
    for comprehensive fraud detection.
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.xgb_model = None
        self.is_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess transaction data for fraud detection.
        
        Args:
            df: Raw transaction dataframe
            
        Returns:
            Preprocessed dataframe
        """
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Handle missing values
        processed_df = processed_df.fillna(0)
        
        # Feature engineering
        if 'amount' in processed_df.columns:
            processed_df['log_amount'] = np.log1p(processed_df['amount'])
            processed_df['amount_zscore'] = (processed_df['amount'] - processed_df['amount'].mean()) / processed_df['amount'].std()
        
        if 'timestamp' in processed_df.columns:
            processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
            processed_df['hour'] = processed_df['timestamp'].dt.hour
            processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
            processed_df['is_weekend'] = (processed_df['day_of_week'] >= 5).astype(int)
        
        return processed_df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract relevant features for fraud detection.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Feature dataframe
        """
        features = []
        
        # Numerical features
        numerical_cols = ['amount', 'log_amount', 'amount_zscore', 'hour', 'day_of_week', 'is_weekend']
        for col in numerical_cols:
            if col in df.columns:
                features.append(col)
        
        # Add any other numerical columns
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and col not in features and col != 'is_fraud':
                features.append(col)
        
        feature_df = df[features].fillna(0)
        return feature_df
    
    def train(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> Dict[str, Any]:
        """
        Train fraud detection models on the provided dataset.
        
        Args:
            df: Training dataframe
            target_col: Name of target column (fraud label)
            
        Returns:
            Training metrics
        """
        logger.info("Starting fraud detection model training...")
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Extract features
        X = self.extract_features(processed_df)
        y = processed_df[target_col] if target_col in processed_df.columns else None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest (unsupervised)
        logger.info("Training Isolation Forest...")
        self.isolation_forest.fit(X_scaled)
        
        results = {
            'isolation_forest_trained': True,
            'feature_count': X.shape[1],
            'sample_count': X.shape[0]
        }
        
        # Train XGBoost if labels are available
        if y is not None:
            logger.info("Training XGBoost classifier...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train XGBoost
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.xgb_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.xgb_model.predict(X_test)
            
            results.update({
                'xgb_trained': True,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            })
            
            logger.info("XGBoost Classification Report:")
            logger.info(f"\n{results['classification_report']}")
        
        self.is_trained = True
        
        # Save models
        self.save_models()
        
        logger.info("Model training completed successfully!")
        return results
    
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict fraud probability for transactions.
        
        Args:
            df: Transaction dataframe
            
        Returns:
            Dictionary with predictions from different models
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Extract features
        X = self.extract_features(processed_df)
        X_scaled = self.scaler.transform(X)
        
        results = {}
        
        # Isolation Forest predictions
        isolation_pred = self.isolation_forest.predict(X_scaled)
        isolation_scores = self.isolation_forest.score_samples(X_scaled)
        
        # Convert to fraud probability (Isolation Forest returns -1 for outliers)
        isolation_fraud_prob = np.where(isolation_pred == -1, 1, 0)
        
        results['isolation_forest'] = {
            'predictions': isolation_fraud_prob,
            'anomaly_scores': isolation_scores
        }
        
        # XGBoost predictions if available
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_scaled)
            xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]  # Probability of fraud
            
            results['xgboost'] = {
                'predictions': xgb_pred,
                'fraud_probability': xgb_proba
            }
        
        return results
    
    def save_models(self):
        """Save trained models to disk."""
        logger.info("Saving models...")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        
        # Save Isolation Forest
        joblib.dump(self.isolation_forest, os.path.join(self.model_dir, 'isolation_forest.pkl'))
        
        # Save XGBoost if trained
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, os.path.join(self.model_dir, 'xgboost.pkl'))
        
        logger.info("Models saved successfully!")
    
    def load_models(self):
        """Load trained models from disk."""
        logger.info("Loading models...")
        
        try:
            # Load scaler
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            
            # Load Isolation Forest
            self.isolation_forest = joblib.load(os.path.join(self.model_dir, 'isolation_forest.pkl'))
            
            # Load XGBoost if available
            xgb_path = os.path.join(self.model_dir, 'xgboost.pkl')
            if os.path.exists(xgb_path):
                self.xgb_model = joblib.load(xgb_path)
            
            self.is_trained = True
            logger.info("Models loaded successfully!")
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise