"""
Dynamic risk scoring system for users and transactions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class RiskScoringEngine:
    """
    Dynamic risk scoring engine that assigns risk scores to users and transactions
    based on behavior patterns and historical data.
    """
    
    def __init__(self):
        self.user_profiles = {}
        self.transaction_history = defaultdict(list)
        self.risk_factors = {
            'amount_threshold': 1000,
            'velocity_threshold': 5,  # max transactions per hour
            'time_window_hours': 24,
            'location_change_threshold': 2,  # max location changes per day
            'unusual_hour_start': 22,
            'unusual_hour_end': 6
        }
    
    def calculate_transaction_risk_score(self, transaction: Dict[str, Any], 
                                       user_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate risk score for a single transaction.
        
        Args:
            transaction: Transaction details
            user_history: Historical transactions for the user
            
        Returns:
            Risk scoring details including score and factors
        """
        risk_score = 0.0
        risk_factors = []
        
        user_id = transaction.get('user_id')
        amount = transaction.get('amount', 0)
        timestamp = transaction.get('timestamp')
        location = transaction.get('location')
        merchant_category = transaction.get('merchant_category')
        
        # Factor 1: Transaction Amount Risk
        amount_risk = self._calculate_amount_risk(amount, user_history)
        risk_score += amount_risk['score']
        if amount_risk['score'] > 0:
            risk_factors.append(amount_risk['reason'])
        
        # Factor 2: Time-based Risk
        time_risk = self._calculate_time_risk(timestamp)
        risk_score += time_risk['score']
        if time_risk['score'] > 0:
            risk_factors.append(time_risk['reason'])
        
        # Factor 3: Velocity Risk (transaction frequency)
        velocity_risk = self._calculate_velocity_risk(user_id, timestamp, user_history)
        risk_score += velocity_risk['score']
        if velocity_risk['score'] > 0:
            risk_factors.append(velocity_risk['reason'])
        
        # Factor 4: Location Risk
        location_risk = self._calculate_location_risk(user_id, location, user_history)
        risk_score += location_risk['score']
        if location_risk['score'] > 0:
            risk_factors.append(location_risk['reason'])
        
        # Factor 5: Merchant Category Risk
        merchant_risk = self._calculate_merchant_risk(merchant_category, user_history)
        risk_score += merchant_risk['score']
        if merchant_risk['score'] > 0:
            risk_factors.append(merchant_risk['reason'])
        
        # Normalize risk score to 0-100 scale
        risk_score = min(risk_score, 100)
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = "CRITICAL"
        elif risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        elif risk_score >= 20:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_amount_risk(self, amount: float, user_history: Optional[List[Dict]]) -> Dict[str, Any]:
        """Calculate risk based on transaction amount."""
        if user_history and len(user_history) > 0:
            # Compare with user's historical spending
            amounts = [t.get('amount', 0) for t in user_history]
            avg_amount = np.mean(amounts)
            std_amount = np.std(amounts)
            
            if std_amount > 0:
                z_score = abs((amount - avg_amount) / std_amount)
                if z_score > 3:  # 3 standard deviations
                    return {
                        'score': min(30, z_score * 5),
                        'reason': f"Amount ${amount} is {z_score:.1f} std devs from user average ${avg_amount:.2f}"
                    }
        
        # Absolute amount thresholds
        if amount > 5000:
            return {'score': 25, 'reason': f"Very high amount: ${amount}"}
        elif amount > 1000:
            return {'score': 15, 'reason': f"High amount: ${amount}"}
        
        return {'score': 0, 'reason': ""}
    
    def _calculate_time_risk(self, timestamp) -> Dict[str, Any]:
        """Calculate risk based on transaction time."""
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        elif not isinstance(timestamp, (datetime, pd.Timestamp)):
            return {'score': 0, 'reason': ""}
        
        hour = timestamp.hour
        
        # Check for unusual hours
        if (hour >= self.risk_factors['unusual_hour_start'] or 
            hour <= self.risk_factors['unusual_hour_end']):
            return {
                'score': 20,
                'reason': f"Transaction at unusual hour: {hour:02d}:00"
            }
        
        return {'score': 0, 'reason': ""}
    
    def _calculate_velocity_risk(self, user_id: str, timestamp, user_history: Optional[List[Dict]]) -> Dict[str, Any]:
        """Calculate risk based on transaction velocity."""
        if not user_history or not timestamp:
            return {'score': 0, 'reason': ""}
        
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Count recent transactions
        time_window = timedelta(hours=self.risk_factors['time_window_hours'])
        recent_transactions = 0
        
        for transaction in user_history:
            if 'timestamp' in transaction:
                txn_time = pd.to_datetime(transaction['timestamp'])
                if abs((timestamp - txn_time).total_seconds()) <= time_window.total_seconds():
                    recent_transactions += 1
        
        if recent_transactions > self.risk_factors['velocity_threshold']:
            return {
                'score': min(25, recent_transactions * 3),
                'reason': f"High velocity: {recent_transactions} transactions in {self.risk_factors['time_window_hours']} hours"
            }
        
        return {'score': 0, 'reason': ""}
    
    def _calculate_location_risk(self, user_id: str, location: str, user_history: Optional[List[Dict]]) -> Dict[str, Any]:
        """Calculate risk based on transaction location."""
        if not user_history or not location:
            return {'score': 0, 'reason': ""}
        
        # Get unique locations from history
        historical_locations = set()
        for transaction in user_history:
            if 'location' in transaction and transaction['location']:
                historical_locations.add(transaction['location'])
        
        # Check if location is new
        if location not in historical_locations:
            return {
                'score': 15,
                'reason': f"New location: {location}"
            }
        
        return {'score': 0, 'reason': ""}
    
    def _calculate_merchant_risk(self, merchant_category: str, user_history: Optional[List[Dict]]) -> Dict[str, Any]:
        """Calculate risk based on merchant category."""
        high_risk_categories = ['gambling', 'adult_entertainment', 'cash_advance', 'cryptocurrency']
        
        if merchant_category in high_risk_categories:
            return {
                'score': 20,
                'reason': f"High-risk merchant category: {merchant_category}"
            }
        
        return {'score': 0, 'reason': ""}
    
    def update_user_profile(self, user_id: str, transaction: Dict[str, Any]):
        """Update user profile with new transaction."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'creation_date': datetime.now().isoformat(),
                'transaction_count': 0,
                'total_amount': 0,
                'avg_amount': 0,
                'locations': set(),
                'merchant_categories': set(),
                'last_transaction': None
            }
        
        profile = self.user_profiles[user_id]
        amount = transaction.get('amount', 0)
        
        # Update statistics
        profile['transaction_count'] += 1
        profile['total_amount'] += amount
        profile['avg_amount'] = profile['total_amount'] / profile['transaction_count']
        
        if 'location' in transaction:
            profile['locations'].add(transaction['location'])
        
        if 'merchant_category' in transaction:
            profile['merchant_categories'].add(transaction['merchant_category'])
        
        profile['last_transaction'] = datetime.now().isoformat()
        
        # Update transaction history
        self.transaction_history[user_id].append(transaction)
        
        # Keep only recent transactions (last 30 days)
        if 'timestamp' in transaction:
            cutoff_date = pd.to_datetime(transaction['timestamp']) - timedelta(days=30)
            self.transaction_history[user_id] = [
                t for t in self.transaction_history[user_id]
                if pd.to_datetime(t.get('timestamp', datetime.now())) > cutoff_date
            ]
    
    def get_user_risk_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive risk profile for a user."""
        if user_id not in self.user_profiles:
            return {'error': 'User not found'}
        
        profile = self.user_profiles[user_id].copy()
        
        # Convert sets to lists for JSON serialization
        profile['locations'] = list(profile['locations'])
        profile['merchant_categories'] = list(profile['merchant_categories'])
        
        # Calculate user risk level
        transaction_count = profile['transaction_count']
        avg_amount = profile['avg_amount']
        
        user_risk_score = 0
        
        # New user risk
        if transaction_count < 5:
            user_risk_score += 20
        
        # High average amount risk
        if avg_amount > 1000:
            user_risk_score += 15
        
        # Multiple locations risk
        if len(profile['locations']) > 10:
            user_risk_score += 10
        
        profile['user_risk_score'] = min(user_risk_score, 100)
        
        return profile
    
    def batch_score_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score multiple transactions in batch.
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            DataFrame with risk scores added
        """
        results = []
        
        for _, transaction in transactions_df.iterrows():
            user_id = transaction.get('user_id')
            user_history = self.transaction_history.get(user_id, [])
            
            # Calculate risk score
            risk_result = self.calculate_transaction_risk_score(
                transaction.to_dict(), user_history
            )
            
            # Update user profile
            self.update_user_profile(user_id, transaction.to_dict())
            
            # Add to results
            result = transaction.to_dict()
            result.update(risk_result)
            results.append(result)
        
        return pd.DataFrame(results)