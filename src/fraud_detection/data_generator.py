"""
Data generator for creating synthetic transaction data for testing and demonstration.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TransactionDataGenerator:
    """
    Generates synthetic transaction data with both legitimate and fraudulent patterns.
    """
    
    def __init__(self, seed: int = 42):
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_legitimate_transactions(self, n_transactions: int) -> List[Dict[str, Any]]:
        """
        Generate legitimate transaction data.
        
        Args:
            n_transactions: Number of transactions to generate
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        
        for i in range(n_transactions):
            # Normal transaction patterns
            amount = np.random.lognormal(mean=3, sigma=1)  # Most transactions are small
            
            # Business hours are more common
            if random.random() < 0.7:  # 70% during business hours
                hour = random.randint(9, 17)
            else:
                hour = random.randint(0, 23)
            
            # Weekdays are more common
            if random.random() < 0.7:  # 70% on weekdays
                day_of_week = random.randint(0, 4)
            else:
                day_of_week = random.randint(5, 6)
            
            transaction = {
                'transaction_id': f'TXN_{i:06d}',
                'user_id': f'USER_{random.randint(1, 1000):04d}',
                'amount': round(amount, 2),
                'merchant_category': random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online']),
                'location': self.fake.city(),
                'timestamp': self.fake.date_time_between(start_date='-30d', end_date='now'),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'card_present': random.choice([0, 1]),
                'is_fraud': 0
            }
            
            transactions.append(transaction)
        
        return transactions
    
    def generate_fraudulent_transactions(self, n_transactions: int) -> List[Dict[str, Any]]:
        """
        Generate fraudulent transaction data with suspicious patterns.
        
        Args:
            n_transactions: Number of fraudulent transactions to generate
            
        Returns:
            List of fraudulent transaction dictionaries
        """
        transactions = []
        
        for i in range(n_transactions):
            # Fraudulent patterns
            fraud_type = random.choice(['high_amount', 'unusual_time', 'rapid_sequence', 'unusual_location'])
            
            if fraud_type == 'high_amount':
                # Unusually high amounts
                amount = np.random.uniform(1000, 10000)
            elif fraud_type == 'unusual_time':
                # Unusual hours (very early morning)
                amount = np.random.lognormal(mean=4, sigma=1)
                hour = random.randint(2, 5)
            elif fraud_type == 'rapid_sequence':
                # Rapid sequence of transactions
                amount = np.random.uniform(100, 500)
            else:  # unusual_location
                amount = np.random.lognormal(mean=3.5, sigma=1)
            
            # More likely during off-hours
            if 'hour' not in locals() or fraud_type != 'unusual_time':
                if random.random() < 0.6:  # 60% during off-hours
                    hour = random.choice(list(range(22, 24)) + list(range(0, 6)))
                else:
                    hour = random.randint(6, 21)
            
            # More likely on weekends
            if random.random() < 0.6:  # 60% on weekends
                day_of_week = random.randint(5, 6)
            else:
                day_of_week = random.randint(0, 4)
            
            transaction = {
                'transaction_id': f'FRAUD_{i:06d}',
                'user_id': f'USER_{random.randint(1, 1000):04d}',
                'amount': round(amount, 2),
                'merchant_category': random.choice(['online', 'cash_advance', 'gambling', 'adult_entertainment']),
                'location': self.fake.city(),
                'timestamp': self.fake.date_time_between(start_date='-30d', end_date='now'),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'card_present': 0,  # More likely to be card-not-present
                'is_fraud': 1
            }
            
            transactions.append(transaction)
        
        return transactions
    
    def generate_dataset(self, n_legitimate: int = 9000, n_fraudulent: int = 1000) -> pd.DataFrame:
        """
        Generate a complete dataset with both legitimate and fraudulent transactions.
        
        Args:
            n_legitimate: Number of legitimate transactions
            n_fraudulent: Number of fraudulent transactions
            
        Returns:
            Combined dataframe with all transactions
        """
        logger.info(f"Generating {n_legitimate} legitimate and {n_fraudulent} fraudulent transactions...")
        
        # Generate transactions
        legitimate = self.generate_legitimate_transactions(n_legitimate)
        fraudulent = self.generate_fraudulent_transactions(n_fraudulent)
        
        # Combine and create DataFrame
        all_transactions = legitimate + fraudulent
        df = pd.DataFrame(all_transactions)
        
        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Add some derived features
        df['log_amount'] = np.log1p(df['amount'])
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        logger.info(f"Generated dataset with {len(df)} transactions ({df['is_fraud'].sum()} fraudulent)")
        
        return df


def create_sample_data(output_path: str = 'data/sample_transactions.csv'):
    """
    Create and save sample transaction data.
    
    Args:
        output_path: Path to save the generated data
    """
    generator = TransactionDataGenerator()
    df = generator.generate_dataset()
    
    # Ensure data directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Sample data saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    df = create_sample_data()
    print(f"Generated {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print("\nDataset info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())