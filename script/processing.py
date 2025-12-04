#!/usr/bin/env python3
"""
Data processing and feature engineering script for ML models
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.data_models.train import TrainType, TrainPriority

logger = setup_logger("data_processing")

class DataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
    def process_train_delays(self, input_file="historical/train_delays.csv"):
        """Process raw delay data into features for ML"""
        logger.info(f"Processing train delays from {input_file}")
        
        # Load data
        df = pd.read_csv(self.data_dir / input_file)
        
        # Convert time columns
        time_cols = ['scheduled_departure', 'actual_departure', 
                    'scheduled_arrival', 'actual_arrival']
        
        for col in time_cols:
            df[col] = pd.to_datetime(df['date'] + ' ' + df[col])
        
        # Feature engineering
        df['departure_delay'] = (df['actual_departure'] - df['scheduled_departure']).dt.total_seconds() / 60
        df['arrival_delay'] = (df['actual_arrival'] - df['scheduled_arrival']).dt.total_seconds() / 60
        
        # Time features
        df['departure_hour'] = df['scheduled_departure'].dt.hour
        df['departure_day_of_week'] = df['scheduled_departure'].dt.dayofweek
        df['departure_month'] = df['scheduled_departure'].dt.month
        df['is_weekend'] = df['departure_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_peak_hour'] = df['departure_hour'].apply(lambda x: 1 if x in [7, 8, 17, 18] else 0)
        
        # Train type encoding
        df['is_express'] = df['train_name'].apply(lambda x: 1 if 'Express' in x else 0)
        df['is_mail'] = df['train_name'].apply(lambda x: 1 if 'Mail' in x else 0)
        
        # Delay reason encoding
        delay_reasons = df['delay_reason'].unique()
        for reason in delay_reasons:
            df[f'delay_reason_{reason}'] = (df['delay_reason'] == reason).astype(int)
        
        # Weather encoding
        weather_conditions = df['weather_condition'].unique()
        for condition in weather_conditions:
            df[f'weather_{condition}'] = (df['weather_condition'] == condition).astype(int)
        
        # Save processed data
        output_file = self.processed_dir / "train_delays_processed.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
        
        return df
    
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic data for testing and development"""
        logger.info(f"Generating {num_samples} synthetic data samples")
        
        np.random.seed(42)
        
        data = []
        train_types = ['express', 'passenger', 'suburban', 'freight']
        priorities = [1, 2, 3, 4, 5]
        weather_conditions = ['clear', 'rain', 'fog', 'cloudy']
        delay_reasons = ['on_time', 'signal_failure', 'crew_delay', 'weather', 'congestion']
        
        for i in range(num_samples):
            train_type = np.random.choice(train_types)
            priority = np.random.choice(priorities)
            
            # Base travel time based on train type
            base_travel_time = {
                'express': np.random.normal(120, 15),
                'passenger': np.random.normal(150, 20),
                'suburban': np.random.normal(60, 10),
                'freight': np.random.normal(180, 25)
            }[train_type]
            
            # Weather impact
            weather = np.random.choice(weather_conditions)
            weather_factor = {
                'clear': 1.0,
                'cloudy': 1.05,
                'rain': 1.15,
                'fog': 1.25
            }[weather]
            
            # Priority impact
            priority_factor = 1.0 + (5 - priority) * 0.05  # Higher priority = slightly faster
            
            # Calculate final travel time with variation
            travel_time = base_travel_time * weather_factor * priority_factor
            travel_time += np.random.normal(0, travel_time * 0.1)  # 10% variation
            
            # Generate delay
            delay_reason = np.random.choice(delay_reasons, p=[0.6, 0.1, 0.1, 0.1, 0.1])
            delay_minutes = 0 if delay_reason == 'on_time' else np.random.exponential(20)
            
            data.append({
                'sample_id': i,
                'train_type': train_type,
                'priority': priority,
                'base_travel_time': base_travel_time,
                'weather_condition': weather,
                'weather_factor': weather_factor,
                'priority_factor': priority_factor,
                'actual_travel_time': travel_time,
                'delay_reason': delay_reason,
                'delay_minutes': delay_minutes,
                'total_travel_time': travel_time + delay_minutes
            })
        
        df = pd.DataFrame(data)
        output_file = self.processed_dir / "synthetic_travel_data.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Synthetic data saved to {output_file}")
        return df
    
    def create_training_dataset(self):
        """Create complete training dataset from multiple sources"""
        logger.info("Creating training dataset")
        
        # Process delays data
        delays_df = self.process_train_delays()
        
        # Load weather data
        weather_df = pd.read_csv(self.data_dir / "historical/weather_data.csv")
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Merge datasets (simplified example)
        # In practice, you'd have more sophisticated merging logic
        
        # Create features for ML
        features = delays_df[[
            'departure_hour', 'departure_day_of_week', 'departure_month',
            'is_weekend', 'is_peak_hour', 'is_express', 'is_mail',
            'delay_minutes'
        ]].copy()
        
        # Add weather features (simplified)
        features['avg_temperature'] = 25  # Placeholder
        features['precipitation'] = 0     # Placeholder
        
        # Save training dataset
        training_file = self.processed_dir / "ml_training_data.csv"
        features.to_csv(training_file, index=False)
        
        logger.info(f"Training dataset saved to {training_file}")
        return features
    
    def validate_data_quality(self):
        """Validate data quality and report issues"""
        logger.info("Validating data quality")
        
        issues = []
        
        # Check if required files exist
        required_files = [
            "historical/train_delays.csv",
            "historical/weather_data.csv",
            "sample/stations.json",
            "sample/trains.json",
            "sample/track_network.json"
        ]
        
        for file_path in required_files:
            if not (self.data_dir / file_path).exists():
                issues.append(f"Missing file: {file_path}")
        
        # Validate CSV files
        for csv_file in ["historical/train_delays.csv", "historical/weather_data.csv"]:
            try:
                df = pd.read_csv(self.data_dir / csv_file)
                
                # Check for missing values
                missing_values = df.isnull().sum().sum()
                if missing_values > 0:
                    issues.append(f"{csv_file}: {missing_values} missing values found")
                
                # Check data types
                for col in df.columns:
                    if df[col].dtype == 'object':
                        unique_count = df[col].nunique()
                        if unique_count > 100 and unique_count < len(df):
                            issues.append(f"{csv_file}: Column '{col}' has {unique_count} unique values - consider categorization")
            
            except Exception as e:
                issues.append(f"Error reading {csv_file}: {e}")
        
        # Report issues
        if issues:
            logger.warning(f"Data quality issues found: {len(issues)}")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Data quality validation passed")
        
        return issues

def main():
    """Main function for data processing"""
    processor = DataProcessor()
    
    # Validate data first
    issues = processor.validate_data_quality()
    if issues:
        print("Data quality issues found. Continuing anyway...")
    
    # Process data
    print("1. Processing train delays...")
    delays_df = processor.process_train_delays()
    
    print("2. Generating synthetic data...")
    synthetic_df = processor.generate_synthetic_data(500)
    
    print("3. Creating training dataset...")
    training_df = processor.create_training_dataset()
    
    print("\n✅ Data processing completed!")
    print(f"   - Processed delays: {len(delays_df)} records")
    print(f"   - Synthetic data: {len(synthetic_df)} records")
    print(f"   - Training data: {len(training_df)} records")
    
    return delays_df, synthetic_df, training_df

if __name__ == "__main__":
    main()
