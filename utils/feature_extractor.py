import pandas as pd
import numpy as np

class KeystrokeFeatureExtractor:
    def __init__(self):
        self.required_features = [
            'dwell_mean', 'dwell_std', 'dwell_median', 'dwell_min', 'dwell_max',
            'dwell_q25', 'dwell_q75', 'dwell_skew', 'dwell_kurtosis',
            'flight_mean', 'flight_std', 'flight_median', 'flight_min', 'flight_max',
            'flight_q25', 'flight_q75', 'flight_skew', 'flight_kurtosis',
            'dwell_flight_ratio', 'total_time', 'num_keystrokes',
            'dwell_cv', 'flight_cv', 'typing_speed', 'pause_ratio'
        ]
    
    def extract_features(self, df):
        """Extract comprehensive keystroke features"""
        dwell = df['dwell_time'].dropna()
        flight = df['flight_time'].dropna()
        
        # Basic statistics
        features = {
            'dwell_mean': dwell.mean(),
            'dwell_std': dwell.std(),
            'dwell_median': dwell.median(),
            'dwell_min': dwell.min(),
            'dwell_max': dwell.max(),
            'dwell_q25': dwell.quantile(0.25),
            'dwell_q75': dwell.quantile(0.75),
            'dwell_skew': dwell.skew(),
            'dwell_kurtosis': dwell.kurtosis(),
            
            'flight_mean': flight.mean(),
            'flight_std': flight.std(),
            'flight_median': flight.median(),
            'flight_min': flight.min(),
            'flight_max': flight.max(),
            'flight_q25': flight.quantile(0.25),
            'flight_q75': flight.quantile(0.75),
            'flight_skew': flight.skew(),
            'flight_kurtosis': flight.kurtosis(),
            
            'dwell_flight_ratio': dwell.mean() / flight.mean() if flight.mean() > 0 else 0,
            'total_time': dwell.sum() + flight.sum(),
            'num_keystrokes': len(df),
        }
        
        # Advanced features
        features['dwell_cv'] = features['dwell_std'] / features['dwell_mean'] if features['dwell_mean'] > 0 else 0
        features['flight_cv'] = features['flight_std'] / features['flight_mean'] if features['flight_mean'] > 0 else 0
        features['typing_speed'] = len(df) / (features['total_time'] / 1000) if features['total_time'] > 0 else 0
        
        # Pause detection
        long_pauses = len(flight[flight > 500])  # pauses longer than 500ms
        features['pause_ratio'] = long_pauses / len(flight) if len(flight) > 0 else 0
        
        return features
    
    def ensure_feature_set(self, features_dict):
        """Ensure all required features are present"""
        for feature in self.required_features:
            if feature not in features_dict:
                features_dict[feature] = 0
        return features_dict