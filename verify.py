import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import accuracy_score
from utils.feature_extractor import KeystrokeFeatureExtractor

class KeystrokeVerifier:
    def __init__(self, model_path='models/keystroke_ensemble_model.pkl'):
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load trained model ensemble"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.best_model = model_data['best_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_extractor = model_data['feature_extractor']
        
        print(f"✅ Model loaded successfully")
        print(f"   Best model: {self.best_model}")
        print(f"   Available models: {list(self.models.keys())}")
    
    def verify_attempt(self, csv_file_path, expected_user=True):
        """Verify a single keystroke attempt"""
        try:
            # Load and preprocess data
            df = pd.read_csv(csv_file_path)
            df.columns = df.columns.str.strip().str.lower()
            
            # Extract features
            features = self.feature_extractor.extract_features(df)
            features = self.feature_extractor.ensure_feature_set(features)
            
            # Create feature vector
            feature_vector = pd.DataFrame([features])[self.feature_names]
            feature_vector = feature_vector.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(feature_vector)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            confidence_scores = {}
            
            for name, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                
                predictions[name] = pred
                probabilities[name] = proba
                # Confidence is the probability of the predicted class
                confidence_scores[name] = max(proba) * 100
            
            # Ensemble prediction (majority voting)
            ensemble_pred = np.round(np.mean(list(predictions.values())))
            ensemble_confidence = np.mean(list(confidence_scores.values()))
            
            # Best model prediction
            best_pred = predictions[self.best_model]
            best_confidence = confidence_scores[self.best_model]
            
            # Security analysis
            security_level = self.analyze_security(probabilities, ensemble_confidence)
            anomaly_score = self.calculate_anomaly_score(features)
            
            result = {
                'file': os.path.basename(csv_file_path),
                'keystrokes': len(df),
                'ensemble_prediction': 'Legitimate' if ensemble_pred == 1 else 'Impostor',
                'ensemble_confidence': ensemble_confidence,
                'best_model_prediction': 'Legitimate' if best_pred == 1 else 'Impostor',
                'best_model_confidence': best_confidence,
                'security_level': security_level,
                'anomaly_score': anomaly_score,
                'all_predictions': predictions,
                'all_confidences': confidence_scores,
                'features': features
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error verifying {csv_file_path}: {e}")
            return None
    
    def analyze_security(self, probabilities, confidence):
        """Analyze security level based on predictions"""
        legit_probs = [proba[1] for proba in probabilities.values()]  # Probability of legitimate class
        
        avg_legit_prob = np.mean(legit_probs)
        std_legit_prob = np.std(legit_probs)
        
        if avg_legit_prob > 0.8 and std_legit_prob < 0.1:
            return "HIGH"
        elif avg_legit_prob > 0.6:
            return "MEDIUM"
        elif avg_legit_prob > 0.4:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def calculate_anomaly_score(self, features):
        """Calculate anomaly score based on feature deviations"""
        # Simple anomaly detection based on dwell and flight times
        dwell_mean = features.get('dwell_mean', 0)
        flight_mean = features.get('flight_mean', 0)
        
        # Expected ranges (can be calibrated based on training data)
        expected_dwell = 100  # ms
        expected_flight = 200  # ms
        
        dwell_deviation = abs(dwell_mean - expected_dwell) / expected_dwell
        flight_deviation = abs(flight_mean - expected_flight) / expected_flight
        
        anomaly_score = (dwell_deviation + flight_deviation) / 2
        return min(anomaly_score * 100, 100)  # Convert to percentage
    
    def batch_verify(self, data_dir='data', file_pattern='sample{}.csv', indices=None):
        """Verify multiple files in batch"""
        if indices is None:
            # Find all sample files
            files = [f for f in os.listdir(data_dir) if f.startswith('sample') and f.endswith('.csv')]
        else:
            files = [file_pattern.format(i) for i in indices]
        
        results = []
        for file in files:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                result = self.verify_attempt(file_path)
                if result:
                    results.append(result)
        
        return results

def print_verification_result(result):
    """Print detailed verification results"""
    print("\n" + "="*70)
    print("🔐 KEYSROKE DYNAMICS VERIFICATION RESULTS")
    print("="*70)
    
    print(f"📁 File: {result['file']}")
    print(f"⌨️  Keystrokes: {result['keystrokes']}")
    print(f"🏆 Ensemble Prediction: {result['ensemble_prediction']}")
    print(f"📊 Ensemble Confidence: {result['ensemble_confidence']:.2f}%")
    print(f"🎯 Best Model Prediction: {result['best_model_prediction']}")
    print(f"📈 Best Model Confidence: {result['best_model_confidence']:.2f}%")
    print(f"🛡️  Security Level: {result['security_level']}")
    print(f"⚠️  Anomaly Score: {result['anomaly_score']:.2f}%")
    
    print("\n📋 Individual Model Predictions:")
    for model, pred in result['all_predictions'].items():
        status = "✅ Legitimate" if pred == 1 else "❌ Impostor"
        print(f"   {model:15}: {status} ({result['all_confidences'][model]:.2f}%)")
    
    print("\n📊 Key Statistics:")
    print(f"   Avg Dwell Time: {result['features']['dwell_mean']:.2f} ms")
    print(f"   Avg Flight Time: {result['features']['flight_mean']:.2f} ms")
    print(f"   Total Time: {result['features']['total_time']:.2f} ms")
    print(f"   Typing Speed: {result['features'].get('typing_speed', 0):.2f} keys/sec")
    
    # Final decision
    if result['ensemble_prediction'] == 'Legitimate' and result['ensemble_confidence'] > 70:
        print("\n🎉 FINAL DECISION: ✅ AUTHENTICATED")
    else:
        print("\n🚫 FINAL DECISION: ❌ REJECTED")
    
    print("="*70)

def main():
    # Test with sample6.csv
    verifier = KeystrokeVerifier()
    
    sample_file = 'data/sample5104.csv'
    if os.path.exists(sample_file):
        result = verifier.verify_attempt(sample_file)
        if result:
            print_verification_result(result)
            
            # Save detailed results
            output_file = 'verification_detailed_results.json'
            with open(output_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_result = {}
                for key, value in result.items():
                    if isinstance(value, (np.int64, np.int32)):
                        json_result[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        json_result[key] = float(value)
                    else:
                        json_result[key] = value
                
                json.dump(json_result, f, indent=2)
            
            print(f"\n💾 Detailed results saved to: {output_file}")
    else:
        print(f"❌ Test file {sample_file} not found!")

if __name__ == "__main__":
    main()