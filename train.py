import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import seaborn as sns
from utils.feature_extractor import KeystrokeFeatureExtractor

class KeystrokeModelTrainer:
    def __init__(self):
        self.feature_extractor = KeystrokeFeatureExtractor()
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_training_data(self, data_dir='data', start_idx=1, end_idx=5106):
        """Load all training data samples"""
        features_list = []
        labels = []
        file_info = []
        
        print(f"📁 Loading training data from {data_dir}...")
        
        for idx in range(start_idx, end_idx + 1):
            file_path = os.path.join(data_dir, f'sample{idx}.csv')
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.strip().str.lower()
                    
                    if 'dwell_time' not in df.columns or 'flight_time' not in df.columns:
                        continue
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(df)
                    features = self.feature_extractor.ensure_feature_set(features)
                    
                    features_list.append(features)
                    labels.append(1)  # 1 for legitimate user
                    file_info.append(f'sample{idx}.csv')
                    
                except Exception as e:
                    print(f"⚠️  Error processing {file_path}: {e}")
                    continue
        
        print(f"✅ Loaded {len(features_list)} legitimate samples")
        
        # Generate synthetic impostor data for training
        impostor_features = self.generate_impostor_data(features_list, n_samples=len(features_list))
        impostor_labels = [0] * len(impostor_features)  # 0 for impostor
        
        # Combine data
        all_features = features_list + impostor_features
        all_labels = labels + impostor_labels
        all_file_info = file_info + ['synthetic_impostor'] * len(impostor_features)
        
        return pd.DataFrame(all_features), np.array(all_labels), all_file_info
    
    def generate_impostor_data(self, legitimate_features, n_samples=1000):
        """Generate synthetic impostor data by perturbing legitimate patterns"""
        legit_df = pd.DataFrame(legitimate_features)
        impostor_features = []
        
        for _ in range(n_samples):
            # Select random legitimate sample
            base_sample = legit_df.sample(1).iloc[0].copy()
            
            # Apply perturbations
            perturbation = np.random.normal(1, 0.3, len(base_sample))  # 30% variation
            impostor_sample = base_sample * perturbation
            
            # Ensure feature values are within reasonable bounds
            for col in ['dwell_mean', 'flight_mean', 'total_time']:
                if col in impostor_sample.index:
                    impostor_sample[col] = max(0, impostor_sample[col])
            
            impostor_features.append(impostor_sample.to_dict())
        
        return impostor_features
    
    def train_models(self, X, y):
        """Train multiple classical ML models"""
        print("🚀 Training multiple models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Define models
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=len(y[y==0])/len(y[y==1])
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Train each model
        trained_models = {}
        for name, model in self.models.items():
            print(f"🏃 Training {name}...")
            model.fit(X_scaled, y)
            trained_models[name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            print(f"   📊 {name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return trained_models
    
    def evaluate_models(self, models, X, y):
        """Evaluate all trained models"""
        X_scaled = self.scaler.transform(X)
        
        best_model = None
        best_score = 0
        evaluation_results = {}
        
        print("\n📈 Model Evaluation Results:")
        print("=" * 60)
        
        for name, model in models.items():
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            
            accuracy = np.mean(y_pred == y)
            auc_score = roc_auc_score(y, y_pred_proba)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"🎯 {name}:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   AUC Score: {auc_score:.4f}")
            print(f"   Classification Report:")
            print(classification_report(y, y_pred, target_names=['Impostor', 'Legitimate']))
            
            if auc_score > best_score:
                best_score = auc_score
                best_model = name
        
        print(f"\n🏆 Best Model: {best_model} (AUC: {best_score:.4f})")
        return best_model, evaluation_results
    
    def save_models(self, models, best_model_name, feature_names, output_dir='models'):
        """Save trained models and metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual models
        for name, model in models.items():
            model_path = os.path.join(output_dir, f'{name.lower()}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save ensemble model data
        ensemble_data = {
            'models': models,
            'best_model': best_model_name,
            'scaler': self.scaler,
            'feature_names': feature_names,
            'feature_extractor': self.feature_extractor
        }
        
        ensemble_path = os.path.join(output_dir, 'keystroke_ensemble_model.pkl')
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        # Save model info
        info_path = os.path.join(output_dir, 'model_info.json')
        model_info = {
            'best_model': best_model_name,
            'available_models': list(models.keys()),
            'feature_count': len(feature_names),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"💾 Models saved to {output_dir}/")
        return ensemble_path
    
    def plot_feature_importance(self, models, feature_names, top_n=15):
        """Plot feature importance for tree-based models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (name, model) in enumerate(models.items()):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]
                
                axes[idx].barh(range(top_n), importances[indices])
                axes[idx].set_yticks(range(top_n))
                axes[idx].set_yticklabels([feature_names[i] for i in indices])
                axes[idx].set_title(f'{name} - Top {top_n} Features')
                axes[idx].set_xlabel('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    trainer = KeystrokeModelTrainer()
    
    # Load data
    X, y, file_info = trainer.load_training_data()
    
    if len(X) == 0:
        print("❌ No training data found!")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print(f"📊 Features: {X_train.shape[1]}")
    
    # Train models
    trained_models = trainer.train_models(X_train, y_train)
    
    # Evaluate models
    best_model, results = trainer.evaluate_models(trained_models, X_test, y_test)
    
    # Save models
    ensemble_path = trainer.save_models(
        trained_models, 
        best_model, 
        list(X.columns)
    )
    
    # Plot feature importance
    trainer.plot_feature_importance(trained_models, list(X.columns))
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Models saved in 'models/' directory")
    print(f"🏆 Best model: {best_model}")

if __name__ == "__main__":
    main()