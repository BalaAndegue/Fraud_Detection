from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(class_weight='balanced', random_state=42),
            'random_forest': RandomForestClassifier(class_weight='balanced', random_state=42),
            'xgboost': XGBClassifier(scale_pos_weight=100, random_state=42),
            'lightgbm': LGBMClassifier(class_weight='balanced', random_state=42)
        }
        self.best_model = None
        self.best_score = 0
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Entraîne et évalue tous les modèles"""
        results = {}
        
        for name, model in self.models.items():
            print(f"🔧 Entraînement de {name}...")
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Métriques
            auc_score = roc_auc_score(y_test, y_proba)
            
            results[name] = {
                'model': model,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            print(f"✅ {name} - AUC: {auc_score:.4f}")
            
            # Mise à jour du meilleur modèle
            if auc_score > self.best_score:
                self.best_score = auc_score
                self.best_model = model
                
        return results
    
    def get_best_model(self):
        """Retourne le meilleur modèle"""
        return self.best_model, self.best_score
