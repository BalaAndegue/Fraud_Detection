import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()  # Plus robuste aux outliers
        
    def fit(self, X, y=None):
        # Si X est un DataFrame, scale les colonnes numériques
        if hasattr(X, 'columns'):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            self.scaler.fit(X[numeric_cols])
        else:
            self.scaler.fit(X)
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        if hasattr(X, 'columns'):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_transformed[numeric_cols] = self.scaler.transform(X[numeric_cols])
        else:
            X_transformed = self.scaler.transform(X)
            
        return X_transformed
    
    def create_features(self, df):
        """Crée de nouvelles features en gérant les NaN"""
        df_engineered = df.copy()
        
        # Features basées sur le temps
        df_engineered['hour_of_day'] = (df_engineered['Time'] // 3600) % 24
        df_engineered['is_night'] = ((df_engineered['hour_of_day'] >= 22) | 
                                   (df_engineered['hour_of_day'] <= 6)).astype(int)
        
        # Features basées sur le montant - GESTION DES NaN
        df_engineered['amount_log'] = np.log1p(df_engineered['Amount'])
        
        # Création de catégories de montant avec gestion des bords
        amount_bins = [0, 10, 50, 100, 500, df_engineered['Amount'].max() + 1]
        df_engineered['amount_category'] = pd.cut(df_engineered['Amount'], 
                                                bins=amount_bins,
                                                labels=[0, 1, 2, 3, 4])
        
        # Remplacer les NaN dans amount_category par la valeur la plus fréquente
        if df_engineered['amount_category'].isnull().any():
            most_frequent = df_engineered['amount_category'].mode()[0]
            df_engineered['amount_category'] = df_engineered['amount_category'].fillna(most_frequent)
        
        # Interactions entre features - avec vérification des colonnes
        for i in range(1, 6):
            col_name = f'V{i}'
            if col_name in df_engineered.columns:
                df_engineered[f'V{i}_amount_interaction'] = df_engineered[col_name] * df_engineered['Amount']
        
        # SUPPRIMER LES COLONNES AVEC NaN OU LES REMPLACER
        df_engineered = df_engineered.dropna(axis=1)  # Supprime les colonnes avec NaN
        
        return df_engineered
    
    def clean_data(self, df):
        """Nettoie les données en supprimant les NaN"""
        # Faire une copie
        df_clean = df.copy()
        
        # Supprimer les colonnes avec trop de NaN
        df_clean = df_clean.dropna(axis=1, how='any')
        
        # Vérifier s'il reste des NaN
        if df_clean.isnull().any().any():
            # Remplacer les NaN restants par la médiane
            for col in df_clean.select_dtypes(include=[np.number]).columns:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        return df_clean