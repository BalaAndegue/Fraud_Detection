import pandas as pd
import numpy as np
from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer

print("🧪 TEST DE NETTOYAGE DES DONNÉES")

loader = DataLoader()
df = loader.load_raw_data()

print(f"1. Données brutes: {df.shape}")
print(f"   NaN dans données brutes: {df.isnull().sum().sum()}")

fe = FeatureEngineer()
df_engineered = fe.create_simple_features(df)

print(f"2. Après feature engineering: {df_engineered.shape}")
print(f"   NaN après engineering: {df_engineered.isnull().sum().sum()}")

# Nettoyage
X = df_engineered.drop('Class', axis=1)
y = df_engineered['Class']
X = X.select_dtypes(include=[np.number])
X = X.dropna(axis=1)

print(f"3. Après nettoyage: {X.shape}")
print(f"   NaN après nettoyage: {X.isnull().sum().sum()}")

print("✅ PRÊT POUR L'ENTRAÎNEMENT!")
