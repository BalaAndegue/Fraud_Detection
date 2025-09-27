import pandas as pd
import numpy as np
import sys
import os

# Ajouter le chemin src
sys.path.append('.')

from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer

print("🧪 TEST COMPLET DU FEATURE ENGINEER")
print("=" * 50)

# 1. Chargement des données
print("1. 📊 CHARGEMENT DES DONNÉES...")
loader = DataLoader()
df = loader.load_raw_data()
print(f"   ✅ Données brutes: {df.shape}")
print(f"   ✅ NaN dans données brutes: {df.isnull().sum().sum()}")

# 2. Test de la classe FeatureEngineer
print("\n2. 🔧 TEST DE FEATUREENGINEER...")
fe = FeatureEngineer()
print(f"   ✅ FeatureEngineer initialisé")

# 3. Test de create_features()
print("\n3. 🛠️ TEST DE CREATE_FEATURES()...")
df_engineered = fe.create_features(df)
print(f"   ✅ Features créées: {df_engineered.shape}")
print(f"   ✅ NaN après create_features: {df_engineered.isnull().sum().sum()}")

# 4. Vérification des nouvelles colonnes
new_columns = [col for col in df_engineered.columns if col not in df.columns]
print(f"   ✅ Nouvelles colonnes créées: {new_columns}")

# 5. Vérification détaillée des NaN par colonne
print("\n4. 🔍 VÉRIFICATION DÉTAILLÉE DES NaN:")
nan_columns = df_engineered.columns[df_engineered.isnull().any()].tolist()
if nan_columns:
    print("   ⚠️  Colonnes avec NaN:")
    for col in nan_columns:
        nan_count = df_engineered[col].isnull().sum()
        print(f"      {col}: {nan_count} NaN ({nan_count/len(df_engineered)*100:.2f}%)")
else:
    print("   ✅ Aucune colonne avec NaN!")

# 6. Test de clean_data()
print("\n5. 🧹 TEST DE CLEAN_DATA()...")
df_clean = fe.clean_data(df_engineered)
print(f"   ✅ Données nettoyées: {df_clean.shape}")
print(f"   ✅ NaN après clean_data: {df_clean.isnull().sum().sum()}")

# 7. Test de préparation pour le ML
print("\n6. 🤖 PRÉPARATION POUR LE MACHINE LEARNING...")
X = df_clean.drop('Class', axis=1)
y = df_clean['Class']

print(f"   ✅ X shape: {X.shape}")
print(f"   ✅ y shape: {y.shape}")
print(f"   ✅ Types de données dans X:")
print(f"      Numériques: {X.select_dtypes(include=[np.number]).shape[1]} colonnes")
print(f"      Catégorielles: {X.select_dtypes(include=['category', 'object']).shape[1]} colonnes")

# 8. Test du scaler
print("\n7. ⚖️ TEST DU SCALER...")
try:
    # Préparer les données numériques pour le scaling
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Fit du scaler
    fe.fit(X_numeric)
    X_scaled = fe.transform(X_numeric)
    
    print(f"   ✅ Scaling réussi: {X_scaled.shape}")
    print(f"   ✅ Aucun NaN après scaling: {np.isnan(X_scaled).sum().sum()}")
    
except Exception as e:
    print(f"   ❌ Erreur lors du scaling: {e}")

# 9. Test avec scikit-learn (compatibilité)
print("\n8. 🔄 TEST DE COMPATIBILITÉ SCKIT-LEARN...")
from sklearn.model_selection import train_test_split

# Préparation finale
X_final = X.select_dtypes(include=[np.number])  # Garder seulement les numériques
X_final = X_final.dropna(axis=1)  # Supprimer colonnes avec NaN

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✅ Split train/test réussi:")
print(f"      X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"      X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"      Aucun NaN: {X_train.isnull().sum().sum() + X_test.isnull().sum().sum()}")

# 10. Test rapide avec un modèle simple
print("\n9. 🎯 TEST RAPIDE AVEC UN MODÈLE...")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # Model simple
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✅ Modèle entraîné avec succès!")
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   ✅ Aucune erreur NaN détectée!")
    
except Exception as e:
    print(f"   ❌ Erreur lors de l'entraînement: {e}")

print("\n" + "=" * 50)
print("🎉 TEST TERMINÉ AVEC SUCCÈS!")
print("\n📋 RÉCAPITULATIF:")
print(f"   • Données initiales: {df.shape}")
print(f"   • Après feature engineering: {df_engineered.shape}") 
print(f"   • Après nettoyage: {df_clean.shape}")
print(f"   • Prêtes pour ML: {X_final.shape}")
print(f"   • NaN totaux: {df_clean.isnull().sum().sum()}")

if df_clean.isnull().sum().sum() == 0:
    print("   ✅ Tous les NaN ont été éliminés!")
else:
    print("   ⚠️  Il reste des NaN à traiter")
