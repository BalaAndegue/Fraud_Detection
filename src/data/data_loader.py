import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataLoader:
    def __init__(self, data_path='./data/'):
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / 'raw'
        self.processed_path = self.data_path / 'processed'
        
        # Création des dossiers si nécessaire
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self):
        """Charge les données brutes ou crée des données simulées"""
        csv_path = self.raw_path / 'creditcard.csv'
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f" Données chargées: {df.shape[0]} transactions")
            return df
        else:
            print(" Création de données simulées...")
            return self._create_simulated_data()
    
    def _create_simulated_data(self, n_samples=100000):
        """Crée des données simulées réalistes"""
        np.random.seed(42)
        
        n_frauds = int(n_samples * 0.0017)  # 0.17% comme le vrai dataset
        
        data = {}
        # Création de features similaires au vrai dataset
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1.5 - (i * 0.05), n_samples)
        
        data['Time'] = np.arange(n_samples)
        data['Amount'] = np.random.exponential(88, n_samples)  # Moyenne ~88€
        
        df = pd.DataFrame(data)
        
        # Création des fraudes avec des patterns différents
        fraud_indices = np.random.choice(n_samples, n_frauds, replace=False)
        df['Class'] = 0
        df.loc[fraud_indices, 'Class'] = 1
        
        # Les fraudes ont des patterns différents
        for i in range(1, 15):
            df.loc[fraud_indices, f'V{i}'] = np.random.normal(1.5, 1, n_frauds)
        
        df.loc[fraud_indices, 'Amount'] = np.random.exponential(200, n_frauds)
        
        # Sauvegarde des données simulées
        df.to_csv(self.raw_path / 'creditcard.csv', index=False)
        print(f"Données simulées créées: {n_frauds} fraudes sur {n_samples} transactions")
        
        return df

# Test simple si le fichier est exécuté directement
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_raw_data()
    print(f"Test réussi! DataFrame shape: {df.shape}")
