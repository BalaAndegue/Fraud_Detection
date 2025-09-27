# Fraud Detection - Système de Détection de Fraude Bancaire

* https://img.shields.io/badge/Python-3.9%252B-blue
* https://img.shields.io/badge/Scikit--learn-1.0%252B-orange
* https://img.shields.io/badge/Streamlit-1.12%252B-red
* https://img.shields.io/badge/License-MIT-green

Un système complet de détection de fraude bancaire utilisant le Machine Learning pour identifier les transactions suspectes en temps réel.
📊 Aperçu du Projet

Ce projet implémente un pipeline complet de détection de fraude, de l'exploration des données au déploiement d'un dashboard interactif. Le système utilise des algorithmes de Machine Learning avancés pour détecter les patterns de fraude avec une précision élevée.

## 🎯 Fonctionnalités Principales

* 🔍 Analyse exploratoire des données transactionnelles

* 🤖 Entraînement de multiples modèles de Machine Learning

* 📊 Dashboard interactif pour la détection en temps réel

* ⚡ API de prédiction pour l'intégration système

* 📈 Monitoring des performances du modèle

## Architecture du Projet

```
Fraud_Detection/
│
├── 📁 data/                           # Gestion des données
│   ├── raw/                          # Données brutes
│   ├── processed/                    # Données transformées
│   └── external/                     # Données externes
│
├── 📁 notebooks/                      # Notebooks d'analyse
│   ├── 01_data_exploration.ipynb     # Exploration des données
│   ├── 02_feature_engineering.ipynb  # Feature engineering
│   ├── 03_model_training.ipynb       # Entraînement des modèles
│   └── 04_model_evaluation.ipynb     # Évaluation des performances
│
├── 📁 src/                           # Code source du projet
│   ├── data/                         # Chargement des données
│   │   └── data_loader.py
│   ├── features/                     # Feature engineering
│   │   └── feature_engineer.py
│   ├── models/                       # Modèles ML
│   │   ├── model_trainer.py
│   │   └── fraud_detector.py
│   ├── visualization/                # Visualisations
│   │   └── plotter.py
│   └── utils/                        # Utilitaires
│       └── helpers.py
│
├── 📁 app/                           # Application Streamlit
│   └── dashboard.py                  # Dashboard principal
│
├── 📁 models/                        # Modèles sauvegardés
├── 📁 tests/                         # Tests unitaires
├── 📁 docs/                          # Documentation
│
├── 📄 requirements.txt               # Dépendances Python
├── 📄 environment.yml                # Environnement Conda
└── 📄 README.md                      # Ce fichier

```
## 🚀 Installation et Configuration
### Prérequis

    Python 3.9+

    Anaconda ou Miniconda

    Git

### 📥 Installation

## Cloner le dépôt ##

```bash

git clone https://github.com/BalaAndegue/Fraud_Detection.git
cd Fraud_Detection
```
## Créer l'environnement Conda ##

```bash

conda create -n fraud-detection python=3.9
conda activate fraud-detection
```
## Installer les dépendances ##

```bash

pip install -r requirements.txt
```
## 🔧 Configuration Rapide ##

```bash

# Script d'installation automatique
chmod +x scripts/setup.sh
./scripts/setup.sh
```
📊 Jeu de Données
Source des Données

Le projet utilise le dataset Credit Card Fraud Detection contenant des transactions bancaires réelles anonymisées.

## Caractéristiques principales :

- 📈 284,807 transactions au total

- 🎯 492 fraudes (0.172% des transactions)

- ⏰ Période : 2 jours de transactions

- 💰 Montants : de $0 à $25,691

## Variables du Dataset

| Variable | Description                                        | Type    |
|----------|--------------------------------------------------|---------|
| Time     | Secondes écoulées depuis la première transaction | Numérique |
| V1-V28   | Features principales obtenues par PCA             | Numérique |
| Amount   | Montant de la transaction                         | Numérique |
| Class    | Cible (1=Fraude, 0=Normal)                       | Binaire |

---

## 🧠 Modèles de Machine Learning Implémentés

### Algorithmes Comparés et leurs Métriques de Performance

| Modèle              | AUC Score | Precision | Recall | F1-Score |
|---------------------|-----------|-----------|--------|----------|
| Random Forest       | 0.95+     | 0.85+     | 0.80+  | 0.82+    |
| XGBoost             | 0.96+     | 0.86+     | 0.82+  | 0.84+    |
| LightGBM            | 0.97+     | 0.87+     | 0.83+  | 0.85+    |
| Logistic Regression | 0.92+     | 0.80+     | 0.75+  | 0.77+    |

---

## 📊 Métriques de Performance

```python

# Exemple de performance attendue

{
    "auc_score": 0.976,
    "precision": 0.864,
    "recall": 0.823, 
    "f1_score": 0.843,
    "accuracy": 0.995
}

```

## 💻 Utilisation
1. 🏃‍♂️ Exécution du Pipeline Complet
bash

# Lancer l'analyse complète
python run_pipeline.py

# Ou exécuter étape par étape
```bash
python -m src.data.data_loader          # Chargement données
python -m src.features.feature_engineer # Feature engineering
python -m src.models.model_trainer      # Entraînement modèles
```
2. 📊 Lancement du Dashboard
bash

streamlit run app/dashboard.py

# Accéder au dashboard : http://localhost:8501

3. 🧪 Utilisation en Mode API
```python

from src.models.fraud_detector import FraudDetector

# Charger le modèle
detector = FraudDetector('models/best_model.pkl')

# Faire une prédiction
transaction = {
    'Time': 406,
    'V1': -1.015, 'V2': 0.240, 'V3': -0.378,
    'Amount': 149.62
}

result = detector.predict(transaction)
print(f"Fraude détectée: {result['is_fraud']}")
print(f"Confiance: {result['probability_fraud']:.2%}")
```
📈 Résultats et Visualisations
Performance des Modèles

- https://docs/images/model_comparison.png
Analyse des Features Importantes

- https://docs/images/feature_importance.png
Courbe ROC

- https://docs/images/roc_curve.png
🛠️ Développement
Structure des Tests

```bash

# Exécuter tous les tests
pytest tests/

# Tests spécifiques
pytest tests/test_data_loader.py
pytest tests/test_feature_engineer.py
pytest tests/test_models.py
```

🏗️ Ajout de Nouveaux Modèles
```python

# Dans src/models/model_trainer.py
class ModelTrainer:
    def add_custom_model(self, name, model):
        """Ajouter un nouveau modèle à l'évaluation"""
        self.models[name] = model
```
🔧 Configuration Avancée

Créer un fichier config.yaml pour personnaliser les paramètres :
```yaml

model:
  test_size: 0.2
  random_state: 42
  scoring: 'roc_auc'

preprocessing:
  smote_sampling: true
  scale_features: true
  feature_selection: true

dashboard:
  port: 8501
  theme: 'dark'
```
🌐 Déploiement
Déploiement Local avec Docker
```dockerfile

# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501"]
```
```bash

# Construction et lancement
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection

Déploiement Cloud (AWS/GCP/Azure)

````
```bash

# Script de déploiement AWS
chmod +x scripts/deploy_aws.sh
./scripts/deploy_aws.sh
```
📚 Documentation Technique
API Reference
FraudDetector Class

```python

class FraudDetector:
    def __init__(self, model_path: str):
        """Charge un modèle pré-entraîné"""
    
    def predict(self, transaction_data: dict) -> dict:
        """Prédit si une transaction est frauduleuse"""
    
    def predict_batch(self, transactions: list) -> list:
        """Prédit plusieurs transactions"""
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modèle"""
``` 
FeatureEngineer Class

```python



class FeatureEngineer:
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée de nouvelles features à partir des données brutes"""
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données et gère les valeurs manquantes"""