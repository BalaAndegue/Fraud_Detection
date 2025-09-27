# app/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import sys
import os

# Ajouter le chemin src
sys.path.append('..')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Détection de Fraude",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .fraud-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .normal-transaction {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00ff00;
    }
</style>
""", unsafe_allow_html=True)

class FraudDashboard:
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Charge les modèles sauvegardés"""
        try:
            # Chemins absolus vers les modèles
            models_base_path = '/home/bala/fraud-detection-project/notebooks/models'
            
            # Vérifier si le dossier existe
            if not os.path.exists(models_base_path):
                st.sidebar.warning(f"⚠️ Dossier models non trouvé: {models_base_path}")
                st.sidebar.info("ℹ️ Utilisation du mode démo")
                self.model = None
                return
            
            # Chercher tous les fichiers .pkl dans le dossier models
            all_pkl_files = [f for f in os.listdir(models_base_path) if f.endswith('.pkl')]
            
            if not all_pkl_files:
                st.sidebar.warning("⚠️ Aucun fichier .pkl trouvé dans le dossier models")
                st.sidebar.info("ℹ️ Utilisation du mode démo")
                self.model = None
                return
            
            # Préférer fraud_pipeline_*.pkl, sinon fraud_detector_*.pkl, sinon autre .pkl
            pipeline_files = [f for f in all_pkl_files if f.startswith('fraud_pipeline_')]
            detector_files = [f for f in all_pkl_files if f.startswith('fraud_detector_')]
            other_pkl_files = [f for f in all_pkl_files if not f.startswith(('fraud_pipeline_', 'fraud_detector_'))]
            
            model_file = None
            model_type = ""
            
            if pipeline_files:
                model_file = sorted(pipeline_files)[-1]  # Le plus récent
                model_type = "pipeline"
            elif detector_files:
                model_file = sorted(detector_files)[-1]  # Le plus récent  
                model_type = "detector"
            elif other_pkl_files:
                model_file = sorted(other_pkl_files)[-1]  # Le plus récent
                model_type = "model"
            
            if model_file:
                full_model_path = os.path.join(models_base_path, model_file)
                st.sidebar.info(f"🔍 Chargement du modèle: {model_file}")
                
                if model_type == "pipeline":
                    # Charger le pipeline complet
                    pipeline = joblib.load(full_model_path)
                    self.model = pipeline.get('model')
                    self.feature_engineer = pipeline.get('feature_engineer')
                    self.metadata = {
                        'model_type': pipeline.get('model_type', 'Inconnu'),
                        'auc_score': pipeline.get('auc_score', 0),
                        'timestamp': pipeline.get('timestamp', 'Inconnu'),
                        'features': pipeline.get('feature_names', [])
                    }
                else:
                    # Charger le modèle seul
                    self.model = joblib.load(full_model_path)
                    self.metadata = {
                        'model_type': type(self.model).__name__,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'features': ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                    }
                
                # Chercher les métadonnées JSON séparées
                json_files = [f for f in os.listdir(models_base_path) if f.endswith('.json')]
                metadata_files = [f for f in json_files if f.startswith('model_metadata_') or f.startswith('metadata_')]
                
                if metadata_files:
                    latest_metadata = sorted(metadata_files)[-1]
                    metadata_path = os.path.join(models_base_path, latest_metadata)
                    with open(metadata_path, 'r') as f:
                        json_metadata = json.load(f)
                        # Fusionner avec les métadonnées existantes
                        if hasattr(self, 'metadata'):
                            self.metadata.update(json_metadata)
                        else:
                            self.metadata = json_metadata
                
                st.sidebar.success(f"✅ Modèle chargé avec succès!")
                st.sidebar.success(f"📊 Type: {self.metadata.get('model_type', 'Inconnu')}")
                st.sidebar.success(f"🎯 AUC: {self.metadata.get('auc_score', 'N/A')}")
                
            else:
                st.sidebar.warning("⚠️ Aucun modèle valide trouvé")
                self.model = None
                    
        except Exception as e:
            st.sidebar.error(f"❌ Erreur chargement modèle: {str(e)}")
            import traceback
            st.sidebar.error(f"🔍 Détails: {traceback.format_exc()}")
            self.model = None

    def load_sample_data(self):
        """Charge des données d'exemple"""
        try:
            from src.data.data_loader import DataLoader
            loader = DataLoader()
            return loader.load_raw_data()
        except:
            # Données simulées en cas d'erreur
            np.random.seed(42)
            n_samples = 1000
            data = {
                'Time': np.arange(n_samples),
                'V1': np.random.normal(0, 1, n_samples),
                'V2': np.random.normal(0, 1, n_samples),
                'Amount': np.random.exponential(100, n_samples),
                'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
            }
            return pd.DataFrame(data)
    
    def predict_fraud(self, transaction_data):
        """Prédit si une transaction est frauduleuse"""
        if self.model is None:
            # Mode démo - simulation
            return {
                'prediction': np.random.choice([0, 1], p=[0.95, 0.05]),
                'probability_fraud': np.random.uniform(0, 0.3),
                'is_fraud': False
            }
        
        try:
            # Prédiction réelle
            if isinstance(transaction_data, dict):
                df = pd.DataFrame([transaction_data])
            else:
                df = transaction_data.copy()
            
            # Sélection des features nécessaires
            features_needed = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29) if f'V{i}' in df.columns]
            df = df[features_needed]
            
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0]
            
            return {
                'prediction': prediction,
                'probability_fraud': float(probability[1]),
                'probability_normal': float(probability[0]),
                'is_fraud': bool(prediction == 1)
            }
        except Exception as e:
            st.error(f"Erreur prédiction: {e}")
            return {'prediction': 0, 'probability_fraud': 0.0, 'is_fraud': False}
    
    def display_overview(self, df):
        """Affiche la vue d'ensemble"""
        st.markdown('<div class="main-header">🎯 Dashboard de Détection de Fraude</div>', unsafe_allow_html=True)
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        
        with col2:
            fraud_count = df['Class'].sum()
            st.metric("Transactions Frauduleuses", f"{fraud_count:,}")
        
        with col3:
            fraud_rate = (fraud_count / len(df)) * 100
            st.metric("Taux de Fraude", f"{fraud_rate:.4f}%")
        
        with col4:
            avg_amount = df['Amount'].mean()
            st.metric("Montant Moyen", f"${avg_amount:.2f}")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribution des Transactions")
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Class'].value_counts().plot.pie(
                autopct='%1.1f%%', 
                colors=['lightgreen', 'lightcoral'],
                labels=['Normales', 'Fraudes'],
                ax=ax
            )
            ax.set_ylabel('')
            st.pyplot(fig)
        
        with col2:
            st.subheader("💰 Distribution des Montants")
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Amount'].hist(bins=50, ax=ax, alpha=0.7, color='skyblue')
            ax.set_xlabel('Montant ($)')
            ax.set_ylabel('Nombre de Transactions')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    def display_real_time_detection(self):
        """Affiche l'outil de détection en temps réel"""
        st.subheader("🔍 Détection de Fraude en Temps Réel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Paramètres de la Transaction")
            
            # Formulaire de saisie
            amount = st.number_input("Montant de la transaction ($)", 
                                   min_value=0.0, 
                                   max_value=10000.0, 
                                   value=100.0,
                                   step=10.0)
            
            time_of_day = st.slider("Heure de la journée", 0, 23, 12)
            
            # Features simulées (dans un vrai cas, elles viendraient du processing)
            v1 = st.slider("V1 (Feature de comportement)", -5.0, 5.0, 0.0, 0.1)
            v2 = st.slider("V2 (Feature de comportement)", -5.0, 5.0, 0.0, 0.1)
            v3 = st.slider("V3 (Feature de comportement)", -5.0, 5.0, 0.0, 0.1)
        
        with col2:
            st.markdown("### Résultat de l'Analyse")
            
            # Préparation des données pour la prédiction
            transaction_data = {
                'Time': time_of_day * 3600,  # Conversion en secondes
                'Amount': amount,
                'V1': v1,
                'V2': v2,
                'V3': v3,
                # Ajouter d'autres features avec des valeurs par défaut
                **{f'V{i}': 0.0 for i in range(4, 29)}
            }
            
            # Prédiction
            if st.button("🔎 Analyser la Transaction", type="primary"):
                with st.spinner("Analyse en cours..."):
                    result = self.predict_fraud(transaction_data)
                    
                    # Affichage du résultat
                    if result['is_fraud']:
                        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
                        st.error("🚨 TRANSACTION SUSPECTE DÉTECTÉE!")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="normal-transaction">', unsafe_allow_html=True)
                        st.success("✅ TRANSACTION NORMALE")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Détails de la prédiction
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Probabilité de Fraude", 
                                f"{result['probability_fraud']*100:.2f}%")
                    
                    with col2:
                        st.metric("Niveau de Confiance", 
                                f"{(1 - abs(result['probability_fraud'] - 0.5))*100:.1f}%")
                    
                    # Barre de probabilité
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.barh(['Risque'], [result['probability_fraud']*100], 
                           color='red' if result['is_fraud'] else 'green', alpha=0.6)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Probabilité de Fraude (%)')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
    
    def display_model_info(self):
        """Affiche les informations du modèle"""
        st.subheader("🤖 Informations du Modèle")
        
        if hasattr(self, 'metadata'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Score AUC", f"{self.metadata.get('auc_score', 0):.4f}")
                st.metric("Type de Modèle", self.metadata.get('model_type', 'Inconnu'))
            
            with col2:
                st.metric("Date d'Entraînement", 
                         self.metadata.get('timestamp', 'Inconnu'))
                st.metric("Nombre de Features", 
                         len(self.metadata.get('features', [])))
            
            # Features importantes (si disponibles)
            if 'features' in self.metadata:
                st.write("**Features utilisées:**")
                features = self.metadata['features']
                cols = st.columns(3)
                for i, feature in enumerate(features[:15]):  # Afficher les 15 premières
                    cols[i % 3].write(f"• {feature}")
        else:
            st.info("ℹ️ Mode démonstration - Données simulées")
    
    def display_data_explorer(self, df):
        """Explorateur de données"""
        st.subheader("📈 Explorateur de Données")
        
        tab1, tab2, tab3 = st.tabs(["Données Brutes", "Statistiques", "Visualisations"])
        
        with tab1:
            st.dataframe(df.head(1000), use_container_width=True)
        
        with tab2:
            st.write("**Statistiques descriptives:**")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.write("**Valeurs manquantes:**")
            missing_data = df.isnull().sum()
            st.dataframe(missing_data[missing_data > 0], use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Heatmap de corrélation
                st.write("**Matrice de Corrélation:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                corr_matrix = df[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix.iloc[:10, :10], annot=True, cmap='coolwarm', 
                           center=0, ax=ax)
                st.pyplot(fig)
            
            with col2:
                # Distribution des montants par classe
                st.write("**Montants par Type de Transaction:**")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for class_val, color, label in [(0, 'blue', 'Normale'), (1, 'red', 'Fraude')]:
                    data = df[df['Class'] == class_val]['Amount']
                    ax.hist(data, bins=30, alpha=0.6, color=color, label=label)
                
                ax.legend()
                ax.set_xlabel('Montant ($)')
                ax.set_ylabel('Fréquence')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
    
    def run(self):
        """Lance le dashboard"""
        # Sidebar
        st.sidebar.title("🎯 Navigation")
        page = st.sidebar.radio("Sections", 
                               ["Vue d'Ensemble", "Détection Temps Réel", "Explorateur de Données", "Info Modèle"])
        
        # Chargement des données
        df = self.load_sample_data()
        
        # Affichage de la page sélectionnée
        if page == "Vue d'Ensemble":
            self.display_overview(df)
        elif page == "Détection Temps Réel":
            self.display_real_time_detection()
        elif page == "Explorateur de Données":
            self.display_data_explorer(df)
        elif page == "Info Modèle":
            self.display_model_info()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **📊 Dashboard de Détection de Fraude**
        
        Ce dashboard permet de:
        - Visualiser les données de transactions
        - Détecter les fraudes en temps réel
        - Explorer les patterns de fraude
        - Monitorer les performances du modèle
        """)

# Lancement du dashboard
if __name__ == "__main__":
    dashboard = FraudDashboard()
    dashboard.run()