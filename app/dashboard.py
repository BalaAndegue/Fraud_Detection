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

# ✅ CORRECTION CRITIQUE : Ajouter le chemin racine du projet
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Remonte d'un niveau (app -> projet)
sys.path.insert(0, project_root)

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
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        border-bottom: 3px solid #3498db;
        padding-bottom: 1rem;
    } 
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px; 
        margin: 0.5rem; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
    } 
    .fraud-alert { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
        color: white; 
        padding: 1.5rem; 
        border-radius: 12px; 
        margin: 1rem 0; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
    } 
    .normal-transaction { 
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%); 
        color: white; 
        padding: 1.5rem; 
        border-radius: 12px; 
        margin: 1rem 0; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
    } 
    .section-header { 
        font-size: 1.8rem; 
        color: #2c3e50; 
        margin: 2rem 0 1rem 0; 
        font-weight: 600; 
        border-left: 4px solid #3498db; 
        padding-left: 1rem; 
    } 
    .sidebar-section { 
        background: #f8f9fa; 
        padding: 1rem; 
        border-radius: 8px; 
        margin: 1rem 0; 
    }
</style>
""", unsafe_allow_html=True)


class FraudDashboard:
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.feature_names = None
        self.required_features = []
        self.metadata = {}
        self._sample_df_cache = None
        self.load_models()

    def load_models(self):
        """Charge les modèles sauvegardés"""
        try:
            models_base_path = '/home/bala/fraud-detection-project/notebooks/models'

            if not os.path.exists(models_base_path):
                st.sidebar.warning(f"Dossier models non trouvé: {models_base_path}")
                st.sidebar.info("Utilisation du mode démo")
                self.model = None
                return

            all_pkl_files = [f for f in os.listdir(models_base_path) if f.endswith('.pkl')]

            if not all_pkl_files:
                st.sidebar.warning("Aucun fichier .pkl trouvé dans le dossier models")
                st.sidebar.info("Utilisation du mode démo")
                self.model = None
                return

            # Choix du modèle : prendre le plus récent alphanumériquement
            pipeline_files = [f for f in all_pkl_files if f.startswith('fraud_pipeline_')]
            detector_files = [f for f in all_pkl_files if f.startswith('fraud_detector_')]
            other_pkl_files = [f for f in all_pkl_files if f not in pipeline_files + detector_files]

            model_file = None
            if pipeline_files:
                model_file = sorted(pipeline_files)[-1]
            elif detector_files:
                model_file = sorted(detector_files)[-1]
            elif other_pkl_files:
                model_file = sorted(other_pkl_files)[-1]

            if model_file:
                full_model_path = os.path.join(models_base_path, model_file)
                st.sidebar.info(f"Chargement du modèle: {model_file}")

                pipeline = joblib.load(full_model_path)

                # Si c'est un pipeline sauvegardé comme dict
                if isinstance(pipeline, dict):
                    self.model = pipeline.get('model')
                    self.feature_engineer = pipeline.get('feature_engineer')
                    self.metadata = {
                        'model_type': pipeline.get('model_type', 'Inconnu'),
                        'auc_score': pipeline.get('auc_score', 0),
                        'timestamp': pipeline.get('timestamp', 'Inconnu'),
                        'features': pipeline.get('feature_names', [])
                    }
                else:
                    # Sinon c'est directement un modèle
                    self.model = pipeline
                    self.metadata = {
                        'model_type': type(self.model).__name__,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'features': []
                    }

                # ✅ Récupérer les features attendues directement depuis le modèle
                if hasattr(self.model, "feature_names_in_"):
                    self.required_features = list(self.model.feature_names_in_)
                elif hasattr(self.model, "feature_names"):
                    self.required_features = list(self.model.feature_names)
                else:
                    # fallback si le modèle ne stocke pas les features
                    self.required_features = self.metadata.get('features', [])

                st.sidebar.success("Modèle chargé avec succès!")
                st.sidebar.success(f"Type: {self.metadata.get('model_type', 'Inconnu')}")
                st.sidebar.success(f"AUC: {self.metadata.get('auc_score', 'N/A')}")
                st.sidebar.info(f"Features requises: {len(self.required_features)}")

            else:
                st.sidebar.warning("Aucun modèle valide trouvé")
                self.model = None

        except Exception as e:
            st.sidebar.error(f"Erreur chargement modèle: {str(e)}")
            self.model = None

    def create_engineered_features(self, transaction_data):
        """Crée les features d'ingénierie nécessaires pour le modèle"""
        try:
            # Créer une copie des données
            df = pd.DataFrame([transaction_data]).copy()

            # Features de temps
            if 'Time' in df.columns:
                df['hour_of_day'] = (df['Time'] / 3600) % 24
                df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
            else:
                df['hour_of_day'] = 0.0
                df['is_night'] = 0

            # Features logarithmiques
            if 'Amount' in df.columns:
                df['amount_log'] = np.log1p(df['Amount'])
            else:
                df['amount_log'] = 0.0

            # Features d'interaction
            for i in range(1, 6):  # V1 à V5
                if f'V{i}' in df.columns:
                    df[f'V{i}_amount_interaction'] = df[f'V{i}'] * df.get('Amount', 0.0)

            # Si certaines V1..V28 manquent, laisser le choix : 0 ou sampling aléatoire
            for i in range(1, 29):
                col = f'V{i}'
                if col not in df.columns:
                    # Valeur par défaut 0.0 (production) ; on peut choisir d'échantillonner pour les tests
                    df[col] = 0.0

            # S'assurer que toutes les features requises sont présentes
            for feature in self.required_features:
                if feature not in df.columns:
                    # Donner une valeur par défaut raisonnée
                    if 'interaction' in feature:
                        df[feature] = 0.0
                    elif 'log' in feature:
                        df[feature] = np.log1p(df.get('Amount', 0.0))
                    elif 'hour' in feature:
                        df[feature] = df.get('hour_of_day', 0.0)
                    elif 'night' in feature:
                        df[feature] = df.get('is_night', 0)
                    else:
                        df[feature] = 0.0

            # Réorganiser les colonnes dans l'ordre attendu par le modèle (si défini)
            if self.required_features:
                df = df.reindex(columns=self.required_features, fill_value=0.0)

            return df

        except Exception as e:
            st.error(f"Erreur création des features: {e}")
            # En cas d'erreur, retourner les données de base
            return pd.DataFrame([transaction_data])

    def load_sample_data(self):
        """Charge des données d'exemple"""
        try:
            # Essayer de charger via le DataLoader de ton projet (si présent)
            from src.data.data_loader import DataLoader
            loader = DataLoader()
            df = loader.load_raw_data()
            st.sidebar.success(f'Données chargées: {len(df):,} transactions')
            return df
        except Exception:
            # Données simulées en cas d'erreur
            np.random.seed(42)
            n_samples = 100000
            data = {
                'Time': np.random.randint(0, 24*3600, size=n_samples),
            }
            # Générer V1..V28 pseudo-PCA
            for i in range(1, 29):
                data[f'V{i}'] = np.random.normal(0, 1, n_samples)
            data['Amount'] = np.random.exponential(100, n_samples)
            data['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
            df = pd.DataFrame(data)
            st.sidebar.info('Données simulées chargées')
            return df

    def sample_real_transaction(self):
        """Retourne une transaction réelle (ou simulée) depuis le dataset pour tester"""
        if self._sample_df_cache is None:
            self._sample_df_cache = self.load_sample_data()

        df = self._sample_df_cache
        # Choisir aléatoirement une ligne (préférer une fraude quand possible)
        fraud_rows = df[df['Class'] == 1]
        if not fraud_rows.empty and np.random.rand() < 0.5:
            row = fraud_rows.sample(1).iloc[0]
        else:
            row = df.sample(1).iloc[0]

        tx = row.to_dict()
        # Time en secondes -> hour
        tx['Time'] = int(tx.get('Time', 0))
        return tx

    def predict_fraud(self, transaction_data):
        """Prédit si une transaction est frauduleuse"""
        if self.model is None:
            # fallback: modèle non chargé
            return {
                'prediction': np.random.choice([0, 1], p=[0.95, 0.05]),
                'probability_fraud': np.random.uniform(0, 0.3),
                'is_fraud': False
            }

        try:
            # Conversion en DataFrame avec features d'ingénierie
            if isinstance(transaction_data, dict):
                df = self.create_engineered_features(transaction_data)
            else:
                df = transaction_data.copy()
                if self.required_features and not all(feat in df.columns for feat in self.required_features):
                    df = self.create_engineered_features(df.iloc[0].to_dict())

            # Debug : afficher ce qu'on envoie au modèle
            st.sidebar.write("⚠️ Debug features envoyées au modèle")
            st.sidebar.write("Shape features envoyées au modèle:", df.shape)
            st.sidebar.write("Colonnes envoyées:", list(df.columns[:10]) + ['...'])
            st.sidebar.write("Premières valeurs:", df.iloc[0].to_dict())

            # Réordonner selon les features attendues par le modèle
            if self.required_features:
                # Afficher différences
                missing = set(self.required_features) - set(df.columns)
                extra = set(df.columns) - set(self.required_features)
                if missing:
                    st.sidebar.warning(f"Features manquantes (remplies par défaut): {sorted(list(missing))[:10]}{'...' if len(missing)>10 else ''}")
                if extra:
                    st.sidebar.info(f"Features en trop (ignorées): {sorted(list(extra))[:10]}{'...' if len(extra)>10 else ''}")

                df = df.reindex(columns=self.required_features, fill_value=0.0)

            # Prédiction
            prediction = self.model.predict(df)[0]
            # Certains modèles ne fournissent pas predict_proba
            try:
                probability = self.model.predict_proba(df)[0]
            except Exception:
                # Si pas de predict_proba, construire un faux score
                probability = [1.0, 0.0] if prediction == 0 else [0.0, 1.0]

            return {
                'prediction': int(prediction),
                'probability_fraud': float(probability[1]),
                'probability_normal': float(probability[0]),
                'is_fraud': bool(prediction == 1)
            }
        except Exception as e:
            st.error(f"Erreur prédiction: {e}")
            import traceback
            st.error(f"Détails: {traceback.format_exc()}")
            return {'prediction': 0, 'probability_fraud': 0.0, 'is_fraud': False}

    def display_overview(self, df):
        """Affiche la vue d'ensemble"""
        st.markdown('<div class="main-header">Dashboard de Détection de Fraude</div>', unsafe_allow_html=True)

        # Métriques principales avec nouveau design
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Transactions", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            fraud_count = df['Class'].sum()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Transactions Frauduleuses", f"{fraud_count:,}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            fraud_rate = (fraud_count / len(df)) * 100
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Taux de Fraude", f"{fraud_rate:.4f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            avg_amount = df['Amount'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Montant Moyen", f"${avg_amount:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Graphiques
        st.markdown('<div class="section-header">Analyses Statistiques</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribution des Transactions")
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Class'].value_counts().plot.pie(
                autopct='%1.1f%%', 
                labels=['Normales', 'Fraudes'],
                ax=ax
            )
            ax.set_ylabel('')
            st.pyplot(fig)

        with col2:
            st.subheader("Distribution des Montants")
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Amount'].hist(bins=50, ax=ax, alpha=0.7)
            ax.set_xlabel('Montant ($)')
            ax.set_ylabel('Nombre de Transactions')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    def display_real_time_detection(self):
        """Affiche l'outil de détection en temps réel"""
        st.markdown('<div class="section-header">Détection de Fraude en Temps Réel</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("Paramètres de la Transaction")

            # Option pour charger une transaction réelle
            if st.button("Charger une transaction réelle du dataset"):
                tx = self.sample_real_transaction()
                st.session_state['sample_tx'] = tx
                st.success("Transaction réelle chargée dans le formulaire")

            sample_tx = st.session_state.get('sample_tx', {})

            amount = st.number_input("Montant de la transaction ($)", 
                                min_value=0.0, 
                                max_value=1000000.0, 
                                value=float(sample_tx.get('Amount', 100.0)),
                                step=1.0)

            # si le sample fournit Time en secondes, on calcule hour
            default_hour = int(sample_tx.get('Time', 12*3600) / 3600) if sample_tx.get('Time') is not None else 12
            time_of_day = st.slider("Heure de la journée", 0, 23, default_hour)

            v_vals = {}
            for i in range(1, 6):
                v_vals[f'V{i}'] = st.slider(f"V{i} (Feature de comportement)", -10.0, 10.0, float(sample_tx.get(f'V{i}', 0.0)), 0.01)

        with col2:
            st.subheader("Résultat de l'Analyse")

            # Construire transaction_data complet (V1..V28)
            transaction_data = {
                'Time': time_of_day * 3600,
                'Amount': float(amount),
            }
            # V1..V5 depuis contrôles, V6..V28 soit depuis sample soit 0
            for i in range(1, 29):
                key = f'V{i}'
                if i <= 5:
                    transaction_data[key] = float(v_vals[key])
                else:
                    transaction_data[key] = float(sample_tx.get(key, 0.0))

            if st.button("Analyser la Transaction", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    result = self.predict_fraud(transaction_data)

                    if result['is_fraud']:
                        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
                        st.markdown("### TRANSACTION SUSPECTE DÉTECTÉE")
                        st.markdown("**Niveau de risque: Élevé**")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="normal-transaction">', unsafe_allow_html=True)
                        st.markdown("### TRANSACTION NORMALE")
                        st.markdown("**Niveau de risque: Faible**")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Métriques de prédiction
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.metric("Probabilité de Fraude", 
                                f"{result['probability_fraud']*100:.2f}%")

                    with col_b:
                        st.metric("Niveau de Confiance", 
                                f"{(1 - abs(result['probability_fraud'] - 0.5))*100:.1f}%")

                    # Barre de probabilité
                    fig, ax = plt.subplots(figsize=(12, 3))
                    ax.barh([''], [result['probability_fraud']*100], 
                        color='#e74c3c' if result['is_fraud'] else '#2ecc71', 
                        alpha=0.8, height=0.5)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Probabilité de Fraude (%)')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

    def display_model_info(self):
        """Affiche les informations du modèle"""
        st.markdown('<div class="section-header">Informations du Modèle</div>', unsafe_allow_html=True)

        if hasattr(self, 'metadata'):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Performances du Modèle")
                st.metric("Score AUC", f"{self.metadata.get('auc_score', 0):.4f}")
                st.metric("Type de Modèle", self.metadata.get('model_type', 'Inconnu'))

            with col2:
                st.markdown("#### Métadonnées")
                st.metric("Date d'Entraînement", 
                        self.metadata.get('timestamp', 'Inconnu'))
                st.metric("Nombre de Features", 
                        len(self.required_features))

            # Features importantes
            if self.required_features:
                st.markdown("#### Features attendues par le modèle")
                cols = st.columns(3)
                for i, feature in enumerate(self.required_features[:60]):
                    cols[i % 3].write(f"• {feature}")
        else:
            st.info("Mode démonstration - Données simulées")

    def display_data_explorer(self, df):
        """Explorateur de données"""
        st.markdown('<div class="section-header">Explorateur de Données</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Données Brutes", "Statistiques", "Visualisations"])

        with tab1:
            st.dataframe(df.head(1000), use_container_width=True)

        with tab2:
            st.subheader("Statistiques Descriptives")
            st.dataframe(df.describe(), use_container_width=True)

            st.subheader("Valeurs Manquantes")
            missing_data = df.isnull().sum()
            st.dataframe(missing_data[missing_data > 0], use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Matrice de Corrélation")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                corr_matrix = df[numeric_cols].corr()

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix.iloc[:10, :10], annot=True, cmap='RdYlBu', 
                        center=0, ax=ax)
                st.pyplot(fig)

            with col2:
                st.subheader("Distribution des Montants par Type")
                fig, ax = plt.subplots(figsize=(10, 6))

                for class_val, label in [(0, 'Normale'), (1, 'Fraude')]:
                    data = df[df['Class'] == class_val]['Amount']
                    ax.hist(data, bins=30, alpha=0.6, label=label)

                ax.legend()
                ax.set_xlabel('Montant ($)')
                ax.set_ylabel('Fréquence')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

    def run(self):
        """Lance le dashboard"""
        # Sidebar
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Page", 
                            ["Vue d'Ensemble", "Détection Temps Réel", "Explorateur de Données", "Info Modèle"],
                            index=0)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

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
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.markdown("""
        **Dashboard de Détection de Fraude**
        
        Fonctionnalités principales:
        - Visualisation des données de transactions
        - Détection en temps réel des fraudes
        - Exploration des patterns de fraude
        - Monitoring des performances
        """)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    dashboard = FraudDashboard()
    dashboard.run()
