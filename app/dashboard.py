import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_loader import DataLoader
from src.models.model_trainer import ModelTrainer

st.title("🎯 Dashboard de Détection de Fraude")

# Chargement des données
loader = DataLoader()
df = loader.load_raw_data()

# Sidebar
st.sidebar.header("Paramètres")
show_raw_data = st.sidebar.checkbox("Afficher les données brutes")

if show_raw_data:
    st.subheader("Données Brutes")
    st.dataframe(df.head(1000))

# Métriques principales
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", f"{len(df):,}")
with col2:
    st.metric("Transactions Frauduleuses", f"{df['Class'].sum():,}")
with col3:
    st.metric("Taux de Fraude", f"{df['Class'].sum()/len(df)*100:.4f}%")

# Visualisations
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df['Class'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0])
ax[0].set_title('Distribution des Classes')
df['Amount'].hist(bins=50, ax=ax[1])
ax[1].set_title('Distribution des Montants')
st.pyplot(fig)