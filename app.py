import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Charger le modèle
model = joblib.load("model_RF.pkl")
features = joblib.load("features.pkl")

st.title("Prédiction du Prix Immobilier en Tunisie 🏠💰")

st.write("Entrez les caractéristiques du bien pour prédire son prix.")

# Formulaire utilisateur
superficie = st.number_input("Superficie (m²)", min_value=20, max_value=600, value=100)
chambres = st.number_input("Nombre de chambres", min_value=1, max_value=10, value=2)
bains = st.number_input("Nombre de salles de bains", min_value=1, max_value=5, value=1)

# Liste des villes (les tiennes !)
# villes = [
#     'tunis', 'nabeul', 'ariana', 'sousse', 'ben arous', 'monastir', 
#     'mahdia', 'bizerte', 'manouba', 'sfax', 'gabes', 'kairouan', 'medenine'
# ]

# city = st.selectbox("Ville", villes)

# Préparer les features d'entrée
row = [superficie, bains, chambres]
row += [0] * (len(features) - len(row))

input_data = pd.DataFrame([row], columns=features)

# Activer la bonne ville en One-Hot Encoding
#col = "city_" + city
# if col in input_data.columns:
#     input_data[col] = 1

# Bouton prédire
if st.button("Prédire le prix"):
    prediction = model.predict(input_data)[0]
    st.success(f"Prix estimé : {prediction:,.0f} DT")
