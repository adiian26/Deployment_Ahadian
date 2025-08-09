import streamlit as st
import pandas as pd
import joblib

# Judul
st.title("Prediksi Profitability Menu Restoran")

# Load model
model = joblib.load("random_forest.pkl")

# Ambil daftar fitur yang model harapkan
try:
    feature_names = model.feature_names_in_
except AttributeError:
    st.error("Model tidak memiliki atribut feature_names_in_. Pastikan model dilatih dengan DataFrame.")
    st.stop()

# Input dari user
price = st.number_input("Harga Menu", min_value=0.0, step=0.1)
menu_category = st.selectbox("Kategori Menu", ["Appetizers", "Main Course", "Desserts", "Beverages"])

# One-hot encoding manual sesuai feature_names
row = {col: 0 for col in feature_names}  # isi semua kolom awalnya 0
row["Price"] = price
if f"MenuCategory_{menu_category}" in row:
    row[f"MenuCategory_{menu_category}"] = 1

# Prediksi
if st.button("Prediksi Profitability"):
    X = pd.DataFrame([row])
    prediction = model.predict(X)[0]
    st.success(f"Prediksi Profitability: {prediction}")
