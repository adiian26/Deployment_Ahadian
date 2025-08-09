import streamlit as st
import pandas as pd
import joblib

# Judul
st.title("Prediksi Profitability Menu Restoran")

# Load model pipeline (sudah termasuk preprocessing)
model = joblib.load("pipeline_model.pkl")  # ganti sesuai nama file pipeline kamu

# Input dari user
price = st.number_input("Harga Menu", min_value=0.0, step=0.1)
menu_category = st.selectbox("Kategori Menu", ["Appetizers", "Main Course", "Desserts", "Beverages"])

# Prediksi
if st.button("Prediksi Profitability"):
    # Masukkan data mentah, pipeline akan preprocessing otomatis
    X = pd.DataFrame([[price, menu_category]], columns=["Price", "MenuCategory"])
    prediction = model.predict(X)[0]
    st.success(f"Prediksi Profitability: {prediction}")
