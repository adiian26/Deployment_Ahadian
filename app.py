import streamlit as st
import pandas as pd
import joblib

# Judul
st.title("Prediksi Profitability Menu Restoran")

# Load model (pastikan kamu sudah simpan model pemenang, misal random_forest.pkl)
model = joblib.load("random_forest.pkl")

# Input dari user
price = st.number_input("Harga Menu", min_value=0.0, step=0.1)
menu_category = st.selectbox("Kategori Menu", ["Appetizers", "Main Course", "Desserts", "Beverages"])

# Konversi kategori ke angka (contoh)
category_map = {
    "Appetizers": 0,
    "Main Course": 1,
    "Desserts": 2,
    "Beverages": 3
}
menu_category_encoded = category_map[menu_category]

# Prediksi
if st.button("Prediksi Profitability"):
    X = pd.DataFrame([[price, menu_category_encoded]], columns=["Price", "MenuCategory"])
    prediction = model.predict(X)[0]
    st.success(f"Prediksi Profitability: {prediction}")
