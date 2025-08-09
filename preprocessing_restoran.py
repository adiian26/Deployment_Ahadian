import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# === 1. Load Dataset ===
df = pd.read_csv("restaurant_menu_optimization_data.csv")
print("Data loaded:", df.shape)

# === 2. Salin dataframe agar aman ===
data = df.copy()

# === 3. Encode target 'Profitability' (ordinal: Low < Medium < High) ===
data['Profitability'] = OrdinalEncoder(categories=[['Low', 'Medium', 'High']]).fit_transform(data[['Profitability']])

# === 4. One-Hot Encoding kolom 'MenuCategory' ===
data = pd.get_dummies(data, columns=['MenuCategory'], drop_first=True)

# === 5. Scaling kolom numerik 'Price' ===
scaler = StandardScaler()
data['Price'] = scaler.fit_transform(data[['Price']])

# === 6. Pisahkan fitur dan target ===
X = data.drop(columns=['Profitability', 'RestaurantID', 'MenuItem', 'Ingredients'])  # fitur
y = data['Profitability']  # target

# === 7. Split data (80% train, 20% test, stratifikasi target) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# === 8. Cek hasil ===
print("X_train shape:", X_train.shape)
print("y_train distribusi kelas:")
print(y_train.value_counts(normalize=True))
