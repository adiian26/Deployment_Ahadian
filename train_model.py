import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# === KONFIGURASI ===
MODEL_FILENAME = "random_forest.pkl"
DATASET_FILENAME = "restaurant_menu_optimization_dataset.csv"

# === 1. Load Dataset ===
df = pd.read_csv(DATASET_FILENAME)

# === 2. Encoding Target ===
oe = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['Profitability'] = oe.fit_transform(df[['Profitability']])

# === 3. Drop kolom yang tidak diperlukan ===
# Misalnya RestaurantID atau MenuItem yang tidak dipakai
if 'RestaurantID' in df.columns:
    df = df.drop(columns=['RestaurantID'])
if 'MenuItem' in df.columns:
    df = df.drop(columns=['MenuItem'])
if 'Ingredients' in df.columns:
    df = df.drop(columns=['Ingredients'])

# === 4. One-Hot Encoding kolom kategorikal nominal ===
cat_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Profitability']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# === 5. Scaling fitur numerik ===
scaler = StandardScaler()
num_cols = [col for col in df.columns if col != 'Profitability']
df[num_cols] = scaler.fit_transform(df[num_cols])

# === 6. Split data ===
X = df.drop('Profitability', axis=1)
y = df['Profitability']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# === 7. Train Model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === 8. Save Model ===
joblib.dump(model, MODEL_FILENAME)
print(f"Model disimpan sebagai {MODEL_FILENAME}")

# Pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, MODEL_FILENAME)