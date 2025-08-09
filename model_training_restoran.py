import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("restaurant_menu_optimization_data.csv")

# Encoding target (Profitability)
profit_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['Profitability'] = df['Profitability'].map(profit_map)

# Pisahkan fitur dan target
X = df[['Price', 'MenuCategory']]
y = df['Profitability']

# One-hot encoding untuk MenuCategory
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), ['MenuCategory'])
], remainder='passthrough')

# Train-test split (stratifikasi)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model yang diuji
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

best_model_name = None
best_model_pipeline = None
best_accuracy = 0

# Fungsi untuk plot confusion matrix
def plot_cm(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Loop semua model
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n===== {name} =====")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Akurasi: {acc:.4f}")
    
    plot_cm(name, y_test, y_pred)
    
    # Cek apakah model ini lebih baik
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model_pipeline = pipeline

# Simpan model terbaik
joblib.dump(best_model_pipeline, "model.pkl")
print(f"\nModel terbaik adalah {best_model_name} dengan akurasi {best_accuracy:.4f}")
print("Model disimpan sebagai 'model.pkl'")
