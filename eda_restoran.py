import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Baca dataset
df = pd.read_csv(r"D:\Semester 4\MPML\UAS\restaurant_menu_optimization_data.csv")

# 1. Preview data
print("5 Data Teratas:")
print(df.head())

# 2. Info dataset
print("\nInfo Dataset:")
print(df.info())

# 3. Cek missing value
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Histogram untuk harga (Price saja yang numerik)
df[['Price']].hist(bins=20, figsize=(6, 4))
plt.suptitle('Distribusi Harga Menu')
plt.show()

# 5. Boxplot untuk harga
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Price'])
plt.title("Boxplot Harga Menu")
plt.show()

# 6. Distribusi target Profitability
sns.countplot(x='Profitability', data=df, order=['Low', 'Medium', 'High'])
plt.title("Distribusi Kelas Target (Profitability)")
plt.show()

# 7. Bar plot rata-rata harga per kategori menu
plt.figure(figsize=(8, 5))
sns.barplot(x='MenuCategory', y='Price', data=df, ci=None)
plt.title("Rata-rata Harga per Kategori Menu")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
